"""Independent CPU references for explicit synthetic functional-inference panels.

No primary admission or production estimators are imported here. Task reductions
use Python sums / Torch FP64 and weighted resample counts; OLS uses full SymPy
normal equations, not the production residualized Fraction implementation.
"""

import math
from collections import defaultdict

import numpy as np
import sympy as sp
import torch

FEATURES = (
    "margin_helpful_mass_share",
    "margin_helpful_attribution_participation_ratio",
    "margin_degrading_attribution_mass",
    "random_50pct_relative_shortlist_ndcg",
)
CONTRASTS = (("muon", "adamw"), ("normuon", "adamw"), ("normuon", "muon"))


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-14), (actual, expected)


def task_vectors(rows, step, *, seed=None):
    groups = defaultdict(list)
    for row in rows:
        if row["step"] == step and row["rotation_seed"] == seed:
            groups[row["task"], row["optimizer"]].append(row)
    tasks = sorted({task for task, _ in groups})
    assert len(tasks) == 14 and all(len(rows) == 4 for rows in groups.values())
    columns, vectors = [], []
    for feature in FEATURES[:3]:
        for treatment, baseline in CONTRASTS:
            columns.append((feature, treatment, baseline))
            vectors.append(
                [
                    math.fsum(r[feature] for r in groups[task, treatment]) / 4
                    - math.fsum(r[feature] for r in groups[task, baseline]) / 4
                    for task in tasks
                ]
            )
    return columns, torch.tensor(vectors, dtype=torch.float64).T


def task_inference(inputs, output, decisions):
    protocol, raw = inputs["protocol"], inputs["tables"]
    step = protocol["inputs"]["checkpoint_stages"][-1]
    columns, matrix = task_vectors(raw["task_summary"], step)
    points = matrix.mean(dim=0)
    se = matrix.std(dim=0, correction=1) / math.sqrt(14)
    assert (se > 0).all().item()
    seed = protocol["corrected_primary_comparison"]["bootstrap_seed"]
    samples = protocol["corrected_primary_comparison"]["bootstrap_samples"]
    # Same declared RNG sample, independently converted to multiplicity weights.
    draws = np.random.default_rng(seed).integers(0, 14, (samples, 14))
    weights = torch.zeros((samples, 14), dtype=torch.float64)
    weights.scatter_add_(1, torch.from_numpy(draws), torch.ones((samples, 14), dtype=torch.float64))
    boot = weights @ matrix / 14
    maximum = ((boot - points) / se).abs().amax(dim=1)
    critical = torch.quantile(maximum, 0.95, interpolation="linear").item()
    indexed, comparisons = {}, []
    for j, key in enumerate(columns):
        row = next(
            r
            for r in output["primary_contrasts"]
            if (r["feature"], r["treatment"], r["baseline"]) == key
        )
        lower, upper = (points[j] - critical * se[j]).item(), (points[j] + critical * se[j]).item()
        for actual, expected in (
            (row["mean_difference"], points[j].item()),
            (row["across_task_standard_error"], se[j].item()),
            (row["max_t_critical_value"], critical),
            (row["simultaneous_ci_95_lower"], lower),
            (row["simultaneous_ci_95_upper"], upper),
        ):
            close(actual, expected)
        assert row["decision"] == (
            "positive" if lower > 0 else "negative" if upper < 0 else "inconclusive"
        )
        indexed[key] = lower > 0 if j // 3 < 2 else upper < 0
        comparisons.append(
            {
                "feature": key[0],
                "treatment": key[1],
                "baseline": key[2],
                "mean": points[j].item(),
                "se": se[j].item(),
                "lower": lower,
                "upper": upper,
                "passed": True,
            }
        )
    constructive = {
        optimizer: all(indexed[f, optimizer, "adamw"] for f in FEATURES[:3])
        for optimizer in ("muon", "normuon")
    }
    assert constructive == decisions["constructive_native_coordinate_use"]
    rotated = []
    for seed in protocol["rotation_control"]["seeds"]:
        keys, values = task_vectors(raw["rotation_summary"], step, seed=seed)
        for j, key in enumerate(keys):
            row = next(
                r
                for r in output["rotation_contrasts"]
                if (r["rotation_seed"], r["feature"], r["treatment"], r["baseline"]) == (seed, *key)
            )
            mean, error = (
                values[:, j].mean().item(),
                values[:, j].std(correction=1).item() / math.sqrt(14),
            )
            close(row["mean_difference"], mean)
            close(row["across_task_standard_error"], error)
            rotated.append(
                {
                    "seed": seed,
                    "feature": key[0],
                    "treatment": key[1],
                    "baseline": key[2],
                    "mean": mean,
                    "passed": True,
                }
            )
    for optimizer in ("muon", "normuon"):
        selected = [r for r in rotated if r["treatment"] == optimizer and r["baseline"] == "adamw"]
        stable = all((1 if r["feature"] in FEATURES[:2] else -1) * r["mean"] > 0 for r in selected)
        assert stable == decisions["direction_stable_across_tested_rotations"][optimizer]
    for point in output["figure_points"]:
        is_removal = point["feature"] == "relative_shortlist_ndcg"
        if is_removal and point["x"] == 0:
            assert point["mean"] == 1
            continue
        selected = [
            r
            for r in raw["random_removal" if is_removal else "task_summary"]
            if r["optimizer"] == point["optimizer"]
            and r["step"] == point["step"]
            and (not is_removal or r["removed_fraction"] == point["x"] / 100)
        ]
        expected = math.fsum(r[point["feature"]] for r in selected) / len(selected)
        close(point["mean"], expected)
    return {
        "method": "Python task means; Torch FP64 multiplicity-weighted shared resampling",
        "contrasts": comparisons,
        "samples": samples,
        "common_critical": critical,
        "rotations": rotated,
        "figure_points_checked": len(output["figure_points"]),
    }


def ranks(vector):
    values = list(vector)
    return sp.Matrix(
        [
            sum(v < x for v in values) + (sum(v == x for v in values) + 1) / sp.Integer(2)
            for x in values
        ]
    )


def correlation(a, b):
    n = len(a)
    a, b = a - sp.ones(n, 1) * sum(a) / n, b - sp.ones(n, 1) * sum(b) / n
    return float((a.dot(b) / sp.sqrt(a.dot(a) * b.dot(b))).evalf(80))


def exact_predictions(tables):
    rows = tables["bridge_rows"]
    baseline = sp.Matrix(
        [
            [
                1,
                int(r["optimizer"] == "muon"),
                int(r["optimizer"] == "normuon"),
                *[int(r["stage"] == s) for s in range(2, 6)],
                sp.Rational(r["centered_log10_learning_rate"]),
            ]
            for r in rows
        ]
    )
    target = sp.Matrix([sp.Rational(r["mean_ndcg_at_10"]) for r in rows])
    saved = {(r["feature"], r["run_id"], r["stage"]): r for r in tables["held_out_predictions"]}
    predictions, systems = 0, []
    for fold in range(1, 5):
        train = [i for i, r in enumerate(rows) if r["dose_index"] != fold]
        test = [i for i, r in enumerate(rows) if r["dose_index"] == fold]
        b = baseline.extract(train, range(8))
        y = target.extract(train, [0])
        bp = baseline.extract(test, range(8)) * ((b.T * b).inv(method="DM") * b.T * y)
        for feature in FEATURES:
            design = baseline.row_join(sp.Matrix([sp.Rational(r[feature]) for r in rows]))
            training = design.extract(train, range(9))
            assert training.rank() == 9  # Explicit nondegenerate fixture; null cases are separate.
            predicted = design.extract(test, range(9)) * (
                (training.T * training).inv(method="DM") * training.T * y
            )
            for j, i in enumerate(test):
                row = saved[feature, rows[i]["run_id"], rows[i]["stage"]]
                assert predicted[j] == sp.Rational(row["feature_prediction_exact"])
                assert bp[j] == sp.Rational(row["baseline_prediction_exact"])
                predictions += 1
            yr = target.extract(test, [0])
            errors, base_errors = yr - predicted, yr - bp
            am, bm = errors.dot(errors) / 15, base_errors.dot(base_errors) / 15
            stored = next(
                r
                for r in tables["leave_dose_fold_metrics"]
                if r["feature"] == feature and r["held_out_dose_index"] == fold
            )
            assert sp.Rational(stored["feature_mse_exact"]) == am
            assert sp.Rational(stored["baseline_mse_exact"]) == bm
            assert sp.Rational(stored["mse_reduction_exact"]) == bm - am
            assert stored["feature_improves"] == bool(bm > am)
            systems.append({"feature": feature, "fold": fold, "predictions": 15, "passed": True})
    projection = baseline * (baseline.T * baseline).inv(method="DM") * baseline.T
    residual_y = target - projection * target
    associations = []
    for feature in FEATURES:
        vector = sp.Matrix([sp.Rational(r[feature]) for r in rows])
        residual_x = vector - projection * vector
        pearson, spearman = (
            correlation(residual_x, residual_y),
            correlation(ranks(residual_x), ranks(residual_y)),
        )
        stored = next(r for r in tables["residual_associations"] if r["feature"] == feature)
        close(stored["pearson_residual_association"], pearson)
        close(stored["spearman_residual_association"], spearman)
        associations.append(
            {"feature": feature, "pearson": pearson, "spearman": spearman, "passed": True}
        )
        folds = [r for r in tables["leave_dose_fold_metrics"] if r["feature"] == feature]
        delta = sum(sp.Rational(r["mse_reduction_exact"]) for r in folds) / 4
        summary = next(r for r in tables["feature_prediction_summary"] if r["feature"] == feature)
        assert sp.Rational(summary["pooled_mse_reduction_exact"]) == delta
        assert summary["predictively_useful"] == (
            bool(delta > 0) and sum(r["feature_improves"] for r in folds) >= 3
        )
    return {
        "method": "SymPy rational full 8/9-column normal equations",
        "exact_predictions": predictions,
        "exact_fold_mse_comparisons": len(systems),
        "exact_pooled_decisions": len(FEATURES),
        "systems": systems,
        "residual_associations": associations,
    }
