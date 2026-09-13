"""Explicit exact-arithmetic/numerical-identifiability amendment for the nine-feature bridge."""

from __future__ import annotations

import math

import numpy as np

from .bridge_exact_arithmetic import (
    Baseline,
    average_ranks,
    correlation,
    design_health,
    displayed,
    dot,
    mse,
    rational,
    rmse_reduction,
    root,
)
from .corrected_retrieval_bridge import FEATURES, OPTIMIZERS, _baseline_design

NUMERICAL_POLICY = {
    "inputs": "Every finite binary64 bridge value is its exact rational value; no decimal rounding or denominator approximation",
    "ols": "Exact rational eight-column baseline and one-feature Frisch-Waugh-Lovell decomposition; no regularization",
    "standardization": "Training-only affine feature standardization is algebraically eliminated using the intercept; retain its exact mean/variance and display scale",
    "support": "Compare exact rational MSE; strictly lower pooled MSE and strictly lower in at least three of four folds; identical mathematical RMSE ordering",
    "redundancy": "Exactly zero training residual yields baseline-equivalent predictions only if that same relation extends to every held-out row; otherwise augmented prediction is unidentified",
    "resolution": "Nonzero added direction must have full rank under eps64 * max(rows,9) * leading singular value on the training-standardized augmented design",
    "undefined": "Any unidentified/unresolved fold makes the full pooled augmented comparison and usefulness flag undefined; retain all rows and never pool only available folds",
    "association": "Exact rational baseline residuals and average-rank ties; zero residual or numerically unresolved nonzero feature/outcome direction has undefined association",
    "display": "80-digit Decimal square roots and cancellation-safe RMSE subtraction converted to finite floats; display precision never decides improvement",
    "boundary": "Numerical resolution is a declared input-stability convention, not statistical significance or causal identification; tiny exact improvements remain tiny",
}


def validate_panel(rows):
    if len(rows) != 60:
        raise ValueError("Require the complete 60-row bridge panel")
    seen, runs = set(), {}
    for row in rows:
        run_id, stage, dose = row.get("run_id"), row.get("stage"), row.get("dose_index")
        if (
            not isinstance(run_id, str)
            or not run_id
            or type(stage) is not int
            or stage not in range(1, 6)
            or type(dose) is not int
            or dose not in range(1, 5)
            or row.get("optimizer") not in OPTIMIZERS
            or (run_id, stage) in seen
        ):
            raise ValueError("Invalid or duplicated bridge panel identity")
        seen.add((run_id, stage))
        rate = float(rational(row["learning_rate"]))
        if rate <= 0 or not 0 <= rational(row["mean_ndcg_at_10"]) <= 1:
            raise ValueError("Invalid learning rate or bounded retrieval outcome")
        for key in (*FEATURES, "centered_log10_learning_rate"):
            rational(row[key])
        identity = (row["optimizer"], dose, rate, row["centered_log10_learning_rate"])
        if run_id in runs and runs[run_id] != identity:
            raise ValueError("Bridge run identity differs across stages")
        runs[run_id] = identity
    if len(runs) != 12 or seen != {(r, s) for r in runs for s in range(1, 6)}:
        raise ValueError("Require every stage of twelve distinct bridge runs")
    for optimizer in OPTIMIZERS:
        members = sorted((r for r in runs.values() if r[0] == optimizer), key=lambda r: r[2])
        if len(members) != 4 or len({r[2] for r in members}) != 4:
            raise ValueError("Require four distinct rates per optimizer")
        mean = float(np.mean([math.log10(r[2]) for r in members]))
        for dose, row in enumerate(members, 1):
            if row[1] != dose or row[3] != math.log10(row[2]) - mean:
                raise ValueError("Dose ordering or centered learning-rate feature differs")
    return sorted(
        rows, key=lambda r: (OPTIMIZERS.index(r["optimizer"]), r["dose_index"], r["stage"])
    )


def error_columns(observed, baseline, feature):
    base_mse = mse(observed, baseline)
    added_mse = None if feature is None else mse(observed, feature)
    return {
        "baseline_rmse": root(base_mse),
        "feature_rmse": None if added_mse is None else root(added_mse),
        "rmse_reduction": None if added_mse is None else rmse_reduction(base_mse, added_mse),
        "feature_improves": None if added_mse is None else base_mse > added_mse,
        "baseline_mse_exact": str(base_mse),
        "feature_mse_exact": None if added_mse is None else str(added_mse),
        "mse_reduction_exact": None if added_mse is None else str(base_mse - added_mse),
    }


def evaluate(rows):
    """Pure complete-panel calculation, not admission of scientific primary artifacts."""
    rows = validate_panel(rows)
    design = [[rational(x) for x in row] for row in _baseline_design(rows).tolist()]
    outcome = [rational(row["mean_ndcg_at_10"]) for row in rows]
    contexts = {
        fold: Baseline(design, [i for i, r in enumerate(rows) if r["dose_index"] != fold])
        for fold in range(1, 5)
    }
    full = Baseline(design, range(60))
    _, outcome_residual = full.project(outcome)
    outcome_energy = dot(outcome_residual, outcome_residual)
    outcome_health = design_health(full.numeric, outcome)
    folds, summaries, associations, predictions, diagnostics = [], [], [], [], []
    for feature in FEATURES:
        values = [rational(row[feature]) for row in rows]
        base_all, added_all = [None] * 60, [None] * 60
        valid, improved = 0, 0
        for fold, context in contexts.items():
            result = context.feature(values, outcome)
            test = [i for i, row in enumerate(rows) if row["dose_index"] == fold]
            baseline = [result["baseline_prediction"][i] for i in test]
            added = result["feature_prediction"]
            selected = None if added is None else [added[i] for i in test]
            errors = error_columns([outcome[i] for i in test], baseline, selected)
            valid += int(selected is not None)
            improved += int(errors["feature_improves"] is True)
            health = result["health"]
            folds.append(
                {
                    "feature": feature,
                    "held_out_dose_index": fold,
                    "train_rows": 45,
                    "test_rows": 15,
                    "status": result["status"],
                    "training_feature_mean": health["training_feature_mean"],
                    "training_feature_scale": health["training_feature_scale"],
                    "augmented_numeric_rank": health["augmented_numeric_rank"],
                    "exact_residual_energy": result["exact_residual_energy"],
                    **errors,
                }
            )
            diagnostics.append({"feature": feature, "fold": fold, **health})
            for i in test:
                base_all[i] = result["baseline_prediction"][i]
                added_all[i] = None if added is None else added[i]
                predictions.append(
                    {
                        "feature": feature,
                        "run_id": rows[i]["run_id"],
                        "stage": rows[i]["stage"],
                        "held_out_dose_index": fold,
                        "status": result["status"],
                        "observed": float(outcome[i]),
                        "baseline_prediction": displayed(base_all[i]),
                        "feature_prediction": None if added is None else displayed(added[i]),
                        "baseline_prediction_exact": str(base_all[i]),
                        "feature_prediction_exact": None if added is None else str(added[i]),
                    }
                )
        pooled = error_columns(outcome, base_all, added_all if valid == 4 else None)
        summaries.append(
            {
                "feature": feature,
                "pooled_rows": 60,
                "pooled_baseline_rmse": pooled["baseline_rmse"],
                "pooled_feature_rmse": pooled["feature_rmse"],
                "pooled_rmse_reduction": pooled["rmse_reduction"],
                "pooled_baseline_mse_exact": pooled["baseline_mse_exact"],
                "pooled_feature_mse_exact": pooled["feature_mse_exact"],
                "pooled_mse_reduction_exact": pooled["mse_reduction_exact"],
                "folds_improved": improved,
                "folds_total": 4,
                "folds_defined": valid,
                "status": "complete" if valid == 4 else "undefined_fold_present",
                "predictively_useful": None
                if valid != 4
                else pooled["feature_improves"] and improved >= 3,
            }
        )
        _, residual = full.project(values)
        feature_energy = dot(residual, residual)
        health = design_health(full.numeric, values)
        if feature_energy == 0:
            status = "feature_in_baseline_span"
        elif outcome_energy == 0:
            status = "outcome_in_baseline_span"
        elif health["augmented_numeric_rank"] != 9:
            status = "unresolved_feature_residual"
        elif outcome_health["augmented_numeric_rank"] != 9:
            status = "unresolved_outcome_residual"
        else:
            status = "resolved"
        associations.append(
            {
                "feature": feature,
                "rows": 60,
                "status": status,
                "feature_residual_energy_exact": str(feature_energy),
                "outcome_residual_energy_exact": str(outcome_energy),
                "pearson_residual_association": correlation(residual, outcome_residual)
                if status == "resolved"
                else None,
                "spearman_residual_association": correlation(
                    average_ranks(residual), average_ranks(outcome_residual)
                )
                if status == "resolved"
                else None,
            }
        )
        diagnostics.append({"feature": feature, "fold": "all", **health})
    return {
        "bridge_rows": rows,
        "leave_dose_fold_metrics": folds,
        "feature_prediction_summary": summaries,
        "residual_associations": associations,
        "held_out_predictions": predictions,
    }, {"feature_designs": diagnostics, "outcome_design": outcome_health}
