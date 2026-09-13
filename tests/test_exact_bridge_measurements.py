import copy
import json
import math
import runpy
from pathlib import Path

import numpy as np
import pytest

from embed_optim import exact_bridge_measurements as measurements
from embed_optim.bridge_named_features import evaluate_named
from embed_optim.bridge_numerics import evaluate
from embed_optim.corrected_retrieval_bridge import FEATURES as ORIGINAL_FEATURES
from embed_optim.corrected_retrieval_bridge import assemble_bridge_rows
from embed_optim.dimension_publication import BRIDGE_FEATURES as DIMENSION_FEATURES
from embed_optim.dimension_publication import _evaluate_bridge as old_dimension_bridge
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_outcomes import recipe_views

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"


@pytest.fixture
def primary():
    return PrimaryV3Contract.load(ROOT / "configs/dense_primary_v3_protocol.json", ROOT, CANDIDATE)


def fixture(primary):
    generator = runpy.run_path(str(ROOT / "tests/test_primary_v3_bridge.py"))["inputs"]
    configs, steps = recipe_views(primary), primary.payload["checkpoint_steps"]
    cp, pairs, scores = generator(primary)
    original = assemble_bridge_rows(cp, pairs, scores, configs)
    exact = []
    for index, source in enumerate(cp):
        row = {key: source[key] for key in ("run_id", "optimizer", "learning_rate", "stage")}
        row.update(
            step=steps[row["stage"] - 1], progress_fraction=row["stage"] / 5, hidden_parameters=100
        )
        for kind in ("saved_segment", "cumulative"):
            prefix = f"exact_{kind}"
            row[f"{prefix}_nonzero_parameters"] = row[f"{prefix}_resolved_parameters"] = 100
            row[f"{prefix}_nonzero_parameter_fraction"] = 1.0
            for status in ("zero", "insufficient_signal_rank", "unresolved_boundary"):
                row[f"{prefix}_{status}_parameters"] = 0
        for fi, (column, _, _) in enumerate(measurements.CHECKPOINT_FEATURES.values()):
            row[column] = 0.2 + 0.1 * math.sin((index + 1) * (fi + 2) * 0.137)
        exact.append(row)
    by_id = {r.run_id: r for r in configs}
    for row in pairs:
        row["first_learning_rate"] = by_id[row["first_run_id"]].optimizer.lr
        row["second_learning_rate"] = by_id[row["second_run_id"]].optimizer.lr
        row["step"] = steps[row["stage"] - 1]
        row["progress_fraction"] = row["stage"] / 5
        row["defined_parameters"] = 100
        row["defined_parameter_fraction"] = 1.0
        row["undefined_zero_parameters"] = 0
        row["undefined_signal_rank_parameters"] = row["undefined_boundary_parameters"] = 0
        row["all_nonzero_pair_subspaces_resolved"] = True
        row["resolved_only_mean_subspace_overlap"] = row["mean_subspace_overlap"]
    return original, exact, pairs, configs, steps


def test_named_family_retains_every_original_numeric_record(primary):
    original, *_ = fixture(primary)
    assert evaluate_named(original, ORIGINAL_FEATURES) == evaluate(original)


def test_exact_family_is_additional_and_full_coverage(primary):
    data = fixture(primary)
    rows, comparisons, coverage = measurements.assemble_exact_panel(*data)
    tables, diagnostic = evaluate_named(rows, measurements.FEATURES)
    before = {(r["run_id"], r["stage"]): r for r in data[0]}
    assert len(rows) == len(coverage) == 60 and len(comparisons) == 300
    assert len(tables["held_out_predictions"]) == 300
    assert len(tables["feature_prediction_summary"]) == 5
    assert len(tables["leave_dose_fold_metrics"]) == 20
    assert len(diagnostic["feature_designs"]) == 25
    for row in rows:
        assert {k: row[k] for k in before[(row["run_id"], row["stage"])]} == before[
            (row["run_id"], row["stage"])
        ]
    assert {r["relationship"] for r in comparisons} == {
        "full_spectrum_and_nonzero_denominator",
        "exact_rank_and_nonzero_denominator",
        "exact_projectors_complete_comparator_population",
    }


def test_every_adamw_comparator_is_retained_and_self_excluded(primary):
    _, _, pairs, configs, steps = fixture(primary)
    values = measurements.pair_features(pairs, configs, steps)
    for config in configs:
        record = values[(config.run_id, 3)]
        for feature, (kind, _) in measurements.PAIR_FEATURES.items():
            others = [
                r.run_id
                for r in configs
                if r.optimizer.name == "adamw" and r.run_id != config.run_id
            ]
            expected = [
                r["mean_subspace_overlap"]
                for r in pairs
                if r["stage"] == 3
                and r["displacement_kind"] == kind
                and config.run_id in (r["first_run_id"], r["second_run_id"])
                and any(other in (r["first_run_id"], r["second_run_id"]) for other in others)
            ]
            assert record["values"][feature] == math.fsum(expected) / len(expected)
            assert record["coverage"][feature]["comparators"] == sorted(others)
            assert len(expected) == (3 if config.optimizer.name == "adamw" else 4)


def test_one_undefined_pair_does_not_use_conditional_or_available_mean(primary):
    data = fixture(primary)
    target = next(
        r for r in data[2] if r["first_optimizer"] == "adamw" and r["second_optimizer"] == "muon"
    )
    target.update(
        mean_subspace_overlap=None,
        defined_parameters=90,
        defined_parameter_fraction=0.9,
        undefined_boundary_parameters=10,
        all_nonzero_pair_subspaces_resolved=False,
    )
    assert target["resolved_only_mean_subspace_overlap"] is not None
    rows, comparisons, _ = measurements.assemble_exact_panel(*data)
    feature = "exact_mean_saved_segment_overlap_to_adamw"
    focal = next(
        r for r in rows if r["run_id"] == target["second_run_id"] and r["stage"] == target["stage"]
    )
    assert focal[feature] is None
    tables, _ = evaluate_named(rows, measurements.FEATURES)
    summary = next(r for r in tables["feature_prediction_summary"] if r["feature"] == feature)
    assert summary["predictively_useful"] is None and summary["folds_defined"] == 0
    assert len(tables["held_out_predictions"]) == 300 and len(comparisons) == 300


def test_zero_checkpoint_preserves_undefined_rank_and_denominator(primary):
    _, rows, _, configs, steps = fixture(primary)
    row = rows[0]
    for kind in ("saved_segment", "cumulative"):
        prefix = f"exact_{kind}"
        row[f"{prefix}_nonzero_parameters"] = row[f"{prefix}_resolved_parameters"] = 0
        row[f"{prefix}_zero_parameters"] = 100
        row[f"{prefix}_nonzero_parameter_fraction"] = 0.0
    for column, _, _ in measurements.CHECKPOINT_FEATURES.values():
        row[column] = None
    mapped = measurements.checkpoint_features(rows, configs, steps)
    assert all(value is None for value in mapped[(row["run_id"], row["stage"])]["values"].values())


@pytest.mark.parametrize(
    "kind",
    [
        "missing_row",
        "duplicate",
        "stage",
        "denominator",
        "zero_imputation",
        "range",
        "pair_missing",
        "pair_relabel",
        "pair_false_defined",
        "pair_drop_rate",
    ],
)
def test_incomplete_inconsistent_or_undefined_relabelled_measurements_refuse(primary, kind):
    data = fixture(primary)
    c, p = data[1], data[2]
    if kind == "missing_row":
        c.pop()
    elif kind == "duplicate":
        c[-1] = copy.deepcopy(c[0])
    elif kind == "stage":
        c[0]["step"] += 1
    elif kind == "denominator":
        c[0]["exact_saved_segment_nonzero_parameters"] = 99
    elif kind == "zero_imputation":
        c[0][next(iter(measurements.CHECKPOINT_FEATURES.values()))[0]] = None
    elif kind == "range":
        c[0][next(iter(measurements.CHECKPOINT_FEATURES.values()))[0]] = 1.1
    elif kind == "pair_missing":
        p.pop()
    elif kind == "pair_relabel":
        p[0]["first_optimizer"] = "muon"
    elif kind == "pair_false_defined":
        p[0]["mean_subspace_overlap"] = None
    else:
        p[:] = [r for r in p if r["first_run_id"] != data[3][0].run_id]
    with pytest.raises((ValueError, KeyError)):
        measurements.assemble_exact_panel(*data)


@pytest.mark.parametrize("names", [[], [""], ["stage"], ["new", "new"], [123]])
def test_named_feature_ids_cannot_be_missing_reserved_or_duplicated(primary, names):
    with pytest.raises(ValueError):
        evaluate_named(fixture(primary)[0], names)


def test_explicit_missing_value_is_not_missing_column(primary):
    rows = fixture(primary)[0]
    with pytest.raises(ValueError, match="Missing named feature"):
        evaluate_named(rows, ["separately_named_feature"])
    for row in rows:
        row["separately_named_feature"] = 0.5
    rows[0]["separately_named_feature"] = float("nan")
    with pytest.raises(ValueError):
        evaluate_named(rows, ["separately_named_feature"])
    rows[0]["separately_named_feature"] = None
    tables, _ = evaluate_named(rows, ["separately_named_feature"])
    assert all(
        r["status"] == "undefined_feature_measurement" for r in tables["leave_dose_fold_metrics"]
    )


def test_dimension_numeric_counterexample_without_relabelling_original_slots():
    path = ROOT / "reports/engineering-archive/dense-v3-exact-geometry-v1/bridge-null-feature.json"
    rows = copy.deepcopy(json.loads(path.read_text())["cases"][1]["rows"])
    original_values = [[r[name] for name in ORIGINAL_FEATURES] for r in rows]
    for row in rows:
        row.update({feature: row["stage"] / 8 for feature in DIMENSION_FEATURES})
    _, old = old_dimension_bridge(rows)
    assert all(r["predictively_useful"] is True for r in old)
    tables, _ = evaluate_named(rows, DIMENSION_FEATURES)
    assert all(
        r["predictively_useful"] is False and r["pooled_mse_reduction_exact"] == "0"
        for r in tables["feature_prediction_summary"]
    )
    assert {r["feature"] for r in tables["feature_prediction_summary"]} == set(DIMENSION_FEATURES)
    assert [[r[name] for name in ORIGINAL_FEATURES] for r in rows] == original_values


def test_named_results_are_order_independent(primary):
    rows, _, _ = measurements.assemble_exact_panel(*fixture(primary))
    expected = evaluate_named(rows, measurements.FEATURES)
    shuffled = [rows[i] for i in np.random.default_rng(20260906).permutation(60)]
    assert evaluate_named(shuffled, measurements.FEATURES) == expected
