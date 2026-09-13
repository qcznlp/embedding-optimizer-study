"""Boundary controls and independent full-system oracles for exact bridge OLS."""

import copy
import json
import math
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
from scipy.linalg import lstsq

from embed_optim.bridge_exact_arithmetic import (
    Baseline,
    average_ranks,
    correlation,
    inverse,
    rational,
    rmse_reduction,
)
from embed_optim.bridge_numerics import evaluate, validate_panel
from embed_optim.corrected_retrieval_bridge import FEATURES, _baseline_design

ROOT = Path(__file__).resolve().parents[1]


def panel(kind="signal"):
    rows = []
    for oi, optimizer in enumerate(("adamw", "muon", "normuon")):
        rates = [1e-6, 3e-6, 1e-5, 3e-5] if oi == 0 else [1e-4, 3e-4, 1e-3, 3e-3]
        mean = float(np.mean([math.log10(x) for x in rates]))
        for dose, rate in enumerate(rates, 1):
            for stage in range(1, 6):
                signal = Q((2 * dose - 5) * (stage - 3), 16)
                value = {
                    "signal": signal,
                    "redundant": Q(stage, 8),
                    "constant": Q(1, 2),
                    "train_only": Q(stage, 8) + (signal if dose == 1 else 0),
                    "near_null": Q(stage, 8) + signal / 2**47,
                }[kind]
                outcome = Q(1, 2) + Q(stage, 128) + Q(oi, 64) + signal / 8
                rows.append(
                    {
                        "run_id": f"synthetic-{optimizer}-{dose}",
                        "optimizer": optimizer,
                        "learning_rate": rate,
                        "dose_index": dose,
                        "centered_log10_learning_rate": math.log10(rate) - mean,
                        "stage": stage,
                        "progress_fraction": stage / 5,
                        "mean_ndcg_at_10": float(outcome),
                        **{feature: float(value) for feature in FEATURES},
                    }
                )
    return rows


def qmatrix(values):
    return [[rational(x) for x in row] for row in values]


@pytest.mark.parametrize("case_index", [0, 1])
def test_both_preserved_counterexamples_have_exactly_zero_gain(case_index):
    source = (
        ROOT / "reports/engineering-archive/dense-v3-exact-geometry-v1/bridge-null-feature.json"
    )
    case = json.loads(source.read_text())["cases"][case_index]
    tables, _ = evaluate(case["rows"])
    assert all(row["status"] == "baseline_equivalent" for row in tables["leave_dose_fold_metrics"])
    assert all(row["mse_reduction_exact"] == "0" for row in tables["leave_dose_fold_metrics"])
    assert all(row["predictively_useful"] is False for row in tables["feature_prediction_summary"])
    assert all(
        row["pooled_mse_reduction_exact"] == "0" for row in tables["feature_prediction_summary"]
    )
    assert all(
        row["pearson_residual_association"] is None for row in tables["residual_associations"]
    )
    assert all(
        row["spearman_residual_association"] is None for row in tables["residual_associations"]
    )


@pytest.mark.parametrize("kind", ["constant", "redundant"])
def test_exact_redundancy_is_not_an_unidentified_extrapolation(kind):
    tables, _ = evaluate(panel(kind))
    assert all(row["feature_improves"] is False for row in tables["leave_dose_fold_metrics"])
    assert all(
        row["feature_prediction_exact"] == row["baseline_prediction_exact"]
        for row in tables["held_out_predictions"]
    )


def test_only_training_fold_collinearity_is_not_filled_with_baseline():
    tables, _ = evaluate(panel("train_only"))
    first = [r for r in tables["leave_dose_fold_metrics"] if r["held_out_dose_index"] == 1]
    assert all(r["status"] == "unidentified_extension" and r["feature_rmse"] is None for r in first)
    assert all(r["predictively_useful"] is None for r in tables["feature_prediction_summary"])
    assert all(
        r["pooled_feature_rmse"] is None and r["folds_defined"] == 3
        for r in tables["feature_prediction_summary"]
    )
    assert len(tables["held_out_predictions"]) == 540


def test_exact_nonzero_near_collinearity_has_distinct_unresolved_status():
    tables, _ = evaluate(panel("near_null"))
    assert all(
        r["status"] == "unresolved_nonzero_direction" for r in tables["leave_dose_fold_metrics"]
    )
    assert all(Q(r["exact_residual_energy"]) > 0 for r in tables["leave_dose_fold_metrics"])
    assert all(r["predictively_useful"] is None for r in tables["feature_prediction_summary"])
    assert all(
        r["status"] == "unresolved_feature_residual" for r in tables["residual_associations"]
    )


def test_true_additional_signal_is_not_suppressed():
    tables, _ = evaluate(panel())
    assert all(
        r["feature_mse_exact"] == "0" and r["feature_improves"] is True
        for r in tables["leave_dose_fold_metrics"]
    )
    assert all(
        r["predictively_useful"] is True and r["folds_improved"] == 4
        for r in tables["feature_prediction_summary"]
    )
    assert all(
        r["pearson_residual_association"] == r["spearman_residual_association"] == 1
        for r in tables["residual_associations"]
    )


def test_outcome_in_baseline_span_has_zero_gain_and_undefined_association():
    rows = panel()
    for row in rows:
        row["mean_ndcg_at_10"] = 0.5 + row["stage"] / 64
    tables, _ = evaluate(rows)
    assert all(
        r["pooled_mse_reduction_exact"] == "0" and r["predictively_useful"] is False
        for r in tables["feature_prediction_summary"]
    )
    assert all(r["status"] == "outcome_in_baseline_span" for r in tables["residual_associations"])


@pytest.mark.parametrize("fold", [1, 2, 3, 4])
def test_fwl_matches_independent_sympy_full_augmented_system_and_scipy(fold):
    rows = panel()
    # Source-fixed non-polynomial noise tests a nonzero fitted residual, not only perfect fits.
    for i, row in enumerate(rows):
        row["mean_ndcg_at_10"] += math.sin((i + 1) * 0.137) / 64
    b = _baseline_design(rows)
    x = np.array([r[FEATURES[0]] for r in rows])
    y = np.array([r["mean_ndcg_at_10"] for r in rows])
    train = [i for i, r in enumerate(rows) if r["dose_index"] != fold]
    test = [i for i, r in enumerate(rows) if r["dose_index"] == fold]
    result = Baseline(qmatrix(b), train).feature([rational(v) for v in x], [rational(v) for v in y])
    design = np.column_stack((b, x))
    full = sp.Matrix([[sp.Rational(float(v)) for v in row] for row in design[train]])
    target = sp.Matrix([sp.Rational(float(v)) for v in y[train]])
    beta = (full.T * full).inv(method="DM") * full.T * target
    predicted = sp.Matrix([[sp.Rational(float(v)) for v in row] for row in design[test]]) * beta
    exact = [Q(int(v.p), int(v.q)) for v in predicted]
    assert exact == [result["feature_prediction"][i] for i in test]
    # A different numerical solver with training-only standardization is only a floating control.
    z = (x - x[train].mean()) / x[train].std()
    scaled = np.column_stack((b, z))
    numeric_beta = lstsq(scaled[train], y[train], lapack_driver="gelsy")[0]
    np.testing.assert_allclose(
        scaled[test] @ numeric_beta, [float(v) for v in exact], rtol=2e-12, atol=2e-14
    )


@pytest.mark.parametrize("exponent", [-500, 0, 500])
def test_exact_predictions_invariant_to_lossless_feature_scale_and_sign(exponent):
    rows = panel()
    b = qmatrix(_baseline_design(rows))
    context = Baseline(b, [i for i, r in enumerate(rows) if r["dose_index"] != 1])
    values = [rational(r[FEATURES[0]]) for r in rows]
    target = [rational(r["mean_ndcg_at_10"]) for r in rows]
    expected = context.feature(values, target)
    scaled = context.feature([v * rational(-(2.0**exponent)) for v in values], target)
    assert expected["feature_prediction"] == scaled["feature_prediction"]
    assert expected["status"] == scaled["status"] == "resolved"


def test_row_permutation_canonicalizes_complete_result():
    rows = panel()
    expected = evaluate(rows)
    shuffled = [rows[i] for i in np.random.default_rng(20260906).permutation(60)]
    assert evaluate(shuffled) == expected


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "optimizer",
        "dose",
        "rate",
        "centered",
        "nonfinite",
        "bool",
        "outcome",
        "stage",
        "run",
    ],
)
def test_invalid_panel_refuses_before_fitting(change):
    rows = copy.deepcopy(panel())
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = dict(rows[0])
    else:
        key, value = {
            "optimizer": ("optimizer", "other"),
            "dose": ("dose_index", 4),
            "rate": ("learning_rate", 0),
            "centered": ("centered_log10_learning_rate", 0),
            "nonfinite": (FEATURES[0], float("nan")),
            "bool": (FEATURES[0], True),
            "outcome": ("mean_ndcg_at_10", 1.1),
            "stage": ("stage", 1.0),
            "run": ("run_id", "rogue"),
        }[change]
        rows[0][key] = value
    with pytest.raises(ValueError):
        validate_panel(rows)


def test_exact_ranks_correlations_and_tiny_error_display():
    assert average_ranks([Q(3), Q(1), Q(1), Q(2)]) == [4, Q(3, 2), Q(3, 2), 3]
    assert correlation([Q(1)] * 4, [Q(i) for i in range(4)]) is None
    assert correlation([Q(i, 2**600) for i in range(4)], [Q(-i, 2**600) for i in range(4)]) == -1
    assert rmse_reduction(Q(1), Q(1) - Q(1, 10**100)) > 0
    assert rmse_reduction(Q(0), Q(0)) == 0
    with pytest.raises(ValueError, match="Singular"):
        inverse([[Q(1), Q(1)], [Q(1), Q(1)]])
