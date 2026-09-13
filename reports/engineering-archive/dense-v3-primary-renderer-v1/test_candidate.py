"""Pure rendering controls only; no checkpoint-backed publication is simulated."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ARCHIVE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "v3_primary_render_candidate", ARCHIVE / "candidate/primary_v3_publication_render.py"
)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
PROBE = json.loads((ARCHIVE / "probe-result.json").read_text())


@pytest.fixture(scope="module")
def rendered():
    return {
        case["kind"]: candidate.render_bridge(case["complete_kernel_tables"])
        for case in PROBE["cases"]
    }


@pytest.mark.parametrize(
    "kind,criterion,defined",
    [
        ("signal", "supported", 4),
        ("constant", "baseline equivalent", 4),
        ("train_only", "undefined", 3),
        ("near_null", "undefined", 0),
    ],
)
def test_complete_nine_feature_render_preserves_resolution(rendered, kind, criterion, defined):
    value = rendered[kind]
    assert len(value["rows"]) == 9
    assert all(
        row["criterion"] == criterion and row["folds_defined"] == defined for row in value["rows"]
    )
    assert value["scientific_completion"] is False and value["manuscript_installed"] is False
    assert value["primary_admission_supplied"] is False
    if criterion == "undefined":
        assert "cannot establish either support or lack of support" in value["latex"]
        assert all(row["pooled_feature_rmse"] is None for row in value["rows"])
    if kind == "constant":
        assert all(
            row["association"]["spearman_residual_association"] is None for row in value["rows"]
        )
        assert "baseline equivalence is distinct" in value["latex"]


@pytest.mark.parametrize("field", ["predictively_useful", "folds_defined", "pooled_rmse_reduction"])
def test_changed_summary_refuses_even_when_rows_and_count_survive(field):
    tables = copy.deepcopy(PROBE["cases"][0]["complete_kernel_tables"])
    row = tables["feature_prediction_summary"][0]
    row[field] = {"predictively_useful": False, "folds_defined": 3, "pooled_rmse_reduction": 0.0}[
        field
    ]
    with pytest.raises(ValueError):
        candidate.render_bridge(tables)


@pytest.mark.parametrize(
    "name", ["leave_dose_fold_metrics", "held_out_predictions", "residual_associations"]
)
def test_omitted_fold_prediction_or_association_is_not_a_renderable_summary(name):
    tables = copy.deepcopy(PROBE["cases"][0]["complete_kernel_tables"])
    tables[name].pop()
    with pytest.raises(ValueError):
        candidate.render_bridge(tables)


@pytest.mark.parametrize(
    "exact,text", [("1/" + str(10**400), "positive"), ("-1/" + str(10**400), "negative")]
)
def test_underflow_display_retains_exact_direction(exact, text):
    assert (
        candidate.delta_text({"pooled_rmse_reduction": 0.0, "pooled_mse_reduction_exact": exact})
        == text + " (below display range)"
    )


def test_partial_undefined_does_not_turn_into_all_negative(rendered):
    rows = copy.deepcopy(rendered["constant"]["rows"])
    rows[0] = copy.deepcopy(rendered["train_only"]["rows"][0])
    text = candidate.finding(rows)
    assert "None of the 8" in text and "criterion remains undefined" in text
    assert "equivalence" in text and "same task suite" in text


def test_undefined_display_is_not_zero():
    assert (
        candidate.delta_text({"pooled_rmse_reduction": None, "pooled_mse_reduction_exact": None})
        == "---"
    )
    with pytest.raises(ValueError):
        candidate.delta_text({"pooled_rmse_reduction": 0.0, "pooled_mse_reduction_exact": None})
