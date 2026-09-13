"""Explicit synthetic scientific panels; no real optimizer comparison is admitted."""

import copy
import json
import math
import runpy
from pathlib import Path

import numpy as np
import pytest

from embed_optim import dimension_inference as inference
from embed_optim.bridge_named_features import evaluate_named
from embed_optim.corrected_retrieval_bridge import assemble_bridge_rows
from embed_optim.dimension_inference_render import render
from embed_optim.dimension_publication import BRIDGE_FEATURES, PRIMARY_FEATURES
from embed_optim.primary_contract import read_json
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_dimension_contract import planned_states
from embed_optim.primary_v3_outcomes import recipe_views

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"


@pytest.fixture(scope="module")
def primary():
    return PrimaryV3Contract.load(ROOT / "configs/dense_primary_v3_protocol.json", ROOT, CANDIDATE)


def fixture(primary):
    protocol = read_json(ROOT / "configs/dense_dimension_utilization_protocol.json")
    tasks = sorted(
        read_json(ROOT / "configs/beir_representation_probe.json")["expected"]["task_counts"]
    )
    tables = {
        name: []
        for name in ("checkpoint_summary", "task_summary", "random_removal", "rotation_summary")
    }
    for index, state in enumerate(planned_states(primary)):
        meta = state["meta"]
        oi = 0 if state["stage"] == 0 else ("adamw", "muon", "normuon").index(meta["optimizer"]) + 1
        rows = []
        for ti, task in enumerate(tasks):
            row = {
                **meta,
                "rotation_seed": None,
                "task": task,
                "baseline_margin": 0.12 + 0.01 * math.sin(index + ti),
                "baseline_shortlist_ndcg": 0.6 + 0.01 * math.cos(index + ti),
                **{
                    feature: 0.4
                    + 0.03 * math.sin((index + 1) * (fi + 2) * 0.137)
                    + 0.003 * oi * (ti + 1) * (fi + 1)
                    for fi, feature in enumerate(PRIMARY_FEATURES)
                },
            }
            rows.append(row)
            tables["task_summary"].append(row)
            if state["stage"] in (0, 5):
                for seed in protocol["rotation_control"]["seeds"]:
                    tables["rotation_summary"].append(
                        {
                            **row,
                            "rotation_seed": seed,
                            **{
                                feature: row[feature] + 0.002 * oi * math.sin(seed + fi + ti)
                                for fi, feature in enumerate(PRIMARY_FEATURES)
                            },
                        }
                    )
            for fraction in protocol["random_removal"]["removed_fractions"]:
                for draw in range(20):
                    ratio = 0.98 + 0.01 * math.sin((index + 1) * 0.157 + ti + draw / 8)
                    tables["random_removal"].append(
                        {
                            **meta,
                            "task": task,
                            "removed_fraction": fraction,
                            "draw": draw,
                            "shortlist_ndcg": ratio * row["baseline_shortlist_ndcg"],
                            "relative_shortlist_ndcg": ratio,
                            "margin": row["baseline_margin"],
                            "relative_margin": 1.0,
                        }
                    )
        tables["checkpoint_summary"].append(
            {
                **meta,
                "tasks": 14,
                **{
                    name: float(np.mean([row[name] for row in rows]))
                    for name in (*PRIMARY_FEATURES, "baseline_margin", "baseline_shortlist_ndcg")
                },
            }
        )
    producer = runpy.run_path(str(ROOT / "tests/test_primary_v3_bridge.py"))["inputs"]
    cp, pairs, scores = producer(primary)
    original = assemble_bridge_rows(cp, pairs, scores, recipe_views(primary))
    return tables, original, protocol, tasks


@pytest.fixture(scope="module")
def complete(primary):
    inputs = fixture(primary)
    result, decisions = inference.summarize(primary, *inputs)
    return inputs, result, decisions


def test_complete_family_and_original_features_retained(complete):
    inputs, result, decisions = complete
    assert {key: len(rows) for key, rows in result.items()} == inference.TABLE_COUNTS
    before = {(r["run_id"], r["stage"]): r for r in inputs[1]}
    for row in result["bridge_rows"]:
        original = before[row["run_id"], row["stage"]]
        assert {key: row[key] for key in original} == original
        assert set(row) - set(original) == set(BRIDGE_FEATURES)
    assert decisions["scientific_completion"] is False
    assert {r["feature"] for r in result["feature_prediction_summary"]} == set(BRIDGE_FEATURES)
    assert len({r["max_t_critical_value"] for r in result["primary_contrasts"]}) == 1
    assert all(
        r["bootstrap_samples"] == 50000 and r["bootstrap_seed"] == 20260905
        for r in result["primary_contrasts"]
    )


@pytest.mark.parametrize(
    "table", ["checkpoint_summary", "task_summary", "random_removal", "rotation_summary"]
)
@pytest.mark.parametrize(
    "kind", ["missing", "duplicate", "historical", "optimizer", "rate", "stage"]
)
def test_complete_identity_not_just_counts(primary, complete, table, kind):
    data = copy.deepcopy(complete[0])
    rows = data[0][table]
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows[-1] = dict(rows[-2])
    else:
        field, value = {
            "historical": ("run_id", "padded-muon-1e-3"),
            "optimizer": ("optimizer", "adamw"),
            "rate": ("learning_rate", 10.0),
            "stage": ("step", 1),
        }[kind]
        rows[-1][field] = value
    with pytest.raises(ValueError):
        inference.validate_tables(primary, data[0], data[2], data[3])


@pytest.mark.parametrize(
    "kind",
    ["bool", "nan", "task_mean", "mask_draw", "mask_fraction", "rotation_seed", "native_rotation"],
)
def test_numerical_pairing_and_invalid_values_fail(primary, complete, kind):
    data = copy.deepcopy(complete[0])
    tables = data[0]
    if kind in ("bool", "nan"):
        tables["task_summary"][-1][PRIMARY_FEATURES[0]] = True if kind == "bool" else math.nan
    elif kind == "task_mean":
        tables["checkpoint_summary"][-1][PRIMARY_FEATURES[0]] += 0.001
    elif kind == "mask_draw":
        tables["random_removal"][-1]["draw"] = True
    elif kind == "mask_fraction":
        tables["random_removal"][-1]["removed_fraction"] += 1e-12
    elif kind == "rotation_seed":
        tables["rotation_summary"][-1]["rotation_seed"] = 42
    else:
        tables["task_summary"][-1]["rotation_seed"] = 314159
    with pytest.raises(ValueError):
        inference.validate_tables(primary, tables, data[2], data[3])


def test_reordering_rows_does_not_change_estimators(primary, complete):
    data = copy.deepcopy(complete[0])
    rng = np.random.default_rng(20260907)
    for rows in data[0].values():
        rng.shuffle(rows)
    rng.shuffle(data[1])
    assert inference.summarize(primary, *data) == (complete[1], complete[2])


def test_zero_se_refuses_without_reducing_the_family(primary, complete):
    data = copy.deepcopy(complete[0])
    for key in ("checkpoint_summary", "task_summary"):
        for row in data[0][key]:
            row[PRIMARY_FEATURES[0]] = 0.5
    with pytest.raises(ValueError, match="positive standard errors"):
        inference.summarize(primary, *data)


def test_cannot_overwrite_original_feature_or_adopt_other_state(primary, complete):
    data = copy.deepcopy(complete[0])
    data[1][0][BRIDGE_FEATURES[0]] = 0.5
    with pytest.raises(ValueError, match="overwrite"):
        inference.add_functional_features(primary, data[0], data[1])
    data = copy.deepcopy(complete[0])
    old_id = data[1][0]["run_id"]
    for row in data[1]:
        if row["run_id"] == old_id:
            row["run_id"] = "historical"
    with pytest.raises(ValueError, match="same complete v3 states"):
        inference.add_functional_features(primary, data[0], data[1])


def test_four_actual_names_keep_exact_null_case_distinct(primary, complete):
    data = copy.deepcopy(complete[0])
    stages = {step: i for i, step in enumerate(data[2]["inputs"]["checkpoint_stages"], 1)}
    for row in data[0]["checkpoint_summary"]:
        if row["step"]:
            row.update({name: stages[row["step"]] / 8 for name in PRIMARY_FEATURES})
    for row in data[0]["random_removal"]:
        if row["step"]:
            row["relative_shortlist_ndcg"] = stages[row["step"]] / 8
    panel = inference.add_functional_features(primary, data[0], data[1])
    actual, _ = evaluate_named(panel, BRIDGE_FEATURES)
    assert all(
        r["pooled_mse_reduction_exact"] == "0" and r["predictively_useful"] is False
        for r in actual["feature_prediction_summary"]
    )
    assert all(r["pearson_residual_association"] is None for r in actual["residual_associations"])


@pytest.mark.parametrize(
    "native,rotation", [(False, False), (False, True), (True, False), (True, True)]
)
def test_text_does_not_turn_negative_or_rotation_controls_into_general_claims(
    complete, native, rotation
):
    result, decisions = copy.deepcopy(complete[1:])
    decisions["constructive_native_coordinate_use"]["muon"] = native
    decisions["direction_stable_across_tested_rotations"]["muon"] = rotation
    for i, row in enumerate(result["feature_prediction_summary"]):
        row["predictively_useful"] = None if i == 0 else False
    rendered = render(result, decisions)
    assert "predictive criterion for helpful mass share is undefined" in rendered
    assert "helpful participation" in rendered.lower()
    assert (
        "does not establish equivalence" in rendered
        if not native
        else "Muon meets the joint" in rendered
    )
    assert "arbitrary-basis" in rendered and "not causal mediation" in rendered
    assert "none of the four" not in rendered
    assert "semantic independence" in rendered
    assert rendered.count("\\addplot+") == 16


def test_frozen_false_or_undefined_predicate_remains_a_distinct_json_value(complete):
    rows = copy.deepcopy(complete[1]["feature_prediction_summary"])
    rows[0]["predictively_useful"] = None
    rows[1]["predictively_useful"] = False
    assert json.loads(json.dumps(rows)) == rows
