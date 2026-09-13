"""Fixture-only scientific arithmetic tests; no synthetic table is a model finding."""

import copy
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pytest

from embed_optim import primary_v3_outcomes as outcome
from embed_optim.corrected_outcome_summary import (
    CONTRASTS,
    paired_max_t_intervals,
    summarize_score_rows,
)
from embed_optim.decontamination import DECONTAMINATED_TASK_NAMES
from embed_optim.primary_contract import digest, file_identity, read_json
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_validation import ValidationContract, select_recipes
from scripts.prepare_dense_v3_outcomes import build_payload

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"


@pytest.fixture
def primary():
    return PrimaryV3Contract.load(ROOT / outcome.PARENTS["primary"][0], ROOT, CANDIDATE)


@pytest.fixture
def fixture(primary):
    rows, metrics = [], []
    for i, r in enumerate(primary.inputs["runs"]):
        recipe = primary.expected_identity(r["run_id"])["recipe"]
        name, lr = recipe["optimizer"]["name"], recipe["optimizer"]["lr"]
        for stage, step in enumerate(primary.payload["checkpoint_steps"], 1):
            for ti, task in enumerate(DECONTAMINATED_TASK_NAMES):
                rows.append(
                    {
                        "run_id": r["run_id"],
                        "optimizer": name,
                        "learning_rate": lr,
                        "model_family": "dense",
                        "stage": stage,
                        "step": step,
                        "fraction": stage / 5,
                        "task": task,
                        "ndcg_at_10": 0.3
                        + 0.01 * stage
                        + 0.0006 * i
                        + 0.00011 * i * ti
                        + 0.000003 * i**2 * ti**2,
                    }
                )
        metrics.append(
            {
                "run_id": r["run_id"],
                "optimizer": name,
                "learning_rate": lr,
                "contrastive_loss": 0.6 - (i % 4) * 0.01,
                "positive_margin": -float(i),
            }
        )
    selection = {
        "scope": "dense_primary_v3_validation_selection",
        "primary_protocol_sha256": primary.sha256,
        "validation_protocol_sha256": outcome.PARENTS["validation"][1],
        "run_metrics": metrics,
        "selected": select_recipes(metrics, primary.inputs["runs"]),
        "scientific_completion": False,
    }
    return rows, selection


def test_adapter_exactly_retains_original_statistical_tables(primary, fixture):
    rows, selection = fixture
    actual = outcome.outcome_tables(primary, rows, selection)
    original = summarize_score_rows(rows, outcome.recipe_views(primary), selection["selected"])
    assert all(actual[k] == v for k, v in original.items())
    assert len(actual["all_task_scores"]) == 840
    assert len(actual["validation_run_metrics"]) == 12
    assert {r["bootstrap_samples"] for r in actual["primary_summary"]} == {50_000}
    assert {r["bootstrap_seed"] for r in actual["primary_summary"]} == {20260903}


def test_primary_is_all_rates_and_independent_of_validation_selection(primary, fixture):
    rows, selection = fixture
    first = outcome.outcome_tables(primary, rows, selection)
    for r in selection["run_metrics"]:
        r["contrastive_loss"] = 1 - r["contrastive_loss"]
        r["positive_margin"] *= -1000
    selection["selected"] = select_recipes(selection["run_metrics"], primary.inputs["runs"])
    second = outcome.outcome_tables(primary, rows, selection)
    assert first["primary_summary"] == second["primary_summary"]
    assert first["primary_task_effects"] == second["primary_task_effects"]
    assert first["secondary_task_effects"] != second["secondary_task_effects"]
    for task_row in first["primary_task_effects"]:
        for name in ("adamw", "muon", "normuon"):
            expected = statistics.fmean(
                r["ndcg_at_10"]
                for r in rows
                if r["optimizer"] == name and r["stage"] == 5 and r["task"] == task_row["task"]
            )
            assert task_row[f"{name}_ndcg_at_10"] == expected


def test_early_dynamics_do_not_change_final_endpoint_or_impute_initialization(primary, fixture):
    rows, selection = fixture
    first = outcome.outcome_tables(primary, rows, selection)
    for r in rows:
        if r["stage"] < 5:
            r["ndcg_at_10"] = 0.05
    second = outcome.outcome_tables(primary, rows, selection)
    assert first["primary_summary"] == second["primary_summary"]
    assert first["secondary_summary"] == second["secondary_summary"]
    assert all(r["progress_fraction"] >= 0.2 for r in second["run_stage_scores"])
    for r in second["run_observed_auc"]:
        final = next(
            x["mean_ndcg_at_10"]
            for x in second["run_stage_scores"]
            if x["run_id"] == r["run_id"] and x["stage"] == 5
        )
        assert r["observed_auc_20_to_100"] == pytest.approx(
            0.6 * 0.05 + 0.2 * (0.05 + final) / 2, abs=1e-14
        )
        assert r["observed_mean_20_to_100"] == r["observed_auc_20_to_100"] / 0.8


@pytest.mark.parametrize(
    "key,value",
    [
        ("ndcg_at_10", float("nan")),
        ("ndcg_at_10", float("inf")),
        ("ndcg_at_10", -0.01),
        ("ndcg_at_10", 1.01),
        ("ndcg_at_10", True),
        ("ndcg_at_10", "0.5"),
        ("stage", 1.9),
        ("stage", True),
        ("stage", 0),
        ("stage", 6),
        ("step", 783),
        ("step", 782.0),
        ("fraction", 0.25),
        ("optimizer", "normuon"),
        ("learning_rate", 9e-4),
        ("learning_rate", "1e-6"),
        ("model_family", "late"),
        ("run_id", "padded-adamw-1e-6"),
        ("task", "Unknown"),
    ],
)
def test_invalid_score_values_are_rejected_before_statistics(primary, fixture, key, value):
    rows, selection = fixture
    rows[0][key] = value
    with pytest.raises(ValueError):
        outcome.outcome_tables(primary, rows, selection)


@pytest.mark.parametrize(
    "kind",
    ["missing_cell", "duplicate_cell", "missing_high_rate", "extra_column", "missing_column"],
)
def test_complete_grid_cannot_drop_or_relabel_any_cell(primary, fixture, kind):
    rows, selection = fixture
    if kind == "missing_cell":
        rows.pop()
    elif kind == "duplicate_cell":
        rows[-1] = copy.deepcopy(rows[0])
    elif kind == "missing_high_rate":
        rows[:] = [r for r in rows if r["run_id"] != "verified-v3-muon-3e-3"]
    elif kind == "extra_column":
        rows[0]["score_on_other_benchmark"] = 0.8
    else:
        del rows[0]["step"]
    with pytest.raises(ValueError):
        outcome.outcome_tables(primary, rows, selection)


def test_score_order_does_not_change_paired_task_calculation(primary, fixture):
    rows, selection = fixture
    assert outcome.outcome_tables(primary, rows, selection) == outcome.outcome_tables(
        primary, rows[::-1], selection
    )


@pytest.mark.parametrize(
    "kind",
    [
        "selected_wrong_rate",
        "old_primary",
        "old_validation",
        "diagnostic",
        "missing_rate",
        "duplicate_rate",
        "fake_scientific",
    ],
)
def test_selection_identity_and_actual_validation_rule_are_required(primary, fixture, kind):
    rows, selection = fixture
    if kind == "selected_wrong_rate":
        selection["selected"]["muon"] = "verified-v3-muon-1e-4"
    elif kind == "old_primary":
        selection["primary_protocol_sha256"] = "0" * 64
    elif kind == "old_validation":
        selection["validation_protocol_sha256"] = "0" * 64
    elif kind == "diagnostic":
        selection["scope"] = "diagnostic"
    elif kind == "missing_rate":
        selection["run_metrics"].pop()
    elif kind == "duplicate_rate":
        selection["run_metrics"][-1] = copy.deepcopy(selection["run_metrics"][0])
    else:
        selection["scientific_completion"] = True
    with pytest.raises(ValueError):
        outcome.outcome_tables(primary, rows, selection)


def scalar_bootstrap_reference(effects, *, samples=50_000, seed=20260903):
    """Separate scalar means/variance/quantile; only the declared index generator is shared."""
    values = [effects[k] for k in CONTRASTS]
    points = [statistics.fmean(x) for x in values]
    se = [statistics.stdev(x) / math.sqrt(14) for x in values]
    indices = np.random.default_rng(seed).integers(0, 14, size=(samples, 14))
    boot = [[] for _ in values]
    maxima = []
    for selected in indices.tolist():
        means = [statistics.fmean(x[i] for i in selected) for x in values]
        for j, mean in enumerate(means):
            boot[j].append(mean)
        maxima.append(max(abs((means[j] - points[j]) / se[j]) for j in range(3)))

    def quantile(items, probability):
        ordered = sorted(items)
        position = (len(ordered) - 1) * probability
        lower = math.floor(position)
        fraction = position - lower
        return (
            ordered[lower] * (1 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction
        )

    critical = quantile(maxima, 0.95)
    return {
        key: {
            "mean": points[j],
            "se": se[j],
            "critical": critical,
            "lower": points[j] - critical * se[j],
            "upper": points[j] + critical * se[j],
            "nominal_lower": quantile(boot[j], 0.025),
            "nominal_upper": quantile(boot[j], 0.975),
        }
        for j, key in enumerate(CONTRASTS)
    }


def test_full_50000_resample_fixed_se_max_t_matches_independent_scalar_reference():
    effects = {
        CONTRASTS[0]: [0.02 + (i - 6) * 0.002 for i in range(14)],
        CONTRASTS[1]: [0.01 + ((i * 3) % 11 - 5) * 0.003 for i in range(14)],
    }
    effects[CONTRASTS[2]] = [
        b - a for a, b in zip(effects[CONTRASTS[0]], effects[CONTRASTS[1]], strict=True)
    ]
    actual, expected = paired_max_t_intervals(effects), scalar_bootstrap_reference(effects)
    fields = {
        "mean": "mean_delta_ndcg_at_10",
        "se": "across_task_standard_error",
        "critical": "simultaneous_critical_value",
        "lower": "simultaneous_ci_95_lower",
        "upper": "simultaneous_ci_95_upper",
        "nominal_lower": "nominal_bootstrap_ci_95_lower",
        "nominal_upper": "nominal_bootstrap_ci_95_upper",
    }
    for key, row in actual.items():
        for ref, field in fields.items():
            assert row[field] == pytest.approx(expected[key][ref], abs=1e-12, rel=1e-12)
        support = (
            "positive"
            if expected[key]["lower"] > 0
            else "negative"
            if expected[key]["upper"] < 0
            else "inconclusive"
        )
        assert row["support"] == support
    assert len({r["simultaneous_critical_value"] for r in actual.values()}) == 1


def test_degenerate_task_variance_fails_without_jitter_or_fallback():
    effects = {k: [0.0] * 14 for k in CONTRASTS}
    with pytest.raises(ValueError, match="positive finite"):
        paired_max_t_intervals(effects)


def make_contract(primary, tmp_path):
    validation = ValidationContract.load(ROOT / outcome.PARENTS["validation"][0], primary)
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(build_payload(ROOT)))
    return outcome.OutcomeContract.load(path, validation)


@pytest.mark.parametrize(
    "keys,value",
    [
        (("scope",), "historical"),
        (("schema_version",), True),
        (("status",), "reviewed_primary_execution_lock"),
        (("scientific_completion",), True),
        (("formal_execution_authorized",), True),
        (("sources",), {}),
        (("parents", "primary", "sha256"), "0" * 64),
        (("parents", "validation", "path"), "wrong"),
        (("parents", "validation_acceptance", "sha256"), "0" * 64),
        (("scientific_rules", "inference", "bootstrap_samples"), 1000),
        (("scientific_rules", "inference", "bootstrap_seed"), 0),
        (("scientific_rules", "inference", "resampling_unit"), "run"),
        (("scientific_rules", "inference", "support_rule"), "use nominal interval"),
        (("scientific_rules", "dynamics", "observed_auc"), "impute initial score"),
        (("table_counts", "all_task_scores"), 756),
    ],
)
def test_wrong_protocol_or_changed_inference_rejected(primary, tmp_path, keys, value):
    contract = make_contract(primary, tmp_path)
    payload = copy.deepcopy(contract.payload)
    node = payload
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        outcome.OutcomeContract.load(changed, contract.validation)


def test_selection_must_finish_before_any_test_score_is_read(primary, tmp_path, monkeypatch):
    contract = make_contract(primary, tmp_path)
    calls = []

    def missing(*args):
        calls.append("selection")
        raise ValueError("missing complete validation fixture")

    monkeypatch.setattr(outcome, "collect_selection", missing)
    monkeypatch.setattr(outcome, "inspect_matrix", lambda *a: calls.append("BEIR"))
    with pytest.raises(ValueError, match="missing complete validation"):
        outcome.collect_evidence(contract, tmp_path, tmp_path, tmp_path, tmp_path)
    assert calls == ["selection"]


def test_actual_absent_primary_cli_creates_no_output(primary, tmp_path):
    contract = make_contract(primary, tmp_path)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="ordinary retained run"):
        outcome.main(
            [
                "build",
                "--protocol",
                str(contract.path),
                "--primary-protocol",
                str(primary.path),
                "--validation-protocol",
                str(contract.validation.path),
                "--repository",
                str(ROOT),
                "--training-root",
                str(CANDIDATE),
                "--experiment-root",
                str(tmp_path),
                "--results-root",
                str(tmp_path / "beir"),
                "--validation-data",
                str(tmp_path / "data"),
                "--validation-root",
                str(tmp_path / "validation"),
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


@pytest.fixture
def synthetic_runs(primary):
    result = []
    for row in primary.inputs["runs"]:
        expected = primary.expected_identity(row["run_id"])
        result.append(
            {
                "run_id": row["run_id"],
                "run_identity_sha256": digest(expected),
                "whole_run_artifacts_verified": True,
                "steps": primary.payload["checkpoint_steps"],
                "accepted_timing": {"segments": 5, "total_wall_time_seconds_max_rank": 5000},
                "system_metrics": {
                    "world_size": 4,
                    "gpu_name": "fixture device, not hardware evidence",
                    "trainer": {"train_samples_per_second": 111, "train_steps_per_second": 0.8},
                    "peak_allocated_bytes_max_rank": 2**30,
                    "peak_reserved_bytes_max_rank": 2**31,
                    "checkpoint_bytes": {
                        f"checkpoint-{s}": 2**30 + s for s in primary.payload["checkpoint_steps"]
                    },
                    "optimizer_state_bytes": {
                        f"checkpoint-{s}": 2**29 + s for s in primary.payload["checkpoint_steps"]
                    },
                },
            }
        )
    return result


def test_systems_use_complete_accepted_time_not_trainer_throughput(primary, synthetic_runs):
    result = outcome.system_rows(primary, synthetic_runs)
    assert len(result) == 12
    assert all(r["samples_per_second"] == 100 for r in result)
    assert all(r["steps_per_second"] == 3907 / 5000 for r in result)
    assert all(r["trainer_reported_samples_per_second"] == 111 for r in result)
    assert all(r["maximum_checkpoint_gib"] == (2**30 + 3907) / 2**30 for r in result)


@pytest.mark.parametrize(
    "kind", ["missing", "duplicate", "identity", "time", "segments", "sizes", "memory", "device"]
)
def test_incomplete_or_changed_systems_rejected(primary, synthetic_runs, kind):
    if kind == "missing":
        synthetic_runs.pop()
    elif kind == "duplicate":
        synthetic_runs[-1] = copy.deepcopy(synthetic_runs[0])
    elif kind == "identity":
        synthetic_runs[0]["run_identity_sha256"] = "0" * 64
    elif kind == "time":
        synthetic_runs[0]["accepted_timing"]["total_wall_time_seconds_max_rank"] = 0
    elif kind == "segments":
        synthetic_runs[0]["accepted_timing"]["segments"] = 4
    elif kind == "sizes":
        synthetic_runs[0]["system_metrics"]["checkpoint_bytes"].pop("checkpoint-782")
    elif kind == "memory":
        synthetic_runs[0]["system_metrics"]["peak_allocated_bytes_max_rank"] = 2**32
    else:
        synthetic_runs[0]["system_metrics"]["gpu_name"] = ""
    with pytest.raises(ValueError):
        outcome.system_rows(primary, synthetic_runs)


def test_saved_tables_recomputed_not_trusted_after_rehashed_mutation(
    primary, fixture, synthetic_runs, tmp_path
):
    rows, selection = fixture
    tables = outcome.outcome_tables(primary, rows, selection)
    tables["system_metrics"] = outcome.system_rows(primary, synthetic_runs)
    plan, evidence = (
        {"scope": "engineering_fixture", "scientific_completion": False},
        {"fixture": True},
    )
    saved = outcome.save_bundle(tmp_path / "bundle", plan, evidence, tables)
    assert saved["recomputed_bytes_verified"] is True
    path = tmp_path / "bundle" / "primary_summary.csv"
    path.write_text(path.read_text().replace("positive", "negative"))
    manifest_path = tmp_path / "bundle" / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["outputs"][path.name] = {"path": path.name, **file_identity(path)}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fresh evidence/statistical recomputation"):
        outcome.inspect_bundle(tmp_path / "bundle", plan, evidence, tables)
    with pytest.raises(ValueError, match="do not overwrite"):
        outcome.save_bundle(tmp_path / "bundle", plan, evidence, tables)


def test_invalid_serialization_creates_no_partial_bundle(tmp_path):
    target = tmp_path / "invalid"
    with pytest.raises(ValueError):
        outcome.save_bundle(target, {}, {}, {"fixture": [{"bad": float("nan")}]})
    assert not target.exists()


def test_complete_fixture_flows_through_collection_build_and_fresh_reader(
    primary, fixture, synthetic_runs, tmp_path, monkeypatch
):
    contract = make_contract(primary, tmp_path)
    rows, selection = fixture
    calls = []

    def chosen(*args):
        calls.append("validation")
        return copy.deepcopy(selection)

    def scored(*args):
        calls.append("BEIR")
        return {"score_rows": copy.deepcopy(rows), "runs": copy.deepcopy(synthetic_runs)}

    monkeypatch.setattr(outcome, "collect_selection", chosen)
    monkeypatch.setattr(outcome, "inspect_matrix", scored)
    plan, evidence, tables = outcome.collect_evidence(
        contract, tmp_path, tmp_path, tmp_path, tmp_path
    )
    assert calls == ["validation", "BEIR"]
    assert {k: len(v) for k, v in tables.items()} == outcome.TABLE_COUNTS
    target = tmp_path / "fixture-bundle"
    outcome.save_bundle(target, plan, evidence, tables)
    fresh = outcome.collect_evidence(contract, tmp_path, tmp_path, tmp_path, tmp_path)
    assert outcome.inspect_bundle(target, *fresh)["recomputed_bytes_verified"] is True
    assert plan["scientific_completion"] is False


def test_changed_expected_raw_evidence_rejects_an_old_saved_bundle(tmp_path):
    plan, evidence, tables = {"scope": "engineering_fixture"}, {"value": 1}, {"fixture": [{"x": 1}]}
    target = tmp_path / "fixture-bundle"
    outcome.save_bundle(target, plan, evidence, tables)
    with pytest.raises(ValueError, match="fresh evidence/statistical recomputation"):
        outcome.inspect_bundle(target, plan, {"value": 2}, tables)
