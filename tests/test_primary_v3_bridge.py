import copy
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim import primary_v3_bridge as bridge
from embed_optim.corrected_retrieval_bridge import assemble_bridge_rows
from embed_optim.primary_contract import file_identity, read_json
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_outcomes import csv_bytes, inspect_bundle, recipe_views, save_bundle
from scripts.prepare_dense_v3_bridge import build_payload

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"


@pytest.fixture
def primary():
    return PrimaryV3Contract.load(ROOT / bridge.PARENTS["primary"][0], ROOT, CANDIDATE)


def inputs(primary):
    configs = recipe_views(primary)
    checkpoints, scores, pairs = [], [], []
    for i, config in enumerate(configs):
        for stage in range(1, 6):
            common = {
                "run_id": config.run_id,
                "optimizer": config.optimizer.name,
                "learning_rate": config.optimizer.lr,
                "stage": stage,
            }
            v = [0.3 + 0.1 * math.sin((i + 1) * (j + 1) * stage * 0.137) for j in range(7)]
            checkpoints.append(
                {
                    **common,
                    "saved_segment_to_weight_ratio": v[0],
                    "saved_segment_stable_rank_fraction_parameter_weighted": v[1],
                    "saved_segment_sketch_effective_rank_fraction_parameter_weighted": v[2],
                    "saved_segment_row_cv_parameter_weighted": v[3],
                    "saved_segment_top_1pct_row_energy_parameter_weighted": v[4],
                    "cumulative_displacement_to_weight_ratio": v[5],
                    "cumulative_stable_rank_fraction_parameter_weighted": v[6],
                }
            )
            scores.append(
                {
                    **common,
                    "mean_ndcg_at_10": 0.5
                    + stage / 128
                    + i / 512
                    + (i % 4 - 1.5) * (stage - 3) / 128,
                }
            )
    for i, j in itertools.combinations(range(12), 2):
        for stage in range(1, 6):
            for kind in ("saved_segment", "cumulative"):
                pairs.append(
                    {
                        "stage": stage,
                        "displacement_kind": kind,
                        "first_run_id": configs[i].run_id,
                        "second_run_id": configs[j].run_id,
                        "first_optimizer": configs[i].optimizer.name,
                        "second_optimizer": configs[j].optimizer.name,
                        "mean_subspace_overlap": 0.3
                        + 0.1
                        * math.cos(
                            (i + 1) * (j + 2) * stage * (0.17 if kind == "saved_segment" else 0.19)
                        ),
                    }
                )
    return checkpoints, pairs, scores


def test_complete_adapter_retains_all_original_feature_values(primary):
    data = inputs(primary)
    tables, numerical = bridge.bridge_tables(primary, *data)
    old = assemble_bridge_rows(*data, recipe_views(primary))

    def key(row):
        return row["run_id"], row["stage"]

    assert sorted(tables["bridge_rows"], key=key) == sorted(old, key=key)
    assert {k: len(v) for k, v in tables.items()} == bridge.TABLE_COUNTS
    assert len(numerical["feature_designs"]) == 45


@pytest.mark.parametrize(
    "kind", ["missing_geometry", "missing_pairs", "missing_score", "relabel", "undefined_overlap"]
)
def test_adapter_refuses_incomplete_or_mislabelled_primary_join(primary, kind):
    c, p, s = inputs(primary)
    if kind == "missing_geometry":
        c.pop()
    elif kind == "missing_pairs":
        p.pop()
    elif kind == "missing_score":
        s.pop()
    elif kind == "relabel":
        s[0]["run_id"] = "historical-run"
    else:
        p[0]["mean_subspace_overlap"] = None
    with pytest.raises((ValueError, TypeError)):
        bridge.bridge_tables(primary, c, p, s)


@pytest.mark.parametrize(
    "kind",
    [
        "policy",
        "feature",
        "count",
        "parent",
        "source",
        "release",
        "science",
        "amendment",
        "old_rule",
    ],
)
def test_protocol_changes_refuse(primary, tmp_path, kind):
    payload = build_payload(ROOT)
    if kind == "policy":
        payload["numerical_policy"]["support"] = "any positive float"
    elif kind == "feature":
        payload["features"].pop()
    elif kind == "count":
        payload["table_counts"]["bridge_rows"] = 59
    elif kind == "parent":
        payload["parents"]["outcomes"]["sha256"] = "0" * 64
    elif kind == "source":
        payload["sources"].pop("src/embed_optim/bridge_exact_arithmetic.py")
    elif kind == "release":
        payload["status"] = "released"
    elif kind == "science":
        payload["scientific_completion"] = True
    elif kind == "amendment":
        payload["amendment"]["numerical_implementation_changed"] = False
    else:
        payload["scientific_rules"]["support_rule"] = "choose a convenient fold"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises((ValueError, KeyError)):
        bridge.BridgeContract.load(path, primary)


def test_actual_prepared_contract_and_missing_runs_before_bundle_reads(
    primary, tmp_path, monkeypatch
):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(build_payload(ROOT)))
    contract = bridge.BridgeContract.load(path, primary)

    def no_bundle(*args, **kwargs):
        raise AssertionError("Opened a bundle before full primary admission")

    monkeypatch.setattr(bridge, "snapshot", no_bundle)
    with pytest.raises(ValueError, match="ordinary retained run"):
        bridge.gather(contract, SimpleNamespace(experiment_root=tmp_path))


@pytest.mark.parametrize(
    "kind", ["prediction", "support", "association", "diagnostic", "extra", "missing", "symlink"]
)
def test_bundle_recomputation_refuses_rehashed_corruption(primary, tmp_path, kind):
    tables, numerical = bridge.bridge_tables(primary, *inputs(primary))
    plan = {"scope": "engineering_synthetic_bridge_fixture", "scientific_completion": False}
    evidence = {"numerical_diagnostics": numerical, "not_primary_evidence": True}
    output = tmp_path / "fixture"
    save_bundle(output, plan, evidence, tables)
    changed = copy.deepcopy(tables)
    if kind in {"prediction", "support", "association"}:
        key, field, value = {
            "prediction": ("held_out_predictions", "feature_prediction", 0.9),
            "support": ("feature_prediction_summary", "predictively_useful", True),
            "association": ("residual_associations", "pearson_residual_association", 0.9),
        }[kind]
        if kind == "support":
            value = not changed[key][0][field]
        changed[key][0][field] = value
        filename = f"{key}.csv"
        (output / filename).write_bytes(csv_bytes(changed[key]))
    elif kind == "diagnostic":
        value = copy.deepcopy(evidence)
        value["numerical_diagnostics"]["feature_designs"][0]["rank_cutoff"] = 0
        filename = "evidence.json"
        (output / filename).write_text(json.dumps(value))
    elif kind == "extra":
        (output / "extra.json").write_text("{}")
    elif kind == "missing":
        (output / "held_out_predictions.csv").unlink()
    else:
        target = output / "held_out_predictions.csv"
        moved = tmp_path / "moved.csv"
        target.rename(moved)
        target.symlink_to(moved)
    if kind in {"prediction", "support", "association", "diagnostic"}:
        manifest = read_json(output / "manifest.json")
        manifest["outputs"][filename] = {"path": filename, **file_identity(output / filename)}
        (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        inspect_bundle(output, plan, evidence, tables)


def test_snapshot_refuses_source_changed_after_admission(tmp_path):
    path = tmp_path / "source.json"
    path.write_text("{}")
    bound = bridge.snapshot(tmp_path, [path.name])
    path.write_text("[]")
    with pytest.raises(ValueError):
        bridge.verify_snapshot(bound)


@pytest.mark.parametrize("mutate_after_reader", [False, True])
def test_complete_gather_wiring_with_explicitly_simulated_upstream_admission(
    primary, tmp_path, monkeypatch, mutate_after_reader
):
    """An orchestration fixture, not a positive claim about real primary input admission."""
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(build_payload(ROOT)))
    contract = bridge.BridgeContract.load(protocol, primary)
    checkpoints, pairs, scores = inputs(primary)
    outcome_tables = {key: [{"synthetic": True}] for key in bridge.OUTCOME_COUNTS}
    outcome_tables["run_stage_scores"] = scores
    plan, evidence = {"scope": "engineering_synthetic_upstream"}, {"simulated_admission": True}
    outcomes = tmp_path / "outcomes"
    save_bundle(outcomes, plan, evidence, outcome_tables)
    geometry = tmp_path / "geometry"
    geometry.mkdir()
    for name in contract.geometry.payload["table_counts"]:
        rows = (
            checkpoints
            if name == "checkpoint_geometry"
            else pairs
            if name == "run_pair_subspace_overlap"
            else [{"synthetic": True}]
        )
        (geometry / f"{name}.csv").write_bytes(csv_bytes(rows))
    (geometry / "raw.control").write_text("synthetic raw binding")
    summary = {
        "raw_bindings": [{"path": "raw.control", **file_identity(geometry / "raw.control")}],
        "simulated_admission": True,
    }
    (geometry / "summary.json").write_text(json.dumps(summary))
    order = []

    def fake_admit(*args):
        order.append("all-primary")

    def fake_outcomes(*args):
        order.append("outcomes")
        return plan, evidence, outcome_tables

    def fake_geometry(*args, **kwargs):
        assert kwargs["produce"] is False
        order.append("geometry")
        if mutate_after_reader:
            (geometry / "raw.control").write_text("altered synthetic raw binding")
        return summary

    monkeypatch.setattr(bridge, "admit_primary", fake_admit)
    monkeypatch.setattr(bridge, "collect_evidence", fake_outcomes)
    monkeypatch.setattr(bridge, "operate", fake_geometry)
    args = SimpleNamespace(
        experiment_root=tmp_path,
        outcomes_root=outcomes,
        geometry_root=geometry,
        results_root=tmp_path,
        validation_data=tmp_path,
        validation_root=tmp_path,
        reference=tmp_path,
    )
    if mutate_after_reader:
        with pytest.raises(ValueError):
            bridge.gather(contract, args)
    else:
        result_plan, result_evidence, tables = bridge.gather(contract, args)
        assert result_plan["scope"] == bridge.SCOPE
        assert result_plan["scientific_completion"] is False
        assert result_evidence["outcome_evidence"]["simulated_admission"] is True
        assert {key: len(rows) for key, rows in tables.items()} == bridge.TABLE_COUNTS
    assert order == ["all-primary", "outcomes", "geometry"]
