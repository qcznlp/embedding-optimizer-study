import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim import primary_v3_exact_bridge as exact
from embed_optim.bridge_numerics import evaluate
from embed_optim.primary_contract import file_identity, read_json
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_outcomes import csv_bytes, inspect_bundle, save_bundle
from scripts.prepare_dense_v3_exact_bridge import payload

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"


@pytest.fixture
def primary():
    return PrimaryV3Contract.load(ROOT / exact.PARENTS["primary"][0], ROOT, CANDIDATE)


def fixture(primary):
    return runpy.run_path(str(ROOT / "tests/test_exact_bridge_measurements.py"))["fixture"](primary)


def test_complete_sensitivity_and_all_original_comparators(primary):
    data = fixture(primary)
    old, _ = evaluate(data[0])
    tables, metadata = exact.sensitivity_tables(primary, old, data[1], data[2])
    assert {key: len(rows) for key, rows in tables.items()} == exact.TABLE_COUNTS
    assert len(metadata["measurement_coverage"]) == 60
    assert len({row["original_feature"] for row in tables["predictive_sensitivity_summary"]}) == 5
    assert all(row["defined"] for row in tables["predictive_sensitivity_summary"])


@pytest.mark.parametrize(
    "kind",
    [
        "mapping",
        "pair_mapping",
        "policy",
        "undefined",
        "measurement",
        "count",
        "parent",
        "source",
        "release",
        "science",
        "amendment",
    ],
)
def test_altered_protocol_refuses(primary, tmp_path, kind):
    value = payload(ROOT)
    if kind == "release":
        value["status"] = "released"
    elif kind == "science":
        value["scientific_completion"] = True
    else:
        key = {
            "mapping": "checkpoint_feature_mapping",
            "pair_mapping": "pair_feature_mapping",
            "policy": "numerical_policy",
            "undefined": "undefined_policy",
            "measurement": "measurement_rules",
            "count": "table_counts",
            "parent": "parents",
            "source": "sources",
            "amendment": "amendment",
        }[kind]
        value[key].pop(next(iter(value[key])))
    path = tmp_path / "altered.json"
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError, KeyError)):
        exact.ExactBridgeContract.load(path, primary)


def test_real_loader_and_actual_missing_runs_before_any_output(primary, tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload(ROOT)))
    contract = exact.ExactBridgeContract.load(path, primary)
    with pytest.raises(ValueError, match="ordinary retained run"):
        exact.gather(contract, SimpleNamespace(experiment_root=tmp_path))


@pytest.mark.parametrize(
    "filename",
    [
        "measurement_comparison",
        "predictive_sensitivity_summary",
        "held_out_predictions",
        "feature_prediction_summary",
        "bridge_rows",
    ],
)
def test_rehashed_sensitivity_values_cannot_self_authenticate(primary, tmp_path, filename):
    data = fixture(primary)
    old, _ = evaluate(data[0])
    tables, metadata = exact.sensitivity_tables(primary, old, data[1], data[2])
    plan = {"scope": "engineering_synthetic_exact_bridge_fixture", "scientific_completion": False}
    output = tmp_path / "bundle"
    save_bundle(output, plan, metadata, tables)
    changed = copy.deepcopy(tables[filename])
    key = next(key for key, value in changed[0].items() if type(value) is float)
    changed[0][key] += 0.0001
    name = filename + ".csv"
    (output / name).write_bytes(csv_bytes(changed))
    manifest = read_json(output / "manifest.json")
    manifest["outputs"][name] = {"path": name, **file_identity(output / name)}
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fresh evidence"):
        inspect_bundle(output, plan, metadata, tables)


@pytest.mark.parametrize("content", ["x,x\n1,2\n", "x,y\n1\n", "x\nNaN\n", "x\nInfinity\n"])
def test_typed_csv_refuses_ambiguous_or_nonfinite_values(tmp_path, content):
    path = tmp_path / "bad.csv"
    path.write_text(content)
    with pytest.raises(ValueError):
        exact.typed_csv(path)


def test_typed_csv_keeps_zero_false_and_undefined_distinct(tmp_path):
    path = tmp_path / "good.csv"
    rows = [{"run_id": "test-a", "stage": 1, "value": None, "flag": False, "zero": 0.0}]
    path.write_bytes(csv_bytes(rows))
    assert exact.typed_csv(path) == rows


@pytest.mark.parametrize("mutate", [False, True])
def test_full_gather_with_explicit_synthetic_upstream_admission(
    primary, tmp_path, monkeypatch, mutate
):
    data = fixture(primary)
    old_tables, old_numeric = evaluate(data[0])
    old_plan = {"scope": "engineering_synthetic_original_bridge", "scientific_completion": False}
    old_evidence = {
        "source_bindings": [],
        "simulated_admission": True,
        "numerical_diagnostics": old_numeric,
    }
    bridge_root = tmp_path / "original"
    save_bundle(bridge_root, old_plan, old_evidence, old_tables)
    exact_root = tmp_path / "exact"
    exact_root.mkdir()
    for name in exact.exact_geometry.TABLE_COUNTS:
        rows = (
            data[1]
            if name == "checkpoint_exact_geometry"
            else data[2]
            if name == "run_pair_exact_subspace_overlap"
            else [{"synthetic": True}]
        )
        (exact_root / f"{name}.csv").write_bytes(csv_bytes(rows))
    (exact_root / "raw.control").write_text("explicit synthetic raw record")
    summary = {
        "simulated_admission": True,
        "raw_bindings": [{"path": "raw.control", **file_identity(exact_root / "raw.control")}],
    }
    (exact_root / "summary.json").write_text(json.dumps(summary))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(payload(ROOT)))
    contract = exact.ExactBridgeContract.load(protocol, primary)

    def original_reader(*args):
        return old_plan, old_evidence, old_tables

    def exact_reader(*args, **kwargs):
        assert kwargs["produce"] is False
        if mutate:
            (exact_root / "raw.control").write_text("altered raw record")
        return summary

    monkeypatch.setattr(exact.original, "gather", original_reader)
    monkeypatch.setattr(exact.exact_geometry, "operate", exact_reader)
    args = SimpleNamespace(
        bridge_root=bridge_root,
        exact_geometry_root=exact_root,
        experiment_root=tmp_path,
        reference=tmp_path,
    )
    if mutate:
        with pytest.raises(ValueError):
            exact.gather(contract, args)
    else:
        plan, evidence, tables = exact.gather(contract, args)
        assert plan["scope"] == exact.SCOPE and plan["scientific_completion"] is False
        assert evidence["original_bridge_evidence"]["simulated_admission"] is True
        assert {key: len(value) for key, value in tables.items()} == exact.TABLE_COUNTS
