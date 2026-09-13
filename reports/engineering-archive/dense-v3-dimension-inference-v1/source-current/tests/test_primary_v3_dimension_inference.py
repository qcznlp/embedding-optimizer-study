"""Complete downstream wiring with explicitly simulated upstream primary admission."""

import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim import primary_v3_dimension_inference as functional
from embed_optim.bridge_numerics import evaluate
from embed_optim.primary_contract import file_identity, read_json
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_outcomes import csv_bytes, inspect_bundle, save_bundle
from scripts.prepare_dense_v3_dimension_inference import payload

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"


@pytest.fixture(scope="module")
def primary():
    return PrimaryV3Contract.load(ROOT / functional.PARENTS["primary"][0], ROOT, CANDIDATE)


@pytest.fixture(scope="module")
def data(primary):
    return runpy.run_path(str(ROOT / "tests/test_dimension_inference.py"))["fixture"](primary)


@pytest.mark.parametrize(
    "kind",
    [
        "scope",
        "version",
        "release",
        "science",
        "execution",
        "install",
        "parent",
        "source",
        "rule",
        "numeric",
        "undefined",
        "count",
        "scientific_rule",
    ],
)
def test_changed_declaration_fails(primary, tmp_path, kind):
    value = payload(ROOT)
    if kind in ("scope", "version", "release", "science", "execution", "install"):
        field, change = {
            "scope": ("scope", "old"),
            "version": ("schema_version", True),
            "release": ("status", "released"),
            "science": ("scientific_completion", True),
            "execution": ("formal_execution_authorized", True),
            "install": ("manuscript_installation_authorized", True),
        }[kind]
        value[field] = change
    else:
        field = {
            "parent": "parents",
            "source": "sources",
            "rule": "rules",
            "numeric": "bridge_numerics",
            "undefined": "undefined_policy",
            "count": "table_counts",
            "scientific_rule": "scientific_rules",
        }[kind]
        value[field].pop(next(iter(value[field])))
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError, KeyError)):
        functional.FunctionalInferenceContract.load(path, primary)


def test_missing_actual_primary_rejected_before_any_later_input(primary, tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload(ROOT)))
    contract = functional.FunctionalInferenceContract.load(path, primary)
    with pytest.raises(ValueError, match="ordinary retained run"):
        functional.gather(contract, SimpleNamespace(experiment_root=tmp_path / "absent"))


@pytest.mark.parametrize("mutation", [None, "raw", "features", "original"])
def test_actual_downstream_reader_under_explicit_upstream_simulation(
    primary, data, tmp_path, monkeypatch, mutation
):
    raw, panel, _, _ = data
    old_tables, old_numeric = evaluate(panel)
    old_plan = {"scope": "engineering_synthetic_original_bridge", "scientific_completion": False}
    old_evidence = {
        "source_bindings": [],
        "simulated_admission": True,
        "numerical_diagnostics": old_numeric,
    }
    bridge_root = tmp_path / "original"
    save_bundle(bridge_root, old_plan, old_evidence, old_tables)
    vectors, features = tmp_path / "vectors", tmp_path / "features"
    vectors.mkdir()
    features.mkdir()
    (vectors / "raw.control").write_text("synthetic raw control; not model vectors")
    for name, rows in raw.items():
        (features / (name + ".csv")).write_bytes(csv_bytes(rows))
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload(ROOT)))
    contract = functional.FunctionalInferenceContract.load(path, primary)

    def original_reader(*args):
        return old_plan, old_evidence, old_tables

    def feature_reader(*args, **kwargs):
        assert (
            kwargs["write"] is False
            and kwargs["expected_vector_manifest_sha256"] == "explicit-fixture-anchor"
        )
        if mutation:
            target = {
                "raw": vectors / "raw.control",
                "features": features / "task_summary.csv",
                "original": bridge_root / "evidence.json",
            }[mutation]
            target.write_text(target.read_text() + "\n")
        return {
            "simulated_upstream_admission": True,
            "primary_models_encoded": 0,
            "scientific_completion": False,
        }

    monkeypatch.setattr(functional.original, "gather", original_reader)
    monkeypatch.setattr(functional.feature_reader, "process", feature_reader)
    args = SimpleNamespace(
        bridge_root=bridge_root,
        dimension_vectors=vectors,
        dimension_features=features,
        experiment_root=tmp_path,
        reference=tmp_path,
        probe_root=tmp_path,
        vector_manifest_sha256="explicit-fixture-anchor",
    )
    if mutation:
        with pytest.raises(ValueError):
            functional.gather(contract, args)
        return
    plan, evidence, tables = functional.gather(contract, args)
    assert evidence["original_bridge_evidence"]["simulated_admission"] is True
    assert evidence["dimension_feature_reader"]["simulated_upstream_admission"] is True
    assert plan["scientific_completion"] is evidence["manuscript_installed"] is False
    assert {name: len(rows) for name, rows in tables.items()} == functional.TABLE_COUNTS
    output = tmp_path / "result"
    assert save_bundle(output, plan, evidence, tables)["recomputed_bytes_verified"] is True
    assert (
        inspect_bundle(output, *functional.gather(contract, args))["recomputed_bytes_verified"]
        is True
    )
    for name in (
        "primary_contrasts",
        "rotation_contrasts",
        "figure_points",
        "held_out_predictions",
        "feature_prediction_summary",
    ):
        changed = copy.deepcopy(tables[name])
        key = next(key for key, value in changed[0].items() if type(value) is float)
        changed[0][key] += 0.001
        filename = name + ".csv"
        (output / filename).write_bytes(csv_bytes(changed))
        manifest = read_json(output / "manifest.json")
        manifest["outputs"][filename] = {"path": filename, **file_identity(output / filename)}
        (output / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="fresh evidence"):
            inspect_bundle(output, plan, evidence, tables)
        (output / filename).write_bytes(csv_bytes(tables[name]))
        manifest["outputs"][filename] = {"path": filename, **file_identity(output / filename)}
        (output / "manifest.json").write_text(json.dumps(manifest))
