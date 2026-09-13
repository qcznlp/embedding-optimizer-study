"""Full-population synthetic wiring and real v3 missing-input refusals."""

import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embed_optim import primary_v3_dimension_exports as exports
from embed_optim import primary_v3_dimensions as features
from embed_optim.dimension_interventions import NUMERICAL_POLICY
from embed_optim.primary_contract import digest, file_identity, read_json
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_dimension_contract import (
    PARENTS,
    TABLE_COUNTS,
    DimensionContract,
    planned_states,
)
from embed_optim.primary_v3_dimension_vector_io import verify_loaded
from scripts.prepare_dense_v3_dimensions import payload

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"
HELPERS = runpy.run_path(str(ROOT / "tests/test_primary_v3_dimension_vectors.py"))


@pytest.fixture(scope="module")
def primary():
    return PrimaryV3Contract.load(ROOT / PARENTS["primary"][0], ROOT, CANDIDATE)


@pytest.fixture
def contract(primary, tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload(ROOT, primary)))
    return DimensionContract.load(path, primary)


def test_real_contract_fixed_population_and_unreleased_boundary(contract):
    states = planned_states(contract.primary)
    assert len(states) == 61
    assert states[0]["cell"] == "pretrained"
    assert sum(state["stage"] == 5 for state in states) == 12
    assert contract.payload["table_counts"] == TABLE_COUNTS
    with pytest.raises(ValueError, match="not execution authorized"):
        contract.require_execution()


@pytest.mark.parametrize(
    "key",
    [
        "encoding",
        "input_execution",
        "numerical_policy",
        "admission",
        "table_counts",
        "states",
        "sources",
        "parents",
        "release",
        "science",
    ],
)
def test_changed_contract_refuses(primary, tmp_path, key):
    value = payload(ROOT, primary)
    if key == "release":
        value["status"] = "released"
        value["formal_execution_authorized"] = True
    elif key == "science":
        value["scientific_completion"] = True
    elif key == "states":
        value[key].pop()
    else:
        value[key].pop(next(iter(value[key])))
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError, KeyError)):
        DimensionContract.load(path, primary, require_released=key == "release")


@pytest.mark.parametrize("action", ["plan", "inspect", "compute", "export"])
def test_real_primary_missing_before_model_or_output(contract, tmp_path, monkeypatch, action):
    monkeypatch.setattr(exports, "encode_state", lambda *args: pytest.fail("model must not load"))
    output = tmp_path / "output"
    inputs = (contract, tmp_path, tmp_path / "reference", tmp_path / "probe")
    with pytest.raises(ValueError):
        if action == "plan":
            exports.admission(*inputs)
        elif action == "inspect":
            exports.inspect_export(*inputs, output, expected_manifest_sha256="a" * 64)
        elif action == "compute":
            features.process(
                *inputs,
                tmp_path / "vectors",
                output,
                expected_vector_manifest_sha256="a" * 64,
                write=True,
            )
        else:
            exports.execute(*inputs, output)
    assert not output.exists()


def synthetic_admission(primary, tmp_path, monkeypatch):
    """Only upstream whole-run/probe/reference admission is simulated and labelled."""
    model, reference, _ = HELPERS["model_fixture"](tmp_path)
    arrays, identities = HELPERS["arrays_fixture"]()
    before = verify_loaded(model, reference)
    observed = {"before": before, "after": before, "parameters_unchanged": True}
    checked_runs = {}
    for run in primary.inputs["runs"]:
        run_id = run["run_id"]
        checkpoints = [
            {
                "step": step,
                "files": [
                    {"path": "model.safetensors", **file_identity(reference / "model.safetensors")}
                ],
            }
            for step in primary.payload["checkpoint_steps"]
        ]
        checked_runs[run_id] = {
            "root": reference,
            "checked": {
                "whole_run_artifacts_verified": True,
                "run_identity_sha256": digest(primary.expected_identity(run_id)),
                "steps": primary.payload["checkpoint_steps"],
                "checkpoints": checkpoints,
                "simulated_admission": True,
                "scientific_completion": False,
            },
        }
        for stage in checkpoints:
            location = reference / f"checkpoint-{stage['step']}"
            if not location.exists():
                location.mkdir()
                (location / "model.safetensors").write_bytes(
                    (reference / "model.safetensors").read_bytes()
                )
                (location / "modules.json").write_bytes((reference / "modules.json").read_bytes())
    contract = SimpleNamespace(
        primary=primary,
        sha256="synthetic-source-not-a-primary-admission",
        payload={"states": planned_states(primary), "numerical_policy": NUMERICAL_POLICY},
        scientific=read_json(ROOT / PARENTS["scientific"][0]),
        admit_runs=lambda _: checked_runs,
        probe=lambda _: (
            {"sample_id": arrays["sample_ids"], "source": arrays["sample_groups"]},
            {
                "files": [],
                "content_sha256": "synthetic",
                "row_identities": identities,
                "row_identities_sha256": digest(identities),
            },
        ),
    )
    contract.require_execution = lambda: contract  # Explicitly simulated; no GPU/real encoder.
    monkeypatch.setattr(exports, "reference_identity", lambda *_: {"simulated_reference": True})
    monkeypatch.setattr(exports, "recheck_contract", lambda _: contract)
    monkeypatch.setattr(features, "recheck_contract", lambda _: contract)
    monkeypatch.setattr(exports, "encode_state", lambda *_: (arrays, observed))
    return contract, reference, arrays, identities, observed, checked_runs


@pytest.fixture(scope="module")
def matrix(primary, tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-complete-dimension-matrix")
    patch = pytest.MonkeyPatch()
    fixture = synthetic_admission(primary, root, patch)
    contract, reference, arrays, identities, observed, runs = fixture
    output = root / "vectors"
    receipt = exports.write_matrix(contract, root, reference, root, output)
    assert receipt["model_encoding_repeated"] is False and len(receipt["states"]) == 61
    patch.undo()
    return root, fixture, output, receipt


def wire_existing(matrix, monkeypatch):
    root, fixture, output, receipt = matrix
    contract, reference, arrays, identities, observed, runs = fixture
    monkeypatch.setattr(exports, "reference_identity", lambda *_: {"simulated_reference": True})
    monkeypatch.setattr(exports, "recheck_contract", lambda _: contract)
    monkeypatch.setattr(features, "recheck_contract", lambda _: contract)
    return root, contract, reference, output, receipt


def test_complete_raw_export_reinspects_without_any_encoder(matrix, monkeypatch):
    root, contract, reference, output, receipt = wire_existing(matrix, monkeypatch)
    monkeypatch.setattr(exports, "encode_state", lambda *_: pytest.fail("readback must not encode"))
    result = exports.inspect_export(
        contract, root, reference, root, output, expected_manifest_sha256=receipt["manifest_sha256"]
    )
    assert result == receipt


@pytest.mark.parametrize("kind", ["missing", "extra", "renamed", "reordered", "unsealed"])
def test_synthetic_whole_run_population_cannot_be_relabelled(primary, tmp_path, monkeypatch, kind):
    contract, reference, _, _, _, runs = synthetic_admission(primary, tmp_path, monkeypatch)
    if kind == "missing":
        runs.pop(next(iter(runs)))
    elif kind == "extra":
        runs["historical-muon"] = next(iter(runs.values()))
    elif kind == "renamed":
        record = runs.pop(next(iter(runs)))
        runs["historical-adamw"] = record
    elif kind == "reordered":
        next(iter(runs.values()))["checked"]["checkpoints"].reverse()
    else:
        next(iter(runs.values()))["checked"]["whole_run_artifacts_verified"] = False
    with pytest.raises((ValueError, KeyError)):
        exports.admission(contract, tmp_path, reference, tmp_path)


def numerical_fixture(meta, arrays, protocol):
    """Explicit 8D synthetic kernel fixture expanded for full-schema wiring only.

    The 768D real numerical checks are the separate d09 parent, not this fixture.
    All 61 states, 14 tasks, 80 masks and every endpoint rotation still traverse IO.
    """
    from embed_optim.dimension_interventions import compute_state

    reduced = {
        key: value[..., :8] if key.endswith("embeddings") else value
        for key, value in arrays.items()
    }
    spec = copy.deepcopy(protocol)
    spec["inputs"]["embedding_dimension"] = 8
    result = compute_state(meta, reduced, spec)
    for values in [result["attributions"], *result["rotated_attributions"]]:
        for key in ("ndcg_removal_gain", "margin_removal_gain"):
            values[key] = np.tile(values[key], (1, 96))
    return result


@pytest.fixture(scope="module")
def feature_matrix(matrix):
    patch = pytest.MonkeyPatch()
    root, contract, reference, output, receipt = wire_existing(matrix, patch)
    patch.setattr(features, "compute_state", numerical_fixture)
    target = root / "features"
    value = features.process(
        contract,
        root,
        reference,
        root,
        output,
        target,
        expected_vector_manifest_sha256=receipt["manifest_sha256"],
        write=True,
    )
    replay = features.process(
        contract,
        root,
        reference,
        root,
        output,
        target,
        expected_vector_manifest_sha256=receipt["manifest_sha256"],
        write=False,
    )
    assert value == replay and value["raw_vector_states_recomputed"] == 61
    patch.undo()
    return target, value


def test_complete_features_retain_every_state_task_mask_rotation(feature_matrix):
    target, result = feature_matrix
    assert result["table_counts"] == TABLE_COUNTS
    assert len(read_json(target / "manifest.json")["states"]) == 61
    assert len(list(target.rglob("coordinate_attribution.npz"))) == 61
    assert result["plan"]["publication_inference_verified"] is False


@pytest.mark.parametrize("table", list(TABLE_COUNTS))
def test_rehashed_aggregate_must_equal_raw_vector_reconstruction(
    matrix, feature_matrix, tmp_path, monkeypatch, table
):
    import shutil

    root, contract, reference, vectors_root, receipt = wire_existing(matrix, monkeypatch)
    monkeypatch.setattr(features, "compute_state", numerical_fixture)
    target = tmp_path / "corrupt"
    shutil.copytree(feature_matrix[0], target)
    path = target / f"{table}.csv"
    path.write_bytes(path.read_bytes().replace(b"pretrained", b"historical", 1))
    manifest = read_json(target / "manifest.json")
    manifest["tables"][table].update(file_identity(path))
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="full raw-vector reconstruction"):
        features.process(
            contract,
            root,
            reference,
            root,
            vectors_root,
            target,
            expected_vector_manifest_sha256=receipt["manifest_sha256"],
            write=False,
        )
