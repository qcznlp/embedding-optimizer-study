import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from embed_optim import primary_v3_geometry as geometry
from embed_optim.corrected_geometry_summary import _basis_overlap, _top_bases
from embed_optim.geometry import TensorStore, _partition_summary, analyze_run, matrix_metrics
from embed_optim.primary_contract import digest, file_identity, read_json, require_same
from embed_optim.primary_geometry_kernels import audited_top_bases, pair_row
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_geometry_io import inspect_run, produce_run
from scripts.prepare_dense_v3_geometry import payload

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"
SETTINGS = {
    **geometry.SETTINGS,
    "sketch_rank": 2,
    "subspace_rank": 2,
    "oversample": 0,
    "power_iterations": 0,
}


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("shape", [(2, 2), (7, 5), (5, 7)])
@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("power", [0, 2])
def test_audited_bases_preserve_old_fp32_outputs_exactly(shape, rank, power):
    matrix = torch.randn(shape, generator=torch.Generator().manual_seed(51))
    settings = dict(rank=rank, oversample=2, power_iterations=power, seed=19)
    old = _top_bases(matrix, **settings)
    actual, health = audited_top_bases(matrix, **settings)
    assert all(torch.equal(x, y) for x, y in zip(old, actual, strict=True))
    assert health["retained_signal_supported"] is True
    assert health["projected_numerical_rank"] >= rank
    assert 0 < health["retained_frobenius_energy"] <= 1


def test_zero_is_undefined_but_null_direction_completion_is_refused():
    settings = dict(rank=2, oversample=0, power_iterations=0, seed=19)
    bases, health = audited_top_bases(torch.zeros(2, 2), **settings)
    assert bases is None and health["retained_signal_supported"] is None
    with pytest.raises(ValueError, match="retained signal rank"):
        audited_top_bases(torch.diag(torch.tensor([1.0, 0.0])), **settings)
    with pytest.raises(ValueError, match="underflowed"):
        audited_top_bases(torch.full((2, 2), 1e-30), **settings)


@pytest.mark.parametrize("kind", ["nan", "inf", "vector", "empty", "rank_boolean"])
def test_subspace_invalid_inputs_are_refused(kind):
    value = torch.eye(2)
    if kind in {"nan", "inf"}:
        value[0, 0] = float(kind)
    elif kind == "vector":
        value = value.flatten()
    elif kind == "empty":
        value = torch.empty(0, 2)
    with pytest.raises(ValueError):
        audited_top_bases(
            value,
            rank=True if kind == "rank_boolean" else 2,
            oversample=0,
            power_iterations=0,
            seed=19,
        )


def test_overlap_matches_projector_trace_and_is_basis_rotation_invariant():
    first = torch.eye(4)[:, :2], torch.eye(5)[:, :2]
    second = torch.eye(4)[:, 1:3], torch.eye(5)[:, 2:4]
    expected = []
    for left, right in zip(first, second, strict=True):
        a, b = left.numpy().astype(np.float64), right.numpy().astype(np.float64)
        expected.append(float(np.trace((a @ a.T) @ (b @ b.T)) / 2))
    assert _basis_overlap(first, second) == (*expected, sum(expected) / 2)
    rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]])
    assert _basis_overlap(tuple(x @ rotation for x in first), second) == pytest.approx(
        _basis_overlap(first, second), abs=2e-7
    )


def test_pair_zero_exclusion_keeps_denominator_and_never_imputes_overlap():
    first = {"run_id": "a", "optimizer": {"name": "adamw", "lr": 1e-5}}
    second = {"run_id": "b", "optimizer": {"name": "muon", "lr": 1e-4}}
    shapes = {"a": [2, 2], "b": [4, 2]}
    basis = (torch.eye(2)[:, :1], torch.eye(2)[:, :1])
    row = pair_row(
        first,
        second,
        stage=1,
        steps=[1],
        kind="saved_segment",
        left_bases={"a": basis, "b": None},
        right_bases={"a": basis, "b": None},
        shapes=shapes,
    )
    assert row["defined_parameter_fraction"] == 1 / 3
    assert row["mean_subspace_overlap"] == 1
    row = pair_row(
        first,
        second,
        stage=1,
        steps=[1],
        kind="saved_segment",
        left_bases={"a": None, "b": None},
        right_bases={"a": basis, "b": None},
        shapes=shapes,
    )
    assert row["defined_parameter_fraction"] == 0 and row["mean_subspace_overlap"] is None


def make_fixture(
    root, run_id="fixture-adamw", optimizer="adamw", lr=1e-5, steps=(1, 2), matrices=1
):
    reference, run = root / "reference", root / run_id
    tensors = {
        f"encoder.layers.{index}.weight": torch.eye(2) * (index + 1) / 32
        for index in range(matrices)
    }
    tensors["encoder.embeddings.weight"] = torch.ones(3, 2)
    if not reference.exists():
        reference.mkdir()
        save_file(tensors, reference / "model.safetensors")
    for index, step in enumerate(steps):
        checkpoint = run / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        save_file(
            {
                name: weight + torch.diag(torch.tensor([0.25, 0.5])) * index
                if name.startswith("encoder.layers.")
                else weight
                for name, weight in tensors.items()
            },
            checkpoint / "model.safetensors",
        )
    with TensorStore(reference) as store:
        shapes = {name: list(store.shape(name)) for name in store.keys()}
        partition = _partition_summary(store)
    ref = {
        "files": [{"path": "model.safetensors", **file_identity(reference / "model.safetensors")}],
        "all_shapes": shapes,
        "hidden_shapes": {
            name: shape for name, shape in shapes.items() if name.startswith("encoder.layers.")
        },
    }
    expected = {
        "recipe": {
            "run_id": run_id,
            "optimizer": {"name": optimizer, "lr": lr},
            "model_family": "dense",
            "checkpoint_fractions": [stage / len(steps) for stage in range(1, len(steps) + 1)],
        }
    }
    checked = {
        "whole_run_artifacts_verified": True,
        "run_identity_sha256": digest(expected),
        "steps": list(steps),
        "checkpoints": [
            {
                "step": step,
                "files": [
                    {
                        "path": "model.safetensors",
                        **file_identity(run / f"checkpoint-{step}/model.safetensors"),
                    }
                ],
            }
            for step in steps
        ],
    }
    (run / "completed.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model_family": "dense",
                "checkpoints": list(steps),
                "dataset_fingerprint": "fixture",
                "optimizer_partition": partition,
            }
        )
    )
    (run / "run_config.json").write_text(json.dumps({"optimizer": expected["recipe"]["optimizer"]}))
    return run, expected, checked, reference, ref


def produce_fixture(tmp_path):
    fixture = make_fixture(tmp_path)
    output = tmp_path / "new-geometry"
    produce_run(*fixture, SETTINGS, output, scope="engineering_fixture", contract_sha="a" * 64)
    return fixture, output


def test_new_raw_records_equal_old_full_tensor_index_seeded_path(tmp_path):
    fixture, output = produce_fixture(tmp_path)
    run, _, _, reference, _ = fixture
    analyze_run(
        run,
        tmp_path / "old-geometry",
        reference=reference,
        **{key: SETTINGS[key] for key in ("sketch_rank", "oversample", "power_iterations", "seed")},
    )
    for step in (1, 2):
        records = [
            json.loads(line)
            for line in (tmp_path / f"old-geometry/records/checkpoint-{step}.jsonl")
            .read_text()
            .splitlines()
        ]
        require_same(records, read_json(output / f"checkpoint-{step}.json")["records"])
    checked = inspect_run(
        output, *fixture, SETTINGS, scope="engineering_fixture", contract_sha="a" * 64
    )
    assert len(checked["checkpoint_rows"]) == 2
    assert checked["checkpoint_rows"][1]["saved_segment_nonzero_parameter_fraction"] == 0.5
    singular = np.linalg.svd(np.diag([0.25, 0.5]), compute_uv=False)
    metric = matrix_metrics(torch.diag(torch.tensor([0.25, 0.5])), sketch_rank=2)
    assert metric["approx_stable_rank"] == pytest.approx(
        float(np.sum(singular**2) / singular[0] ** 2)
    )


@pytest.mark.parametrize(
    "kind", ["metric", "summary", "entry", "health", "basis", "anchor", "extra", "identity"]
)
def test_rehashed_outputs_fail_fresh_numeric_recomputation(tmp_path, kind):
    fixture, original = produce_fixture(tmp_path)
    output = tmp_path / "changed-geometry"
    shutil.copytree(original, output)
    manifest = read_json(output / "manifest.json")
    binding = manifest["outputs"][1]
    record_path = output / binding["records"]["path"]
    record = read_json(record_path)
    if kind == "metric":
        record["records"][0]["weight"]["frobenius_norm"] += 1
    elif kind == "summary":
        record["checkpoint_row"]["saved_segment_to_weight_ratio"] += 1
    elif kind == "entry":
        record["entry_records"][0]["saved_segment_nonzero_parameters"] -= 1
    elif kind == "health":
        record["subspace_health"][0]["relative_boundary_gap"] = 1
    elif kind == "anchor":
        record["previous_step"] = 0
    elif kind == "basis":
        path = output / binding["bases"]["path"]
        values = load_file(path)
        values[next(iter(values))][0, 0] += 0.1
        save_file(
            values,
            path,
            metadata={"scope": "engineering_fixture", "run_identity_sha256": digest(fixture[1])},
        )
        binding["bases"].update(file_identity(path))
    elif kind == "identity":
        manifest["plan"]["run_identity_sha256"] = "b" * 64
    else:
        (output / "undeclared.json").write_text("{}")
    record_path.write_text(json.dumps(record))
    binding["records"].update(file_identity(record_path))
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        inspect_run(output, *fixture, SETTINGS, scope="engineering_fixture", contract_sha="a" * 64)


def test_partial_geometry_is_not_adopted(tmp_path):
    fixture = make_fixture(tmp_path)
    output = tmp_path / "partial"
    output.mkdir()
    with pytest.raises(ValueError, match="never adopt"):
        produce_run(*fixture, SETTINGS, output, scope="engineering_fixture", contract_sha="a" * 64)
    with pytest.raises(ValueError):
        inspect_run(output, *fixture, SETTINGS, scope="engineering_fixture", contract_sha="a" * 64)


@pytest.fixture
def primary():
    return PrimaryV3Contract.load(ROOT / geometry.PARENTS["primary"][0], ROOT, CANDIDATE)


@pytest.mark.parametrize(
    "keys,value",
    [
        (("status",), "released"),
        (("schema_version",), True),
        (("scientific_completion",), True),
        (("formal_execution_authorized",), True),
        (("settings", "subspace_rank"), 1),
        (("settings", "sketch_rank"), 32),
        (("settings", "seed"), 42),
        (("settings", "power_iterations"), 0),
        (("settings", "cpu_threads"), 8),
        (("sources",), {}),
        (("parents", "primary", "sha256"), "0" * 64),
        (("weight_space_rules", "rate_pair_policy"), "Choose best BEIR rate"),
        (("subspace_validity", "zero"), "Impute zero"),
        (("subspace_validity", "nonzero_signal_rank"), "Drop deficient matrices"),
        (("table_counts", "checkpoint_geometry"), 15),
    ],
)
def test_protocol_refuses_changed_rule_or_identity(primary, tmp_path, keys, value):
    contents = payload(ROOT)
    node = contents
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(contents))
    with pytest.raises(ValueError):
        geometry.GeometryContract.load(path, primary)


def test_baseline_protocol_loads(primary, tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(payload(ROOT)))
    contract = geometry.GeometryContract.load(path, primary)
    assert contract.payload["table_counts"] == geometry.TABLE_COUNTS


@pytest.mark.parametrize("produce", [True, False])
def test_absent_primary_population_refused_before_reference_or_output(
    primary, tmp_path, monkeypatch, produce
):
    contract = SimpleNamespace(primary=primary)
    calls = []
    monkeypatch.setattr(geometry, "reference_identity", lambda *args: calls.append("reference"))
    with pytest.raises(ValueError, match="ordinary retained run"):
        geometry.operate(contract, tmp_path, tmp_path, tmp_path / "not-created", produce=produce)
    assert not calls and not (tmp_path / "not-created").exists()


def test_complete_synthetic_12_run_five_stage_production_and_reader(tmp_path, monkeypatch):
    # Only orchestration is simulated. These tiny matrices are not primary checkpoints.
    admitted = []
    reference = ref = None
    for optimizer, rates in (
        ("adamw", (1e-6, 3e-6, 1e-5, 3e-5)),
        ("muon", (1e-4, 3e-4, 1e-3, 3e-3)),
        ("normuon", (1e-4, 3e-4, 1e-3, 3e-3)),
    ):
        for lr in rates:
            run, expected, checked, reference, ref = make_fixture(
                tmp_path,
                f"fixture-{optimizer}-{lr}",
                optimizer,
                lr,
                steps=(1, 2, 3, 4, 5),
                matrices=88,
            )
            admitted.append((run, expected, checked))
    contract = SimpleNamespace(
        primary=SimpleNamespace(sha256="a" * 64),
        payload={"settings": SETTINGS},
        sha256="b" * 64,
        path=tmp_path / "fixture-protocol.json",
    )
    monkeypatch.setattr(geometry, "admit_primary", lambda *args: admitted)
    monkeypatch.setattr(geometry, "reference_identity", lambda *args: ref)
    monkeypatch.setattr(geometry.GeometryContract, "load", lambda *args: contract)
    output = tmp_path / "complete-fixture"
    first = geometry.operate(contract, tmp_path, reference, output, produce=True)
    second = geometry.operate(contract, tmp_path, reference, output, produce=False)
    assert first == second
    assert first["scientific_completion"] is False
    assert first["table_counts"] == {
        "checkpoint_geometry": 60,
        "run_pair_subspace_overlap": 660,
        "optimizer_pair_subspace_summary": 60,
        "subspace_health": 10560,
    }
    for name, count in first["table_counts"].items():
        assert len((output / f"{name}.csv").read_text().splitlines()) == count + 1
    assert math.comb(12, 2) * 5 * 2 == 660
