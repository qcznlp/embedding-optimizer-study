import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from embed_optim import primary_v3_exact_geometry as geometry
from embed_optim.primary_contract import file_identity
from embed_optim.primary_exact_geometry_kernels import (
    SETTINGS,
    exact_matrix,
    exact_overlap,
    optimizer_rows,
    pair_row,
)
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_exact_geometry_io import compute_stage, inspect_run, produce_run
from scripts.prepare_dense_v3_exact_geometry import payload
from tests.test_primary_v3_geometry import make_fixture

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"
SMALL = {**SETTINGS, "subspace_rank": 2}


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (4, 4)])
def test_full_spectra_match_torch_and_all_singular_values_are_saved(shape):
    matrix = np.random.default_rng(123).normal(size=shape)
    metrics, singular, bases = exact_matrix(matrix, SMALL)
    expected = torch.linalg.svd(torch.from_numpy(matrix), full_matrices=False)
    np.testing.assert_allclose(singular, expected.S.numpy(), atol=1e-12, rtol=1e-12)
    assert len(singular) == min(shape) and metrics["basis_status"] == "resolved"
    assert metrics["stable_rank"] == pytest.approx(np.sum(matrix**2) / singular[0] ** 2)
    assert metrics["full_entropy_effective_rank"] > 2
    assert exact_overlap(
        bases, (expected.U[:, :2].numpy(), expected.Vh[:2].T.numpy())
    ) == pytest.approx((1, 1, 1), abs=1e-12)


@pytest.mark.parametrize(
    "diagonal,status",
    [
        ([0, 0, 0, 0], "zero"),
        ([3, 0, 0, 0], "insufficient_signal_rank"),
        ([3, 2, 2, 1], "unresolved_boundary"),
        ([4, 3, 2, 1], "resolved"),
    ],
)
def test_every_subspace_validity_state_retains_full_spectrum(diagonal, status):
    metrics, singular, bases = exact_matrix(np.diag(diagonal), SMALL)
    assert metrics["basis_status"] == status
    assert len(singular) == 4
    assert (bases is not None) == (status == "resolved")
    assert (metrics["stable_rank"] is None) == (status == "zero")


@pytest.mark.parametrize("gap,status", [(1e-8, "unresolved_boundary"), (4e-8, "resolved")])
def test_boundary_resolution_is_declared_not_posthoc(gap, status):
    metrics, _, _ = exact_matrix(np.diag([2.0, 2 * (1 - gap), 1]), {**SMALL, "subspace_rank": 1})
    assert metrics["basis_status"] == status


def test_full_dimension_basis_has_no_arbitrary_external_boundary():
    metrics, _, bases = exact_matrix(np.eye(2), SETTINGS)
    assert metrics["basis_status"] == "resolved"
    assert metrics["boundary_gap_to_leading"] is None and bases is not None


@pytest.mark.parametrize(
    "key,value",
    [
        ("subspace_rank", True),
        ("subspace_rank", 0),
        ("boundary_gap_to_leading_tolerance", -1),
        ("boundary_gap_to_leading_tolerance", float("nan")),
        ("boundary_gap_to_leading_tolerance", True),
    ],
)
def test_invalid_exact_settings_fail_closed(key, value):
    with pytest.raises(ValueError):
        exact_matrix(np.eye(2), {**SMALL, key: value})


def test_fast_overlap_matches_explicit_projectors_and_basis_rotation():
    rng = np.random.default_rng(42)
    first = tuple(np.linalg.qr(rng.normal(size=(size, 2)))[0] for size in (8, 7))
    second = tuple(np.linalg.qr(rng.normal(size=(size, 2)))[0] for size in (8, 7))
    exact = [float(np.trace((a @ a.T) @ (b @ b.T)) / 2) for a, b in zip(first, second, strict=True)]
    assert exact_overlap(first, second) == pytest.approx((*exact, sum(exact) / 2), abs=1e-13)
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
    assert exact_overlap(tuple(x @ rotation for x in first), second) == pytest.approx(
        exact_overlap(first, second), abs=1e-13
    )
    with pytest.raises(ValueError, match="orthonormal"):
        exact_overlap(tuple(x * 1.001 for x in first), second)


def pair(unsafe):
    def recipe(name):
        return {"run_id": name, "optimizer": {"name": "adamw", "lr": 1e-5}}

    basis = (np.eye(4)[:, :1],) * 2
    first = {"a": {"status": "resolved", "bases": basis}, "b": {"status": unsafe, "bases": None}}
    second = {name: {"status": "resolved", "bases": basis} for name in ("a", "b")}
    return pair_row(
        recipe("a"),
        recipe("b"),
        stage=1,
        steps=[1],
        kind="saved_segment",
        first_views=first,
        second_views=second,
        shapes={"a": [4, 4], "b": [4, 4]},
        rank=1,
    )


@pytest.mark.parametrize("status", ["insufficient_signal_rank", "unresolved_boundary"])
def test_unresolved_nonzero_population_is_not_silently_dropped(status):
    row = pair(status)
    assert row["mean_subspace_overlap"] is None
    assert row["resolved_only_mean_subspace_overlap"] == 1
    assert row["defined_parameter_fraction"] == 0.5
    assert row["all_nonzero_pair_subspaces_resolved"] is False
    summary = optimizer_rows(
        [row, {**row, "mean_subspace_overlap": 0.8, "all_nonzero_pair_subspaces_resolved": True}]
    )[0]
    assert summary["mean_subspace_overlap"] is None
    assert summary["run_pairs"] == 2 and summary["defined_run_pairs"] == 1


def test_zero_pair_exclusion_is_distinct_and_denominator_is_visible():
    row = pair("zero")
    assert row["mean_subspace_overlap"] == 1
    assert row["defined_parameter_fraction"] == 0.5
    assert row["undefined_zero_parameters"] == 16
    assert row["all_nonzero_pair_subspaces_resolved"] is True


def output_fixture(tmp_path):
    fixture = make_fixture(tmp_path)
    output = tmp_path / "exact"
    produce_run(*fixture, SMALL, output, scope="engineering_fixture", contract_sha="a" * 64)
    return fixture, output


def test_exact_run_roundtrip_keeps_anchors_zero_semantics_and_complete_arrays(tmp_path):
    fixture, output = output_fixture(tmp_path)
    first = inspect_run(output, *fixture, SMALL, scope="engineering_fixture", contract_sha="a" * 64)
    assert len(first["matrix_rows"]) == 4 and len(first["checkpoint_rows"]) == 2
    assert (
        first["checkpoint_rows"][0]["exact_saved_segment_stable_rank_parameter_weighted_nonzero"]
        is None
    )
    assert first["checkpoint_rows"][0]["exact_saved_segment_zero_parameters"] == 4
    for step, previous in ((1, 0), (2, 1)):
        record = json.loads((output / f"checkpoint-{step}.json").read_text())
        assert record["previous_step"] == previous and record["cumulative_anchor_step"] == 0
    assert len(load_file(output / "checkpoint-1-exact.safetensors")) == 2
    assert len(load_file(output / "checkpoint-2-exact.safetensors")) == 6


@pytest.mark.parametrize(
    "change",
    [
        "metric",
        "status",
        "anchor",
        "missing_matrix",
        "singular",
        "basis",
        "extra_array",
        "extra_file",
        "identity",
        "partial",
    ],
)
def test_numerically_false_or_partial_rehashed_outputs_are_refused(tmp_path, change):
    fixture, output = output_fixture(tmp_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if change in {"metric", "status", "anchor", "missing_matrix"}:
        bound = manifest["outputs"][1]["records"]
        path = output / bound["path"]
        value = json.loads(path.read_text())
        if change == "metric":
            value["records"][0]["stable_rank"] += 0.125
        elif change == "status":
            value["records"][0]["basis_status"] = "zero"
        elif change == "anchor":
            value["previous_step"] = 0
        else:
            value["records"].pop()
        path.write_text(json.dumps(value))
        bound.update(file_identity(path))
    elif change in {"singular", "basis", "extra_array"}:
        bound = manifest["outputs"][1]["arrays"]
        path = output / bound["path"]
        arrays = load_file(path)
        if change == "extra_array":
            arrays["undeclared"] = torch.ones(2, dtype=torch.float64)
        else:
            name = next(
                name
                for name in arrays
                if name.endswith("singular" if change == "singular" else "left")
            )
            arrays[name].flatten()[0] += 0.125
        save_file(
            arrays,
            path,
            metadata={
                "scope": "engineering_fixture",
                "run_identity_sha256": manifest["plan"]["run_identity_sha256"],
            },
        )
        bound.update(file_identity(path))
    elif change == "extra_file":
        (output / "extra.json").write_text("{}")
    elif change == "identity":
        manifest["plan"]["run_identity_sha256"] = "0" * 64
    else:
        manifest["outputs"].pop()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        inspect_run(output, *fixture, SMALL, scope="engineering_fixture", contract_sha="a" * 64)


def test_equal_displacement_reuse_does_not_change_measurement_or_nonmatching_anchor(
    tmp_path, monkeypatch
):
    import embed_optim.primary_v3_exact_geometry_io as io

    run, expected, checked, reference, ref = make_fixture(tmp_path, steps=(1, 2, 3))
    calls = []
    original = io.exact_matrix

    def counted(*args):
        calls.append(args[0].copy())
        return original(*args)

    monkeypatch.setattr(io, "exact_matrix", counted)
    compute_stage(run, expected["recipe"], checked["steps"], 2, reference, ref, SMALL)
    assert len(calls) == 1  # Checkpoint 1 is identical to base in the fixture.
    calls.clear()
    record, _ = compute_stage(run, expected["recipe"], checked["steps"], 3, reference, ref, SMALL)
    assert len(calls) == 2
    assert record["checkpoint_row"]["exact_cumulative_frobenius_norm"] == pytest.approx(
        2 * record["checkpoint_row"]["exact_saved_segment_frobenius_norm"]
    )


@pytest.fixture
def primary():
    return PrimaryV3Contract.load(ROOT / "configs/dense_primary_v3_protocol.json", ROOT, CANDIDATE)


def test_exact_protocol_loads_with_unchanged_approximate_and_bridge_parents(primary, tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload(ROOT)))
    contract = geometry.ExactGeometryContract.load(path, primary)
    assert contract.payload["settings"] == SETTINGS


@pytest.mark.parametrize(
    "keys,value",
    [
        (("settings", "subspace_rank"), 8),
        (("settings", "boundary_gap_to_leading_tolerance"), 1e-3),
        (("settings", "cpu_threads"), 1),
        (("measurement_rules", "undefined"), "Impute zero"),
        (("measurement_rules", "optimizer_mean"), "Average available"),
        (("measurement_rules", "old_features"), "Replace silently"),
        (("parents",), {}),
        (("sources",), {}),
        (("table_counts", "exact_matrix_geometry"), 5280),
        (("formal_execution_authorized",), True),
    ],
)
def test_changed_measurement_contract_is_refused(primary, tmp_path, keys, value):
    record = copy.deepcopy(payload(ROOT))
    node = record
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        geometry.ExactGeometryContract.load(path, primary)


@pytest.mark.parametrize("produce", [True, False])
def test_missing_primary_matrix_rejects_before_tensor_read_or_output(
    primary, tmp_path, monkeypatch, produce
):
    calls = []
    monkeypatch.setattr(geometry, "reference_identity", lambda *args: calls.append(True))
    with pytest.raises(ValueError, match="ordinary retained run"):
        geometry.operate(
            SimpleNamespace(primary=primary),
            tmp_path,
            tmp_path,
            tmp_path / "absent",
            produce=produce,
        )
    assert not calls and not (tmp_path / "absent").exists()


def test_full_synthetic_twelve_run_five_stage_88_matrix_chain(tmp_path, monkeypatch):
    # Full orchestration topology, synthetic two-dimensional matrices; not primary admission.
    admitted, reference, ref = [], None, None
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
        payload={"settings": SMALL},
        sha256="b" * 64,
        path=tmp_path / "plan.json",
    )
    monkeypatch.setattr(geometry, "admit_primary", lambda *args: admitted)
    monkeypatch.setattr(geometry, "reference_identity", lambda *args: ref)
    monkeypatch.setattr(geometry.ExactGeometryContract, "load", lambda *args: contract)
    output = tmp_path / "full-exact"
    first = geometry.operate(contract, tmp_path, reference, output, produce=True)
    assert geometry.operate(contract, tmp_path, reference, output, produce=False) == first
    assert first["table_counts"] == geometry.TABLE_COUNTS
    assert first["scientific_completion"] is False
    assert math.comb(12, 2) * 5 * 2 == 660
    for name, count in geometry.TABLE_COUNTS.items():
        assert len((output / f"{name}.csv").read_text().splitlines()) == count + 1
