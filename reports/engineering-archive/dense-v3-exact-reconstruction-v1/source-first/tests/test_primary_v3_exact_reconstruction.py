"""New exact reconstruction controls; all generated model fixtures are synthetic."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from embed_optim import primary_v3_exact_geometry as exact
from embed_optim import primary_v3_geometry as original
from embed_optim.primary_contract import file_identity, read_json
from embed_optim.primary_exact_geometry_kernels import SETTINGS, exact_matrix
from embed_optim.primary_v3_exact_geometry_io import produce_run as produce_exact
from embed_optim.primary_v3_exact_geometry_primitives import (
    inspect_run_primitives,
    spectrum_metrics,
)
from embed_optim.primary_v3_geometry_io import produce_run as produce_original
from tests.test_primary_v3_geometry import SETTINGS as ORIGINAL_SETTINGS
from tests.test_primary_v3_geometry import make_fixture


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("shape", [(3, 5), (5, 3), (4, 4)])
@pytest.mark.parametrize("rank", [1, 2, 16])
def test_all_spectrum_defined_metrics_match_unchanged_full_svd(shape, rank):
    matrix = np.random.default_rng(127).normal(size=shape)
    settings = {**SETTINGS, "subspace_rank": rank}
    before, singular, _ = exact_matrix(matrix, settings)
    actual = spectrum_metrics(singular, shape, settings, before["relative_reconstruction_residual"])
    assert actual == before
    independent = torch.linalg.svdvals(torch.from_numpy(matrix)).numpy()
    np.testing.assert_allclose(singular, independent, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize(
    "diagonal,status",
    [
        ([0, 0, 0, 0], "zero"),
        ([3, 0, 0, 0], "insufficient_signal_rank"),
        ([3, 2, 2, 1], "unresolved_boundary"),
        ([4, 3, 2, 1], "resolved"),
    ],
)
def test_all_statuses_recomputed_from_full_spectrum(diagonal, status):
    settings = {**SETTINGS, "subspace_rank": 2}
    before, singular, _ = exact_matrix(np.diag(diagonal), settings)
    actual = spectrum_metrics(
        singular, (4, 4), settings, before["relative_reconstruction_residual"]
    )
    assert actual == before and actual["basis_status"] == status


@pytest.mark.parametrize("kind", ["float32", "short", "unsorted", "negative", "nan", "inf"])
def test_invalid_or_truncated_spectrum_is_not_accepted(kind):
    singular = np.array([4.0, 3.0, 2.0, 1.0])
    if kind == "float32":
        singular = singular.astype(np.float32)
    elif kind == "short":
        singular = singular[:2]
    elif kind == "unsorted":
        singular[2] = 3.5
    elif kind == "negative":
        singular[-1] = -1
    else:
        singular[-1] = float(kind)
    with pytest.raises(ValueError):
        spectrum_metrics(singular, (4, 4), SETTINGS, 0.0)


@pytest.mark.parametrize("residual", [-1.0, float("nan"), True, None])
def test_invalid_recorded_residual_refused(residual):
    with pytest.raises(ValueError):
        spectrum_metrics(np.array([4.0, 3.0, 2.0, 1.0]), (4, 4), SETTINGS, residual)


def test_residual_is_retained_measurement_not_falsely_recomputed():
    values = np.array([4.0, 3.0, 2.0, 1.0])
    a = spectrum_metrics(values, (4, 4), SETTINGS, 1e-15)
    b = spectrum_metrics(values, (4, 4), SETTINGS, 2e-15)
    assert a.pop("relative_reconstruction_residual") == 1e-15
    assert b.pop("relative_reconstruction_residual") == 2e-15
    assert a == b


def fixture(root):
    raw = make_fixture(root)
    contract = SimpleNamespace(payload={"settings": SETTINGS}, sha256="a" * 64)
    old = SimpleNamespace(payload={"settings": ORIGINAL_SETTINGS}, sha256="b" * 64)
    produce_exact(*raw, SETTINGS, root / "exact", scope=exact.SCOPE, contract_sha=contract.sha256)
    produce_original(
        *raw, ORIGINAL_SETTINGS, root / "original", scope=original.SCOPE, contract_sha=old.sha256
    )
    return raw, contract, old


def inspect(root, raw, contract, old):
    _, expected, checked, _, reference = raw
    return inspect_run_primitives(
        root / "exact", root / "original", expected, checked, reference, contract, old
    )


def test_actual_small_saved_weight_outputs_reconstruct_without_tensor_loading(
    tmp_path, monkeypatch
):
    raw, contract, old = fixture(tmp_path)
    import embed_optim.geometry as geometry

    monkeypatch.setattr(
        geometry.TensorStore,
        "__init__",
        lambda *a, **k: pytest.fail("Portable reader opened checkpoint weights"),
    )
    result = inspect(tmp_path, raw, contract, old)
    assert result["checkpoint_rows"] == result["stored_checkpoint_rows"]
    assert len(result["matrix_rows"]) == 4
    assert result["checkpoint_rows"][0]["exact_saved_segment_zero_parameters"] == 4


@pytest.mark.parametrize(
    "change",
    [
        "stable_rank",
        "entropy",
        "status",
        "threshold",
        "shape",
        "parameters",
        "run",
        "anchor",
        "missing_row",
        "extra_row",
        "float32",
        "singular",
        "missing_singular",
        "basis",
        "missing_basis",
        "extra_array",
        "zero_basis",
        "extra_file",
        "plan",
    ],
)
def test_rehashed_structural_and_numerical_exact_changes_refused(tmp_path, change):
    raw, contract, old = fixture(tmp_path)
    directory = tmp_path / "exact"
    manifest = read_json(directory / "manifest.json")
    entry = manifest["outputs"][1]
    record_path = directory / entry["records"]["path"]
    arrays_path = directory / entry["arrays"]["path"]
    value = read_json(record_path)
    arrays = load_file(arrays_path)
    if change in {"stable_rank", "entropy", "status", "threshold", "shape", "parameters", "run"}:
        field = {
            "entropy": "full_entropy_effective_rank",
            "status": "basis_status",
            "threshold": "numerical_threshold",
            "run": "run_id",
        }.get(change, change)
        value["records"][0][field] = {"status": "zero", "shape": [3, 2], "run": "held-run"}.get(
            change, 999.0
        )
    elif change == "anchor":
        value["previous_step"] = 0
    elif change == "missing_row":
        value["records"].pop()
    elif change == "extra_row":
        value["records"].append(copy.deepcopy(value["records"][0]))
    elif change == "extra_file":
        (directory / "undeclared.json").write_text("{}")
    elif change == "plan":
        manifest["plan"]["run_identity_sha256"] = "e" * 64
    else:
        singular = next(key for key in arrays if key.endswith("|singular"))
        basis = next(key for key in arrays if key.endswith("|left"))
        if change == "float32":
            arrays[singular] = arrays[singular].float()
        elif change == "singular":
            arrays[singular][0] += 0.1
        elif change == "missing_singular":
            del arrays[singular]
        elif change == "basis":
            arrays[basis][0, 0] += 0.1
        elif change == "missing_basis":
            del arrays[basis]
        elif change == "extra_array":
            arrays["undeclared"] = torch.zeros(1, dtype=torch.float64)
        elif change == "zero_basis":
            zero_entry = manifest["outputs"][0]
            zero_path = directory / zero_entry["arrays"]["path"]
            zero_arrays = load_file(zero_path)
            zero_arrays[basis] = torch.eye(2, dtype=torch.float64)
            save_file(
                zero_arrays,
                zero_path,
                metadata={
                    "scope": exact.SCOPE,
                    "run_identity_sha256": manifest["plan"]["run_identity_sha256"],
                },
            )
            zero_entry["arrays"].update(file_identity(zero_path))
    record_path.write_text(json.dumps(value))
    save_file(
        arrays,
        arrays_path,
        metadata={
            "scope": exact.SCOPE,
            "run_identity_sha256": manifest["plan"]["run_identity_sha256"],
        },
    )
    entry["records"].update(file_identity(record_path))
    entry["arrays"].update(file_identity(arrays_path))
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        inspect(tmp_path, raw, contract, old)
