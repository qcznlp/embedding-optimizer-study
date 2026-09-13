import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from embed_optim import primary_weight_entries as entries
from embed_optim.corrected_geometry_summary import (
    _basis_overlap,
    _nonzero_parameter_fraction,
    _top_bases,
)
from embed_optim.primary_contract import digest, file_identity
from embed_optim.primary_v3_contract import PrimaryV3Contract
from scripts.audit_dense_weight_entries import reference_bindings
from scripts.prepare_dense_weight_entries import payload

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"


@pytest.fixture
def primary():
    return PrimaryV3Contract.load(ROOT / "configs/dense_primary_v3_protocol.json", ROOT, CANDIDATE)


def test_old_matrix_proxy_is_not_individual_entry_fraction():
    matrix = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    old = [{"parameters": 4, "displacement": {"frobenius_norm": 1.0}}]
    assert _nonzero_parameter_fraction(old, "displacement") == 1
    assert entries.changed_entries(matrix, torch.zeros_like(matrix)) == 1
    rows = [
        {
            "tensor": "a",
            "shape": [2, 2],
            "parameters": 4,
            "saved_segment_nonzero_parameters": 1,
            "cumulative_nonzero_parameters": 1,
        }
    ]
    summary = entries.summarize_entries(rows, {"a": [2, 2]})
    assert summary["saved_segment_nonzero_parameter_fraction"] == 0.25
    assert summary["saved_segment_parameter_mass_in_nonzero_matrices"] == 1


def test_global_fraction_weights_individual_entries_and_retains_zero_matrices():
    rows = [
        {
            "tensor": "a",
            "shape": [2, 2],
            "parameters": 4,
            "saved_segment_nonzero_parameters": 1,
            "cumulative_nonzero_parameters": 4,
        },
        {
            "tensor": "b",
            "shape": [2, 4],
            "parameters": 8,
            "saved_segment_nonzero_parameters": 0,
            "cumulative_nonzero_parameters": 2,
        },
    ]
    summary = entries.summarize_entries(rows, {"a": [2, 2], "b": [2, 4]})
    assert summary["hidden_parameters"] == 12
    assert summary["saved_segment_nonzero_parameter_fraction"] == 1 / 12
    assert summary["cumulative_nonzero_parameter_fraction"] == 0.5
    assert summary["saved_segment_parameter_mass_in_nonzero_matrices"] == 1 / 3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1, 1), (2, 5), (8, 4)])
def test_exact_saved_value_counts_match_numpy_without_tolerance(dtype, shape):
    initial = torch.arange(np.prod(shape), dtype=torch.float64).reshape(shape).to(dtype)
    current = initial.clone()
    current.flatten()[::3] += 0.25
    before = initial.clone(), current.clone()
    expected = int(np.count_nonzero(current.double().numpy() != initial.double().numpy()))
    assert entries.changed_entries(current, initial) == expected
    assert torch.equal(initial, before[0]) and torch.equal(current, before[1])


def test_below_arbitrary_norm_threshold_changes_still_count():
    initial = torch.zeros((2, 2), dtype=torch.float64)
    current = initial.clone()
    current[0, 0] = 1e-250
    assert entries.changed_entries(current, initial) == 1
    current[1, 1] = -0.0
    assert entries.changed_entries(current, initial) == 1


@pytest.mark.parametrize(
    "kind", ["nan", "inf", "integer", "empty", "vector", "shape", "not_tensor"]
)
def test_invalid_tensor_inputs_are_rejected(kind):
    initial = torch.zeros((2, 2))
    current = initial.clone()
    if kind == "nan":
        current[0, 0] = float("nan")
    elif kind == "inf":
        current[0, 0] = float("inf")
    elif kind == "integer":
        current = current.long()
    elif kind == "empty":
        current, initial = torch.empty((0, 2)), torch.empty((0, 2))
    elif kind == "vector":
        current, initial = current.flatten(), initial.flatten()
    elif kind == "shape":
        current = torch.zeros((2, 3))
    else:
        current = [[1, 2]]
    with pytest.raises(ValueError):
        entries.changed_entries(current, initial)


@pytest.mark.parametrize(
    "kind", ["missing", "duplicate", "shape", "count", "negative", "boolean", "parameters"]
)
def test_population_and_integer_denominators_are_strict(kind):
    records = [
        {
            "tensor": "a",
            "shape": [2, 2],
            "parameters": 4,
            "saved_segment_nonzero_parameters": 1,
            "cumulative_nonzero_parameters": 1,
        }
    ]
    if kind == "missing":
        records.clear()
    elif kind == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif kind == "shape":
        records[0]["shape"] = [2, 3]
    elif kind == "count":
        records[0]["saved_segment_nonzero_parameters"] = 5
    elif kind == "negative":
        records[0]["saved_segment_nonzero_parameters"] = -1
    elif kind == "boolean":
        records[0]["saved_segment_nonzero_parameters"] = True
    else:
        records[0]["parameters"] = False
    with pytest.raises(ValueError):
        entries.summarize_entries(records, {"a": [2, 2]})


def test_rank_deficient_full_basis_overlap_is_not_signal_direction_overlap():
    # Retained as a limitation of the unchanged fixed-rank subspace measure, not a new correction.
    first, second = torch.diag(torch.tensor([1.0, 0.0])), torch.diag(torch.tensor([0.0, 1.0]))
    settings = dict(rank=2, oversample=0, power_iterations=0, seed=20260903)
    assert _basis_overlap(_top_bases(first, **settings), _top_bases(second, **settings)) == (
        1.0,
        1.0,
        1.0,
    )
    settings["rank"] = 1
    overlap = _basis_overlap(_top_bases(first, **settings), _top_bases(second, **settings))
    assert overlap == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert _top_bases(torch.zeros_like(first), **settings) is None


def test_saved_segment_uses_previous_but_cumulative_always_uses_initial(tmp_path):
    shape = {"encoder.layers.0.weight": [2, 2]}
    initial = torch.zeros((2, 2))
    first = initial.clone()
    first[0, 0] = 1
    second = first.clone()
    second[1, 1] = 2
    reference, root = tmp_path / "reference", tmp_path / "run"
    for path, tensor in (
        (reference, initial),
        (root / "checkpoint-1", first),
        (root / "checkpoint-2", second),
    ):
        path.mkdir(parents=True)
        save_file({"encoder.layers.0.weight": tensor}, path / "model.safetensors")
    expected = {"recipe": {"run_id": "engineering_fixture"}}
    checked = {
        "whole_run_artifacts_verified": True,
        "run_identity_sha256": digest(expected),
        "steps": [1, 2],
        "checkpoints": [
            {
                "step": s,
                "files": [
                    {
                        "path": "model.safetensors",
                        **file_identity(root / f"checkpoint-{s}/model.safetensors"),
                    }
                ],
            }
            for s in (1, 2)
        ],
    }
    ref = {
        "files": [{"path": "model.safetensors", **file_identity(reference / "model.safetensors")}],
        "all_shapes": shape,
        "hidden_shapes": shape,
    }
    result = entries.stream_run_census(
        root, expected, checked, reference, ref, scope="engineering_fixture"
    )
    assert result["checkpoints"][0]["previous_step"] == 0
    assert result["checkpoints"][1]["previous_step"] == 1
    assert result["checkpoints"][1]["summary"]["saved_segment_nonzero_parameters"] == 1
    assert result["checkpoints"][1]["summary"]["cumulative_nonzero_parameters"] == 2
    assert result["scientific_completion"] is False


@pytest.mark.parametrize(
    "keys,value",
    [
        (("scope",), "historical"),
        (("schema_version",), True),
        (("status",), "reviewed_primary_execution_lock"),
        (("scientific_completion",), True),
        (("formal_execution_authorized",), True),
        (("sources",), {}),
        (("parents", "analysis", "sha256"), "0" * 64),
        (("primary", "sha256"), "0" * 64),
        (("measurement", "numerator"), "Count entire nonzero matrices"),
        (("measurement", "comparison"), "Norm greater than 1e-6"),
        (("measurement", "saved_segment_anchor"), "Always initialization"),
        (("measurement", "legacy_artifacts_rewritten"), True),
    ],
)
def test_amendment_rejects_changed_measurement_or_parent(primary, tmp_path, keys, value):
    value_dict = payload(ROOT, primary)
    node = value_dict
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(value_dict))
    with pytest.raises(ValueError):
        entries.load_amendment(path, primary)


def test_missing_primary_run_stops_before_any_tensor_read(primary, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(entries, "reference_identity", lambda *a: calls.append("reference"))
    with pytest.raises(ValueError, match="ordinary retained run"):
        entries.collect_primary(primary, payload(ROOT, primary), tmp_path, tmp_path)
    assert not calls


def test_diagnostic_reference_bindings_do_not_overwrite_absolute_path(tmp_path):
    files = [{"path": "model.safetensors", "bytes": 123, "sha256": "a" * 64}]
    result = reference_bindings(tmp_path, files)
    assert result == [
        {"path": str(tmp_path / "model.safetensors"), "bytes": 123, "sha256": "a" * 64}
    ]
    assert files[0]["path"] == "model.safetensors"
