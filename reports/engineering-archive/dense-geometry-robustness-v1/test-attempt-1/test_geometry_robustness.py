import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from embed_optim.geometry_robustness import (
    exact_reference,
    finite_matrix,
    orthogonal_basis,
    projector_comparison,
    spectrum_summary,
    subspace_quality,
)
from scripts.audit_dense_geometry_robustness import compare_arrays, pair_measure
from scripts.prepare_dense_geometry_robustness import VARIANTS, load_protocol, payload

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("shape", [(8, 5), (5, 8), (6, 6)])
def test_two_full_svd_backends_and_projectors_agree(shape):
    matrix = np.random.default_rng(123).normal(size=shape)
    first, metrics = exact_reference(matrix)
    second, other = exact_reference(matrix, backend="torch")
    np.testing.assert_allclose(first[1], second[1], rtol=1e-12, atol=1e-12)
    for field in ("spectral_norm", "frobenius_norm", "stable_rank", "full_entropy_effective_rank"):
        assert metrics[field] == pytest.approx(other[field], rel=1e-12)
    for rank in (1, 3):
        result = subspace_quality(
            matrix, first, (second[0][:, :rank], second[2][:, :rank]), rank=rank
        )
        for side in ("left", "right"):
            assert result[side]["overlap"] == pytest.approx(1, abs=1e-12)
            assert result[side]["projector_rms_sine"] < 1e-12


@pytest.mark.parametrize("factor", [1e-120, 1, 1e120])
def test_full_spectrum_metrics_are_scale_invariant_and_analytic(factor):
    result = spectrum_summary(np.array([4, 3, 0], dtype=np.float64) * factor)
    assert result["stable_rank"] == 25 / 16
    assert result["frobenius_norm"] == pytest.approx(5 * factor, rel=1e-14, abs=0)
    probabilities = np.array([4 / 7, 3 / 7])
    assert result["full_entropy_effective_rank"] == pytest.approx(
        np.exp(-np.sum(probabilities * np.log(probabilities)))
    )


def test_zero_spectrum_has_no_rank_or_signal_subspace():
    exact, metrics = exact_reference(np.zeros((3, 2)))
    assert metrics["zero"] and metrics["stable_rank"] is None
    assert metrics["full_entropy_effective_rank"] is None
    with pytest.raises(ValueError, match="Zero displacement"):
        subspace_quality(np.zeros((3, 2)), exact, (np.eye(3)[:, :1], np.eye(2)[:, :1]), rank=1)


def test_rank_deficiency_is_not_completed_with_null_directions():
    matrix = np.diag([3.0, 0, 0])
    exact, _ = exact_reference(matrix)
    with pytest.raises(ValueError, match="does not support"):
        subspace_quality(matrix, exact, (np.eye(3)[:, :2], np.eye(3)[:, :2]), rank=2)


def test_equal_energy_does_not_establish_same_top_subspace_at_degenerate_boundary():
    matrix = np.eye(4)
    exact, _ = exact_reference(matrix)
    result = subspace_quality(matrix, exact, (np.eye(4)[:, 2:], np.eye(4)[:, 2:]), rank=2)
    assert result["exact_relative_boundary_gap"] == 0
    assert result["left_energy_relative_shortfall"] == 0
    assert result["left"]["overlap"] == 0
    assert result["left"]["projector_rms_sine"] == 1


def test_projector_measurement_matches_trace_and_is_basis_rotation_invariant():
    random = np.random.default_rng(3)
    first = np.linalg.qr(random.normal(size=(11, 3)))[0]
    second = np.linalg.qr(random.normal(size=(11, 3)))[0]
    rotation = np.linalg.qr(random.normal(size=(3, 3)))[0]
    result = projector_comparison(first, second)
    expected = np.trace((first @ first.T) @ (second @ second.T)) / 3
    assert result["overlap"] == pytest.approx(expected, abs=1e-14)
    changed = projector_comparison(first @ rotation, -second)
    for field in ("overlap", "projector_rms_sine", "largest_principal_angle_degrees"):
        assert changed[field] == pytest.approx(result[field], abs=1e-12)


def test_comparison_qr_does_not_modify_saved_fp32_basis():
    first = np.linalg.qr(np.random.default_rng(5).normal(size=(7, 2)))[0].astype(np.float32)
    unchanged = first.copy()
    result = projector_comparison(first, first)
    np.testing.assert_array_equal(first, unchanged)
    assert result["overlap"] == pytest.approx(1)
    assert result["projector_rms_sine"] == 0


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((0, 2)),
        np.ones(3),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.array([[1j]]),
        np.array([["abc"]]),
    ],
)
def test_invalid_matrices_are_refused(value):
    with pytest.raises(ValueError, match="finite real matrix"):
        finite_matrix(value)


@pytest.mark.parametrize("value", [np.zeros((3, 2)), np.eye(3) * 2, np.ones((2, 3))])
def test_nonorthogonal_basis_is_not_silently_repaired(value):
    with pytest.raises(ValueError):
        orthogonal_basis(value)


@pytest.mark.parametrize("value", [[1, 2], [-1], [np.inf], [np.nan], [], [[1]]])
def test_invalid_spectra_are_refused(value):
    with pytest.raises(ValueError):
        spectrum_summary(value)


@pytest.mark.parametrize("value", [1e200, 1e-200])
def test_unsupported_global_energy_range_fails_closed(value):
    with np.errstate(over="ignore", under="ignore"):
        with pytest.raises(ValueError, match="supported scale"):
            exact_reference(np.full((2, 2), value))


def test_declared_rank_and_shape_are_required():
    matrix = np.diag([3.0, 2.0, 1.0])
    exact, _ = exact_reference(matrix)
    for rank in (True, 0, 4):
        with pytest.raises(ValueError, match="rank"):
            subspace_quality(matrix, exact, (np.eye(3)[:, :1], np.eye(3)[:, :1]), rank=rank)
    with pytest.raises(ValueError, match="shape and rank"):
        subspace_quality(matrix, exact, (np.eye(3)[:, :1], np.eye(3)[:, :1]), rank=2)
    with pytest.raises(ValueError, match="same dimension"):
        projector_comparison(np.eye(3)[:, :1], np.eye(4)[:, :1])


def test_pair_measure_and_rehashed_array_controls():
    first = np.eye(4)[:, :2], np.eye(5)[:, :2]
    second = np.eye(4)[:, 2:], np.eye(5)[:, 1:3]
    assert pair_measure(first, second)["mean_overlap"] == 0.25
    array = {"a": torch.eye(3)}
    compare_arrays(array, {"a": torch.eye(3)})
    for invalid in ({}, {"a": torch.ones(3, 3)}, {"a": torch.eye(2)}):
        with pytest.raises(ValueError, match="recomputed"):
            compare_arrays(array, invalid)


def test_plan_fixes_every_case_without_primary_feature_changes(tmp_path):
    value = payload(ROOT)
    assert len(VARIANTS) == 13
    assert {row["rank"] for row in VARIANTS} == {8, 16, 32, 64}
    assert value["selection"]["matrices"] == 12
    assert value["interpretation"]["primary_features_changed"] is False
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(value))
    assert load_protocol(path, ROOT) == value


@pytest.mark.parametrize(
    "change",
    [
        "rank",
        "seed",
        "iterations",
        "select",
        "algorithm",
        "tolerance",
        "source",
        "parent",
        "release",
        "primary_feature",
    ],
)
def test_changed_diagnostic_plans_are_refused(tmp_path, change):
    value = copy.deepcopy(payload(ROOT))
    if change in {"rank", "seed", "iterations"}:
        key = {"rank": "rank", "seed": "seed_offset", "iterations": "power_iterations"}[change]
        value["settings"]["variants"][0][key] += 1
    elif change == "select":
        value["selection"]["tensor_names"].pop()
    elif change == "algorithm":
        value["selection"]["algorithms"].pop()
    elif change == "tolerance":
        value["settings"]["reference_rtol"] = 1e-3
    elif change == "source":
        value["sources"].pop(next(iter(value["sources"])))
    elif change == "parent":
        value["parents"]["geometry_acceptance"]["sha256"] = "0" * 64
    elif change == "release":
        value["formal_execution_authorized"] = True
    else:
        value["interpretation"]["primary_features_changed"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_protocol(path, ROOT)
