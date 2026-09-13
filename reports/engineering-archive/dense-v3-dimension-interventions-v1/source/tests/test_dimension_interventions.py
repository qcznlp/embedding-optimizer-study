import copy
from pathlib import Path

import numpy as np
import pytest

from embed_optim import dimension_interventions as new
from embed_optim.dimension_utilization import _cosine_scores, _leave_one_out_metrics, _query_metrics
from embed_optim.primary_contract import read_json
from scripts.audit_dimension_deletion_boundary import cases, control, scalar_reference

ROOT = Path(__file__).resolve().parents[1]


def vectors(dimension=8, count=8):
    rng = np.random.default_rng(20260906)
    q = rng.standard_normal((count, dimension)).astype(np.float32)
    d = rng.standard_normal((count, 8, dimension)).astype(np.float32)
    return q, d


def test_old_fixed_boundary_is_preserved_and_confirmed():
    assert len(control()) == 2


def test_zero_remaining_direction_is_refused_not_imputed():
    with pytest.raises(ValueError, match="Coordinate 0:.*zero-norm"):
        new.leave_one_out_metrics(*cases()["zero_remaining_norm"])


def test_nonzero_dyadic_remainder_is_retained():
    q, d = cases()["nonzero_dyadic_remaining_direction"]
    actual = new.leave_one_out_metrics(q, d)
    for coordinate in range(3):
        expected = scalar_reference(q, d, coordinate)
        np.testing.assert_array_equal(actual[0][:, coordinate], expected["ndcg"])
        np.testing.assert_allclose(
            actual[1][:, coordinate], expected["margin"], rtol=1e-9, atol=1e-12
        )
    assert actual[0][0, 0] < 1


@pytest.mark.parametrize("dimension", [3, 8, 768])
def test_every_coordinate_agrees_with_independent_scalar_deletion(dimension):
    q, d = vectors(dimension, 3)
    actual = new.leave_one_out_metrics(q, d)
    for coordinate in range(dimension):
        expected = scalar_reference(q, d, coordinate)
        np.testing.assert_array_equal(actual[0][:, coordinate], expected["ndcg"])
        np.testing.assert_allclose(
            actual[1][:, coordinate], expected["margin"], rtol=1e-9, atol=1e-12
        )


def test_ordinary_case_agrees_with_legacy_without_adopting_its_boundary():
    q, d = vectors()
    for new_values, old_values in zip(
        new.leave_one_out_metrics(q, d), _leave_one_out_metrics(q, d), strict=True
    ):
        np.testing.assert_allclose(new_values, old_values, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("mutation", ["dimension", "candidates", "rows", "integer", "nan", "empty"])
def test_invalid_vectors_fail(mutation):
    q, d = vectors()
    if mutation == "dimension":
        q = q[:, :1]
    elif mutation == "candidates":
        d = d[:, :7]
    elif mutation == "rows":
        q = q[:-1]
    elif mutation == "integer":
        q = q.astype(int)
    elif mutation == "nan":
        d[0, 0, 0] = np.nan
    else:
        q, d = q[:0], d[:0]
    with pytest.raises(ValueError, match="finite floating"):
        new.leave_one_out_metrics(q, d)


def state_fixture(step=782):
    q, d = vectors()
    protocol = copy.deepcopy(read_json(ROOT / "configs/dense_dimension_utilization_protocol.json"))
    protocol["inputs"]["embedding_dimension"] = (
        8  # Explicit synthetic numerical fixture, not admission.
    )
    arrays = {
        "query_embeddings": q,
        "document_embeddings": d,
        "sample_ids": np.arange(len(q), dtype=np.int64),
        "sample_groups": np.asarray(["a"] * 4 + ["b"] * 4),
    }
    meta = {"run_id": "synthetic-muon", "optimizer": "muon", "learning_rate": 0.001, "step": step}
    return meta, arrays, protocol


@pytest.mark.parametrize("step,rotations", [(782, 0), (3907, 3), (0, 3)])
def test_complete_state_retains_all_masks_task_pairing_and_rotations(step, rotations):
    meta, arrays, protocol = state_fixture(step)
    if step == 0:
        meta = {"run_id": "pretrained", "optimizer": "pretrained", "learning_rate": None, "step": 0}
    result = new.compute_state(meta, arrays, protocol)
    assert {key: len(rows) for key, rows in result["tables"].items()} == {
        "checkpoint_summary": 1,
        "task_summary": 2,
        "random_removal": 160,
        "rotation_summary": rotations * 2,
    }
    assert len(result["rotation_checks"]) == len(result["rotated_attributions"]) == rotations
    assert result["attributions"]["margin_removal_gain"].dtype == np.float64
    assert result["attributions"]["margin_removal_gain"].shape == (2, 8)
    assert result["scientific_completion"] is False
    assert all(row["positive_ranks_identical"] for row in result["rotation_checks"])


def test_task_attribution_is_mean_before_split_not_mean_of_positive_parts():
    groups = np.asarray(["a", "a"])
    gains = np.asarray([[2.0, -1.0], [-2.0, -3.0]])
    result = new.task_attributions(groups, gains, gains)
    np.testing.assert_array_equal(result["margin_removal_gain"], [[0.0, -2.0]])


def test_relative_margin_is_explicitly_undefined_without_losing_absolute_values():
    meta, arrays, protocol = state_fixture()
    # Equal candidate vectors give exactly zero baseline and ablated margins.
    arrays["document_embeddings"] = np.repeat(arrays["document_embeddings"][:, :1], 8, axis=1)
    result = new.compute_state(meta, arrays, protocol)
    assert len(result["tables"]["random_removal"]) == 160
    assert all(
        row["relative_margin"] is None and row["margin"] == 0
        for row in result["tables"]["random_removal"]
    )


def test_positive_ties_follow_original_strict_greater_rule():
    q, d = vectors()
    d[:] = d[:, :1]
    ndcg, margin = new.leave_one_out_metrics(q, d)
    np.testing.assert_array_equal(ndcg, np.ones_like(ndcg))
    np.testing.assert_array_equal(margin, np.zeros_like(margin))


def test_full_cosine_does_not_change_the_frozen_shortlist_definition():
    q, d = vectors()
    ndcg, margin, ranks = _query_metrics(_cosine_scores(q, d))
    assert np.all((ranks >= 1) & (ranks <= 8))
    np.testing.assert_array_equal(ndcg, 1 / np.log2(ranks + 1))
    assert np.isfinite(margin).all()


@pytest.mark.parametrize(
    "field,value", [("step", -1), ("learning_rate", True), ("optimizer", "other"), ("run_id", "")]
)
def test_invalid_explicit_state_metadata_fails(field, value):
    meta, arrays, protocol = state_fixture()
    meta[field] = value
    with pytest.raises(ValueError):
        new.compute_state(meta, arrays, protocol)


@pytest.mark.parametrize("mutation", ["duplicate_ids", "group_count", "negative_ids", "dimension"])
def test_malformed_state_arrays_fail(mutation):
    meta, arrays, protocol = state_fixture()
    if mutation == "duplicate_ids":
        arrays["sample_ids"][0] = arrays["sample_ids"][1]
    elif mutation == "negative_ids":
        arrays["sample_ids"][0] = -1
    elif mutation == "group_count":
        arrays["sample_groups"] = arrays["sample_groups"][:-1]
    else:
        protocol["inputs"]["embedding_dimension"] = 768
    with pytest.raises(ValueError):
        new.compute_state(meta, arrays, protocol)
