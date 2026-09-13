from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from embed_optim.dimension_utilization import (
    _attribution_summary,
    _cosine_scores,
    _covariance_summary,
    _identity,
    _leave_one_out_metrics,
    _orthogonal_matrix,
    _parse_archive,
    _query_metrics,
    _require_rotation_invariance,
    _shared_masks,
    _validate_export,
    audit_outputs,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/dense_dimension_utilization_protocol.json"


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(7)
    queries = generator.normal(size=(5, 6))
    documents = generator.normal(size=(5, 4, 6))
    documents[:, 0] += queries
    return queries, documents


def test_leave_one_out_matches_explicit_coordinate_deletion() -> None:
    queries, documents = _fixture()
    ndcg, margin = _leave_one_out_metrics(queries, documents)
    for dimension in range(queries.shape[1]):
        keep = np.arange(queries.shape[1]) != dimension
        expected_ndcg, expected_margin, _ = _query_metrics(_cosine_scores(queries, documents, keep))
        np.testing.assert_allclose(ndcg[:, dimension], expected_ndcg, atol=1e-12)
        np.testing.assert_allclose(margin[:, dimension], expected_margin, atol=1e-12)


def test_shared_orthogonal_rotation_preserves_cosine_scores() -> None:
    queries, documents = _fixture()
    rotation = _orthogonal_matrix(queries.shape[1], 11)
    expected = _cosine_scores(queries, documents)
    observed = _cosine_scores(queries @ rotation, documents @ rotation)
    np.testing.assert_allclose(observed, expected, atol=1e-12)


def test_rotation_guard_rejects_rank_flip_even_with_tiny_score_change():
    original = np.asarray([[0.5, 0.5]])
    rotated = np.asarray([[0.5, 0.5 + 1e-12]])
    with pytest.raises(ValueError, match="changed positive ranks"):
        _require_rotation_invariance(original, rotated, "fixture")


def test_rotation_guard_retains_the_frozen_score_tolerance():
    original = np.asarray([[0.5, 0.2]])
    _require_rotation_invariance(original, original + 1e-12, "fixture")
    with pytest.raises(ValueError, match="changed full scores"):
        _require_rotation_invariance(original, original + 1e-5, "fixture")


@pytest.mark.parametrize("changed", [np.asarray([[0.5]]), np.asarray([[np.nan, 0.2]])])
def test_rotation_guard_rejects_malformed_score_arrays(changed):
    with pytest.raises(ValueError, match="shape or finiteness"):
        _require_rotation_invariance(np.asarray([[0.5, 0.2]]), changed, "fixture")


def test_attribution_summary_separates_helpful_and_degrading_mass() -> None:
    summary = _attribution_summary(np.asarray([-2.0, -1.0, 0.0, 1.0]), 1e-12)
    assert summary["degrading_coordinate_fraction"] == 0.25
    assert summary["degrading_attribution_mass"] == 1.0
    assert summary["helpful_attribution_mass"] == 3.0
    assert summary["helpful_mass_share"] == 0.75
    assert 0 < summary["absolute_attribution_participation_ratio"] <= 1
    assert 0 < summary["helpful_attribution_participation_ratio"] <= 1


def test_protocol_freezes_paired_masks_and_result_blind_corrected_role() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    visibility = protocol["visibility_at_freeze"]
    assert protocol["status"] == "prospective_corrected_dimension_utilization_lock"
    assert visibility["corrected_beir_outputs_visible"] is False
    assert visibility["corrected_dimension_utilization_outputs_visible"] is False
    assert protocol["rotation_control"]["seeds"] == [314159, 271828, 161803]
    first = _shared_masks(12, protocol)
    second = _shared_masks(12, protocol)
    assert len(first) == len(second) == 80
    for left, right in zip(first, second, strict=True):
        assert left["removed_fraction"] == right["removed_fraction"]
        assert left["draw"] == right["draw"]
        np.testing.assert_array_equal(left["keep"], right["keep"])


def test_archive_parser_accepts_corrected_and_historical_run_names(tmp_path: Path) -> None:
    corrected = _parse_archive(tmp_path / "padded-muon-3e-4" / "checkpoint-3907.npz", tmp_path)
    historical = _parse_archive(tmp_path / "adamw-lr1e-6" / "checkpoint-782.npz", tmp_path)

    assert corrected == {
        "run_id": "padded-muon-3e-4",
        "optimizer": "muon",
        "learning_rate": 3e-4,
        "step": 3907,
    }
    assert historical == {
        "run_id": "adamw-lr1e-6",
        "optimizer": "adamw",
        "learning_rate": 1e-6,
        "step": 782,
    }


def test_covariance_summary_recovers_two_equally_used_axes() -> None:
    vectors = np.asarray(((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)))

    summary = _covariance_summary(vectors)

    assert summary["entropy_effective_rank"] == pytest.approx(2.0)
    assert summary["normalized_effective_rank"] == pytest.approx(1.0)
    assert summary["stable_rank"] == pytest.approx(2.0)
    assert summary["leading_variance_fraction"] == pytest.approx(0.5)


def test_output_audit_rejects_content_tampering(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol = {
        "status": "prospective_corrected_dimension_utilization_lock",
        "inputs": {
            "historical_expected_checkpoints": 0,
            "task_groups": 1,
            "embedding_dimension": 2,
        },
        "claim_boundary": "test boundary",
    }
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    implementation = tmp_path / "implementation.py"
    implementation.write_text("# bound implementation\n", encoding="utf-8")
    archive = tmp_path / "pretrained.npz"
    archive.write_bytes(b"archive")
    export_manifest = tmp_path / "pretrained.npz.manifest.json"
    export_manifest.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "report"
    output_dir.mkdir()
    outputs = {}
    for name in (
        "checkpoint_summary",
        "task_summary",
        "random_removal",
        "rotation_summary",
        "coordinate_attribution",
    ):
        path = output_dir / f"{name}.dat"
        path.write_text(f"{name}\n", encoding="utf-8")
        outputs[name] = _identity(path, tmp_path)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "scope": "historical_dense_dimension_utilization",
        "implementation": _identity(implementation, tmp_path),
        "protocol": _identity(protocol_path, tmp_path),
        "inputs": [
            {
                "archive": _identity(archive, tmp_path),
                "manifest": _identity(export_manifest, tmp_path),
            }
        ],
        "coverage": {
            "checkpoint_exports": 0,
            "pretrained_exports": 1,
            "tasks": 1,
            "dimensions": 2,
            "task_checkpoint_rows": 1,
        },
        "outputs": outputs,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output_dir / "summary_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    args = Namespace(
        protocol=protocol_path,
        analysis_scope="historical",
        output_dir=output_dir,
        skip_rotation=False,
        repository=tmp_path,
    )

    assert audit_outputs(args)["status"] == "complete"
    (output_dir / "task_summary.dat").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        audit_outputs(args)


@pytest.mark.parametrize("mutation", [None, "ids", "groups", "nonfinite", "probe"])
def test_export_identity_is_checked_even_after_payload_is_rehashed(tmp_path, mutation):
    sample_ids = np.arange(224, dtype=np.int64)
    groups = np.asarray([f"task-{index // 16}" for index in range(224)])
    sample_digest = hashlib.sha256("".join(f"{i}\n" for i in sample_ids).encode()).hexdigest()
    expected = {
        "manifest_sha256": "a" * 64,
        "selection_sha256": "b" * 64,
        "selected_sample_ids_sha256": sample_digest,
        "task_counts": {f"task-{i}": 16 for i in range(14)},
    }
    spec_path = tmp_path / "probe.json"
    spec_path.write_text(json.dumps({"expected": expected}), encoding="utf-8")
    probe = {
        **expected,
        "frozen_spec": {"sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest()},
    }
    queries = np.ones((224, 4))
    if mutation == "ids":
        sample_ids = sample_ids[::-1]
    elif mutation == "groups":
        groups[0] = "task-1"
    elif mutation == "nonfinite":
        queries[0, 0] = np.nan
    elif mutation == "probe":
        probe["selection_sha256"] = "c" * 64
    path = tmp_path / "pretrained.npz"
    np.savez(
        path,
        sample_ids=sample_ids,
        sample_groups=groups,
        query_embeddings=queries,
        document_embeddings=np.ones((224, 8, 4)),
    )
    manifest = {
        "schema_version": 1,
        "family": "dense",
        "probe": probe,
        "encoding": {"positive_candidate_index": 0},
        "output": {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    }
    path.with_suffix(".npz.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if mutation is None:
        arrays, _ = _validate_export(path, 4, probe_spec=spec_path)
        assert np.array_equal(arrays["sample_ids"], sample_ids)
    else:
        with pytest.raises(ValueError):
            _validate_export(path, 4, probe_spec=spec_path)
