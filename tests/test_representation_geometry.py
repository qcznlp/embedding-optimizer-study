import json
from pathlib import Path

import numpy as np
import pytest
import torch

from embed_optim.representation_geometry import (
    analyze_probe,
    dense_probe_metrics,
    late_probe_metrics,
    late_ragged_probe_metrics,
    representation_summary,
)


def test_representation_summary_is_deterministic_and_scale_aware():
    vectors = torch.arange(80, dtype=torch.float32).reshape(20, 4) + 1

    first = representation_summary(vectors, max_vectors=8, seed=17)
    second = representation_summary(vectors, max_vectors=8, seed=17)

    assert first == second
    assert first["original_vectors"] == 20
    assert first["analyzed_vectors"] == 8
    assert first["sampled"] is True
    assert 0 <= first["normalized_effective_rank"] <= 1
    assert -1 <= first["mean_pairwise_cosine"] <= 1


def test_dense_probe_reports_exact_margin_and_reference_stability():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    documents = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        ]
    )
    reference = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])

    result = dense_probe_metrics(
        queries,
        documents,
        max_representation_vectors=100,
        seed=42,
        top_k=2,
        reference_scores=reference,
        sample_groups=["a", "b"],
    )

    score = result["score_geometry"]
    assert score["positive_hardest_negative_margin"]["mean"] == pytest.approx(0.0)
    assert score["top1_accuracy"] == pytest.approx(0.5)
    assert score["mean_reciprocal_rank"] == pytest.approx(0.75)
    assert score["reference_ranking"]["top1_agreement"] == pytest.approx(1.0)
    assert score["reference_ranking"]["score_drift_rms"] == pytest.approx(0.0)
    assert score["by_group"]["a"]["top1_accuracy"] == pytest.approx(1.0)
    assert score["by_group"]["b"]["top1_accuracy"] == pytest.approx(0.0)


def test_late_probe_reports_mean_maxsim_and_token_utilization():
    queries = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    documents = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]],
            ]
        ]
    )
    query_mask = torch.tensor([[True, True]])
    document_mask = torch.tensor([[[True, True, False], [True, True, False]]])

    result = late_probe_metrics(
        queries,
        documents,
        query_mask,
        document_mask,
        batch_size=1,
        max_representation_vectors=100,
        seed=42,
        top_k=2,
    )

    score = result["score_geometry"]
    assert score["positive_score"]["mean"] == pytest.approx(1.0)
    assert score["hardest_negative_score"]["mean"] == pytest.approx(0.5)
    assert score["positive_hardest_negative_margin"]["mean"] == pytest.approx(0.5)
    utilization = result["token_utilization"]
    assert utilization["positive_document_token_coverage"]["mean"] == pytest.approx(1.0)
    assert utilization["positive_document_token_repeated_selection_dominance"][
        "mean"
    ] == pytest.approx(0.5)

    ragged = late_ragged_probe_metrics(
        queries.reshape(-1, 2),
        documents[:, :, :2].reshape(-1, 2),
        torch.tensor([0, 2]),
        torch.tensor([0, 2, 4]),
        samples=1,
        batch_size=1,
        max_representation_vectors=100,
        seed=42,
        top_k=2,
    )
    assert ragged["storage"] == "ragged_offsets"
    assert ragged["score_geometry"] == result["score_geometry"]
    assert ragged["token_utilization"] == result["token_utilization"]


def test_analyze_probe_hashes_input_and_writes_atomic_json(tmp_path: Path):
    source = tmp_path / "dense.npz"
    output = tmp_path / "metrics.json"
    np.savez_compressed(
        source,
        sample_ids=np.array([101, 102]),
        sample_groups=np.array(["fiqa", "nq"]),
        query_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        document_embeddings=np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ],
            dtype=np.float32,
        ),
    )

    payload = analyze_probe(source, output, family="dense", label="checkpoint-782")

    assert json.loads(output.read_text()) == payload
    assert payload["schema_version"] == 1
    assert payload["label"] == "checkpoint-782"
    assert len(payload["input"]["sha256"]) == 64
    assert payload["input"]["arrays"]["document_embeddings"]["shape"] == [2, 2, 2]
    assert set(payload["metrics"]["score_geometry"]["by_group"]) == {"fiqa", "nq"}


def test_analyze_probe_rejects_reordered_reference_samples(tmp_path: Path):
    current = tmp_path / "current.npz"
    reference = tmp_path / "reference.npz"
    arrays = {
        "sample_ids": np.array([101, 102]),
        "query_embeddings": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "document_embeddings": np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ],
            dtype=np.float32,
        ),
    }
    np.savez(current, **arrays)
    np.savez(reference, **{**arrays, "sample_ids": arrays["sample_ids"][::-1]})

    with pytest.raises(ValueError, match="sample_ids differ or are reordered"):
        analyze_probe(
            current,
            tmp_path / "metrics.json",
            family="dense",
            reference_source=reference,
        )


def test_analyze_probe_rejects_invalid_late_masks(tmp_path: Path):
    source = tmp_path / "late.npz"
    np.savez(
        source,
        sample_ids=np.array([1]),
        query_embeddings=np.ones((1, 2, 3), dtype=np.float32),
        document_embeddings=np.ones((1, 2, 2, 3), dtype=np.float32),
        query_mask=np.array([[1, 1]], dtype=np.int8),
        document_mask=np.zeros((1, 2, 2), dtype=np.int8),
    )

    with pytest.raises(ValueError, match="no valid token"):
        analyze_probe(source, tmp_path / "metrics.json", family="late")
