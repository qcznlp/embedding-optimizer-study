from __future__ import annotations

import json

import numpy as np
import pytest
from datasets import Dataset

from embed_optim.candidate_breadth_evaluation import (
    METRICS,
    _audit_existing_evaluation,
    _baseline_check,
    _group_summaries,
    _identity,
    _sample_records_from_scores,
    candidate_width_metrics,
)
from embed_optim.geometry import SCHEMA_VERSION, _atomic_jsonl


def test_candidate_metrics_are_nested_and_rank_ties_are_adverse() -> None:
    scores = np.asarray(
        [
            [0.8, 0.7, 0.6, 0.9, 0.1],
            [0.5, 0.5, 0.4, 0.3, 0.2],
        ],
        dtype=np.float32,
    )
    metrics = candidate_width_metrics(scores, [1, 2, 4], temperature=0.02)
    for row in range(2):
        assert metrics[1]["contrastive_loss"][row] <= metrics[2]["contrastive_loss"][row]
        assert metrics[2]["contrastive_loss"][row] <= metrics[4]["contrastive_loss"][row]
        assert metrics[1]["positive_margin"][row] >= metrics[2]["positive_margin"][row]
        assert metrics[2]["positive_margin"][row] >= metrics[4]["positive_margin"][row]
    assert metrics[2]["top1_accuracy"].tolist() == [1.0, 0.0]
    assert metrics[4]["reciprocal_rank"].tolist() == pytest.approx([0.5, 0.5])


def test_candidate_metrics_reject_invalid_widths_and_nonfinite_scores() -> None:
    scores = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="widths"):
        candidate_width_metrics(scores, [2, 1], temperature=0.02)
    scores[0, 0] = np.nan
    with pytest.raises(ValueError, match="scores"):
        candidate_width_metrics(scores, [1], temperature=0.02)


def test_width_seven_baseline_check_is_sample_exact(tmp_path) -> None:
    records = []
    baseline = tmp_path / "sample_metrics.jsonl"
    with baseline.open("w", encoding="utf-8") as handle:
        for sample_id in (10, 20):
            row = {"sample_id": sample_id, **{metric: sample_id / 100 for metric in METRICS}}
            handle.write(json.dumps(row) + "\n")
            records.append(
                {
                    "sample_id": sample_id,
                    "negative_width": 7,
                    **{metric: sample_id / 100 + 1e-7 for metric in METRICS},
                }
            )
    result = _baseline_check(records, baseline, tolerance=1e-5)
    assert result["samples"] == 2
    assert result["maximum_absolute_error"] < 1e-5

    records[0]["positive_margin"] += 1e-3
    with pytest.raises(ValueError, match="do not reproduce"):
        _baseline_check(records, baseline, tolerance=1e-5)


def _evaluation_fixture(tmp_path):
    output = tmp_path / "evaluation"
    output.mkdir()
    queries = Dataset.from_list(
        [
            {"sample_id": 10, "source": "fiqa"},
            {"sample_id": 20, "source": "hotpotqa"},
        ]
    )
    widths = [1, 2]
    scores = np.asarray([[0.8, 0.7, 0.6], [0.5, 0.4, 0.6]], dtype=np.float32)
    sample_records = _sample_records_from_scores(scores, queries, widths, temperature=0.02)
    group_records = _group_summaries(sample_records)
    sample_path = output / "sample_metrics.jsonl"
    group_path = output / "group_metrics.jsonl"
    score_path = output / "scores.npz"
    _atomic_jsonl(sample_path, sample_records)
    _atomic_jsonl(group_path, group_records)
    np.savez_compressed(
        score_path,
        scores=scores,
        sample_ids=np.asarray(queries["sample_id"], dtype=np.int64),
        negative_widths=np.asarray(widths, dtype=np.int64),
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "checkpoint": {
            "path": str(tmp_path / "run" / "checkpoint-1"),
            "inputs": [],
            "run_config": {},
        },
        "protocol": {"path": "protocol.json", "bytes": 1, "sha256": "a" * 64},
        "data": {"path": "data", "audit": {"status": "complete"}},
        "negative_widths": widths,
        "temperature": 0.02,
    }
    manifest = {
        **identity,
        "sample_records": len(sample_records),
        "group_records": len(group_records),
        "baseline_reproduction": None,
        "outputs": {
            "sample_metrics": _identity(sample_path, output),
            "group_metrics": _identity(group_path, output),
            "scores": _identity(score_path, output),
        },
        "runtime": {
            "torch": "test",
            "sentence_transformers": "test",
            "cuda": None,
            "device": "cpu",
            "gpu_name": None,
        },
    }
    return output, queries, identity, manifest


def test_resume_audit_recomputes_metrics_from_raw_scores(tmp_path) -> None:
    output, queries, identity, manifest = _evaluation_fixture(tmp_path)
    assert (
        _audit_existing_evaluation(
            manifest,
            identity,
            output,
            queries,
            baseline_root=None,
            device="cpu",
        )
        == manifest
    )

    sample_path = output / "sample_metrics.jsonl"
    rows = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["positive_margin"] += 0.1
    _atomic_jsonl(sample_path, rows)
    manifest["outputs"]["sample_metrics"] = _identity(sample_path, output)
    with pytest.raises(ValueError, match="do not reproduce raw scores"):
        _audit_existing_evaluation(
            manifest,
            identity,
            output,
            queries,
            baseline_root=None,
            device="cpu",
        )
