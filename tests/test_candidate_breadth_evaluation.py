from __future__ import annotations

import json

import numpy as np
import pytest

from embed_optim.candidate_breadth_evaluation import (
    METRICS,
    _baseline_check,
    candidate_width_metrics,
)


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
