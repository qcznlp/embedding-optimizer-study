from __future__ import annotations

import pytest

from embed_optim.candidate_breadth_summary import (
    candidate_breadth_decision,
    spearman,
)


def test_spearman_uses_average_ranks() -> None:
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486832981)
    with pytest.raises(ValueError, match="constant"):
        spearman([1, 1, 1], [1, 2, 3])


def _contrasts(*, broad_reversal: bool, broad_scale: float = 1.0) -> list[dict]:
    rows = []
    for optimizer in ("muon", "normuon"):
        rows.append(
            {
                "optimizer": optimizer,
                "negative_width": 7,
                "contrastive_loss_delta": -0.2,
                "positive_margin_delta": 0.1,
            }
        )
        rows.append(
            {
                "optimizer": optimizer,
                "negative_width": 2048,
                "contrastive_loss_delta": (0.2 if broad_reversal else -0.2) * broad_scale,
                "positive_margin_delta": (-0.1 if broad_reversal else 0.1) * broad_scale,
            }
        )
    return rows


def test_candidate_breadth_support_requires_endpoint_reversal_for_both_challengers() -> None:
    supported = candidate_breadth_decision(_contrasts(broad_reversal=True), baseline_pass=True)
    assert supported["decision"] == "supported"
    assert supported["width_2048_reversal_pass"] is True

    failed_baseline = candidate_breadth_decision(
        _contrasts(broad_reversal=True), baseline_pass=False
    )
    assert failed_baseline["decision"] == "not_supported"


def test_candidate_breadth_reports_attenuation_without_promoting_it_to_support() -> None:
    partial = candidate_breadth_decision(
        _contrasts(broad_reversal=False, broad_scale=0.4), baseline_pass=True
    )
    assert partial["decision"] == "partial_attenuation"
    assert partial["halfway_attenuation_pass"] is True

    unchanged = candidate_breadth_decision(_contrasts(broad_reversal=False), baseline_pass=True)
    assert unchanged["decision"] == "not_supported"
