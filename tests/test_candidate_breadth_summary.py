from __future__ import annotations

import pytest

from embed_optim.candidate_breadth_summary import (
    _candidate_breadth_figure,
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


def test_candidate_breadth_publication_figure_requires_and_writes_complete_rows(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    widths = [7, 10, 32, 128, 512, 2048]
    calibration = [
        {
            "optimizer": optimizer,
            "negative_width": width,
            "loss_beir_spearman": -0.8 + index * 0.05,
            "margin_beir_spearman": 0.8 - index * 0.05,
        }
        for optimizer in ("adamw", "muon", "normuon")
        for index, width in enumerate(widths)
    ]
    contrasts = [
        {
            "optimizer": optimizer,
            "negative_width": width,
            "contrastive_loss_delta": -0.1 + index * 0.03,
            "positive_margin_delta": 0.1 - index * 0.03,
        }
        for optimizer in ("muon", "normuon")
        for index, width in enumerate(widths)
    ]

    outputs = _candidate_breadth_figure(calibration, contrasts, tmp_path)

    assert set(outputs) == {"svg", "pdf"}
    for suffix, record in outputs.items():
        path = tmp_path / record["path"]
        assert path.suffix == f".{suffix}"
        assert path.stat().st_size == record["bytes"] > 0
        assert len(record["sha256"]) == 64

    repeated = _candidate_breadth_figure(calibration, contrasts, tmp_path)
    assert {suffix: item["sha256"] for suffix, item in repeated.items()} == {
        suffix: item["sha256"] for suffix, item in outputs.items()
    }

    with pytest.raises(ValueError, match="complete frozen width coverage"):
        _candidate_breadth_figure(calibration[:-1], contrasts, tmp_path)
