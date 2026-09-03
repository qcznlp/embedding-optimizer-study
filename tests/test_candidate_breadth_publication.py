from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from embed_optim.candidate_breadth_publication import (
    CANDIDATE_BREADTH_MARKERS,
    _paired_transition_records,
    candidate_breadth_latex,
    candidate_breadth_markdown,
    render_candidate_breadth_publication,
)
from embed_optim.geometry import _sha256


def _rows() -> tuple[list[dict], list[dict]]:
    widths = (7, 10, 32, 128, 512, 2048)
    calibration = [
        {
            "optimizer": optimizer,
            "negative_width": width,
            "loss_beir_spearman": -0.8 + index * 0.1,
            "margin_beir_spearman": 0.8 - index * 0.1,
        }
        for optimizer in ("adamw", "muon", "normuon")
        for index, width in enumerate(widths)
    ]
    contrasts = [
        {
            "optimizer": optimizer,
            "negative_width": width,
            "samples": 224,
            "contrastive_loss_delta": -0.1 + index * 0.03,
            "contrastive_loss_delta_ci95_lower": -0.11 + index * 0.03,
            "contrastive_loss_delta_ci95_upper": -0.09 + index * 0.03,
            "positive_margin_delta": 0.1 - index * 0.03,
            "positive_margin_delta_ci95_lower": 0.09 - index * 0.03,
            "positive_margin_delta_ci95_upper": 0.11 - index * 0.03,
            "contrastive_loss_high_dose_better_fraction": 0.8 - index * 0.1,
            "positive_margin_high_dose_better_fraction": 0.75 - index * 0.09,
        }
        for optimizer in ("muon", "normuon")
        for index, width in enumerate(widths)
    ]
    return calibration, contrasts


def _summary(decision: str = "supported") -> dict:
    return {
        "baseline_maximum_absolute_error": 1e-6,
        "decision": {"decision": decision},
        "paired_uncertainty": {
            "recorded_at_utc": "2026-09-01T16:21:44Z",
            "candidate_breadth_data_or_scores_visible": False,
            "decision_rule_changed": False,
            "metrics": ["contrastive_loss", "positive_margin"],
            "method": "source-stratified paired percentile bootstrap",
            "strata": (
                "the seven training-data sources, resampled independently at their fixed "
                "32-query sizes"
            ),
            "replicates": 50_000,
            "seed": 20_260_902,
            "confidence": 0.95,
            "role": (
                "descriptive uncertainty only; intervals do not enter the frozen support rule"
            ),
        },
        "claim_boundary": "Post hoc only.",
    }


def test_publication_blocks_report_decision_endpoints_and_claim_boundary() -> None:
    calibration, contrasts = _rows()
    markdown = candidate_breadth_markdown(calibration, contrasts, _summary())
    latex = candidate_breadth_latex(calibration, contrasts, _summary())

    assert "**Frozen decision: Supported.**" in markdown
    assert "| NorMuon | 2,048 |" in markdown
    assert "candidate_breadth_calibration.svg" in markdown
    assert "-0.100000 [-0.110000, -0.090000]" in markdown
    assert "source-stratified paired percentile bootstrap intervals" in markdown
    assert "do not enter the frozen support rule" in markdown
    assert "earliest retained width" in markdown
    assert "Muon: 512 negatives" in markdown
    assert "80.0%/75.0% at width 7 → 30.0%/30.0% at width 2,048" in markdown
    assert "not an estimated threshold" in markdown
    assert "post hoc" in markdown
    assert "contributing to the shortlist--corpus gap" in markdown
    assert "does not establish that contribution causally" in markdown
    assert r"\label{fig:candidate-breadth}" in latex
    assert "candidate_breadth_calibration.pdf" in latex
    assert "-0.10000 [-0.11000, -0.09000]" in latex
    assert "shaded bands are descriptive 95\\% source-stratified" in latex
    assert "do not enter the frozen support rule" in latex
    assert r"\paragraph{Candidate-breadth uncertainty and paired prevalence.}" in latex
    assert r"\paragraph{Descriptive paired prevalence.}" in latex
    assert r"80.0\%/75.0\% at width 7 to 30.0\%/30.0\% at width 2,048" in latex
    assert latex.count(r"\newcommand{\CandidateBreadthFigure}") == 1
    assert "Frozen decision: Supported. Width 7" in latex
    assert "post hoc" in latex
    assert "does not establish that contribution causally" in latex
    assert r"\newcommand{\CandidateBreadthConclusion}" in latex
    assert r"\newcommand{\CandidateBreadthDiscussion}" in latex
    assert "supplements but does not alter" in latex


def test_paired_transition_reports_nonmonotone_grid_crossing_and_validates_prevalence() -> None:
    _, contrasts = _rows()
    records = _paired_transition_records(contrasts)

    assert [row["earliest_joint_mean_reversal_width"] for row in records] == [512, 512]

    contrasts[4]["contrastive_loss_delta"] = -0.01
    contrasts[5]["positive_margin_delta"] = 0.01
    records = _paired_transition_records(contrasts)
    assert records[0]["earliest_joint_mean_reversal_width"] is None

    contrasts[0]["samples"] = 223
    with pytest.raises(ValueError, match="sample count"):
        _paired_transition_records(contrasts)


def test_publication_rejects_reversed_paired_bootstrap_interval() -> None:
    calibration, contrasts = _rows()
    contrasts[0]["contrastive_loss_delta_ci95_lower"] = 0.2
    with pytest.raises(ValueError, match="confidence interval is reversed"):
        candidate_breadth_markdown(calibration, contrasts, _summary())


@pytest.mark.parametrize(
    ("decision", "label", "outcome"),
    (
        ("supported", "supported", "does not establish that contribution causally"),
        (
            "partial_attenuation",
            "partial attenuation",
            "required broad-set reversal does not occur",
        ),
        ("not_supported", "not supported", "do not support missing-candidate coverage"),
    ),
)
def test_candidate_conclusion_macro_reports_every_frozen_decision(
    decision: str, label: str, outcome: str
) -> None:
    calibration, contrasts = _rows()
    latex = candidate_breadth_latex(calibration, contrasts, _summary(decision))

    assert latex.count(r"\newcommand{\CandidateBreadthConclusion}") == 1
    assert latex.count(r"\newcommand{\CandidateBreadthDiscussion}") == 1
    assert latex.count(r"\newcommand{\CandidateBreadthFigure}") == 1
    assert f"nested-candidate decision was {label}" in latex
    assert f"candidate-breadth decision was {label}" in latex
    assert outcome in latex
    assert "supplements but does not alter" in latex


def test_failed_width_seven_bridge_is_disclosed() -> None:
    calibration, contrasts = _rows()
    summary = _summary("not_supported")
    summary["baseline_maximum_absolute_error"] = 8.286418572068214
    summary["decision"]["baseline_reproduction_pass"] = False

    markdown = candidate_breadth_markdown(calibration, contrasts, summary)
    latex = candidate_breadth_latex(calibration, contrasts, summary)

    for text in (markdown, latex):
        assert "prerequisite width-7 bridge failed" in text
        assert "maximum error 8.286419" in text
        assert "high-dose advantage is already absent at width 7" in text
        assert "Missing-candidate coverage therefore does not explain" in text


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _record(path: Path) -> dict:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def test_renderer_round_trips_blog_paper_and_manifest(monkeypatch, tmp_path: Path) -> None:
    protocol = tmp_path / "configs" / "candidate_breadth_probe.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("{}\n", encoding="utf-8")
    summary_dir = tmp_path / "reports" / "candidate-breadth"
    calibration_path = summary_dir / "calibration_by_width.csv"
    contrasts_path = summary_dir / "high_dose_contrasts.csv"
    svg_path = summary_dir / "candidate_breadth_calibration.svg"
    pdf_path = summary_dir / "candidate_breadth_calibration.pdf"
    calibration, contrasts = _rows()
    _write_csv(calibration_path, calibration)
    _write_csv(contrasts_path, contrasts)
    svg_path.write_text("<svg/>\n", encoding="utf-8")
    pdf_path.write_bytes(b"%PDF-fake\n")
    summary = {
        **_summary(),
        "outputs": {
            "calibration": _record(calibration_path),
            "contrasts": _record(contrasts_path),
            "figure_svg": _record(svg_path),
            "figure_pdf": _record(pdf_path),
        },
    }
    (summary_dir / "summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )
    blog = tmp_path / "docs" / "blog.md"
    blog.parent.mkdir(parents=True)
    blog.write_text(
        f"before\n{CANDIDATE_BREADTH_MARKERS[0]}\npending\n{CANDIDATE_BREADTH_MARKERS[1]}\nafter\n",
        encoding="utf-8",
    )
    paper = tmp_path / "paper" / "generated" / "candidate-breadth.tex"
    paper.parent.mkdir(parents=True)
    paper.write_text("pending\n", encoding="utf-8")
    manifest = summary_dir / "publication_manifest.json"
    monkeypatch.setattr(
        "embed_optim.candidate_breadth_publication.build_candidate_breadth_summary",
        lambda *_args, **_kwargs: summary,
    )

    rendered = render_candidate_breadth_publication(
        protocol,
        summary_dir=summary_dir,
        blog_path=blog,
        paper_path=paper,
        manifest_path=manifest,
    )
    audited = render_candidate_breadth_publication(
        protocol,
        summary_dir=summary_dir,
        blog_path=blog,
        paper_path=paper,
        manifest_path=manifest,
        audit_only=True,
    )

    assert rendered == audited
    assert rendered["status"] == "complete"
    assert "Frozen decision: Supported" in blog.read_text(encoding="utf-8")
    paper_text = paper.read_text(encoding="utf-8")
    assert r"\label{fig:candidate-breadth}" in paper_text
    assert r"\newcommand{\CandidateBreadthConclusion}" in paper_text
    assert r"\newcommand{\CandidateBreadthDiscussion}" in paper_text
    assert r"\newcommand{\CandidateBreadthFigure}" in paper_text
    assert "The post-hoc nested-candidate decision was supported" in paper_text
    assert json.loads(manifest.read_text(encoding="utf-8")) == rendered
