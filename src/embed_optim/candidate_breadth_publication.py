from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from .candidate_breadth_summary import build_candidate_breadth_summary
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .mechanism_report import _atomic_text

WIDTHS = (7, 10, 32, 128, 512, 2048)
OPTIMIZERS = ("adamw", "muon", "normuon")
CHALLENGERS = ("muon", "normuon")
LABELS = {"adamw": "AdamW", "muon": "Muon", "normuon": "NorMuon"}
EXPECTED_SAMPLES = 224
UNCERTAINTY_METRICS = ("contrastive_loss", "positive_margin")


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": _relative(path, root),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite(value: Any, *, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Non-numeric candidate-breadth {context}: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"Non-finite candidate-breadth {context}: {value!r}")
    return number


def _paired_uncertainty(summary: dict[str, Any]) -> dict[str, Any]:
    uncertainty = summary.get("paired_uncertainty")
    if (
        not isinstance(uncertainty, dict)
        or set(uncertainty)
        != {
            "recorded_at_utc",
            "candidate_breadth_data_or_scores_visible",
            "decision_rule_changed",
            "metrics",
            "method",
            "strata",
            "replicates",
            "seed",
            "confidence",
            "role",
        }
        or uncertainty.get("recorded_at_utc") != "2026-09-01T16:21:44Z"
        or uncertainty.get("candidate_breadth_data_or_scores_visible") is not False
        or uncertainty.get("decision_rule_changed") is not False
        or uncertainty.get("metrics") != list(UNCERTAINTY_METRICS)
        or uncertainty.get("method") != "source-stratified paired percentile bootstrap"
        or uncertainty.get("strata")
        != "the seven training-data sources, resampled independently at their fixed 32-query sizes"
        or uncertainty.get("replicates") != 50_000
        or uncertainty.get("seed") != 20_260_902
        or uncertainty.get("confidence") != 0.95
        or uncertainty.get("role")
        != "descriptive uncertainty only; intervals do not enter the frozen support rule"
    ):
        raise ValueError("Candidate-breadth publication uncertainty contract changed")
    return uncertainty


def _validate_interval(row: dict[str, Any], metric: str) -> tuple[float, float]:
    lower = _finite(row[f"{metric}_delta_ci95_lower"], context=f"{metric} CI lower")
    upper = _finite(row[f"{metric}_delta_ci95_upper"], context=f"{metric} CI upper")
    if lower > upper:
        raise ValueError(f"Candidate-breadth {metric} confidence interval is reversed")
    return lower, upper


def _delta_interval(row: dict[str, Any], metric: str, *, digits: int) -> str:
    delta = _finite(row[f"{metric}_delta"], context=f"{metric} contrast")
    lower, upper = _validate_interval(row, metric)
    return f"{delta:+.{digits}f} [{lower:+.{digits}f}, {upper:+.{digits}f}]"


def load_candidate_breadth_publication_rows(
    summary_dir: Path,
    summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _paired_uncertainty(summary)
    outputs = summary.get("outputs", {})
    expected_outputs = {"calibration", "contrasts", "figure_svg", "figure_pdf"}
    if set(outputs) != expected_outputs:
        raise ValueError(
            "Candidate-breadth publication requires the complete summary output ledger"
        )
    for name, item in outputs.items():
        path = summary_dir / str(item.get("path", ""))
        if (
            not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or _sha256(path) != item.get("sha256")
        ):
            raise ValueError(f"Candidate-breadth publication input changed: {name}")

    calibration = []
    for row in _read_rows(summary_dir / outputs["calibration"]["path"]):
        calibration.append(
            {
                "optimizer": str(row["optimizer"]),
                "negative_width": int(row["negative_width"]),
                "loss_beir_spearman": _finite(
                    row["loss_beir_spearman"], context="loss correlation"
                ),
                "margin_beir_spearman": _finite(
                    row["margin_beir_spearman"], context="margin correlation"
                ),
            }
        )
    contrasts = []
    for row in _read_rows(summary_dir / outputs["contrasts"]["path"]):
        parsed = {
            "optimizer": str(row["optimizer"]),
            "negative_width": int(row["negative_width"]),
            "samples": int(row["samples"]),
            "contrastive_loss_delta": _finite(
                row["contrastive_loss_delta"], context="loss contrast"
            ),
            "positive_margin_delta": _finite(
                row["positive_margin_delta"], context="margin contrast"
            ),
            "contrastive_loss_delta_ci95_lower": _finite(
                row["contrastive_loss_delta_ci95_lower"], context="loss CI lower"
            ),
            "contrastive_loss_delta_ci95_upper": _finite(
                row["contrastive_loss_delta_ci95_upper"], context="loss CI upper"
            ),
            "positive_margin_delta_ci95_lower": _finite(
                row["positive_margin_delta_ci95_lower"], context="margin CI lower"
            ),
            "positive_margin_delta_ci95_upper": _finite(
                row["positive_margin_delta_ci95_upper"], context="margin CI upper"
            ),
            "contrastive_loss_high_dose_better_fraction": _finite(
                row["contrastive_loss_high_dose_better_fraction"],
                context="loss high-dose-better fraction",
            ),
            "positive_margin_high_dose_better_fraction": _finite(
                row["positive_margin_high_dose_better_fraction"],
                context="margin high-dose-better fraction",
            ),
        }
        if parsed["samples"] != EXPECTED_SAMPLES or any(
            not 0 <= parsed[field] <= 1
            for field in (
                "contrastive_loss_high_dose_better_fraction",
                "positive_margin_high_dose_better_fraction",
            )
        ):
            raise ValueError("Candidate-breadth paired prevalence row is invalid")
        for metric in UNCERTAINTY_METRICS:
            _validate_interval(parsed, metric)
        contrasts.append(parsed)
    expected_calibration = {(optimizer, width) for optimizer in OPTIMIZERS for width in WIDTHS}
    expected_contrasts = {(optimizer, width) for optimizer in CHALLENGERS for width in WIDTHS}
    if (
        len(calibration) != len(expected_calibration)
        or {(row["optimizer"], row["negative_width"]) for row in calibration}
        != expected_calibration
    ):
        raise ValueError("Candidate-breadth calibration publication rows are incomplete")
    if (
        len(contrasts) != len(expected_contrasts)
        or {(row["optimizer"], row["negative_width"]) for row in contrasts} != expected_contrasts
    ):
        raise ValueError("Candidate-breadth contrast publication rows are incomplete")
    return calibration, contrasts


def _paired_transition_records(contrast_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize grid-resolved mean sign crossings and paired query prevalence."""

    indexed = {(str(row["optimizer"]), int(row["negative_width"])): row for row in contrast_rows}
    expected = {(optimizer, width) for optimizer in CHALLENGERS for width in WIDTHS}
    if len(contrast_rows) != len(expected) or set(indexed) != expected:
        raise ValueError("Candidate-breadth paired transition rows are incomplete")
    records = []
    for optimizer in CHALLENGERS:
        rows = [indexed[(optimizer, width)] for width in WIDTHS]
        for row in rows:
            if int(row.get("samples", -1)) != EXPECTED_SAMPLES:
                raise ValueError("Candidate-breadth paired transition sample count differs")
            for field in (
                "contrastive_loss_high_dose_better_fraction",
                "positive_margin_high_dose_better_fraction",
            ):
                fraction = _finite(row.get(field), context=field.replace("_", " "))
                if not 0 <= fraction <= 1:
                    raise ValueError("Candidate-breadth paired transition fraction is invalid")
        crossing = next(
            (
                int(row["negative_width"])
                for row in rows
                if _finite(row["contrastive_loss_delta"], context="loss contrast") > 0
                and _finite(row["positive_margin_delta"], context="margin contrast") < 0
            ),
            None,
        )
        narrow = indexed[(optimizer, WIDTHS[0])]
        broad = indexed[(optimizer, WIDTHS[-1])]
        records.append(
            {
                "optimizer": optimizer,
                "earliest_joint_mean_reversal_width": crossing,
                "narrow_loss_high_dose_better_fraction": float(
                    narrow["contrastive_loss_high_dose_better_fraction"]
                ),
                "narrow_margin_high_dose_better_fraction": float(
                    narrow["positive_margin_high_dose_better_fraction"]
                ),
                "broad_loss_high_dose_better_fraction": float(
                    broad["contrastive_loss_high_dose_better_fraction"]
                ),
                "broad_margin_high_dose_better_fraction": float(
                    broad["positive_margin_high_dose_better_fraction"]
                ),
            }
        )
    return records


def _paired_transition_markdown(contrast_rows: list[dict[str, Any]]) -> str:
    parts = []
    for record in _paired_transition_records(contrast_rows):
        crossing = record["earliest_joint_mean_reversal_width"]
        crossing_text = "not observed" if crossing is None else f"{crossing:,} negatives"
        parts.append(
            f"{LABELS[record['optimizer']]}: {crossing_text}; paired high-dose win fractions "
            f"(loss/margin) {record['narrow_loss_high_dose_better_fraction']:.1%}/"
            f"{record['narrow_margin_high_dose_better_fraction']:.1%} at width 7 → "
            f"{record['broad_loss_high_dose_better_fraction']:.1%}/"
            f"{record['broad_margin_high_dose_better_fraction']:.1%} at width 2,048"
        )
    return (
        "**Descriptive paired prevalence:** the earliest retained width at which both mean "
        "contrasts favor the retrieval-optimal 3e-4 dose is "
        + "; ".join(parts)
        + ". This is a crossing on the six-point frozen grid, not an estimated threshold; the "
        "curves need not be monotone."
    )


def _paired_transition_latex(contrast_rows: list[dict[str, Any]]) -> str:
    parts = []
    for record in _paired_transition_records(contrast_rows):
        crossing = record["earliest_joint_mean_reversal_width"]
        crossing_text = "not observed" if crossing is None else f"{crossing:,} negatives"
        fractions = (
            f"{record['narrow_loss_high_dose_better_fraction']:.1%}/"
            f"{record['narrow_margin_high_dose_better_fraction']:.1%} at width 7 to "
            f"{record['broad_loss_high_dose_better_fraction']:.1%}/"
            f"{record['broad_margin_high_dose_better_fraction']:.1%} at width 2,048"
        ).replace("%", r"\%")
        parts.append(
            f"{LABELS[record['optimizer']]}: {crossing_text}; paired high-dose win fractions "
            f"(loss/margin) {fractions}"
        )
    return (
        r"\paragraph{Descriptive paired prevalence.} The earliest retained width at which both "
        r"mean contrasts favor the retrieval-optimal $3\!\times\!10^{-4}$ dose is "
        + "; ".join(parts)
        + ". This is a crossing on the six-point frozen grid, not an estimated threshold; the "
        "curves need not be monotone."
    )


def _decision_text(summary: dict[str, Any]) -> tuple[str, str]:
    decision = summary.get("decision", {}).get("decision")
    if decision == "supported":
        return (
            "Supported",
            "Both Muon-family optimizers reverse their high-dose ordering at 2,048 negatives, "
            "which is consistent with missing-candidate coverage contributing to the "
            "shortlist--corpus gap. Because this diagnostic is post hoc, it does not establish "
            "that contribution causally.",
        )
    if decision == "partial_attenuation":
        return (
            "Partial attenuation",
            "Both Muon-family contrasts move at least halfway toward zero, but the required broad-set "
            "reversal does not occur; candidate coverage is suggestive but not established.",
        )
    if decision == "not_supported":
        if summary.get("decision", {}).get("baseline_reproduction_pass") is False:
            error = _finite(
                summary.get("baseline_maximum_absolute_error"),
                context="baseline maximum error",
            )
            return (
                "Not supported",
                "The prerequisite width-7 bridge failed: independently padded scoring did not "
                f"reproduce the legacy packed validation outputs (maximum error {error:.6f}). "
                "On the padded path, the Muon-family high-dose advantage is already absent at "
                "width 7, and widening to 2,048 candidates does not produce the required joint "
                "reversal. Missing-candidate coverage therefore does not explain the observed "
                "shortlist--corpus gap.",
            )
        return (
            "Not supported",
            "The frozen reproduction and endpoint rules do not support missing-candidate coverage as "
            "the explanation of the shortlist--corpus gap.",
        )
    raise ValueError(f"Unknown candidate-breadth decision: {decision!r}")


def candidate_breadth_markdown(
    calibration_rows: list[dict[str, Any]],
    contrast_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    _paired_uncertainty(summary)
    title, sentence = _decision_text(summary)
    calibration = {(row["optimizer"], row["negative_width"]): row for row in calibration_rows}
    contrasts = {(row["optimizer"], row["negative_width"]): row for row in contrast_rows}
    lines = [
        "### Candidate-breadth outcome",
        "",
        f"**Frozen decision: {title}.** {sentence}",
        "",
        "![Candidate-breadth calibration](../reports/candidate-breadth/"
        "candidate_breadth_calibration.svg)",
        "",
        "| Optimizer | Negatives | loss↔BEIR ρ | margin↔BEIR ρ | "
        "high-dose loss Δ [95% CI] | high-dose margin Δ [95% CI] |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for optimizer in OPTIMIZERS:
        for width in (7, 2048):
            row = calibration[(optimizer, width)]
            contrast = contrasts.get((optimizer, width))
            loss_delta = (
                "—" if contrast is None else _delta_interval(contrast, "contrastive_loss", digits=6)
            )
            margin_delta = (
                "—" if contrast is None else _delta_interval(contrast, "positive_margin", digits=6)
            )
            lines.append(
                f"| {LABELS[optimizer]} | {width:,} | {row['loss_beir_spearman']:+.3f} | "
                f"{row['margin_beir_spearman']:+.3f} | {loss_delta} | {margin_delta} |"
            )
    baseline_error = _finite(
        summary.get("baseline_maximum_absolute_error"), context="baseline maximum error"
    )
    lines.extend(
        [
            "",
            f"The maximum width-7 reproduction error is `{baseline_error:.3e}`. High-dose deltas are "
            "3e-3 minus 3e-4 on the same 224 paired queries. This diagnostic was designed after the "
            "shortlist--corpus gap was observed, so it remains post hoc regardless of its outcome "
            "and cannot replace the frozen three-seed full-corpus comparison.",
            "",
            "Brackets and shaded bands are descriptive 95% source-stratified paired percentile "
            "bootstrap intervals (50,000 resamples; each of the seven 32-query source strata is "
            "resampled independently). They do not enter the frozen support rule.",
            "",
            _paired_transition_markdown(contrast_rows),
        ]
    )
    return "\n".join(lines)


def candidate_breadth_latex(
    calibration_rows: list[dict[str, Any]],
    contrast_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    _paired_uncertainty(summary)
    title, sentence = _decision_text(summary)
    if summary.get("decision", {}).get("baseline_reproduction_pass") is False:
        conclusion = (
            "The post-hoc nested-candidate decision was not supported; it supplements but does not "
            "alter the validation-frozen three-seed comparison."
        )
        discussion = (
            "Because padded width-7 scoring failed to reproduce legacy packed validation, the "
            "frozen post-hoc candidate-breadth decision was not supported and missing-candidate "
            "coverage is not an explanation here."
        )
    else:
        conclusion = (
            f"The post-hoc nested-candidate decision was {title.lower()}. {sentence} "
            "This diagnostic supplements but does not alter the validation-frozen three-seed "
            "comparison."
        )
        discussion = (
            f"The frozen post-hoc candidate-breadth decision was {title.lower()}. {sentence} "
            "This constrains the missing-candidate account without altering the validation-frozen "
            "three-seed comparison."
        )
    calibration = {(row["optimizer"], row["negative_width"]): row for row in calibration_rows}
    contrasts = {(row["optimizer"], row["negative_width"]): row for row in contrast_rows}
    endpoint_points = []
    endpoint_intervals = []
    for optimizer in CHALLENGERS:
        narrow = contrasts[(optimizer, 7)]
        broad = contrasts[(optimizer, 2048)]
        endpoint_points.append(
            f"{LABELS[optimizer]} "
            f"{narrow['contrastive_loss_delta']:+.5f}/{narrow['positive_margin_delta']:+.5f}"
            r"$\rightarrow$"
            f"{broad['contrastive_loss_delta']:+.5f}/{broad['positive_margin_delta']:+.5f}"
        )
        endpoint_intervals.append(
            f"{LABELS[optimizer]} loss/margin deltas move from "
            f"{_delta_interval(narrow, 'contrastive_loss', digits=5)}/"
            f"{_delta_interval(narrow, 'positive_margin', digits=5)} to "
            f"{_delta_interval(broad, 'contrastive_loss', digits=5)}/"
            f"{_delta_interval(broad, 'positive_margin', digits=5)}"
        )
    narrow_calibration = "; ".join(
        f"{LABELS[optimizer]} $({calibration[(optimizer, 7)]['loss_beir_spearman']:+.2f},"
        f"{calibration[(optimizer, 7)]['margin_beir_spearman']:+.2f})$"
        for optimizer in OPTIMIZERS
    )
    return "\n".join(
        (
            r"\newcommand{\CandidateBreadthConclusion}{%",
            conclusion,
            r"}",
            "",
            r"\newcommand{\CandidateBreadthDiscussion}{%",
            discussion,
            r"}",
            "",
            r"\newcommand{\CandidateBreadthFigure}{%",
            r"\begin{figure*}[t]",
            r"\centering",
            r"\includegraphics[width=0.88\textwidth]{../reports/candidate-breadth/"
            r"candidate_breadth_calibration.pdf}",
            r"\caption{Post-hoc nested candidate-breadth diagnostic on 224 balanced, query-disjoint "
            r"queries. Top: within-optimizer validation-metric correlation with discovery BEIR over "
            r"four learning rates. Bottom: paired high-dose ($3\!\times\!10^{-3}$) minus "
            r"retrieval-optimal ($3\!\times\!10^{-4}$) contrasts. Width 7 exactly nests inside every "
            r"broader set; shaded bands are descriptive 95\% source-stratified paired percentile "
            rf"bootstrap intervals. Frozen decision: {title}.}}",
            r"\label{fig:candidate-breadth}",
            r"\end{figure*}",
            r"\paragraph{Candidate-breadth uncertainty and paired prevalence.}",
            sentence.replace("--", r"--")
            + " Exact endpoint intervals are "
            + "; ".join(endpoint_intervals)
            + ". Intervals use 50,000 resamples, independently preserving each of the seven "
            "32-query source strata; they are descriptive and do not enter the frozen support "
            "rule.",
            "",
            _paired_transition_latex(contrast_rows),
            r"}",
            "",
            r"\paragraph{Candidate-breadth decision.}",
            f"Frozen decision: {title}. Width 7 "
            r"$\rightarrow$ 2,048 loss/margin deltas are "
            + "; ".join(endpoint_points)
            + ". Source-stratified intervals and paired prevalence appear in "
            r"Figure~\ref{fig:candidate-breadth}; this post-hoc diagnostic cannot alter the "
            "three-seed comparison. Width-7 loss/margin Spearman pairs are "
            + narrow_calibration
            + ".",
            "",
        )
    )


def _publication_manifest(
    *,
    root: Path,
    protocol_path: Path,
    summary_dir: Path,
    summary: dict[str, Any],
    paper_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "post_hoc_mechanism_diagnostic",
        "protocol": _file_record(protocol_path, root),
        "summary": _file_record(summary_dir / "summary.json", root),
        "summary_outputs": {
            name: _file_record(summary_dir / item["path"], root)
            for name, item in sorted(summary["outputs"].items())
        },
        "decision": summary["decision"],
        "outputs": {
            "paper_tex": _file_record(paper_path, root),
        },
        "claim_boundary": summary["claim_boundary"],
    }


def render_candidate_breadth_publication(
    protocol_path: str | Path = "configs/candidate_breadth_probe.json",
    *,
    summary_dir: str | Path = "reports/candidate-breadth",
    paper_path: str | Path = "paper/generated/candidate-breadth.tex",
    manifest_path: str | Path = "reports/candidate-breadth/publication_manifest.json",
    audit_only: bool = False,
) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    root = protocol_path.parent.parent.resolve()

    def resolve(path: str | Path) -> Path:
        value = Path(path)
        return value.resolve() if value.is_absolute() else (root / value).resolve()

    summary_dir = resolve(summary_dir)
    paper_path = resolve(paper_path)
    manifest_path = resolve(manifest_path)
    summary = build_candidate_breadth_summary(
        protocol_path, output_dir=summary_dir, audit_only=True
    )
    calibration, contrasts = load_candidate_breadth_publication_rows(summary_dir, summary)
    latex = candidate_breadth_latex(calibration, contrasts, summary)

    if audit_only:
        if not paper_path.is_file() or paper_path.read_text(encoding="utf-8") != latex:
            raise ValueError("Candidate-breadth paper block differs from recomputed evidence")
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        expected = _publication_manifest(
            root=root,
            protocol_path=protocol_path,
            summary_dir=summary_dir,
            summary=summary,
            paper_path=paper_path,
        )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != expected:
            raise ValueError("Candidate-breadth publication manifest changed")
        return existing

    _atomic_text(paper_path, latex)
    manifest = _publication_manifest(
        root=root,
        protocol_path=protocol_path,
        summary_dir=summary_dir,
        summary=summary,
        paper_path=paper_path,
    )
    _atomic_json(manifest_path, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render or audit the post-hoc candidate-breadth paper artifact"
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/candidate_breadth_probe.json")
    )
    parser.add_argument("--summary-dir", type=Path, default=Path("reports/candidate-breadth"))
    parser.add_argument("--paper", type=Path, default=Path("paper/generated/candidate-breadth.tex"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/candidate-breadth/publication_manifest.json"),
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = render_candidate_breadth_publication(
        args.protocol,
        summary_dir=args.summary_dir,
        paper_path=args.paper,
        manifest_path=args.manifest,
        audit_only=args.audit_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
