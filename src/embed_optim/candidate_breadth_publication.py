from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .candidate_breadth_summary import build_candidate_breadth_summary
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .mechanism_report import _atomic_text

CANDIDATE_BREADTH_MARKERS = (
    "<!-- CANDIDATE-BREADTH:BEGIN -->",
    "<!-- CANDIDATE-BREADTH:END -->",
)
WIDTHS = (7, 10, 32, 128, 512, 2048)
OPTIMIZERS = ("adamw", "muon", "normuon")
CHALLENGERS = ("muon", "normuon")
LABELS = {"adamw": "AdamW", "muon": "Muon", "normuon": "NorMuon"}


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


def load_candidate_breadth_publication_rows(
    summary_dir: Path,
    summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
        contrasts.append(
            {
                "optimizer": str(row["optimizer"]),
                "negative_width": int(row["negative_width"]),
                "contrastive_loss_delta": _finite(
                    row["contrastive_loss_delta"], context="loss contrast"
                ),
                "positive_margin_delta": _finite(
                    row["positive_margin_delta"], context="margin contrast"
                ),
            }
        )
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
        "| Optimizer | Negatives | loss↔BEIR ρ | margin↔BEIR ρ | high-dose loss Δ | "
        "high-dose margin Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for optimizer in OPTIMIZERS:
        for width in (7, 2048):
            row = calibration[(optimizer, width)]
            contrast = contrasts.get((optimizer, width))
            loss_delta = "—" if contrast is None else f"{contrast['contrastive_loss_delta']:+.6f}"
            margin_delta = "—" if contrast is None else f"{contrast['positive_margin_delta']:+.6f}"
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
        ]
    )
    return "\n".join(lines)


def candidate_breadth_latex(
    calibration_rows: list[dict[str, Any]],
    contrast_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    title, sentence = _decision_text(summary)
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
    endpoints = []
    for optimizer in CHALLENGERS:
        narrow = contrasts[(optimizer, 7)]
        broad = contrasts[(optimizer, 2048)]
        endpoints.append(
            f"{LABELS[optimizer]} loss/margin deltas move from "
            f"{narrow['contrastive_loss_delta']:+.5f}/{narrow['positive_margin_delta']:+.5f} "
            f"to {broad['contrastive_loss_delta']:+.5f}/{broad['positive_margin_delta']:+.5f}"
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
            rf"broader set. Frozen decision: {title}.}}",
            r"\label{fig:candidate-breadth}",
            r"\end{figure*}",
            r"}",
            "",
            r"\paragraph{Candidate-breadth decision.}",
            sentence.replace("--", r"--")
            + " At width 7, loss/margin Spearman pairs are "
            + narrow_calibration
            + ". "
            + "; ".join(endpoints)
            + ". The diagnostic is post hoc and does not alter the frozen three-seed comparison.",
            "",
        )
    )


def _replace_marked(text: str, content: str) -> str:
    begin, end = CANDIDATE_BREADTH_MARKERS
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError("Expected exactly one candidate-breadth marker pair in the blog")
    before, remainder = text.split(begin, 1)
    _, after = remainder.split(end, 1)
    return f"{before}{begin}\n\n{content}\n\n{end}{after}"


def _block_record(path: Path, root: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    begin, end = CANDIDATE_BREADTH_MARKERS
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError("Expected exactly one candidate-breadth marker pair in the blog")
    start = text.index(begin)
    stop = text.index(end, start) + len(end)
    payload = text[start:stop].encode("utf-8")
    return {
        "path": _relative(path, root),
        "markers": list(CANDIDATE_BREADTH_MARKERS),
        "block_bytes": len(payload),
        "block_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _publication_manifest(
    *,
    root: Path,
    protocol_path: Path,
    summary_dir: Path,
    summary: dict[str, Any],
    blog_path: Path,
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
            "blog": _block_record(blog_path, root),
            "paper_tex": _file_record(paper_path, root),
        },
        "claim_boundary": summary["claim_boundary"],
    }


def render_candidate_breadth_publication(
    protocol_path: str | Path = "configs/candidate_breadth_probe.json",
    *,
    summary_dir: str | Path = "reports/candidate-breadth",
    blog_path: str | Path = "docs/blog.md",
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
    blog_path = resolve(blog_path)
    paper_path = resolve(paper_path)
    manifest_path = resolve(manifest_path)
    summary = build_candidate_breadth_summary(
        protocol_path, output_dir=summary_dir, audit_only=True
    )
    calibration, contrasts = load_candidate_breadth_publication_rows(summary_dir, summary)
    markdown = candidate_breadth_markdown(calibration, contrasts, summary)
    latex = candidate_breadth_latex(calibration, contrasts, summary)
    expected_blog = _replace_marked(blog_path.read_text(encoding="utf-8"), markdown)

    if audit_only:
        if blog_path.read_text(encoding="utf-8") != expected_blog:
            raise ValueError("Candidate-breadth blog block differs from recomputed evidence")
        if not paper_path.is_file() or paper_path.read_text(encoding="utf-8") != latex:
            raise ValueError("Candidate-breadth paper block differs from recomputed evidence")
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        expected = _publication_manifest(
            root=root,
            protocol_path=protocol_path,
            summary_dir=summary_dir,
            summary=summary,
            blog_path=blog_path,
            paper_path=paper_path,
        )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != expected:
            raise ValueError("Candidate-breadth publication manifest changed")
        return existing

    _atomic_text(blog_path, expected_blog)
    _atomic_text(paper_path, latex)
    manifest = _publication_manifest(
        root=root,
        protocol_path=protocol_path,
        summary_dir=summary_dir,
        summary=summary,
        blog_path=blog_path,
        paper_path=paper_path,
    )
    _atomic_json(manifest_path, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render or audit the post-hoc candidate-breadth blog and paper artifacts"
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/candidate_breadth_probe.json")
    )
    parser.add_argument("--summary-dir", type=Path, default=Path("reports/candidate-breadth"))
    parser.add_argument("--blog", type=Path, default=Path("docs/blog.md"))
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
        blog_path=args.blog,
        paper_path=args.paper,
        manifest_path=args.manifest,
        audit_only=args.audit_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
