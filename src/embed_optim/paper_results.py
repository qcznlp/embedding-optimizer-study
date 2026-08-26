from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .mechanism_report import (
    OPTIMIZER_LABELS,
    _atomic_text,
    _bridge_rows,
    _common_state_rows,
    _read_declared_csv,
    _retrieval_rows,
    _spectrum_rows,
)
from .outcome_report import (
    _confirmation_rows,
    _functional_rows,
    _hybrid_rows,
    _short_branch_rows,
)
from .paper_audit import (
    HEADLINE_MACROS,
    _macros,
    audit_paper,
    expected_constant_macros,
    load_paper_claim_protocol,
)

FAMILY_LABELS = {"dense": "DenseOn", "late": "LateOn"}
FAMILIES = ("dense", "late")
OPTIMIZERS = ("adamw", "muon", "normuon")
CONTRAST_LABELS = ("Muon - AdamW", "NorMuon - AdamW", "NorMuon - Muon")


def _finite(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid paper result value: {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite paper result value: {value!r}")
    return parsed


def _indexed(rows: list[list[str]], *, context: str) -> dict[tuple[str, str], list[str]]:
    result: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        if len(row) < 2 or (row[0], row[1]) in result:
            raise ValueError(f"Invalid or duplicate {context} row: {row}")
        result[(row[0], row[1])] = row
    return result


def _discovery_final_medians(
    retrieval_dir: Path,
    manifest: dict[str, Any],
) -> tuple[dict[tuple[str, str], float], Path]:
    repository_root = retrieval_dir.resolve().parents[1]
    rows, table = _read_declared_csv(
        repository_root,
        manifest,
        "checkpoint_dynamics",
        required_fields={"model_family", "optimizer", "stage", "mean_ndcg_at_10"},
    )
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        family = row["model_family"]
        optimizer = row["optimizer"]
        try:
            stage = int(row["stage"])
        except ValueError as error:
            raise ValueError(f"Invalid discovery checkpoint stage: {row}") from error
        if family not in FAMILIES or optimizer not in OPTIMIZERS or not 1 <= stage <= 5:
            raise ValueError(f"Invalid discovery checkpoint identity: {row}")
        if stage == 5:
            grouped.setdefault((family, optimizer), []).append(_finite(row["mean_ndcg_at_10"]))
    expected = {(family, optimizer) for family in FAMILIES for optimizer in OPTIMIZERS}
    if (
        len(rows) != 120
        or set(grouped) != expected
        or any(len(values) != 4 for values in grouped.values())
    ):
        raise ValueError("Discovery headline requires four final points for all six family groups")
    return {key: float(statistics.median(values)) for key, values in grouped.items()}, table


def _ci_classification(cell: str) -> str:
    if not (cell.startswith("[") and cell.endswith("]") and "," in cell):
        raise ValueError(f"Invalid confirmatory interval cell: {cell!r}")
    lower_text, upper_text = cell[1:-1].split(",", 1)
    lower = _finite(lower_text.strip())
    upper = _finite(upper_text.strip())
    if lower > upper:
        raise ValueError(f"Reversed confirmatory interval: {cell!r}")
    if lower > 0:
        return "positive"
    if upper < 0:
        return "negative"
    return "inconclusive"


def build_headline_macros(
    *,
    final_medians: dict[tuple[str, str], float],
    retrieval_rows: list[list[str]],
    common_rows: list[list[str]],
    spectrum_rows: list[list[str]],
    representation_rows: list[list[str]],
    correlation_rows: list[list[str]],
    functional_rows: list[list[str]],
    hybrid_rows: list[list[str]],
    short_rows: list[list[str]],
    confirmation_rows: list[list[str]],
) -> dict[str, str]:
    retrieval = _indexed(retrieval_rows, context="retrieval")
    common = _indexed(common_rows, context="common-state")
    spectra = _indexed(spectrum_rows, context="spectrum")
    representation = _indexed(representation_rows, context="representation")
    functional = {(row[0], row[1], row[2]): row for row in functional_rows if len(row) >= 8}
    short = _indexed(short_rows, context="short-branch")
    confirmation = _indexed(confirmation_rows, context="confirmation")

    optimizer_labels = tuple(OPTIMIZER_LABELS[name] for name in OPTIMIZERS)
    discovery_parts = []
    for family in FAMILIES:
        family_label = FAMILY_LABELS[family]
        medians = "/".join(f"{final_medians[(family, optimizer)]:.4f}" for optimizer in OPTIMIZERS)
        reached = "/".join(
            retrieval[(family_label, optimizer_label)][3].split("/", 1)[0]
            for optimizer_label in optimizer_labels
        )
        discovery_parts.append(
            f"{family_label} median final nDCG@10 was {medians}, with {reached} of four "
            "learning rates reaching the frozen AdamW reference"
        )
    discovery = "For AdamW/Muon/NorMuon respectively, " + "; ".join(discovery_parts) + "."

    common_parts = []
    for family_label in FAMILY_LABELS.values():
        row_cv = "/".join(common[(family_label, operator)][2] for operator in ("Muon", "NorMuon"))
        stable = "/".join(spectra[(family_label, operator)][2] for operator in ("Muon", "NorMuon"))
        common_parts.append(
            f"{family_label} Muon/NorMuon row-CV ratios to AdamW were {row_cv} and their "
            f"normalized exact stable ranks were {stable}"
        )
    common_headline = "At shared-gradient common states, " + "; ".join(common_parts) + "."

    correlation = {(row[0], row[1], row[2]): row for row in correlation_rows if len(row) >= 5}
    representation_parts = []
    rho_parts = []
    for family_label in FAMILY_LABELS.values():
        margins = "/".join(
            representation[(family_label, operator)][3] for operator in optimizer_labels
        )
        representation_parts.append(f"{family_label} unseen margins were {margins}")
        rho_parts.append(correlation[(family_label, "unseen margin", "mean BEIR nDCG@10")][4])
    late_coverage = "/".join(
        representation[("LateOn", operator)][6] for operator in optimizer_labels
    )
    representation_headline = (
        "For AdamW/Muon/NorMuon respectively, "
        + "; ".join(representation_parts)
        + f", and LateOn document-token coverage was {late_coverage}. The descriptive within-run "
        f"margin-to-BEIR Spearman rho was {rho_parts[0]}/{rho_parts[1]} for DenseOn/LateOn."
    )

    intervention_parts = []
    for family_label in FAMILY_LABELS.values():
        margins = "/".join(
            functional[(family_label, operator, "descent")][4] for operator in optimizer_labels
        )
        muon_margin = short[(family_label, "Muon - AdamW")][3]
        normuon_margin = short[(family_label, "NorMuon - Muon")][3]
        hybrid_deltas = [
            _finite(row[4]) for row in hybrid_rows if len(row) >= 6 and row[0] == family_label
        ]
        if len(hybrid_deltas) != 4:
            raise ValueError(f"Hybrid headline requires four learning rates for {family_label}")
        intervention_parts.append(
            f"{family_label} matched-step margin changes were {margins}, final shared-start "
            f"Muon--AdamW/NorMuon--Muon margin contrasts were {muon_margin}/{normuon_margin}, "
            f"and mean hybrid-routing AdamW change was {statistics.mean(hybrid_deltas):.4f}"
        )
    intervention_headline = (
        "For AdamW/Muon/NorMuon respectively, " + "; ".join(intervention_parts) + "."
    )

    confirmation_parts = []
    for family_label in FAMILY_LABELS.values():
        cells = []
        for contrast in CONTRAST_LABELS:
            row = confirmation[(family_label, contrast)]
            cells.append(
                f"{contrast.replace(' - ', '--')} {row[2]} {row[3]} ({_ci_classification(row[3])})"
            )
        confirmation_parts.append(f"{family_label}: " + ", ".join(cells))
    confirmation_headline = (
        "The validation-frozen three-seed paired nDCG@10 contrasts were "
        + "; ".join(confirmation_parts)
        + "."
    )

    result = {
        "DiscoveryHeadline": discovery,
        "CommonStateHeadline": common_headline,
        "RepresentationHeadline": representation_headline,
        "InterventionHeadline": intervention_headline,
        "ConfirmationHeadline": confirmation_headline,
    }
    if set(result) != set(HEADLINE_MACROS) or any("\n" in value for value in result.values()):
        raise ValueError("Generated paper headlines violate the single-line macro contract")
    return result


def _replace_headlines(text: str, headlines: dict[str, str]) -> str:
    lines = text.splitlines()
    replaced = set()
    for index, line in enumerate(lines):
        for name, value in headlines.items():
            if line.startswith(f"\\newcommand{{\\{name}}}"):
                if name in replaced:
                    raise ValueError(f"Duplicate paper headline macro: {name}")
                lines[index] = f"\\newcommand{{\\{name}}}{{{value}}}"
                replaced.add(name)
    missing = set(HEADLINE_MACROS) - replaced
    if missing:
        raise ValueError(f"Paper headline macros are absent: {sorted(missing)}")
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def render_paper_results(
    *,
    repo_root: Path = Path("."),
    results_path: Path = Path("paper/results.tex"),
    output_manifest: Path = Path("reports/paper-results.manifest.json"),
) -> dict[str, Any]:
    root = repo_root.resolve()
    current_audit = audit_paper(repo_root=root)
    if current_audit["incomplete_evidence"]:
        raise ValueError(
            "Paper headlines require every frozen evidence tier: "
            f"{current_audit['incomplete_evidence']}"
        )
    claim_path, claim_protocol, _claim_sources = load_paper_claim_protocol(repo_root=root)
    paper_results = (root / results_path).resolve()
    current_macros = _macros(paper_results)
    expected_constants, _ = expected_constant_macros(
        root / "configs/experiment.yaml",
        root / "reports/weight-space",
        root / "reports/training-dynamics",
        repo_root=root,
    )
    mismatches = {
        name: (expected, current_macros.get(name))
        for name, expected in expected_constants.items()
        if current_macros.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"Paper constants differ before headline rendering: {mismatches}")

    retrieval_dir = root / "reports/retrieval-dynamics"
    retrieval_rows, retrieval_manifest, retrieval_table, _retrieval_figure = _retrieval_rows(
        retrieval_dir
    )
    final_medians, checkpoint_table = _discovery_final_medians(retrieval_dir, retrieval_manifest)
    common_rows, _common_manifest, common_table = _common_state_rows(root / "reports/common-state")
    spectrum_rows, _spectrum_manifest, spectrum_table = _spectrum_rows(
        root / "results/common-state-spectra/summary"
    )
    representation_rows, correlation_rows, _bridge_manifest, bridge_tables = _bridge_rows(
        root / "reports/mechanism-bridge"
    )
    functional_rows, functional_table, _functional_manifest = _functional_rows(
        root / "reports/functional-intervention"
    )
    hybrid_rows, hybrid_table, _hybrid_manifest = _hybrid_rows(root / "reports/hybrid-adamw")
    short_rows, short_table, _short_manifest = _short_branch_rows(root / "reports/short-branch")
    confirmation_rows, confirmation_table, _confirmation_manifest = _confirmation_rows(
        root / "reports/confirmatory"
    )
    headlines = build_headline_macros(
        final_medians=final_medians,
        retrieval_rows=retrieval_rows,
        common_rows=common_rows,
        spectrum_rows=spectrum_rows,
        representation_rows=representation_rows,
        correlation_rows=correlation_rows,
        functional_rows=functional_rows,
        hybrid_rows=hybrid_rows,
        short_rows=short_rows,
        confirmation_rows=confirmation_rows,
    )
    _atomic_text(
        paper_results,
        _replace_headlines(paper_results.read_text(encoding="utf-8"), headlines),
    )
    if any(_macros(paper_results).get(name) != value for name, value in headlines.items()):
        raise ValueError("Rendered paper headline macros do not round-trip")

    evidence_paths = sorted(
        {
            Path(item["path"]).resolve()
            for items in current_audit["evidence"].values()
            for item in items
        }
    )
    source_tables = [
        checkpoint_table,
        retrieval_table,
        common_table,
        spectrum_table,
        *bridge_tables,
        functional_table,
        hybrid_table,
        short_table,
        confirmation_table,
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "claim_protocol": {
            **_source(claim_path),
            "status": claim_protocol["status"],
            "frozen_at": claim_protocol["frozen_at"],
        },
        "evidence_manifests": [_source(path) for path in evidence_paths],
        "source_tables": [_source(path) for path in source_tables],
        "headlines": headlines,
        "results_tex": _source(paper_results),
        "claim_boundary": (
            "These macros report the complete prespecified contrasts and interval classifications; "
            "they do not convert descriptive checkpoint associations into causal evidence."
        ),
    }
    _atomic_json((root / output_manifest).resolve(), manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render audited final evidence into the frozen NAACL headline macros"
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--results", type=Path, default=Path("paper/results.tex"))
    parser.add_argument("--output", type=Path, default=Path("reports/paper-results.manifest.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = render_paper_results(
        repo_root=args.repo_root,
        results_path=args.results,
        output_manifest=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
