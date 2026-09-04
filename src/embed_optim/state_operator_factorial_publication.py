"""Render the prospective state-by-operator factorial into the paper.

The scientific estimands are defined by the existing factorial protocol.  This
module only validates the complete summary, maps the three frozen decisions to
their predeclared interpretation, and writes one deterministic LaTeX include.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

from .geometry import SCHEMA_VERSION, _sha256

PUBLICATION_PROTOCOL = Path(
    "configs/dense_no_packing_state_operator_factorial_publication_protocol.json"
)
SUMMARY_ROOT = Path("reports/state-operator-factorial")
SUMMARY_MANIFEST = SUMMARY_ROOT / "summary_manifest.json"
PAPER_LATEX = Path("paper/generated/state-operator-factorial.tex")
PUBLICATION_MANIFEST = SUMMARY_ROOT / "publication_manifest.json"

ESTIMANDS = (
    "weight_state_effect",
    "operator_effect",
    "state_operator_interaction",
)
ESTIMAND_LABELS = {
    "weight_state_effect": "Muon-state $-$ AdamW-state",
    "operator_effect": "Muon operator $-$ AdamW operator",
    "state_operator_interaction": "State $\\times$ operator interaction",
}
DECISIONS = {"supported_positive", "supported_negative", "inconclusive"}


def _identity(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _record_matches(record: Any, path: Path, root: Path) -> bool:
    if not isinstance(record, dict):
        return False
    try:
        declared = Path(str(record["path"]))
        if declared.is_absolute():
            expected_suffix = path.resolve().relative_to(root.resolve()).parts
            if tuple(declared.parts[-len(expected_suffix) :]) != expected_suffix:
                return False
        elif (root / declared).resolve() != path.resolve():
            return False
        return int(record["bytes"]) == path.stat().st_size and str(record["sha256"]) == _sha256(
            path
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _load_protocol(path: Path, root: Path) -> dict[str, Any]:
    protocol = _load_json(path)
    if protocol.get("status") != "prospective_state_operator_publication_lock":
        raise ValueError("State-operator publication protocol status differs")
    bindings = [
        *protocol.get("parent_bindings", {}).values(),
        *protocol.get("source_bindings", {}).values(),
    ]
    if not bindings:
        raise ValueError("State-operator publication protocol has no source bindings")
    for binding in bindings:
        source = root / str(binding.get("path", ""))
        if not _record_matches(binding, source, root):
            raise ValueError(f"State-operator publication source changed: {source}")
    return protocol


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_summary(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    manifest_path = root / SUMMARY_MANIFEST
    manifest = _load_json(manifest_path)
    scientific = root / protocol["parent_bindings"]["scientific_protocol"]["path"]
    implementation = root / protocol["parent_bindings"]["implementation_protocol"]["path"]
    coverage = manifest.get("coverage", {})
    inference = manifest.get("inference", {})
    if (
        manifest.get("status") != "complete"
        or not _record_matches(manifest.get("scientific_protocol"), scientific, root)
        or not _record_matches(manifest.get("implementation_protocol"), implementation, root)
        or coverage
        != {
            "training_runs": 12,
            "beir_seed_task_scores": 168,
            "estimand_seed_task_contrasts": 126,
            "estimands": 3,
            "probe_checkpoints": 60,
            "probe_task_rows": 840,
        }
        or inference.get("samples") != 100_000
        or inference.get("seed") != 20_260_904
    ):
        raise ValueError("State-operator summary manifest violates the frozen contract")

    expected_outputs = {
        "beir_seed_task_scores": (SUMMARY_ROOT / "beir_seed_task_scores.csv", 168),
        "factorial_cell_summary": (SUMMARY_ROOT / "factorial_cell_summary.csv", 4),
        "estimand_seed_task_contrasts": (
            SUMMARY_ROOT / "estimand_seed_task_contrasts.csv",
            126,
        ),
        "estimand_summary": (SUMMARY_ROOT / "estimand_summary.csv", 3),
        "probe_checkpoint_metrics": (SUMMARY_ROOT / "probe_checkpoint_metrics.csv", 60),
        "probe_task_metrics": (SUMMARY_ROOT / "probe_task_metrics.csv", 840),
    }
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(expected_outputs):
        raise ValueError("State-operator summary output set differs")
    tables: dict[str, list[dict[str, str]]] = {}
    for name, (relative, rows) in expected_outputs.items():
        path = root / relative
        if not _record_matches(outputs[name], path, root):
            raise ValueError(f"State-operator summary output changed: {name}")
        tables[name] = _read_csv(path)
        if len(tables[name]) != rows:
            raise ValueError(f"State-operator summary row count differs: {name}")

    estimands = tables["estimand_summary"]
    if {row.get("estimand") for row in estimands} != set(ESTIMANDS):
        raise ValueError("State-operator estimand identities differ")
    indexed: dict[str, dict[str, Any]] = {}
    for raw in estimands:
        name = str(raw["estimand"])
        try:
            row = {
                "estimand": name,
                "point_estimate": float(raw["point_estimate"]),
                "bootstrap_ci_95_lower": float(raw["bootstrap_ci_95_lower"]),
                "bootstrap_ci_95_upper": float(raw["bootstrap_ci_95_upper"]),
                "decision": str(raw["decision"]),
                "bootstrap_samples": int(raw["bootstrap_samples"]),
                "bootstrap_seed": int(raw["bootstrap_seed"]),
                "seed_clusters": int(raw["seed_clusters"]),
                "task_clusters": int(raw["task_clusters"]),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid state-operator estimand row: {name}") from error
        numeric = (
            row["point_estimate"],
            row["bootstrap_ci_95_lower"],
            row["bootstrap_ci_95_upper"],
        )
        if (
            not all(math.isfinite(value) for value in numeric)
            or row["bootstrap_ci_95_lower"] > row["bootstrap_ci_95_upper"]
            or row["decision"] not in DECISIONS
            or row["bootstrap_samples"] != 100_000
            or row["bootstrap_seed"] != 20_260_904
            or row["seed_clusters"] != 3
            or row["task_clusters"] != 14
        ):
            raise ValueError(f"Invalid state-operator inference row: {name}")
        lower = row["bootstrap_ci_95_lower"]
        upper = row["bootstrap_ci_95_upper"]
        expected_decision = (
            "supported_positive"
            if lower > 0
            else "supported_negative"
            if upper < 0
            else "inconclusive"
        )
        if row["decision"] != expected_decision:
            raise ValueError(f"State-operator decision does not follow the frozen interval: {name}")
        indexed[name] = row

    cells = tables["factorial_cell_summary"]
    expected_cells = {
        (state, operator)
        for state in ("adamw_state", "muon_state")
        for operator in ("adamw", "muon")
    }
    if {(row.get("state"), row.get("operator")) for row in cells} != expected_cells:
        raise ValueError("State-operator factorial cell identities differ")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "tables": tables,
        "estimands": indexed,
    }


def _effect(row: dict[str, Any]) -> str:
    return (
        f"{row['point_estimate']:+.4f} "
        f"[{row['bootstrap_ci_95_lower']:+.4f}, {row['bootstrap_ci_95_upper']:+.4f}]"
    )


def _decision_label(decision: str) -> str:
    return {
        "supported_positive": "positive",
        "supported_negative": "negative",
        "inconclusive": "inconclusive",
    }[decision]


def _interpretation(rows: dict[str, dict[str, Any]]) -> str:
    positive = {name for name, row in rows.items() if row["decision"] == "supported_positive"}
    negative = {name for name, row in rows.items() if row["decision"] == "supported_negative"}
    if "state_operator_interaction" in positive:
        prefix = (
            "Muon-created weights and Muon continuation are complementary, supporting the "
            "predeclared closed-loop state--operator feedback account"
        )
        additions = []
        if "weight_state_effect" in positive:
            additions.append("a carried Muon-state benefit")
        if "operator_effect" in positive:
            additions.append("an averaged Muon-operator benefit")
        return prefix + (" alongside " + " and ".join(additions) if additions else "") + "."
    if positive == {"weight_state_effect"}:
        return (
            "The practical gain is primarily inherited in the Muon-reached weight state, not in "
            "a universally better local Muon direction."
        )
    if positive == {"operator_effect"}:
        return (
            "The continuation transform is the main supported contributor over this branch horizon."
        )
    if positive == {"weight_state_effect", "operator_effect"}:
        return (
            "The branch supports additive carried-state and continuation-operator benefits, but "
            "not state-specific complementarity."
        )
    if positive:
        labels = ", ".join(sorted(name.replace("_", " ") for name in positive))
        return f"Positive evidence is confined to {labels}; no positive interaction is supported."
    if negative:
        labels = ", ".join(sorted(name.replace("_", " ") for name in negative))
        return (
            f"No positive state-feedback account is supported; {labels} is supported in the "
            "opposite direction."
        )
    return (
        "The corrected factorial does not support a stable carried-state, continuation-operator, "
        "or state--operator feedback effect; the retrieval result remains without a positive "
        "mechanism claim."
    )


def _render_latex(rows: dict[str, dict[str, Any]]) -> str:
    weight = rows["weight_state_effect"]
    operator = rows["operator_effect"]
    interaction = rows["state_operator_interaction"]
    interpretation = _interpretation(rows)
    abstract = (
        "A crossed reset-state continuation test separates the gain into carried weight-state, "
        f"operator, and interaction effects of {_effect(weight)}, {_effect(operator)}, and "
        f"{_effect(interaction)} nDCG@10, respectively. {interpretation}"
    )
    main = (
        "Averaged over continuation operators, the Muon-reached versus AdamW-reached state changes "
        f"final nDCG@10 by {_effect(weight)} ({_decision_label(weight['decision'])}); averaged "
        "over source states, Muon versus AdamW continuation changes it by "
        f"{_effect(operator)} ({_decision_label(operator['decision'])}). The state--operator "
        f"interaction is {_effect(interaction)} ({_decision_label(interaction['decision'])}). "
        f"{interpretation}"
    )
    conclusion = (
        "After optimizer-state reset and first-update scale matching, the crossed continuation "
        f"experiment finds weight-state, operator, and interaction effects of {_effect(weight)}, "
        f"{_effect(operator)}, and {_effect(interaction)}. {interpretation}"
    )
    effect_lines = []
    for name in ESTIMANDS:
        row = rows[name]
        effect_lines.append(
            f"{ESTIMAND_LABELS[name]} & {row['point_estimate']:+.4f} & "
            f"[{row['bootstrap_ci_95_lower']:+.4f}, {row['bootstrap_ci_95_upper']:+.4f}] & "
            f"{_decision_label(row['decision'])} \\\\"
        )
    return "\n".join(
        (
            "% Generated only from the complete, source-bound state-by-operator summary.",
            f"\\newcommand{{\\StateOperatorAbstractFinding}}{{%\n{abstract}}}",
            f"\\newcommand{{\\StateOperatorMechanismFinding}}{{%\n{main}}}",
            f"\\newcommand{{\\StateOperatorConclusionFinding}}{{%\n{conclusion}}}",
            "\\newcommand{\\StateOperatorAppendixTable}{%",
            "\\begin{table}[t]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{lrrl}",
            "\\toprule",
            "Factorial contrast & Effect & 95\\% CI & Decision \\\\",
            "\\midrule",
            *effect_lines,
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Prospective reset-state factorial on final decontaminated-BEIR nDCG@10. "
            "Effects average the fixed three order seeds and 14 tasks; intervals use the frozen "
            "two-way seed/task cluster bootstrap.}",
            "\\label{tab:state-operator-factorial}",
            "\\end{table}%",
            "}",
            "",
        )
    )


def _expected_manifest(
    root: Path,
    protocol_path: Path,
    summary: dict[str, Any],
    paper_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "paper-only state-by-operator mechanism rendering",
        "publication_protocol": _identity(protocol_path, root),
        "summary_manifest": _identity(summary["manifest_path"], root),
        "source_outputs": {
            name: _identity(root / SUMMARY_ROOT / f"{name}.csv", root)
            for name in (
                "factorial_cell_summary",
                "estimand_seed_task_contrasts",
                "estimand_summary",
                "probe_checkpoint_metrics",
                "probe_task_metrics",
            )
        },
        "estimands": [summary["estimands"][name] for name in ESTIMANDS],
        "interpretation": _interpretation(summary["estimands"]),
        "paper_latex": _identity(paper_path, root),
        "claim_boundary": (
            "The crossed branch distinguishes a carried state effect, an averaged continuation "
            "effect, and their interaction for one source-checkpoint pair and one branch horizon; "
            "it is not universal mediation of the full 500K trajectory."
        ),
    }


def render(
    *,
    repo_root: Path,
    protocol_path: Path = PUBLICATION_PROTOCOL,
    paper_path: Path = PAPER_LATEX,
    manifest_path: Path = PUBLICATION_MANIFEST,
    audit_only: bool = False,
) -> dict[str, Any]:
    root = repo_root.resolve()
    protocol_path = (
        (root / protocol_path).resolve() if not protocol_path.is_absolute() else protocol_path
    )
    paper_path = (root / paper_path).resolve() if not paper_path.is_absolute() else paper_path
    manifest_path = (
        (root / manifest_path).resolve() if not manifest_path.is_absolute() else manifest_path
    )
    protocol = _load_protocol(protocol_path, root)
    summary = _load_summary(root, protocol)
    latex = _render_latex(summary["estimands"])
    if not audit_only:
        paper_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = paper_path.with_name(f".{paper_path.name}.tmp.{os.getpid()}")
        temporary.write_text(latex, encoding="utf-8")
        os.replace(temporary, paper_path)
    if not paper_path.is_file() or paper_path.read_text(encoding="utf-8") != latex:
        raise ValueError("State-operator paper include differs from its source-bound rendering")
    expected = _expected_manifest(root, protocol_path, summary, paper_path)
    if not audit_only:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, manifest_path)
    observed = _load_json(manifest_path)
    if observed != expected:
        raise ValueError("State-operator publication manifest differs from recomputation")
    return observed


def audit_state_operator_publication(repo_root: Path) -> dict[str, Any]:
    return render(repo_root=repo_root, audit_only=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, default=PUBLICATION_PROTOCOL)
    parser.add_argument("--paper-latex", type=Path, default=PAPER_LATEX)
    parser.add_argument("--manifest", type=Path, default=PUBLICATION_MANIFEST)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(
        json.dumps(
            render(
                repo_root=args.repo_root,
                protocol_path=args.protocol,
                paper_path=args.paper_latex,
                manifest_path=args.manifest,
                audit_only=args.audit_only,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
