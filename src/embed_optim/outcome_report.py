from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .mechanism_report import (
    FAMILY_LABELS,
    OPTIMIZER_LABELS,
    _atomic_text,
    _finite,
    _format,
    _load_manifest,
    _read_declared_csv,
    _table,
)

OUTCOME_MARKERS = ("<!-- OUTCOMES:BEGIN -->", "<!-- OUTCOMES:END -->")
MECHANISM_MARKERS = ("<!-- MECHANISM:BEGIN -->", "<!-- MECHANISM:END -->")
FAMILIES = ("dense", "late")
OPTIMIZERS = ("adamw", "muon", "normuon")
CONTRASTS = (("muon", "adamw"), ("normuon", "adamw"), ("normuon", "muon"))


def _replace_marked(text: str, content: str) -> str:
    begin, end = OUTCOME_MARKERS
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError("Expected exactly one outcome marker pair in the blog")
    before, remainder = text.split(begin)
    _, after = remainder.split(end)
    return f"{before}{begin}\n\n{content}\n\n{end}{after}"


def _source(manifest_path: Path, **coverage: Any) -> dict[str, Any]:
    return {
        "path": str(manifest_path.resolve()),
        "bytes": manifest_path.stat().st_size,
        "sha256": _sha256(manifest_path),
        **coverage,
    }


def _validate_mechanism_section(report_path: Path, blog_path: Path) -> Path:
    report_path = report_path.resolve()
    manifest_path = report_path.with_suffix(".manifest.json")
    manifest = _load_manifest(manifest_path)
    output = manifest.get("output", {})
    if (
        manifest.get("complete") is not True
        or Path(str(output.get("path", ""))).resolve() != report_path
        or not report_path.is_file()
        or report_path.stat().st_size != output.get("bytes")
        or _sha256(report_path) != output.get("sha256")
    ):
        raise ValueError("Mechanism report differs from its strict manifest")
    blog = blog_path.read_text(encoding="utf-8")
    begin, end = MECHANISM_MARKERS
    if blog.count(begin) != 1 or blog.count(end) != 1:
        raise ValueError("Expected exactly one mechanism marker pair in the blog")
    rendered = blog.split(begin, 1)[1].split(end, 1)[0].strip()
    if rendered != report_path.read_text(encoding="utf-8").strip():
        raise ValueError("Final blog mechanism marker differs from its rendered report")
    return manifest_path


def _hybrid_rows(root: Path) -> tuple[list[list[str]], Path, dict[str, Any]]:
    root = root.resolve()
    manifest = _load_manifest(root / "summary_manifest.json")
    evaluations = manifest.get("evaluations", {})
    if (
        manifest.get("complete") is not True
        or evaluations.get("native_five_stage_units") != 560
        or evaluations.get("native_final_units") != 112
        or evaluations.get("hybrid_final_units") != 112
        or evaluations.get("tasks") != 14
    ):
        raise ValueError("Hybrid AdamW report is not the frozen 2x4x14 control")
    required = {
        "model_family",
        "learning_rate",
        "tasks",
        "adamw_mean_ndcg_at_10",
        "hybrid_adamw_mean_ndcg_at_10",
        "hybrid_minus_adamw_mean",
        "hybrid_task_wins",
        "task_ties",
        "hybrid_task_losses",
    }
    rows, table = _read_declared_csv(root, manifest, "final_summary", required_fields=required)
    expected = {
        (family, learning_rate) for family in FAMILIES for learning_rate in (1e-6, 3e-6, 1e-5, 3e-5)
    }
    indexed = {(row["model_family"], float(row["learning_rate"])): row for row in rows}
    if len(rows) != 8 or set(indexed) != expected:
        raise ValueError("Hybrid AdamW summary identities differ from the frozen control")
    output = []
    for family, learning_rate in sorted(expected):
        row = indexed[(family, learning_rate)]
        wins = int(row["hybrid_task_wins"])
        ties = int(row["task_ties"])
        losses = int(row["hybrid_task_losses"])
        if int(row["tasks"]) != 14 or wins + ties + losses != 14:
            raise ValueError("Hybrid AdamW task counts are invalid")
        output.append(
            [
                FAMILY_LABELS[family],
                f"{learning_rate:.0e}",
                _format(_finite(row, "adamw_mean_ndcg_at_10")),
                _format(_finite(row, "hybrid_adamw_mean_ndcg_at_10")),
                _format(_finite(row, "hybrid_minus_adamw_mean")),
                f"{wins}/{ties}/{losses}",
            ]
        )
    return output, table, manifest


def _functional_rows(root: Path) -> tuple[list[list[str]], Path, dict[str, Any]]:
    root = root.resolve()
    manifest = _load_manifest(root / "manifest.json")
    if (
        manifest.get("complete") is not True
        or manifest.get("anchors") != 20
        or manifest.get("conditions_per_anchor") != 13
        or manifest.get("anchor_effect_records") != 240
        or manifest.get("optimizer_contrast_records") != 160
        or manifest.get("family_summary_records") != 24
    ):
        raise ValueError("Functional intervention report is not the frozen 20-anchor matrix")
    required = {
        "family",
        "algorithm",
        "direction",
        "relative_scale",
        "anchors",
        "mean_anchor_delta_contrastive_loss",
        "mean_anchor_delta_positive_margin",
        "mean_anchor_delta_reciprocal_rank",
        "mean_anchor_delta_top1_accuracy",
        "anchors_with_lower_loss_fraction",
    }
    rows, table = _read_declared_csv(root, manifest, "family_summary", required_fields=required)
    expected = {
        (family, optimizer, direction, scale)
        for family in FAMILIES
        for optimizer in OPTIMIZERS
        for direction, scales in (
            ("descent", (0.0001, 0.0003, 0.001)),
            ("sign_reversal", (0.001,)),
        )
        for scale in scales
    }
    indexed = {
        (row["family"], row["algorithm"], row["direction"], float(row["relative_scale"])): row
        for row in rows
    }
    if len(rows) != 24 or set(indexed) != expected:
        raise ValueError("Functional intervention summary identities differ from the protocol")
    output = []
    for family in FAMILIES:
        for optimizer in OPTIMIZERS:
            for direction in ("descent", "sign_reversal"):
                row = indexed[(family, optimizer, direction, 0.001)]
                if int(row["anchors"]) != 10:
                    raise ValueError("Functional intervention family/operator anchor count differs")
                output.append(
                    [
                        FAMILY_LABELS[family],
                        OPTIMIZER_LABELS[optimizer],
                        direction.replace("_", " "),
                        _format(_finite(row, "mean_anchor_delta_contrastive_loss")),
                        _format(_finite(row, "mean_anchor_delta_positive_margin")),
                        _format(_finite(row, "mean_anchor_delta_reciprocal_rank")),
                        _format(_finite(row, "mean_anchor_delta_top1_accuracy")),
                        _format(_finite(row, "anchors_with_lower_loss_fraction"), 2),
                    ]
                )
    return output, table, manifest


def _short_branch_rows(root: Path) -> tuple[list[list[str]], Path, dict[str, Any]]:
    root = root.resolve()
    manifest = _load_manifest(root / "summary_manifest.json")
    coverage = manifest.get("coverage", {})
    if (
        manifest.get("complete") is not True
        or coverage.get("runs") != 18
        or coverage.get("checkpoints") != 90
        or coverage.get("paired_checkpoint_contrasts") != 90
        or coverage.get("paired_dynamics_summaries") != 120
    ):
        raise ValueError("Short-branch report is not the frozen 18-run shared-start study")
    required = {
        "family",
        "stage",
        "fraction",
        "treatment",
        "baseline",
        "metric",
        "seeds",
        "mean_delta",
        "treatment_seed_wins",
        "seed_ties",
        "treatment_seed_losses",
        "beneficial_direction",
    }
    rows, table = _read_declared_csv(root, manifest, "paired_summary", required_fields=required)
    expected = {
        (family, stage, treatment, baseline, metric)
        for family in FAMILIES
        for stage in range(1, 6)
        for treatment, baseline in CONTRASTS
        for metric in ("contrastive_loss", "positive_margin", "reciprocal_rank", "top1_accuracy")
    }
    indexed = {
        (
            row["family"],
            int(row["stage"]),
            row["treatment"],
            row["baseline"],
            row["metric"],
        ): row
        for row in rows
    }
    if len(rows) != 120 or set(indexed) != expected:
        raise ValueError("Short-branch summary identities differ from the frozen design")
    output = []
    metrics = ("contrastive_loss", "positive_margin", "reciprocal_rank", "top1_accuracy")
    for family in FAMILIES:
        for treatment, baseline in CONTRASTS:
            cells = []
            for metric in metrics:
                row = indexed[(family, 5, treatment, baseline, metric)]
                wins = int(row["treatment_seed_wins"])
                ties = int(row["seed_ties"])
                losses = int(row["treatment_seed_losses"])
                if int(row["seeds"]) != 3 or wins + ties + losses != 3:
                    raise ValueError("Short-branch seed counts are invalid")
                cells.append(f"{_format(_finite(row, 'mean_delta'))} ({wins}/{ties}/{losses})")
            output.append(
                [
                    FAMILY_LABELS[family],
                    f"{OPTIMIZER_LABELS[treatment]} - {OPTIMIZER_LABELS[baseline]}",
                    *cells,
                ]
            )
    return output, table, manifest


def _confirmation_rows(root: Path) -> tuple[list[list[str]], Path, dict[str, Any]]:
    root = root.resolve()
    manifest = _load_manifest(root / "summary_manifest.json")
    coverage = manifest.get("coverage", {})
    if manifest.get("complete") is not True or coverage != {
        "seeds": 3,
        "runs": 18,
        "tasks": 14,
        "evaluation_units": 252,
        "paired_contrast_units": 252,
    }:
        raise ValueError("Confirmatory report is not the frozen 3-seed 252-unit study")
    required = {
        "model_family",
        "treatment",
        "baseline",
        "seeds",
        "tasks",
        "mean_delta_ndcg_at_10",
        "bootstrap_ci_95_lower",
        "bootstrap_ci_95_upper",
        "seed_wins",
        "seed_ties",
        "seed_losses",
        "task_wins_after_seed_average",
        "task_ties_after_seed_average",
        "task_losses_after_seed_average",
    }
    rows, table = _read_declared_csv(root, manifest, "paired_summary", required_fields=required)
    expected = {
        (family, treatment, baseline) for family in FAMILIES for treatment, baseline in CONTRASTS
    }
    indexed = {(row["model_family"], row["treatment"], row["baseline"]): row for row in rows}
    if len(rows) != 6 or set(indexed) != expected:
        raise ValueError("Confirmatory paired summaries differ from the frozen contrasts")
    output = []
    for family, treatment, baseline in sorted(expected):
        row = indexed[(family, treatment, baseline)]
        seed_counts = (int(row["seed_wins"]), int(row["seed_ties"]), int(row["seed_losses"]))
        task_counts = (
            int(row["task_wins_after_seed_average"]),
            int(row["task_ties_after_seed_average"]),
            int(row["task_losses_after_seed_average"]),
        )
        if (
            int(row["seeds"]) != 3
            or sum(seed_counts) != 3
            or int(row["tasks"]) != 14
            or sum(task_counts) != 14
        ):
            raise ValueError("Confirmatory seed/task counts are invalid")
        lower = _finite(row, "bootstrap_ci_95_lower")
        upper = _finite(row, "bootstrap_ci_95_upper")
        output.append(
            [
                FAMILY_LABELS[family],
                f"{OPTIMIZER_LABELS[treatment]} - {OPTIMIZER_LABELS[baseline]}",
                _format(_finite(row, "mean_delta_ndcg_at_10")),
                f"[{_format(lower)}, {_format(upper)}]",
                "/".join(map(str, seed_counts)),
                "/".join(map(str, task_counts)),
            ]
        )
    return output, table, manifest


def render_outcome_report(
    functional_dir: Path,
    hybrid_dir: Path,
    short_branch_dir: Path,
    confirmatory_dir: Path,
    mechanism_report: Path,
    blog_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    blog_path = blog_path.resolve()
    mechanism_manifest_path = _validate_mechanism_section(mechanism_report, blog_path)
    functional, functional_table, functional_manifest = _functional_rows(functional_dir)
    hybrid, hybrid_table, hybrid_manifest = _hybrid_rows(hybrid_dir)
    short, short_table, short_manifest = _short_branch_rows(short_branch_dir)
    confirmation, confirmation_table, confirmation_manifest = _confirmation_rows(confirmatory_dir)
    content = "\n\n".join(
        [
            "## Causal controls and confirmation",
            "The tables in this section are generated only after all frozen routing, local-step, "
            "shared-start, and confirmatory manifests pass their cardinality and content-hash "
            "contracts. They separate four questions that a single optimizer leaderboard cannot.",
            "### Does AdamW parameter routing explain the result?\n\n"
            + _table(
                ["Family", "LR", "AdamW", "hybrid AdamW", "difference", "task W/T/L"],
                hybrid,
            )
            + "\n\nAll four native AdamW learning rates are retained. The paired difference isolates "
            "Muon-style hidden/auxiliary parameter routing; it does not isolate orthogonalization.",
            "### Do matched optimizer directions have immediate functional effects?\n\n"
            + _table(
                [
                    "Family",
                    "Direction source",
                    "Applied sign",
                    "delta loss",
                    "delta margin",
                    "delta MRR",
                    "delta top-1",
                    "anchors lowering loss",
                ],
                functional,
            )
            + "\n\nEvery row uses the common relative scale 0.001 at fixed weights with per-tensor "
            "Frobenius matching; the sign-reversal row is the directionality control. These are "
            "immediate virtual-step effects, not claims that one step reproduces a native trajectory.",
            "### Do direction effects accumulate from a shared checkpoint?\n\n"
            + _table(
                [
                    "Family",
                    "Final-stage contrast",
                    "delta loss (W/T/L)",
                    "delta margin (W/T/L)",
                    "delta MRR (W/T/L)",
                    "delta top-1 (W/T/L)",
                ],
                short,
            )
            + "\n\nThese are final-stage means over three independently ordered 50K-query branches "
            "starting from the same 60% AdamW checkpoint and calibrated to the same hidden "
            "update-to-weight target. They use frozen probes rather than a second full BEIR run.",
            "### Does the validation-frozen recipe replicate?\n\n"
            + _table(
                [
                    "Family",
                    "Contrast",
                    "mean delta nDCG@10",
                    "hierarchical 95% CI",
                    "seed W/T/L",
                    "task W/T/L",
                ],
                confirmation,
            )
            + "\n\nRecipes were selected on the query-disjoint validation set before these runs. "
            "Intervals independently resample seeds and tasks; aggregate MTEB files do not support "
            "a query-level significance claim. The renderer reports every prespecified contrast and "
            "does not convert an interval or win count into an automatic narrative conclusion.",
        ]
    )
    output_path = output_path.resolve()
    _atomic_text(output_path, content + "\n")
    _atomic_text(blog_path, _replace_marked(blog_path.read_text(encoding="utf-8"), content))
    source_manifests = {
        "mechanism_report": _source(mechanism_manifest_path),
        "functional_intervention": _source(
            functional_dir / "manifest.json", anchors=functional_manifest["anchors"]
        ),
        "hybrid_adamw": _source(
            hybrid_dir / "summary_manifest.json",
            hybrid_units=hybrid_manifest["evaluations"]["hybrid_final_units"],
        ),
        "short_branch": _source(
            short_branch_dir / "summary_manifest.json",
            runs=short_manifest["coverage"]["runs"],
        ),
        "confirmation": _source(
            confirmatory_dir / "summary_manifest.json",
            units=confirmation_manifest["coverage"]["evaluation_units"],
        ),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "sources": source_manifests,
        "source_tables": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (functional_table, hybrid_table, short_table, confirmation_table)
        ],
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        },
        "blog": {
            "path": str(blog_path),
            "bytes": blog_path.stat().st_size,
            "sha256": _sha256(blog_path),
            "markers": list(OUTCOME_MARKERS),
        },
        "claim_boundary": (
            "Routing and local-step tables are controls; short branches test accumulation on frozen "
            "probes; only the validation-frozen three-seed BEIR table is confirmatory retrieval evidence."
        ),
    }
    _atomic_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render strict causal-control and confirmation tables into the final blog"
    )
    parser.add_argument(
        "--functional-dir", type=Path, default=Path("reports/functional-intervention")
    )
    parser.add_argument("--hybrid-dir", type=Path, default=Path("reports/hybrid-adamw"))
    parser.add_argument("--short-branch-dir", type=Path, default=Path("reports/short-branch"))
    parser.add_argument("--confirmatory-dir", type=Path, default=Path("reports/confirmatory"))
    parser.add_argument(
        "--mechanism-report", type=Path, default=Path("reports/mechanism-summary.md")
    )
    parser.add_argument("--blog", type=Path, default=Path("docs/blog.md"))
    parser.add_argument("--output", type=Path, default=Path("reports/outcome-summary.md"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = render_outcome_report(
        args.functional_dir,
        args.hybrid_dir,
        args.short_branch_dir,
        args.confirmatory_dir,
        args.mechanism_report,
        args.blog,
        args.output,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
