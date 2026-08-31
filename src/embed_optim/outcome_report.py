from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .causal_chain_rendering import (
    causal_chain_display_contract,
    render_causal_chain_markdown,
)
from .causal_chain_reporting import load_causal_chain_evidence
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .mechanism_report import (
    FAMILY_LABELS,
    OPTIMIZER_LABELS,
    _atomic_text,
    _finite,
    _format,
    _load_manifest,
    _marked_block_complete,
    _marked_block_record,
    _portable_path,
    _read_declared_csv,
    _repository_root,
    _table,
)
from .scope import ALL_FAMILIES, resolve_scope

OUTCOME_MARKERS = ("<!-- OUTCOMES:BEGIN -->", "<!-- OUTCOMES:END -->")
MECHANISM_MARKERS = ("<!-- MECHANISM:BEGIN -->", "<!-- MECHANISM:END -->")
FAMILIES = ALL_FAMILIES
OPTIMIZERS = ("adamw", "muon", "normuon")
CONTRASTS = (("muon", "adamw"), ("normuon", "adamw"), ("normuon", "muon"))
SPECTRAL_METRICS = (
    "contrastive_loss",
    "positive_score",
    "hardest_negative_score",
    "positive_margin",
    "reciprocal_rank",
    "top1_accuracy",
)
SPECTRAL_TAIL_CONDITIONS = (
    "muon-native",
    "adam-basis__spectrum-lambda-0.25",
    "adam-basis__spectrum-lambda-0.50",
    "adam-basis__spectrum-lambda-0.75",
    "adam-basis__muon-spectrum",
    "muon-basis__adam-spectrum",
    "adam-basis__muon-head-spectrum",
    "adam-basis__muon-middle-spectrum",
    "adam-basis__muon-tail-spectrum",
)


def _replace_marked(text: str, content: str) -> str:
    begin, end = OUTCOME_MARKERS
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError("Expected exactly one outcome marker pair in the blog")
    before, remainder = text.split(begin)
    _, after = remainder.split(end)
    return f"{before}{begin}\n\n{content}\n\n{end}{after}"


def _source(
    manifest_path: Path, *, repository_root: Path | None = None, **coverage: Any
) -> dict[str, Any]:
    return {
        "path": (
            str(manifest_path.resolve())
            if repository_root is None
            else _portable_path(manifest_path, repository_root)
        ),
        "bytes": manifest_path.stat().st_size,
        "sha256": _sha256(manifest_path),
        **coverage,
    }


def _validate_mechanism_section(
    report_path: Path,
    blog_path: Path,
    families: tuple[str, ...] = FAMILIES,
    scope_amendment: dict[str, Any] | None = None,
) -> Path:
    report_path = report_path.resolve()
    repository_root = _repository_root(report_path)
    manifest_path = report_path.with_suffix(".manifest.json")
    manifest = _load_manifest(manifest_path)
    output = manifest.get("output", {})
    output_path = Path(str(output.get("path", "")))
    resolved_output = (
        output_path.resolve()
        if output_path.is_absolute()
        else (repository_root / output_path).resolve()
    )
    if (
        manifest.get("complete") is not True
        or (
            families != FAMILIES
            and (
                manifest.get("families") != list(families)
                or manifest.get("scope_amendment") != scope_amendment
            )
        )
        or resolved_output != report_path
        or not report_path.is_file()
        or report_path.stat().st_size != output.get("bytes")
        or _sha256(report_path) != output.get("sha256")
    ):
        raise ValueError("Mechanism report differs from its strict manifest")
    if not _marked_block_complete(
        blog_path,
        manifest.get("blog"),
        MECHANISM_MARKERS,
        repository_root=repository_root,
    ):
        raise ValueError("Final blog mechanism marker differs from its rendered manifest")
    blog = blog_path.read_text(encoding="utf-8")
    begin, end = MECHANISM_MARKERS
    if blog.count(begin) != 1 or blog.count(end) != 1:
        raise ValueError("Expected exactly one mechanism marker pair in the blog")
    rendered = blog.split(begin, 1)[1].split(end, 1)[0].strip()
    if rendered != report_path.read_text(encoding="utf-8").strip():
        raise ValueError("Final blog mechanism marker differs from its rendered report")
    return manifest_path


def _strict_output_paths(
    root: Path,
    manifest: dict[str, Any],
    expected: dict[str, tuple[str, int | None]],
) -> dict[str, Path]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(expected):
        raise ValueError("Strict report output identities differ from the frozen contract")
    resolved: dict[str, Path] = {}
    for name, (filename, rows) in expected.items():
        record = outputs.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"Strict report output is malformed: {name}")
        declared = Path(record["path"])
        path = declared.resolve() if declared.is_absolute() else (root / declared).resolve()
        if (
            path != (root / filename).resolve()
            or not path.is_file()
            or isinstance(record.get("bytes"), bool)
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != _sha256(path)
            or (rows is not None and record.get("rows") != rows)
            or (rows is None and "rows" in record)
        ):
            raise ValueError(f"Strict report output differs from its manifest: {path}")
        resolved[name] = path
    return resolved


def _require_active_scope(
    manifest: dict[str, Any],
    families: tuple[str, ...],
    scope_amendment: dict[str, Any] | None,
    *,
    context: str,
) -> None:
    if families != FAMILIES and (
        manifest.get("families") != list(families)
        or manifest.get("scope_amendment") != scope_amendment
    ):
        raise ValueError(f"{context} is not bound to the active scope amendment")


def _hybrid_rows(
    root: Path,
    families: tuple[str, ...] = FAMILIES,
    scope_amendment: dict[str, Any] | None = None,
) -> tuple[list[list[str]], Path, dict[str, Any]]:
    root = root.resolve()
    manifest = _load_manifest(root / "summary_manifest.json")
    _require_active_scope(manifest, families, scope_amendment, context="Hybrid AdamW report")
    evaluations = manifest.get("evaluations", {})
    family_count = len(families)
    if (
        manifest.get("complete") is not True
        or evaluations.get("native_five_stage_units") != 280 * family_count
        or evaluations.get("native_final_units") != 56 * family_count
        or evaluations.get("hybrid_final_units") != 56 * family_count
        or evaluations.get("tasks") != 14
    ):
        raise ValueError("Hybrid AdamW report differs from the active 4-run-per-family control")
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
        (family, learning_rate) for family in families for learning_rate in (1e-6, 3e-6, 1e-5, 3e-5)
    }
    indexed = {(row["model_family"], float(row["learning_rate"])): row for row in rows}
    if len(rows) != 4 * family_count or set(indexed) != expected:
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


def _functional_rows(
    root: Path,
    families: tuple[str, ...] = FAMILIES,
) -> tuple[list[list[str]], Path, dict[str, Any]]:
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
        (
            row["family"],
            row["algorithm"],
            row["direction"],
            float(row["relative_scale"]),
        ): row
        for row in rows
    }
    if len(rows) != 24 or set(indexed) != expected:
        raise ValueError("Functional intervention summary identities differ from the protocol")
    output = []
    for family in families:
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


def _short_branch_rows(
    root: Path,
    families: tuple[str, ...] = FAMILIES,
    scope_amendment: dict[str, Any] | None = None,
) -> tuple[list[list[str]], Path, dict[str, Any]]:
    root = root.resolve()
    manifest = _load_manifest(root / "summary_manifest.json")
    _require_active_scope(manifest, families, scope_amendment, context="Short-branch report")
    coverage = manifest.get("coverage", {})
    family_count = len(families)
    if (
        manifest.get("complete") is not True
        or coverage.get("runs") != 9 * family_count
        or coverage.get("checkpoints") != 45 * family_count
        or coverage.get("paired_checkpoint_contrasts") != 45 * family_count
        or coverage.get("paired_dynamics_summaries") != 60 * family_count
    ):
        raise ValueError("Short-branch report differs from the active 9-run-per-family study")
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
        for family in families
        for stage in range(1, 6)
        for treatment, baseline in CONTRASTS
        for metric in (
            "contrastive_loss",
            "positive_margin",
            "reciprocal_rank",
            "top1_accuracy",
        )
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
    if len(rows) != 60 * family_count or set(indexed) != expected:
        raise ValueError("Short-branch summary identities differ from the frozen design")
    output = []
    metrics = (
        "contrastive_loss",
        "positive_margin",
        "reciprocal_rank",
        "top1_accuracy",
    )
    for family in families:
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


def _confirmation_rows(
    root: Path,
    families: tuple[str, ...] = FAMILIES,
    scope_amendment: dict[str, Any] | None = None,
) -> tuple[list[list[str]], Path, dict[str, Any]]:
    root = root.resolve()
    manifest = _load_manifest(root / "summary_manifest.json")
    _require_active_scope(manifest, families, scope_amendment, context="Confirmatory report")
    coverage = manifest.get("coverage", {})
    family_count = len(families)
    if manifest.get("complete") is not True or coverage != {
        "seeds": 3,
        "runs": 9 * family_count,
        "tasks": 14,
        "evaluation_units": 126 * family_count,
        "paired_contrast_units": 126 * family_count,
    }:
        raise ValueError("Confirmatory report differs from the active three-seed study")
    required = {
        "model_family",
        "treatment",
        "baseline",
        "seeds",
        "tasks",
        "mean_delta_ndcg_at_10",
        "bootstrap_ci_95_lower",
        "bootstrap_ci_95_upper",
        "familywise_method",
        "familywise_contrasts",
        "familywise_ci_95_lower",
        "familywise_ci_95_upper",
        "seed_wins",
        "seed_ties",
        "seed_losses",
        "task_wins_after_seed_average",
        "task_ties_after_seed_average",
        "task_losses_after_seed_average",
    }
    rows, table = _read_declared_csv(root, manifest, "paired_summary", required_fields=required)
    expected = {
        (family, treatment, baseline) for family in families for treatment, baseline in CONTRASTS
    }
    indexed = {(row["model_family"], row["treatment"], row["baseline"]): row for row in rows}
    if len(rows) != 3 * family_count or set(indexed) != expected:
        raise ValueError("Confirmatory paired summaries differ from the frozen contrasts")
    output = []
    for family, treatment, baseline in sorted(expected):
        row = indexed[(family, treatment, baseline)]
        seed_counts = (
            int(row["seed_wins"]),
            int(row["seed_ties"]),
            int(row["seed_losses"]),
        )
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
        familywise_lower = _finite(row, "familywise_ci_95_lower")
        familywise_upper = _finite(row, "familywise_ci_95_upper")
        if (
            row["familywise_method"] != "bonferroni"
            or int(row["familywise_contrasts"]) != 6
            or familywise_lower > lower
            or familywise_upper < upper
        ):
            raise ValueError("Confirmatory familywise interval contract is invalid")
        output.append(
            [
                FAMILY_LABELS[family],
                f"{OPTIMIZER_LABELS[treatment]} - {OPTIMIZER_LABELS[baseline]}",
                _format(_finite(row, "mean_delta_ndcg_at_10")),
                f"[{_format(lower)}, {_format(upper)}]",
                f"[{_format(familywise_lower)}, {_format(familywise_upper)}]",
                "/".join(map(str, seed_counts)),
                "/".join(map(str, task_counts)),
            ]
        )
    return output, table, manifest


def _tail_stability_rows(
    root: Path,
    families: tuple[str, ...] = FAMILIES,
    scope_amendment: dict[str, Any] | None = None,
) -> tuple[list[list[str]], list[list[str]], tuple[Path, Path], dict[str, Any]]:
    root = root.resolve()
    manifest = _load_manifest(root / "summary_manifest.json")
    family_count = len(families)
    expected_outputs = {
        "discovery_anchor_tail": ("discovery_anchor_tail.csv", 30 * family_count),
        "discovery_family_contrasts": (
            "discovery_family_contrasts.csv",
            2 * family_count,
        ),
        "discovery_cross_tail": ("discovery_cross_tail.csv", 20 * family_count),
        "discovery_cross_tail_summary": (
            "discovery_cross_tail_summary.csv",
            2 * family_count,
        ),
        "short_branch_checkpoint_tail": (
            "short_branch_checkpoint_tail.csv",
            45 * family_count,
        ),
        "short_branch_checkpoint_contrasts": (
            "short_branch_checkpoint_contrasts.csv",
            30 * family_count,
        ),
        "short_branch_final_summary": (
            "short_branch_final_summary.csv",
            2 * family_count,
        ),
        "readme": ("README.md", None),
    }
    if (
        manifest.get("status") != "complete"
        or manifest.get("complete") is not True
        or manifest.get("discovery_complete") is not True
        or manifest.get("short_branch_confirmation_complete") is not True
        or manifest.get("analysis_status")
        != "post_hoc_discovery_with_prospective_short_branch_confirmation"
        or manifest.get("families") != list(families)
        or manifest.get("scope_amendment") != scope_amendment
        or manifest.get("pending_reason") is not None
        or manifest.get("discovery_anchors") != 10 * family_count
        or manifest.get("discovery_anchor_operator_rows") != 30 * family_count
        or manifest.get("discovery_contrasts") != 2 * family_count
        or manifest.get("discovery_cross_tail_rows") != 20 * family_count
        or manifest.get("discovery_cross_tail_summaries") != 2 * family_count
        or manifest.get("short_branch_checkpoint_rows") != 45 * family_count
        or manifest.get("short_branch_contrast_rows") != 30 * family_count
        or manifest.get("short_branch_final_rows") != 2 * family_count
        or not isinstance(manifest.get("claim_boundary"), str)
    ):
        raise ValueError("Tail-stability report is not the complete active-scope analysis")
    paths = _strict_output_paths(root, manifest, expected_outputs)
    discovery_fields = {
        "family",
        "challenger",
        "reference",
        "anchors",
        "tail_fraction",
        "tail_size",
        "adam_tail_contrast_mean",
        "challenger_tail_contrast_mean",
        "tail_jaccard_mean",
        "tail_identity_regime",
    }
    discovery, discovery_table = _read_declared_csv(
        root,
        manifest,
        "discovery_cross_tail_summary",
        required_fields=discovery_fields,
    )
    final_fields = {
        "family",
        "challenger",
        "reference",
        "stage",
        "seeds",
        "median_delta_validation_loss_p95",
        "validation_loss_p95_seed_wins",
        "median_delta_unseen_margin_p05",
        "unseen_margin_p05_seed_wins",
        "tail_stability_decision",
    }
    final, final_table = _read_declared_csv(
        root,
        manifest,
        "short_branch_final_summary",
        required_fields=final_fields,
    )
    expected = {(family, challenger) for family in families for challenger in ("muon", "normuon")}
    discovery_indexed = {(row["family"], row["challenger"]): row for row in discovery}
    final_indexed = {(row["family"], row["challenger"]): row for row in final}
    if (
        len(discovery) != len(expected)
        or set(discovery_indexed) != expected
        or len(final) != len(expected)
        or set(final_indexed) != expected
    ):
        raise ValueError("Tail-stability key summaries do not cover the frozen contrasts")
    discovery_output: list[list[str]] = []
    final_output: list[list[str]] = []
    for family, challenger in sorted(expected):
        row = discovery_indexed[(family, challenger)]
        if (
            row["reference"] != "adamw"
            or int(row["anchors"]) != 10
            or _finite(row, "tail_fraction") != 0.05
            or int(row["tail_size"]) != 12
            or row["tail_identity_regime"]
            not in {"shared-tail severity suppression", "tail redistribution", "mixed"}
        ):
            raise ValueError("Tail-stability discovery identity or tail rule is invalid")
        discovery_output.append(
            [
                FAMILY_LABELS[family],
                OPTIMIZER_LABELS[challenger],
                _format(_finite(row, "adam_tail_contrast_mean")),
                _format(_finite(row, "challenger_tail_contrast_mean")),
                _format(_finite(row, "tail_jaccard_mean")),
                row["tail_identity_regime"],
            ]
        )
        row = final_indexed[(family, challenger)]
        loss_wins = int(row["validation_loss_p95_seed_wins"])
        margin_wins = int(row["unseen_margin_p05_seed_wins"])
        loss_p95 = _finite(row, "median_delta_validation_loss_p95")
        margin_p05 = _finite(row, "median_delta_unseen_margin_p05")
        if loss_p95 < 0 and margin_p05 > 0:
            expected_decision = "supported"
        elif (loss_p95 < 0) != (margin_p05 > 0):
            expected_decision = "mixed"
        else:
            expected_decision = "not-supported"
        if (
            row["reference"] != "adamw"
            or int(row["stage"]) != 5
            or int(row["seeds"]) != 3
            or not 0 <= loss_wins <= 3
            or not 0 <= margin_wins <= 3
            or row["tail_stability_decision"] != expected_decision
        ):
            raise ValueError("Tail-stability final summary identity or decision is invalid")
        final_output.append(
            [
                FAMILY_LABELS[family],
                OPTIMIZER_LABELS[challenger],
                _format(loss_p95),
                f"{loss_wins}/3",
                _format(margin_p05),
                f"{margin_wins}/3",
                row["tail_stability_decision"],
            ]
        )
    if (
        discovery_table != paths["discovery_cross_tail_summary"]
        or final_table != paths["short_branch_final_summary"]
    ):
        raise ValueError("Tail-stability key table paths differ from the strict output contract")
    return discovery_output, final_output, (discovery_table, final_table), manifest


def _spectral_transplant_rows(
    root: Path,
    families: tuple[str, ...] = FAMILIES,
    scope_amendment: dict[str, Any] | None = None,
) -> tuple[list[list[str]], list[list[str]], tuple[Path, Path], dict[str, Any]]:
    root = root.resolve()
    manifest = _load_manifest(root / "summary_manifest.json")
    family_count = len(families)
    expected_outputs = {
        "anchor_condition_effects": (
            "anchor_condition_effects.csv",
            100 * family_count,
        ),
        "family_condition_summary": ("family_condition_summary.csv", 60 * family_count),
        "anchor_factorial_effects": ("anchor_factorial_effects.csv", 60 * family_count),
        "family_factorial_summary": ("family_factorial_summary.csv", 6 * family_count),
        "anchor_spectral_path": ("anchor_spectral_path.csv", 300 * family_count),
        "family_spectral_path": ("family_spectral_path.csv", 30 * family_count),
        "anchor_band_effects": ("anchor_band_effects.csv", 180 * family_count),
        "family_band_summary": ("family_band_summary.csv", 18 * family_count),
        "anchor_query_tail_effects": (
            "anchor_query_tail_effects.csv",
            90 * family_count,
        ),
        "family_query_tail_summary": (
            "family_query_tail_summary.csv",
            9 * family_count,
        ),
    }
    if (
        manifest.get("status") != "complete"
        or manifest.get("complete") is not True
        or manifest.get("analysis_status") != "post_hoc_explanatory_intervention"
        or manifest.get("families") != list(families)
        or manifest.get("scope_amendment") != scope_amendment
        or manifest.get("anchors") != 10 * family_count
        or manifest.get("anchor_effect_records") != 100 * family_count
        or manifest.get("anchor_tail_effect_records") != 90 * family_count
        or not isinstance(manifest.get("claim_boundary"), str)
    ):
        raise ValueError("Spectral-transplant report is not the complete active-scope intervention")
    paths = _strict_output_paths(root, manifest, expected_outputs)
    factorial_fields = {
        "family",
        "metric",
        "anchors",
        "median_spectrum_main_effect",
        "median_basis_main_effect",
        "median_spectrum_basis_interaction",
    }
    factorial, factorial_table = _read_declared_csv(
        root, manifest, "family_factorial_summary", required_fields=factorial_fields
    )
    tail_fields = {
        "family",
        "condition",
        "anchors",
        "median_p95_pairwise_loss_contrast",
        "median_p05_pairwise_margin_contrast",
        "median_mean_loss_contrast_on_adam_tail",
        "median_mean_loss_contrast_on_condition_tail",
        "median_worst_loss_tail_jaccard",
    }
    tail, tail_table = _read_declared_csv(
        root, manifest, "family_query_tail_summary", required_fields=tail_fields
    )
    expected_factorial = {(family, metric) for family in families for metric in SPECTRAL_METRICS}
    factorial_indexed = {(row["family"], row["metric"]): row for row in factorial}
    expected_tail = {
        (family, condition) for family in families for condition in SPECTRAL_TAIL_CONDITIONS
    }
    tail_indexed = {(row["family"], row["condition"]): row for row in tail}
    if (
        len(factorial) != len(expected_factorial)
        or set(factorial_indexed) != expected_factorial
        or len(tail) != len(expected_tail)
        or set(tail_indexed) != expected_tail
    ):
        raise ValueError("Spectral-transplant key summaries differ from the frozen grid")
    factorial_output: list[list[str]] = []
    for family in families:
        for metric in ("contrastive_loss", "positive_margin"):
            row = factorial_indexed[(family, metric)]
            if int(row["anchors"]) != 10:
                raise ValueError("Spectral factorial summary requires ten anchors per metric")
            factorial_output.append(
                [
                    FAMILY_LABELS[family],
                    metric.replace("_", " "),
                    _format(_finite(row, "median_spectrum_main_effect")),
                    _format(_finite(row, "median_basis_main_effect")),
                    _format(_finite(row, "median_spectrum_basis_interaction")),
                ]
            )
    condition_labels = {
        "muon-native": "Muon native",
        "adam-basis__muon-spectrum": "Adam basis + Muon spectrum",
        "muon-basis__adam-spectrum": "Muon basis + Adam spectrum",
    }
    tail_output: list[list[str]] = []
    for family in families:
        for condition, label in condition_labels.items():
            row = tail_indexed[(family, condition)]
            if int(row["anchors"]) != 10:
                raise ValueError("Spectral tail summary requires ten anchors per condition")
            tail_output.append(
                [
                    FAMILY_LABELS[family],
                    label,
                    _format(_finite(row, "median_p95_pairwise_loss_contrast")),
                    _format(_finite(row, "median_p05_pairwise_margin_contrast")),
                    _format(_finite(row, "median_mean_loss_contrast_on_adam_tail")),
                    _format(_finite(row, "median_mean_loss_contrast_on_condition_tail")),
                    _format(_finite(row, "median_worst_loss_tail_jaccard")),
                ]
            )
    if (
        factorial_table != paths["family_factorial_summary"]
        or tail_table != paths["family_query_tail_summary"]
    ):
        raise ValueError(
            "Spectral-transplant key table paths differ from the strict output contract"
        )
    return factorial_output, tail_output, (factorial_table, tail_table), manifest


def render_outcome_report(
    functional_dir: Path,
    hybrid_dir: Path,
    short_branch_dir: Path,
    confirmatory_dir: Path,
    mechanism_report: Path,
    blog_path: Path,
    output_path: Path,
    *,
    tail_stability_dir: Path = Path("reports/tail-stability"),
    spectral_transplant_dir: Path = Path("reports/spectral-transplant"),
    families: tuple[str, ...] = FAMILIES,
    scope_amendment: str | Path | None = None,
) -> dict[str, Any]:
    families, scope = resolve_scope(families, scope_amendment)
    repository_root = _repository_root(mechanism_report)
    causal_evidence = load_causal_chain_evidence(repository_root, allow_pending=True)
    causal_section = render_causal_chain_markdown(causal_evidence, detailed=False, heading_level=3)
    causal_display = causal_chain_display_contract(causal_evidence)
    blog_path = blog_path.resolve()
    mechanism_manifest_path = _validate_mechanism_section(
        mechanism_report, blog_path, families, scope
    )
    causal_chain: dict[str, dict[str, Any]] = {}
    causal_table_paths: list[Path] = []
    if causal_evidence["complete"]:
        mechanism_manifest = _load_manifest(mechanism_manifest_path)
        if mechanism_manifest.get("causal_chain") != causal_display:
            raise ValueError(
                "Mechanism report causal-chain display differs from fresh strict evidence"
            )
        for label in ("temporal_short_branch", "dose_band"):
            branch = causal_evidence[label]
            record = branch["manifest"]
            path = Path(record["path"])
            causal_chain[label] = {
                "path": _portable_path(path, repository_root),
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "status": branch["status"],
                "claimable": branch["claimable"],
                "supported": branch["supported"],
                "claim_boundary": branch["claim_boundary"],
            }
        causal_table_paths = [
            Path(record["path"]) for record in causal_evidence["source_table_records"]
        ]
    functional, functional_table, functional_manifest = _functional_rows(functional_dir, families)
    hybrid, hybrid_table, hybrid_manifest = _hybrid_rows(hybrid_dir, families, scope)
    short, short_table, short_manifest = _short_branch_rows(short_branch_dir, families, scope)
    tail_discovery, tail_final, tail_tables, tail_manifest = _tail_stability_rows(
        tail_stability_dir, families, scope
    )
    spectral_factorial, spectral_tail, spectral_tables, spectral_manifest = (
        _spectral_transplant_rows(spectral_transplant_dir, families, scope)
    )
    confirmation, confirmation_table, confirmation_manifest = _confirmation_rows(
        confirmatory_dir, families, scope
    )
    familywise_sentence = (
        "familywise 95% interval over all six comparisons prespecified before the post-hoc "
        "Dense-only scope amendment. Only the familywise "
    )
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
            "### Does the tail signature survive accumulation?\n\n"
            + _table(
                [
                    "Family",
                    "Challenger",
                    "delta on AdamW tail",
                    "delta on challenger tail",
                    "tail Jaccard",
                    "post-hoc regime",
                ],
                tail_discovery,
            )
            + "\n\nThe fixed-state cross-tail identity diagnostic is post hoc: it distinguishes "
            "severity suppression on a shared fragile-query set from redistribution to a new worst "
            "set. The separately frozen three-seed endpoint rule tests whether the loss-tail and "
            "unseen-margin signs persist after shared-start accumulation:\n\n"
            + _table(
                [
                    "Family",
                    "Challenger",
                    "validation loss p95 delta",
                    "loss seed wins",
                    "unseen margin p05 delta",
                    "margin seed wins",
                    "decision",
                ],
                tail_final,
            )
            + "\n\nThis accumulated persistence test is prospective relative to the branch outcomes, "
            "but it does not establish that tail stability mediates a full-training BEIR gain.",
            "### Post-hoc spectrum-versus-basis causal decomposition\n\n"
            + _table(
                [
                    "Family",
                    "Immediate metric",
                    "spectrum main effect",
                    "basis main effect",
                    "interaction",
                ],
                spectral_factorial,
            )
            + "\n\nThe 2x2 transplant holds the checkpoint and evaluation examples fixed while swapping "
            "singular values and singular vectors. It therefore causally decomposes the immediate "
            "functional difference at these fixed states, but it is a post-hoc explanatory "
            "intervention rather than a confirmatory retrieval analysis. Its query-tail readout is:\n\n"
            + _table(
                [
                    "Family",
                    "Condition",
                    "loss p95 delta",
                    "margin p05 delta",
                    "delta on AdamW tail",
                    "delta on condition tail",
                    "tail Jaccard",
                ],
                spectral_tail,
            )
            + "\n\nThese fixed-state contrasts can attribute an immediate effect to spectrum versus "
            "basis; they cannot show that either component causes the full-training BEIR outcome.",
            "### Does the validation-frozen recipe replicate?\n\n"
            + _table(
                [
                    "Family",
                    "Contrast",
                    "mean delta nDCG@10",
                    "hierarchical 95% CI",
                    "familywise 95% CI",
                    "seed W/T/L",
                    "task W/T/L",
                ],
                confirmation,
            )
            + "\n\nRecipes were selected on the query-disjoint validation set before these runs. "
            "Intervals independently resample seeds and tasks; aggregate MTEB files do not support "
            "a query-level significance claim. The nominal interval is shown beside a Bonferroni "
            + familywise_sentence
            + "interval determines positive, negative, or inconclusive headline language; every "
            "contrast and all win counts remain visible.",
        ]
    )
    content += "\n\n" + causal_section
    output_path = output_path.resolve()
    _atomic_text(output_path, content + "\n")
    _atomic_text(blog_path, _replace_marked(blog_path.read_text(encoding="utf-8"), content))
    source_manifests = {
        "mechanism_report": _source(mechanism_manifest_path, repository_root=repository_root),
        "functional_intervention": _source(
            functional_dir / "manifest.json",
            repository_root=repository_root,
            anchors=functional_manifest["anchors"],
        ),
        "hybrid_adamw": _source(
            hybrid_dir / "summary_manifest.json",
            repository_root=repository_root,
            hybrid_units=hybrid_manifest["evaluations"]["hybrid_final_units"],
        ),
        "short_branch": _source(
            short_branch_dir / "summary_manifest.json",
            repository_root=repository_root,
            runs=short_manifest["coverage"]["runs"],
        ),
        "tail_stability": _source(
            tail_stability_dir / "summary_manifest.json",
            repository_root=repository_root,
            anchors=tail_manifest["discovery_anchors"],
            final_contrasts=tail_manifest["short_branch_final_rows"],
        ),
        "spectral_transplant": _source(
            spectral_transplant_dir / "summary_manifest.json",
            repository_root=repository_root,
            anchors=spectral_manifest["anchors"],
            anchor_effect_records=spectral_manifest["anchor_effect_records"],
        ),
        "confirmation": _source(
            confirmatory_dir / "summary_manifest.json",
            repository_root=repository_root,
            units=confirmation_manifest["coverage"]["evaluation_units"],
        ),
    }
    source_manifests.update(causal_chain)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "sources": source_manifests,
        "source_tables": [
            {
                "path": _portable_path(path, repository_root),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (
                functional_table,
                hybrid_table,
                short_table,
                *tail_tables,
                *spectral_tables,
                confirmation_table,
                *causal_table_paths,
            )
        ],
        "causal_chain": causal_display,
        "output": {
            "path": _portable_path(output_path, repository_root),
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        },
        "blog": {
            **_marked_block_record(blog_path, OUTCOME_MARKERS),
            "path": _portable_path(blog_path, repository_root),
        },
        "claim_boundary": (
            "Routing and local-step tables are controls; short branches test accumulation on frozen "
            "probes; the tail and spectrum-versus-basis analyses are post-hoc causal decomposition "
            "at fixed states and do not establish mediation of BEIR; only the validation-frozen "
            "three-seed BEIR table is confirmatory retrieval evidence."
        ),
    }
    if scope is not None:
        manifest["families"] = list(families)
        manifest["scope_amendment"] = scope
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
    parser.add_argument("--tail-stability-dir", type=Path, default=Path("reports/tail-stability"))
    parser.add_argument(
        "--spectral-transplant-dir",
        type=Path,
        default=Path("reports/spectral-transplant"),
    )
    parser.add_argument("--confirmatory-dir", type=Path, default=Path("reports/confirmatory"))
    parser.add_argument(
        "--mechanism-report", type=Path, default=Path("reports/mechanism-summary.md")
    )
    parser.add_argument("--blog", type=Path, default=Path("docs/blog.md"))
    parser.add_argument("--output", type=Path, default=Path("reports/outcome-summary.md"))
    parser.add_argument(
        "--families", nargs="+", choices=("dense", "late"), default=["dense", "late"]
    )
    parser.add_argument("--scope-amendment", type=Path)
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
        tail_stability_dir=args.tail_stability_dir,
        spectral_transplant_dir=args.spectral_transplant_dir,
        families=tuple(args.families),
        scope_amendment=args.scope_amendment,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
