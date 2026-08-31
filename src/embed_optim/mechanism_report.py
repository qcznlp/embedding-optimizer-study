from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .basis_sensitivity import audit_basis_sensitivity
from .causal_chain_rendering import (
    causal_chain_display_contract,
    render_causal_chain_markdown,
)
from .causal_chain_reporting import load_causal_chain_evidence
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .scope import ALL_FAMILIES, normalize_families, resolve_scope

MECHANISM_MARKERS = ("<!-- MECHANISM:BEGIN -->", "<!-- MECHANISM:END -->")
FAMILIES = ALL_FAMILIES
OPTIMIZERS = ("adamw", "muon", "normuon")
FAMILY_LABELS = {"dense": "DenseOn", "late": "LateOn"}
OPTIMIZER_LABELS = {"adamw": "AdamW", "muon": "Muon", "normuon": "NorMuon"}


def _resolve_declared(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _repository_root(path: Path) -> Path:
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    for candidate in (resolved, *resolved.parents):
        if candidate.name == "reports":
            return candidate.parent
    raise ValueError(f"Cannot resolve repository root from {path}")


def _portable_path(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repository_root.resolve()))
    except ValueError:
        return str(resolved)


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema: {path}")
    return payload


def _rehash_references(value: Any, *, context: str) -> int:
    if isinstance(value, list):
        return sum(
            _rehash_references(item, context=f"{context}[{index}]")
            for index, item in enumerate(value)
        )
    if not isinstance(value, dict):
        return 0
    verified = 0
    path_value = value.get("path")
    digest = value.get("sha256")
    if isinstance(path_value, str) and isinstance(digest, str):
        path = Path(path_value).resolve()
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"Referenced source differs in {context}: {path}")
        verified += 1
    return verified + sum(
        _rehash_references(item, context=f"{context}.{name}")
        for name, item in value.items()
        if name not in {"path", "sha256"}
    )


def _read_declared_csv(
    root: Path,
    manifest: dict[str, Any],
    name: str,
    *,
    required_fields: set[str],
) -> tuple[list[dict[str, str]], Path]:
    declared = manifest.get("outputs", {}).get(name)
    if not isinstance(declared, dict) or not isinstance(declared.get("path"), str):
        raise ValueError(f"Manifest does not declare {name}")
    path = _resolve_declared(root, declared["path"])
    if (
        not path.is_file()
        or path.stat().st_size != declared.get("bytes")
        or _sha256(path) != declared.get("sha256")
    ):
        raise ValueError(f"Declared table differs from its manifest: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not required_fields.issubset(fields):
            raise ValueError(f"Required fields are absent from {path}")
        rows = list(reader)
    if len(rows) != declared.get("rows"):
        raise ValueError(f"Declared row count differs for {path}")
    return rows, path


def _finite(row: dict[str, str], field: str, *, allow_empty: bool = False) -> float | None:
    value = row.get(field, "")
    if allow_empty and value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value for {field}: {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite numeric value for {field}: {value!r}")
    return parsed


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty metric group")
    return float(statistics.median(values))


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| "
            + " | ".join("---" if index == 0 else "---:" for index in range(len(headers)))
            + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def _format(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _common_state_rows(
    summary_dir: Path,
    families: tuple[str, ...] = FAMILIES,
) -> tuple[list[list[str]], dict[str, Any], Path]:
    families = normalize_families(families)
    summary_dir = summary_dir.resolve()
    manifest_path = summary_dir / "summary_manifest.json"
    manifest = _load_manifest(manifest_path)
    expected_rows = {
        "gradient_tensor_metrics": 1_760,
        "update_tensor_metrics": 5_280,
        "pairwise_tensor_cosines": 5_280,
        "gradient_anchor_metrics": 20,
        "anchor_metrics": 60,
        "pairwise_anchor_cosines": 60,
        "update_gradient_contrasts": 60,
        "anchor_contrasts": 40,
    }
    if (
        manifest.get("complete") is not True
        or manifest.get("allow_partial") is not False
        or manifest.get("expected_anchors") != 20
        or manifest.get("valid_anchors") != 20
        or manifest.get("missing_labels") != []
        or {name: item.get("rows") for name, item in manifest.get("outputs", {}).items()}
        != expected_rows
    ):
        raise ValueError("Common-state summary is not the complete frozen 20-anchor matrix")
    fields = {
        "family",
        "label",
        "update_operator",
        "row_norm_cv_parameter_weighted_to_adamw_ratio",
        "top_1pct_row_energy_parameter_weighted_to_adamw_ratio",
        "approx_stable_rank_parameter_weighted_to_adamw_ratio",
        "spectral_norm_parameter_weighted_to_adamw_ratio",
        "cosine_with_adamw_parameter_weighted",
    }
    rows, path = _read_declared_csv(
        summary_dir, manifest, "anchor_contrasts", required_fields=fields
    )
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("family", ""), row.get("update_operator", ""))
        if key[0] not in FAMILIES or key[1] not in {"muon", "normuon"}:
            raise ValueError(f"Unexpected common-state contrast identity: {key}")
        grouped[key].append(row)
    if set(grouped) != {
        (family, optimizer) for family in FAMILIES for optimizer in ("muon", "normuon")
    } or any(len(values) != 10 for values in grouped.values()):
        raise ValueError("Common-state contrasts do not cover ten anchors per family/operator")
    output: list[list[str]] = []
    metrics = (
        "row_norm_cv_parameter_weighted_to_adamw_ratio",
        "top_1pct_row_energy_parameter_weighted_to_adamw_ratio",
        "approx_stable_rank_parameter_weighted_to_adamw_ratio",
        "spectral_norm_parameter_weighted_to_adamw_ratio",
        "cosine_with_adamw_parameter_weighted",
    )
    for family in families:
        for optimizer in ("muon", "normuon"):
            values = grouped[(family, optimizer)]
            medians = [_median([float(_finite(row, field)) for row in values]) for field in metrics]
            output.append(
                [
                    FAMILY_LABELS[family],
                    OPTIMIZER_LABELS[optimizer],
                    *[_format(value, 3) for value in medians],
                ]
            )
    return output, manifest, path


def _basis_rows(
    summary_dir: Path,
    families: tuple[str, ...] = FAMILIES,
) -> tuple[list[list[str]], dict[str, Any], Path]:
    families = normalize_families(families)
    summary_dir = summary_dir.resolve()
    manifest_path = summary_dir / "summary_manifest.json"
    manifest = _load_manifest(manifest_path)
    if (
        manifest.get("coverage")
        != {
            "anchors": 20,
            "tensor_sequences": 60,
            "records": 540,
            "head_records": 3_240,
            "summary_rows": 6,
        }
        or manifest.get("complete") is not True
    ):
        raise ValueError("Basis diagnostic is not the complete frozen Cartesian grid")
    protocol_path = Path(str(manifest.get("protocol", {}).get("path", "")))
    audited = audit_basis_sensitivity(protocol_path, output_dir=summary_dir)
    if audited != manifest:
        raise ValueError("Basis diagnostic changed during its strict audit")
    fields = {
        "family",
        "optimizer",
        "records",
        "median_mapped_direction_cosine",
        "median_mapped_relative_frobenius_error",
        "median_absolute_norm_ratio_error",
        "median_predicted_descent_relative_error",
        "median_head_spectrum_relative_l2_error",
    }
    rows, path = _read_declared_csv(summary_dir, manifest, "summary", required_fields=fields)
    indexed = {(row.get("family", ""), row.get("optimizer", "")): row for row in rows}
    expected = {(family, optimizer) for family in FAMILIES for optimizer in OPTIMIZERS}
    if len(rows) != 6 or set(indexed) != expected:
        raise ValueError("Basis summary does not cover every family/optimizer group")
    output = []
    for family in families:
        for optimizer in OPTIMIZERS:
            row = indexed[(family, optimizer)]
            if int(row["records"]) != 90:
                raise ValueError("Basis summary must contain 90 records per family/operator")
            output.append(
                [
                    FAMILY_LABELS[family],
                    OPTIMIZER_LABELS[optimizer],
                    _format(_finite(row, "median_mapped_direction_cosine"), 5),
                    _format(_finite(row, "median_mapped_relative_frobenius_error"), 5),
                    _format(_finite(row, "median_absolute_norm_ratio_error"), 5),
                    _format(_finite(row, "median_predicted_descent_relative_error"), 5),
                    _format(_finite(row, "median_head_spectrum_relative_l2_error"), 5),
                ]
            )
    return output, manifest, path


def _spectrum_rows(
    summary_dir: Path,
    families: tuple[str, ...] = FAMILIES,
) -> tuple[list[list[str]], dict[str, Any], Path]:
    families = normalize_families(families)
    summary_dir = summary_dir.resolve()
    manifest_path = summary_dir / "summary_manifest.json"
    manifest = _load_manifest(manifest_path)
    if (
        manifest.get("complete") is not True
        or manifest.get("allow_partial") is not False
        or manifest.get("expected_anchors") != 20
        or manifest.get("valid_anchors") != 20
        or manifest.get("expected_spectra") != 360
        or manifest.get("valid_spectra") != 360
        or manifest.get("missing_labels") != []
    ):
        raise ValueError("Exact-spectrum summary is not the frozen 360-spectrum matrix")
    fields = {
        "family",
        "label",
        "update_operator",
        "tensor",
        "rank",
        "stable_rank",
        "entropy_effective_rank",
        "condition_number",
    }
    rows, path = _read_declared_csv(
        summary_dir, manifest, "spectrum_metrics", required_fields=fields
    )
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("family", ""), row.get("update_operator", ""))
        if key[0] not in FAMILIES or key[1] not in OPTIMIZERS:
            raise ValueError(f"Unexpected exact-spectrum identity: {key}")
        grouped[key].append(row)
    if set(grouped) != {
        (family, optimizer) for family in FAMILIES for optimizer in OPTIMIZERS
    } or any(len(values) != 60 for values in grouped.values()):
        raise ValueError("Exact spectra do not cover 60 selected matrices per family/operator")
    output: list[list[str]] = []
    for family in families:
        for optimizer in OPTIMIZERS:
            values = grouped[(family, optimizer)]
            normalized_stable = []
            normalized_entropy = []
            conditions = []
            for row in values:
                rank = _finite(row, "rank")
                stable = _finite(row, "stable_rank")
                entropy = _finite(row, "entropy_effective_rank")
                assert rank is not None and stable is not None and entropy is not None
                if rank <= 0:
                    raise ValueError("Exact spectrum has a non-positive rank")
                normalized_stable.append(stable / rank)
                normalized_entropy.append(entropy / rank)
                condition = _finite(row, "condition_number", allow_empty=True)
                if condition is not None:
                    conditions.append(condition)
            output.append(
                [
                    FAMILY_LABELS[family],
                    OPTIMIZER_LABELS[optimizer],
                    _format(_median(normalized_stable)),
                    _format(_median(normalized_entropy)),
                    _format(_median(conditions) if conditions else None, 2),
                ]
            )
    return output, manifest, path


def _bridge_rows_with_coverage(
    bridge_dir: Path,
    families: tuple[str, ...] = FAMILIES,
) -> tuple[
    list[list[str]],
    list[list[str]],
    dict[str, Any],
    list[Path],
    dict[str, int],
]:
    families = normalize_families(families)
    bridge_dir = bridge_dir.resolve()
    manifest_path = bridge_dir / "summary_manifest.json"
    manifest = _load_manifest(manifest_path)
    if (
        manifest.get("complete") is not True
        or manifest.get("checkpoints") != 120
        or manifest.get("within_run_transitions") != 96
        or manifest.get("correlations") != 216
        or not {"training_dynamics", "loss_retrieval_protocol"}.issubset(
            manifest.get("sources", {})
        )
    ):
        raise ValueError("Mechanism bridge is not the strict 120-checkpoint join")
    if _rehash_references(manifest.get("sources"), context="mechanism_bridge.sources") < 6:
        raise ValueError("Mechanism bridge does not bind all strict source manifests and tables")
    checkpoint_fields = {
        "model_family",
        "optimizer",
        "stage",
        "mean_training_loss",
        "training_margin_mean",
        "unseen_margin_mean",
        "unseen_query_normalized_effective_rank",
        "unseen_reference_top1_agreement",
        "unseen_document_token_coverage_mean",
        "mean_beir_ndcg_at_10",
    }
    checkpoints, checkpoint_path = _read_declared_csv(
        bridge_dir, manifest, "checkpoint_bridge", required_fields=checkpoint_fields
    )
    final: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    checkpoint_groups: dict[tuple[str, str, int], int] = defaultdict(int)
    for row in checkpoints:
        family = row.get("model_family", "")
        optimizer = row.get("optimizer", "")
        try:
            stage = int(row.get("stage", ""))
        except ValueError as error:
            raise ValueError(f"Invalid mechanism checkpoint stage: {row}") from error
        if family not in FAMILIES or optimizer not in OPTIMIZERS or not 1 <= stage <= 5:
            raise ValueError(f"Invalid mechanism checkpoint identity: {row}")
        checkpoint_groups[(family, optimizer, stage)] += 1
        if stage == 5:
            final[(family, optimizer)].append(row)
    expected_checkpoint_groups = {
        (family, optimizer, stage)
        for family in FAMILIES
        for optimizer in OPTIMIZERS
        for stage in range(1, 6)
    }
    if set(checkpoint_groups) != expected_checkpoint_groups or any(
        count != 4 for count in checkpoint_groups.values()
    ):
        raise ValueError(
            "Mechanism bridge does not contain four learning rates per checkpoint group"
        )
    if set(final) != {
        (family, optimizer) for family in FAMILIES for optimizer in OPTIMIZERS
    } or any(len(values) != 4 for values in final.values()):
        raise ValueError("Final representation bridge does not cover four learning rates")
    representation: list[list[str]] = []
    fields = (
        "training_margin_mean",
        "unseen_margin_mean",
        "unseen_query_normalized_effective_rank",
        "unseen_reference_top1_agreement",
        "mean_beir_ndcg_at_10",
    )
    include_token_coverage = "late" in families
    for family in families:
        for optimizer in OPTIMIZERS:
            values = final[(family, optimizer)]
            summaries = [
                _median([float(_finite(row, field)) for row in values]) for field in fields
            ]
            coverage: float | None = None
            if family == "late":
                coverage = _median(
                    [float(_finite(row, "unseen_document_token_coverage_mean")) for row in values]
                )
            rendered = [
                FAMILY_LABELS[family],
                OPTIMIZER_LABELS[optimizer],
                *[_format(value) for value in summaries[:4]],
            ]
            if include_token_coverage:
                rendered.append(_format(coverage))
            rendered.append(_format(summaries[4]))
            representation.append(rendered)

    correlation_fields = {
        "model_family",
        "scope",
        "optimizer",
        "analysis",
        "predictor",
        "outcome",
        "observations",
        "spearman_rho",
    }
    correlations, correlation_path = _read_declared_csv(
        bridge_dir,
        manifest,
        "descriptive_correlations",
        required_fields=correlation_fields,
    )
    correlation_counts = {
        family: sum(row.get("model_family", "") == family for row in correlations)
        for family in FAMILIES
    }
    if correlation_counts != {"dense": 96, "late": 120}:
        raise ValueError("Mechanism correlations do not cover the frozen family-specific grid")
    indexed = {
        (
            row["model_family"],
            row["scope"],
            row["optimizer"],
            row["analysis"],
            row["predictor"],
            row["outcome"],
        ): row
        for row in correlations
    }
    selected = [
        (family, predictor, outcome)
        for family in families
        for predictor, outcome in (
            ("reference_delta_row_cv_parameter_weighted", "unseen_margin_mean"),
            ("unseen_margin_mean", "mean_beir_ndcg_at_10"),
            ("unseen_query_normalized_effective_rank", "mean_beir_ndcg_at_10"),
        )
    ]
    if "late" in families:
        selected.append(("late", "unseen_document_token_coverage_mean", "mean_beir_ndcg_at_10"))
    selected.extend((family, "mean_training_loss", "mean_beir_ndcg_at_10") for family in families)
    labels = {
        "reference_delta_row_cv_parameter_weighted": "weight-delta row CV",
        "unseen_margin_mean": "unseen margin",
        "unseen_query_normalized_effective_rank": "unseen query effective rank",
        "unseen_document_token_coverage_mean": "document-token coverage",
        "mean_training_loss": "trailing training loss (post-hoc)",
        "mean_beir_ndcg_at_10": "mean BEIR nDCG@10",
    }
    correlation_output: list[list[str]] = []
    for family, predictor, outcome in selected:
        key = (
            family,
            "all_optimizers",
            "all",
            "within_run_first_differences",
            predictor,
            outcome,
        )
        if key not in indexed:
            raise ValueError(f"Prespecified mechanism correlation is absent: {key}")
        row = indexed[key]
        observations = int(row["observations"])
        if observations != 48:
            raise ValueError(f"Unexpected within-run observation count for {key}: {observations}")
        rho = _finite(row, "spearman_rho", allow_empty=True)
        correlation_output.append(
            [
                FAMILY_LABELS[family],
                labels[predictor],
                labels[outcome],
                str(observations),
                _format(rho, 3),
            ]
        )
    selected_correlations = sum(correlation_counts[family] for family in families)
    selected_checkpoints = sum(
        count
        for (family, _optimizer, _stage), count in checkpoint_groups.items()
        if family in families
    )
    bridge_tables = [checkpoint_path, correlation_path]
    selected_transitions = manifest["within_run_transitions"] * len(families) // len(FAMILIES)
    if families != FAMILIES:
        transition_fields = {
            "model_family",
            "optimizer",
            "learning_rate",
            "stage",
            "previous_stage",
        }
        transitions, transition_path = _read_declared_csv(
            bridge_dir,
            manifest,
            "within_run_changes",
            required_fields=transition_fields,
        )
        transition_groups: dict[tuple[str, str, int], int] = defaultdict(int)
        for row in transitions:
            family = row.get("model_family", "")
            optimizer = row.get("optimizer", "")
            try:
                stage = int(row.get("stage", ""))
                previous_stage = int(row.get("previous_stage", ""))
                float(row.get("learning_rate", ""))
            except ValueError as error:
                raise ValueError(f"Invalid mechanism transition identity: {row}") from error
            if (
                family not in FAMILIES
                or optimizer not in OPTIMIZERS
                or not 2 <= stage <= 5
                or previous_stage != stage - 1
            ):
                raise ValueError(f"Invalid mechanism transition identity: {row}")
            transition_groups[(family, optimizer, stage)] += 1
        expected_transition_groups = {
            (family, optimizer, stage)
            for family in FAMILIES
            for optimizer in OPTIMIZERS
            for stage in range(2, 6)
        }
        if set(transition_groups) != expected_transition_groups or any(
            count != 4 for count in transition_groups.values()
        ):
            raise ValueError(
                "Mechanism transitions do not contain four learning rates per adjacent stage"
            )
        selected_transitions = sum(
            count
            for (family, _optimizer, _stage), count in transition_groups.items()
            if family in families
        )
        bridge_tables.append(transition_path)
    coverage = {
        "checkpoints": selected_checkpoints,
        "within_run_transitions": selected_transitions,
        "correlations": selected_correlations,
    }
    return (
        representation,
        correlation_output,
        manifest,
        bridge_tables,
        coverage,
    )


def _bridge_rows(
    bridge_dir: Path,
    families: tuple[str, ...] = FAMILIES,
) -> tuple[list[list[str]], list[list[str]], dict[str, Any], list[Path]]:
    representation, correlations, manifest, tables, _coverage = _bridge_rows_with_coverage(
        bridge_dir, families
    )
    return representation, correlations, manifest, tables


def _validate_figure(path: Path, *, spectra: bool = False) -> dict[str, Any]:
    path = path.resolve()
    sidecar = path.with_suffix(".manifest.json")
    manifest = _load_manifest(sidecar)
    output = manifest.get("output") or {}
    if (
        not path.is_file()
        or Path(output.get("path", "")).resolve() != path
        or path.stat().st_size != output.get("bytes")
        or _sha256(path) != output.get("sha256")
        or (not spectra and manifest.get("complete") is not True)
        or (spectra and (manifest.get("anchors") != 20 or manifest.get("spectra") != 360))
    ):
        raise ValueError(f"Mechanism figure differs from its strict manifest: {path}")
    sources = {name: value for name, value in manifest.items() if name != "output"}
    if _rehash_references(sources, context=f"figure.{path.name}") == 0:
        raise ValueError(f"Mechanism figure does not bind a source manifest: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "manifest_sha256": _sha256(sidecar),
    }


def _retrieval_rows(
    retrieval_dir: Path,
    families: tuple[str, ...] = FAMILIES,
) -> tuple[list[list[str]], dict[str, Any], Path, dict[str, Any]]:
    families = normalize_families(families)
    retrieval_dir = retrieval_dir.resolve()
    repository_root = retrieval_dir.parents[1]
    manifest_path = retrieval_dir / "summary_manifest.json"
    manifest = _load_manifest(manifest_path)
    if (
        manifest.get("complete") is not True
        or manifest.get("coverage")
        != {
            "runs": 24,
            "checkpoints": 120,
            "tasks": 14,
            "evaluation_units": 1_680,
            "optimizer_family_groups": 6,
            "best_config_task_delta_rows": 280,
            "adjacent_stage_task_stability_rows": 16,
        }
        or _rehash_references(manifest.get("sources"), context="retrieval_dynamics.sources") < 1_685
    ):
        raise ValueError("Retrieval dynamics is not the strict 1,680-unit completion report")
    fields = {
        "model_family",
        "optimizer",
        "learning_rate_points",
        "adamw_median_final_target",
        "points_reaching_target",
        "points_right_censored",
        "fastest_observed_useful_wall_time_hours",
        "median_observed_useful_wall_time_hours",
        "target_definition",
        "interpolation",
    }
    rows, table = _read_declared_csv(
        repository_root,
        manifest,
        "optimizer_first_passage",
        required_fields=fields,
    )
    indexed = {(row.get("model_family", ""), row.get("optimizer", "")): row for row in rows}
    expected = {(family, optimizer) for family in FAMILIES for optimizer in OPTIMIZERS}
    if len(rows) != 6 or set(indexed) != expected:
        raise ValueError("Retrieval first-passage table does not cover six optimizer/family groups")
    output = []
    for family in families:
        for optimizer in OPTIMIZERS:
            row = indexed[(family, optimizer)]
            learning_rates = int(row["learning_rate_points"])
            reached = int(row["points_reaching_target"])
            censored = int(row["points_right_censored"])
            target = _finite(row, "adamw_median_final_target")
            fastest = _finite(row, "fastest_observed_useful_wall_time_hours", allow_empty=True)
            median = _finite(row, "median_observed_useful_wall_time_hours", allow_empty=True)
            if (
                learning_rates != 4
                or reached + censored != 4
                or row["target_definition"] != "within-family-median-of-four-adamw-final-points"
                or row["interpolation"] != "none-five-observed-checkpoints-only"
            ):
                raise ValueError(f"Invalid retrieval first-passage group: {(family, optimizer)}")
            output.append(
                [
                    FAMILY_LABELS[family],
                    OPTIMIZER_LABELS[optimizer],
                    _format(target),
                    f"{reached}/4",
                    _format(fastest, 3),
                    _format(median, 3),
                    str(censored),
                ]
            )
    figure_item = manifest.get("outputs", {}).get("quality_vs_useful_wall_time", {})
    figure_path = _resolve_declared(repository_root, str(figure_item.get("path", "")))
    if (
        figure_path != retrieval_dir / "quality_vs_useful_wall_time.svg"
        or not figure_path.is_file()
        or figure_path.stat().st_size != figure_item.get("bytes")
        or _sha256(figure_path) != figure_item.get("sha256")
    ):
        raise ValueError("Retrieval dynamics figure differs from its strict manifest")
    figure = {"path": str(figure_path), "sha256": _sha256(figure_path)}
    return output, manifest, table, figure


def _replace_marked(text: str, content: str) -> str:
    begin, end = MECHANISM_MARKERS
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError("Expected exactly one mechanism marker pair in the blog")
    before, remainder = text.split(begin)
    _, after = remainder.split(end)
    return f"{before}{begin}\n\n{content}\n\n{end}{after}"


def _marked_block_bytes(text: str, markers: tuple[str, str]) -> bytes:
    """Return the exact UTF-8 bytes owned by one blog renderer."""

    begin, end = markers
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"Expected exactly one marker pair {markers}")
    start = text.index(begin)
    stop = text.index(end, start + len(begin)) + len(end)
    return text[start:stop].encode("utf-8")


def _marked_block_record(path: Path, markers: tuple[str, str]) -> dict[str, Any]:
    block = _marked_block_bytes(path.read_text(encoding="utf-8"), markers)
    return {
        "path": str(path.resolve()),
        "markers": list(markers),
        "block_bytes": len(block),
        "block_sha256": hashlib.sha256(block).hexdigest(),
    }


def _marked_block_complete(
    path: Path,
    record: Any,
    markers: tuple[str, str],
    *,
    repository_root: Path | None = None,
) -> bool:
    declared = Path(str(record.get("path", ""))) if isinstance(record, dict) else Path()
    declared_path = (
        declared.resolve()
        if declared.is_absolute() or repository_root is None
        else (repository_root.resolve() / declared).resolve()
    )
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("path"), str)
        or declared_path != path.resolve()
        or record.get("markers") != list(markers)
    ):
        return False
    try:
        block = _marked_block_bytes(path.read_text(encoding="utf-8"), markers)
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    return bool(
        record.get("block_bytes") == len(block)
        and record.get("block_sha256") == hashlib.sha256(block).hexdigest()
    )


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def ensure_retrieval_dynamics(
    retrieval_dir: Path,
    *,
    builder: Any | None = None,
) -> Path:
    retrieval_dir = retrieval_dir.resolve()
    manifest_path = retrieval_dir / "summary_manifest.json"
    if manifest_path.is_file():
        return manifest_path
    if builder is None:
        from .retrieval_dynamics import build_retrieval_dynamics

        builder = build_retrieval_dynamics
    builder(output_dir=retrieval_dir)
    if not manifest_path.is_file():
        raise ValueError("Retrieval-dynamics builder did not produce its strict summary manifest")
    return manifest_path


def render_mechanism_report(
    common_state_dir: Path,
    spectrum_dir: Path,
    bridge_dir: Path,
    retrieval_dir: Path,
    blog_path: Path,
    output_path: Path,
    *,
    basis_dir: Path,
    spectrum_figure: Path,
    representation_figure: Path,
    late_token_figure: Path,
    families: tuple[str, ...] = FAMILIES,
    scope_amendment: str | Path | None = None,
) -> dict[str, Any]:
    families = normalize_families(families)
    families, scope = resolve_scope(families, scope_amendment)
    repository_root = _repository_root(common_state_dir)
    causal_evidence = load_causal_chain_evidence(repository_root, allow_pending=True)
    causal_section = render_causal_chain_markdown(causal_evidence, detailed=True, heading_level=3)
    causal_display = causal_chain_display_contract(causal_evidence)
    causal_manifests = {}
    causal_table_paths: list[Path] = []
    if causal_evidence["complete"]:
        for label in ("temporal_short_branch", "dose_band"):
            branch = causal_evidence[label]
            record = branch["manifest"]
            path = Path(record["path"])
            causal_manifests[label] = {
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
    family_count = len(families)
    common_rows, common_manifest, common_table = _common_state_rows(common_state_dir, families)
    basis_rows, basis_manifest, basis_table = _basis_rows(basis_dir, families)
    spectrum_rows, spectrum_manifest, spectrum_table = _spectrum_rows(spectrum_dir, families)
    (
        representation_rows,
        correlation_rows,
        bridge_manifest,
        bridge_tables,
        bridge_coverage,
    ) = _bridge_rows_with_coverage(bridge_dir, families)
    retrieval_rows, retrieval_manifest, retrieval_table, retrieval_figure = _retrieval_rows(
        retrieval_dir, families
    )
    figures = {"retrieval_dynamics": retrieval_figure}
    if families == FAMILIES:
        figures.update(
            {
                "exact_spectra": _validate_figure(spectrum_figure, spectra=True),
                "representation_dynamics": _validate_figure(representation_figure),
                "late_token_dynamics": _validate_figure(late_token_figure),
            }
        )
    for record in figures.values():
        record["path"] = _portable_path(Path(record["path"]), repository_root)
    content = "\n\n".join(
        [
            "The formal mechanism tier evaluates every optimizer transform at the same frozen weights "
            "and on the same ordered eight-gradient history. The values below are generated only after "
            "the complete 20-anchor matrix, 540 basis comparisons, 360 exact spectra, both 122-job "
            "representation tiers, and the 1,680-unit retrieval matrix pass their content-hash audits.",
            "### Retrieval time to an AdamW reference\n\n"
            "![Retrieval quality versus useful wall time]"
            "(../reports/retrieval-dynamics/quality_vs_useful_wall_time.svg)\n\n"
            + _table(
                [
                    "Family",
                    "Optimizer",
                    "AdamW reference",
                    "LR points reaching",
                    "fastest hours",
                    "median hours",
                    "right-censored",
                ],
                retrieval_rows,
            )
            + "\n\nThe reference is the within-family median final nDCG@10 of the four AdamW "
            "learning-rate points. Passage is observed only at the five saved checkpoints; no "
            "interpolation is used, and non-reaching points remain right-censored. Checkpoint time "
            "is a step-proportional estimate from audited useful terminal wall time. The rule was "
            "locked after 160/1,680 discovery units were visible, so this is exploratory rather "
            "than a preregistration or a substitute for the three-seed confirmation.",
            "### Same-state optimizer fingerprints\n\n"
            + _table(
                [
                    "Family",
                    "Operator",
                    "row CV / AdamW",
                    "top-1% row energy / AdamW",
                    "stable rank / AdamW",
                    "spectral norm / AdamW",
                    "cosine with AdamW",
                ],
                common_rows,
            )
            + "\n\nEach cell is the median over ten frozen anchors. Ratios use raw optimizer "
            "directions but are scale-invariant except for the explicitly reported spectral-norm ratio; "
            "the exact-spectrum intervention below uses per-tensor Frobenius-matched directions. Weight "
            "decay is excluded from this comparison.",
            "### Function-preserving basis sensitivity\n\n"
            + _table(
                [
                    "Family",
                    "Operator",
                    "mapped cosine",
                    "relative direction error",
                    "absolute norm-ratio error",
                    "predicted-descent error",
                    "Q/K spectrum error",
                ],
                basis_rows,
            )
            + "\n\nEach row is the median over 90 fixed comparisons: ten common-state anchors, "
            "three QKV layers, and three seeded RoPE-commuting rotations. Query and key share "
            "each split-half plane rotation, value rows are unchanged, and every direction is "
            "inverse-mapped before comparison. The transform preserves attention logits, so this "
            "table measures implementation-level coordinate dependence rather than retrieval "
            "quality; bfloat16 Newton--Schulz rounding is retained as part of the Muon runtime.",
            "### Exact update spectra\n\n"
            "![Exact common-state update spectra](../reports/common-state/exact-update-spectra.svg)\n\n"
            + _table(
                [
                    "Family",
                    "Operator",
                    "stable rank / rank",
                    "entropy rank / rank",
                    "condition number",
                ],
                spectrum_rows,
            )
            + "\n\nThe six matrices were fixed by early/middle/final depth and attention/MLP role "
            "before formal spectra existed. Values are medians over 60 exact spectra per "
            "family/operator; the figure shows the full normalized curves and interquartile bands.",
            "### Representation and score geometry\n\n"
            "![Representation dynamics](../reports/representation-space/representation-dynamics.svg)\n\n"
            + _table(
                [
                    "Family",
                    "Optimizer",
                    "training margin",
                    "unseen margin",
                    "unseen query rank",
                    "pretrained top-1 agreement",
                    "Late document-token coverage",
                    "mean BEIR nDCG@10",
                ],
                representation_rows,
            )
            + "\n\nRows are final-stage medians across all four frozen learning rates, not "
            "test-selected winners. Training and unseen probes remain separate; the latter contains "
            "224 fixed examples balanced over all 14 decontaminated tasks.",
            "### Late-interaction token utilization\n\n"
            "![LateOn token-utilization dynamics](../reports/representation-space/late-token-dynamics.svg)\n\n"
            "This panel reports the four prespecified MaxSim evidence summaries on both probe tiers. "
            "It is kept separate from the shared DenseOn/LateOn figure so a LateOn-only signal cannot "
            "change the cross-architecture metric definition after results are visible.",
            "### Descriptive temporal bridge\n\n"
            + _table(
                [
                    "Family",
                    "Predictor change",
                    "Outcome change",
                    "Transitions",
                    "Spearman ρ",
                ],
                correlation_rows,
            )
            + "\n\nThe first seven geometry associations were fixed in the renderer and use "
            "within-run first differences across all optimizers. The final two training-loss rows "
            "are explicitly post-hoc diagnostics added after 1,456/1,680 discovery units were "
            "visible. All nine are one-seed observational summaries, not a causal mediation "
            "analysis. The same-state fingerprints identify what each update rule does; causal "
            "claims about later retrieval still require matched short branches or optimizer-switch "
            "interventions.",
        ]
    )
    if families != FAMILIES:
        content = "\n\n".join(
            [
                "Under the disclosed post-hoc DenseOn scope, the formal mechanism tier evaluates "
                "every optimizer transform at the same frozen weights and on the same ordered "
                "eight-gradient history. The complete historical source artifacts still pass their "
                "content-hash and cardinality audits before the renderer selects the active DenseOn "
                "slice: 10 common-state anchors, 270 basis comparisons, 180 exact spectra, 60 bridge "
                "checkpoints, and 840 retrieval evaluation units.",
                "### Retrieval time to an AdamW reference\n\n"
                + _table(
                    [
                        "Family",
                        "Optimizer",
                        "AdamW reference",
                        "LR points reaching",
                        "fastest hours",
                        "median hours",
                        "right-censored",
                    ],
                    retrieval_rows,
                )
                + "\n\nThe reference is the DenseOn median final nDCG@10 of the four AdamW "
                "learning-rate points. Passage is observed only at the five saved checkpoints; no "
                "interpolation is used, and non-reaching points remain right-censored. Checkpoint "
                "time is a step-proportional estimate from audited useful terminal wall time. This "
                "one-seed discovery analysis remains exploratory rather than a substitute for the "
                "validation-frozen three-seed confirmation.",
                "### Same-state optimizer fingerprints\n\n"
                + _table(
                    [
                        "Family",
                        "Operator",
                        "row CV / AdamW",
                        "top-1% row energy / AdamW",
                        "stable rank / AdamW",
                        "spectral norm / AdamW",
                        "cosine with AdamW",
                    ],
                    common_rows,
                )
                + "\n\nEach cell is the median over ten frozen DenseOn anchors. Ratios use raw "
                "optimizer directions but are scale-invariant except for the explicitly reported "
                "spectral-norm ratio; the exact-spectrum intervention uses per-tensor "
                "Frobenius-matched directions. Weight decay is excluded from this comparison.",
                "### Function-preserving basis sensitivity\n\n"
                + _table(
                    [
                        "Family",
                        "Operator",
                        "mapped cosine",
                        "relative direction error",
                        "absolute norm-ratio error",
                        "predicted-descent error",
                        "Q/K spectrum error",
                    ],
                    basis_rows,
                )
                + "\n\nEach row is the median over 90 fixed comparisons: ten common-state "
                "anchors, three QKV layers, and three seeded RoPE-commuting rotations. Query and key "
                "share each split-half plane rotation, value rows are unchanged, and every direction "
                "is inverse-mapped before comparison. The transform preserves attention logits, so "
                "this table measures implementation-level coordinate dependence rather than "
                "retrieval quality; bfloat16 Newton--Schulz rounding is retained as part of the Muon "
                "runtime.",
                "### Exact update spectra\n\n"
                + _table(
                    [
                        "Family",
                        "Operator",
                        "stable rank / rank",
                        "entropy rank / rank",
                        "condition number",
                    ],
                    spectrum_rows,
                )
                + "\n\nThe six matrices were fixed by early/middle/final depth and attention/MLP "
                "role before formal spectra existed. Values are medians over 60 exact spectra per "
                "optimizer on the active DenseOn anchors.",
                "### Representation and score geometry\n\n"
                + _table(
                    [
                        "Family",
                        "Optimizer",
                        "training margin",
                        "unseen margin",
                        "unseen query rank",
                        "pretrained top-1 agreement",
                        "mean BEIR nDCG@10",
                    ],
                    representation_rows,
                )
                + "\n\nRows are final-stage medians across all four frozen learning rates, not "
                "test-selected winners. Training and unseen probes remain separate; the latter "
                "contains 224 fixed examples balanced over all 14 decontaminated tasks.",
                "### Descriptive temporal bridge\n\n"
                + _table(
                    [
                        "Family",
                        "Predictor change",
                        "Outcome change",
                        "Transitions",
                        "Spearman ρ",
                    ],
                    correlation_rows,
                )
                + "\n\nThe first three geometry associations were fixed in the renderer and use "
                "within-run first differences across all optimizers. The final training-loss row is "
                "an explicitly post-hoc diagnostic. All four are one-seed observational summaries, "
                "not a causal mediation analysis. Same-state fingerprints identify what each update "
                "rule does; causal claims about accumulated retrieval behavior still require the "
                "matched shared-start branches and fixed-state spectral interventions.",
            ]
        )
    content += "\n\n" + causal_section
    output_path = output_path.resolve()
    _atomic_text(output_path, content + "\n")
    blog_path = blog_path.resolve()
    rendered_blog = _replace_marked(blog_path.read_text(encoding="utf-8"), content)
    _atomic_text(blog_path, rendered_blog)
    source_manifests = {
        "common_state": {
            "path": _portable_path(common_state_dir / "summary_manifest.json", repository_root),
            "sha256": _sha256((common_state_dir / "summary_manifest.json").resolve()),
            "anchors": common_manifest["valid_anchors"] * family_count // len(FAMILIES),
        },
        "retrieval_dynamics": {
            "path": _portable_path(retrieval_dir / "summary_manifest.json", repository_root),
            "sha256": _sha256((retrieval_dir / "summary_manifest.json").resolve()),
            "evaluation_units": (
                retrieval_manifest["coverage"]["evaluation_units"] * family_count // len(FAMILIES)
            ),
        },
        "exact_spectra": {
            "path": _portable_path(spectrum_dir / "summary_manifest.json", repository_root),
            "sha256": _sha256((spectrum_dir / "summary_manifest.json").resolve()),
            "spectra": spectrum_manifest["valid_spectra"] * family_count // len(FAMILIES),
        },
        "basis_sensitivity": {
            "path": _portable_path(basis_dir / "summary_manifest.json", repository_root),
            "sha256": _sha256((basis_dir / "summary_manifest.json").resolve()),
            "records": basis_manifest["coverage"]["records"] * family_count // len(FAMILIES),
            "head_records": (
                basis_manifest["coverage"]["head_records"] * family_count // len(FAMILIES)
            ),
        },
        "mechanism_bridge": {
            "path": _portable_path(bridge_dir / "summary_manifest.json", repository_root),
            "sha256": _sha256((bridge_dir / "summary_manifest.json").resolve()),
            "checkpoints": bridge_manifest["checkpoints"] * family_count // len(FAMILIES),
        },
    }
    source_manifests.update(causal_manifests)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "sources": source_manifests,
        "source_tables": [
            {
                "path": _portable_path(path, repository_root),
                "sha256": _sha256(path),
            }
            for path in [
                retrieval_table,
                common_table,
                basis_table,
                spectrum_table,
                *bridge_tables,
                *causal_table_paths,
            ]
        ],
        "causal_chain": causal_display,
        "figures": figures,
        "output": {
            "path": _portable_path(output_path, repository_root),
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        },
        "blog": {
            **_marked_block_record(blog_path, MECHANISM_MARKERS),
            "path": _portable_path(blog_path, repository_root),
        },
        "aggregation": {
            "retrieval_dynamics": "six-family-optimizer-groups-over-four-learning-rate-points",
            "common_state": "median-over-ten-frozen-anchors-per-family-operator",
            "basis_sensitivity": "median-over-ninety-fixed-comparisons-per-family-operator",
            "exact_spectra": "median-over-sixty-prespecified-spectra-per-family-operator",
            "representation": "final-stage-median-over-four-frozen-learning-rates",
            "bridge": (
                "seven-prespecified-geometry-and-two-posthoc-training-loss-within-run-"
                "first-difference-spearman-associations"
            ),
        },
        "interpretation": (
            "Common-state transforms identify optimizer fingerprints and the bridge remains "
            "descriptive one-seed evidence; causal retrieval claims require short interventions."
        ),
    }
    if scope is not None:
        source_manifests["basis_sensitivity"]["summary_rows"] = 3 * family_count
        source_manifests["mechanism_bridge"].update(
            {
                "within_run_transitions": bridge_coverage["within_run_transitions"],
                "correlations": bridge_coverage["correlations"],
            }
        )
        manifest["families"] = list(families)
        manifest["scope_amendment"] = scope
        manifest["aggregation"] = {
            "retrieval_dynamics": (
                "three-optimizer-groups-over-four-learning-rate-points-in-active-dense-scope"
            ),
            "common_state": "median-over-ten-frozen-anchors-per-operator-in-active-dense-scope",
            "basis_sensitivity": (
                "median-over-ninety-fixed-comparisons-per-operator-in-active-dense-scope"
            ),
            "exact_spectra": (
                "median-over-sixty-prespecified-spectra-per-operator-in-active-dense-scope"
            ),
            "representation": "final-stage-median-over-four-frozen-learning-rates",
            "bridge": (
                "three-prespecified-geometry-and-one-posthoc-training-loss-within-run-"
                "first-difference-spearman-associations-in-active-dense-scope"
            ),
        }
    _atomic_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the strict update-to-retrieval mechanism section into the final blog"
    )
    parser.add_argument("--common-state-dir", type=Path, default=Path("reports/common-state"))
    parser.add_argument("--basis-dir", type=Path, default=Path("reports/basis-sensitivity"))
    parser.add_argument(
        "--spectrum-dir",
        type=Path,
        default=Path("results/common-state-spectra/summary"),
    )
    parser.add_argument("--bridge-dir", type=Path, default=Path("reports/mechanism-bridge"))
    parser.add_argument("--retrieval-dir", type=Path, default=Path("reports/retrieval-dynamics"))
    parser.add_argument("--blog", type=Path, default=Path("docs/blog.md"))
    parser.add_argument("--output", type=Path, default=Path("reports/mechanism-summary.md"))
    parser.add_argument(
        "--spectrum-figure",
        type=Path,
        default=Path("reports/common-state/exact-update-spectra.svg"),
    )
    parser.add_argument(
        "--representation-figure",
        type=Path,
        default=Path("reports/representation-space/representation-dynamics.svg"),
    )
    parser.add_argument(
        "--late-token-figure",
        type=Path,
        default=Path("reports/representation-space/late-token-dynamics.svg"),
    )
    parser.add_argument(
        "--families", nargs="+", choices=("dense", "late"), default=["dense", "late"]
    )
    parser.add_argument("--scope-amendment", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    families = normalize_families(args.families)
    resolve_scope(families, args.scope_amendment)
    ensure_retrieval_dynamics(args.retrieval_dir)
    manifest = render_mechanism_report(
        args.common_state_dir,
        args.spectrum_dir,
        args.bridge_dir,
        args.retrieval_dir,
        args.blog,
        args.output,
        basis_dir=args.basis_dir,
        spectrum_figure=args.spectrum_figure,
        representation_figure=args.representation_figure,
        late_token_figure=args.late_token_figure,
        families=families,
        scope_amendment=args.scope_amendment,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
