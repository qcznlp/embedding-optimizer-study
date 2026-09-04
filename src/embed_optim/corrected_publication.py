"""Render the complete corrected Dense evidence into report and paper artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .geometry import _atomic_json, _sha256

SCHEMA_VERSION = 1
OPTIMIZERS = ("adamw", "muon", "normuon")
CONTRASTS = (("muon", "adamw"), ("normuon", "adamw"), ("normuon", "muon"))
LABELS = {"adamw": "AdamW", "muon": "Muon", "normuon": "NorMuon"}
FEATURE_LABELS = {
    "log_saved_segment_to_weight_ratio": "log segment/weight norm",
    "saved_segment_stable_rank_fraction": "segment stable-rank fraction",
    "saved_segment_sketch_effective_rank_fraction": "segment effective-rank fraction",
    "saved_segment_row_norm_cv": "segment row-norm CV",
    "saved_segment_top_1pct_row_energy": "segment top-1% row energy",
    "cumulative_displacement_to_weight_ratio": "cumulative/weight norm",
    "cumulative_stable_rank_fraction": "cumulative stable-rank fraction",
    "mean_saved_segment_subspace_overlap_to_adamw": "segment subspace overlap to AdamW",
    "mean_cumulative_subspace_overlap_to_adamw": "cumulative subspace overlap to AdamW",
}
SYSTEM_FIELDS = (
    "wall_time_hours",
    "samples_per_second",
    "steps_per_second",
    "peak_allocated_gib",
    "peak_reserved_gib",
    "checkpoint_gib",
    "optimizer_state_gib",
)


def _finite(value: Any, *, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value for {context}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value for {context}: {result}")
    return result


def _optional_finite(value: Any, *, context: str) -> float | None:
    if value in (None, ""):
        return None
    return _finite(value, context=context)


def _boolean(value: Any, *, context: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid Boolean value for {context}: {value!r}")


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_summary_manifest(
    directory: Path,
    *,
    expected_protocol_sha: str,
    context: str,
) -> tuple[Path, dict[str, Any]]:
    directory = directory.resolve()
    path = directory / "summary_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "complete"
        or payload.get("protocol", {}).get("sha256") != expected_protocol_sha
    ):
        raise ValueError(f"Incomplete or unexpected {context} summary manifest: {path}")
    return path, payload


def _read_bound_csv(
    directory: Path,
    manifest: dict[str, Any],
    *,
    key: str,
    filename: str,
    expected_rows: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    path = directory.resolve() / filename
    identity = manifest.get("outputs", {}).get(key)
    if (
        not isinstance(identity, dict)
        or int(identity.get("rows", -1)) != expected_rows
        or not path.is_file()
        or path.stat().st_size != int(identity.get("bytes", -1))
        or _sha256(path) != identity.get("sha256")
    ):
        raise ValueError(f"Publication input provenance mismatch: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in {path}, found {len(rows)}")
    return rows, {**_file_record(path), "rows": len(rows), "manifest_key": key}


def _read_bound_json(
    directory: Path,
    manifest: dict[str, Any],
    *,
    key: str,
    filename: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = directory.resolve() / filename
    identity = manifest.get("outputs", {}).get(key)
    if (
        not isinstance(identity, dict)
        or not path.is_file()
        or path.stat().st_size != int(identity.get("bytes", -1))
        or _sha256(path) != identity.get("sha256")
    ):
        raise ValueError(f"Publication JSON input provenance mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8")), {
        **_file_record(path),
        "manifest_key": key,
    }


def _load_publication_protocol(path: Path, repository: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "corrected_publication_implementation_lock":
        raise ValueError(f"Unexpected corrected publication protocol status: {path}")
    for group in ("parent_bindings", "source_bindings"):
        for identity in payload.get(group, {}).values():
            source = repository / identity["path"]
            if (
                not source.is_file()
                or _sha256(source) != identity["sha256"]
                or ("bytes" in identity and source.stat().st_size != int(identity["bytes"]))
            ):
                raise ValueError(f"Corrected publication {group} mismatch: {source}")
    return payload


def _validate_contrasts(rows: list[dict[str, Any]], *, secondary: bool) -> list[dict[str, Any]]:
    indexed = {}
    critical_values = set()
    for row in rows:
        key = (str(row.get("treatment")), str(row.get("baseline")))
        mean = _finite(row.get("mean_delta_ndcg_at_10"), context=f"{key} mean")
        lower = _finite(row.get("simultaneous_ci_95_lower"), context=f"{key} lower CI")
        upper = _finite(row.get("simultaneous_ci_95_upper"), context=f"{key} upper CI")
        critical = _finite(row.get("simultaneous_critical_value"), context=f"{key} critical")
        support = str(row.get("support"))
        expected_support = "positive" if lower > 0 else "negative" if upper < 0 else "inconclusive"
        if (
            key in indexed
            or key not in CONTRASTS
            or lower > mean
            or mean > upper
            or support != expected_support
            or int(row.get("bootstrap_samples", -1)) != 50_000
            or int(row.get("bootstrap_seed", -1)) != 20_260_903
            or int(row.get("tasks", -1)) != 14
            or (secondary and (not row.get("treatment_run_id") or not row.get("baseline_run_id")))
        ):
            raise ValueError(f"Invalid corrected publication contrast: {key}")
        critical_values.add(critical)
        indexed[key] = {
            **row,
            "mean": mean,
            "lower": lower,
            "upper": upper,
            "support": support,
        }
    if set(indexed) != set(CONTRASTS) or len(critical_values) != 1:
        raise ValueError("Corrected publication contrasts are incomplete or use different max-T")
    return [indexed[key] for key in CONTRASTS]


def _validate_optimizer_stage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = {(optimizer, stage) for optimizer in OPTIMIZERS for stage in range(1, 6)}
    indexed = {}
    for row in rows:
        key = (str(row.get("optimizer")), int(row.get("stage", -1)))
        mean = _finite(row.get("mean_ndcg_at_10_across_rates"), context=f"{key} mean")
        median = _finite(row.get("median_ndcg_at_10_across_rates"), context=f"{key} median")
        if (
            key in indexed
            or key not in expected
            or int(row.get("learning_rates", -1)) != 4
            or not 0 <= mean <= 1
            or not 0 <= median <= 1
        ):
            raise ValueError(f"Invalid optimizer-stage publication row: {key}")
        indexed[key] = {**row, "mean": mean, "median": median}
    if set(indexed) != expected:
        raise ValueError("Optimizer-stage publication rows are incomplete")
    return [indexed[(optimizer, stage)] for stage in range(1, 6) for optimizer in OPTIMIZERS]


def _selected_validation_rows(
    rows: list[dict[str, Any]], selection: dict[str, Any]
) -> list[dict[str, Any]]:
    selected = selection.get("selected_run_ids")
    if (
        selection.get("schema_version") != SCHEMA_VERSION
        or selection.get("status") != "complete"
        or not isinstance(selected, dict)
        or set(selected) != set(OPTIMIZERS)
    ):
        raise ValueError("Corrected validation selection receipt is invalid")
    indexed = {}
    for row in rows:
        run_id = str(row.get("run_id"))
        optimizer = str(row.get("optimizer"))
        key = (optimizer, run_id)
        if (
            key in indexed
            or optimizer not in OPTIMIZERS
            or not run_id
            or _finite(row.get("learning_rate"), context=f"{key} rate") <= 0
        ):
            raise ValueError(f"Invalid validation publication row: {key}")
        indexed[key] = row
    if len(rows) != 12:
        raise ValueError("Corrected validation publication requires 12 rows")
    output = []
    for optimizer in OPTIMIZERS:
        row = indexed.get((optimizer, str(selected[optimizer])))
        if row is None:
            raise ValueError(f"Selected validation run is absent for {optimizer}")
        output.append(
            {
                **row,
                "learning_rate_value": _finite(
                    row["learning_rate"], context=f"{optimizer} selected rate"
                ),
                "loss": _finite(row["contrastive_loss"], context=f"{optimizer} validation loss"),
                "margin": _finite(row["positive_margin"], context=f"{optimizer} validation margin"),
            }
        )
    return output


def _system_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for row in rows:
        optimizer = str(row.get("optimizer"))
        run_id = str(row.get("run_id"))
        if optimizer not in OPTIMIZERS or not run_id or run_id in seen:
            raise ValueError(f"Invalid corrected system row: {run_id}")
        seen.add(run_id)
        if int(row.get("world_size", -1)) != 4:
            raise ValueError(f"Corrected system row has unexpected world size: {run_id}")
        grouped[optimizer].append(row)
    if len(rows) != 12 or any(len(grouped[optimizer]) != 4 for optimizer in OPTIMIZERS):
        raise ValueError("Corrected system publication requires four runs per optimizer")
    output = []
    for optimizer in OPTIMIZERS:
        result: dict[str, Any] = {"optimizer": optimizer, "runs": 4}
        for field in SYSTEM_FIELDS:
            values = [
                _finite(row.get(field), context=f"{optimizer} {field}")
                for row in grouped[optimizer]
            ]
            if any(value < 0 for value in values):
                raise ValueError(f"Corrected system metric is negative: {optimizer} {field}")
            result[f"mean_{field}"] = statistics.fmean(values)
        output.append(result)
    return output


def _final_geometry(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "saved_segment_to_weight_ratio",
        "saved_segment_stable_rank_fraction_parameter_weighted",
        "saved_segment_sketch_effective_rank_fraction_parameter_weighted",
        "saved_segment_row_cv_parameter_weighted",
        "saved_segment_top_1pct_row_energy_parameter_weighted",
        "cumulative_displacement_to_weight_ratio",
        "cumulative_stable_rank_fraction_parameter_weighted",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities = set()
    for row in rows:
        optimizer = str(row.get("optimizer"))
        stage = int(row.get("stage", -1))
        run_id = str(row.get("run_id"))
        identity = (run_id, stage)
        if optimizer not in OPTIMIZERS or stage not in range(1, 6) or identity in identities:
            raise ValueError(f"Invalid geometry publication row: {identity}")
        identities.add(identity)
        if stage == 5:
            grouped[optimizer].append(row)
    if len(rows) != 60 or any(len(grouped[optimizer]) != 4 for optimizer in OPTIMIZERS):
        raise ValueError("Corrected geometry publication requires 60 rows and four final rates")
    output = []
    for optimizer in OPTIMIZERS:
        result: dict[str, Any] = {"optimizer": optimizer, "learning_rates": 4}
        for metric in metrics:
            values = [
                _finite(row.get(metric), context=f"{optimizer} {metric}")
                for row in grouped[optimizer]
            ]
            result[metric] = statistics.fmean(values)
        output.append(result)
    return output


def _bridge_table(
    summary_rows: list[dict[str, Any]], association_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    summaries = {str(row.get("feature")): row for row in summary_rows}
    associations = {str(row.get("feature")): row for row in association_rows}
    if set(summaries) != set(FEATURE_LABELS) or set(associations) != set(FEATURE_LABELS):
        raise ValueError("Corrected bridge publication feature set differs from the frozen nine")
    output = []
    for feature in FEATURE_LABELS:
        summary = summaries[feature]
        association = associations[feature]
        if (
            int(summary.get("pooled_rows", -1)) != 60
            or int(summary.get("folds_total", -1)) != 4
            or int(summary.get("folds_improved", -1)) not in range(0, 5)
            or int(association.get("rows", -1)) != 60
        ):
            raise ValueError(f"Invalid corrected bridge publication row: {feature}")
        baseline_rmse = _finite(summary.get("pooled_baseline_rmse"), context=f"{feature} baseline")
        feature_rmse = _finite(summary.get("pooled_feature_rmse"), context=f"{feature} RMSE")
        reduction = _finite(summary.get("pooled_rmse_reduction"), context=f"{feature} reduction")
        supported = _boolean(summary.get("predictively_useful"), context=f"{feature} support")
        folds_improved = int(summary["folds_improved"])
        expected_support = reduction > 0 and folds_improved >= 3
        if (
            baseline_rmse < 0
            or feature_rmse < 0
            or not math.isclose(baseline_rmse - feature_rmse, reduction, abs_tol=1e-12)
            or supported != expected_support
        ):
            raise ValueError(f"Corrected bridge support rule mismatch: {feature}")
        output.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "baseline_rmse": baseline_rmse,
                "feature_rmse": feature_rmse,
                "reduction": reduction,
                "folds_improved": folds_improved,
                "supported": supported,
                "pearson": _optional_finite(
                    association.get("pearson_residual_association"),
                    context=f"{feature} Pearson",
                ),
                "spearman": _optional_finite(
                    association.get("spearman_residual_association"),
                    context=f"{feature} Spearman",
                ),
            }
        )
    return output


def _sensitivity_tables(
    contrast_rows: list[dict[str, Any]], ranking_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contrast_index = {}
    for row in contrast_rows:
        key = (str(row.get("optimizer")), int(row.get("stage", -1)))
        historical = _finite(
            row.get("historical_optimizer_minus_adamw"), context=f"{key} historical"
        )
        corrected = _finite(row.get("corrected_optimizer_minus_adamw"), context=f"{key} corrected")
        shift = _finite(
            row.get("corrected_minus_historical_contrast_shift"), context=f"{key} shift"
        )
        if (
            key in contrast_index
            or key[0] not in ("muon", "normuon")
            or key[1] not in range(1, 6)
            or not math.isclose(corrected - historical, shift, abs_tol=1e-12)
        ):
            raise ValueError(f"Invalid corrected execution-sensitivity contrast: {key}")
        contrast_index[key] = {
            **row,
            "historical": historical,
            "corrected": corrected,
            "shift": shift,
        }
    expected_contrasts = {
        (optimizer, stage) for optimizer in ("muon", "normuon") for stage in range(1, 6)
    }
    if set(contrast_index) != expected_contrasts:
        raise ValueError("Corrected execution-sensitivity contrasts are incomplete")
    ranking_index = {}
    for row in ranking_rows:
        stage = int(row.get("stage", -1))
        historical_order = str(row.get("historical_optimizer_order"))
        corrected_order = str(row.get("corrected_optimizer_order"))
        changed = _boolean(row.get("ranking_changed"), context=f"stage {stage} ranking")
        if (
            stage in ranking_index
            or stage not in range(1, 6)
            or set(historical_order.split(">")) != set(OPTIMIZERS)
            or set(corrected_order.split(">")) != set(OPTIMIZERS)
        ):
            raise ValueError(f"Invalid corrected execution-sensitivity ranking: {stage}")
        ranking_index[stage] = {
            **row,
            "stage": stage,
            "historical_order": historical_order,
            "corrected_order": corrected_order,
            "changed": changed,
        }
    if set(ranking_index) != set(range(1, 6)):
        raise ValueError("Corrected execution-sensitivity rankings are incomplete")
    return (
        [
            contrast_index[(optimizer, stage)]
            for stage in range(1, 6)
            for optimizer in ("muon", "normuon")
        ],
        [ranking_index[stage] for stage in range(1, 6)],
    )


def load_publication_evidence(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    parents = protocol["parent_bindings"]
    sources: dict[str, Any] = {}

    outcome_path, outcome_manifest = _load_summary_manifest(
        args.outcomes_dir,
        expected_protocol_sha=parents["outcome_protocol"]["sha256"],
        context="outcome",
    )
    if outcome_manifest.get("coverage", {}).get("task_units") != 840:
        raise ValueError("Corrected publication outcome coverage is not 840 task units")
    sources["outcome_manifest"] = _file_record(outcome_path)
    primary_rows, sources["primary_summary"] = _read_bound_csv(
        args.outcomes_dir,
        outcome_manifest,
        key="primary_summary",
        filename="primary_summary.csv",
        expected_rows=3,
    )
    secondary_rows, sources["secondary_summary"] = _read_bound_csv(
        args.outcomes_dir,
        outcome_manifest,
        key="secondary_summary",
        filename="secondary_summary.csv",
        expected_rows=3,
    )
    optimizer_stage_rows, sources["optimizer_stage_scores"] = _read_bound_csv(
        args.outcomes_dir,
        outcome_manifest,
        key="optimizer_stage_scores",
        filename="optimizer_stage_scores.csv",
        expected_rows=15,
    )
    validation_rows, sources["validation_run_metrics"] = _read_bound_csv(
        args.outcomes_dir,
        outcome_manifest,
        key="validation_run_metrics",
        filename="validation_run_metrics.csv",
        expected_rows=12,
    )
    system_rows, sources["system_metrics"] = _read_bound_csv(
        args.outcomes_dir,
        outcome_manifest,
        key="system_metrics",
        filename="system_metrics.csv",
        expected_rows=12,
    )
    selection, sources["validation_recipe_selection"] = _read_bound_json(
        args.outcomes_dir,
        outcome_manifest,
        key="validation_recipe_selection",
        filename="validation_recipe_selection.json",
    )

    geometry_path, geometry_manifest = _load_summary_manifest(
        args.geometry_dir,
        expected_protocol_sha=parents["analysis_protocol"]["sha256"],
        context="geometry",
    )
    if (
        geometry_manifest.get("checkpoint_rows") != 60
        or geometry_manifest.get("run_pair_subspace_rows") != 660
    ):
        raise ValueError("Corrected publication geometry coverage is incomplete")
    sources["geometry_manifest"] = _file_record(geometry_path)
    geometry_rows, sources["checkpoint_geometry"] = _read_bound_csv(
        args.geometry_dir,
        geometry_manifest,
        key="checkpoint_geometry.csv",
        filename="checkpoint_geometry.csv",
        expected_rows=60,
    )

    bridge_path, bridge_manifest = _load_summary_manifest(
        args.bridge_dir,
        expected_protocol_sha=parents["bridge_protocol"]["sha256"],
        context="bridge",
    )
    if bridge_manifest.get("coverage") != {
        "runs": 12,
        "stages": 5,
        "bridge_rows": 60,
        "features": 9,
        "leave_dose_fold_rows": 36,
    }:
        raise ValueError("Corrected publication bridge coverage is incomplete")
    sources["bridge_manifest"] = _file_record(bridge_path)
    bridge_summary_rows, sources["feature_prediction_summary"] = _read_bound_csv(
        args.bridge_dir,
        bridge_manifest,
        key="feature_prediction_summary.csv",
        filename="feature_prediction_summary.csv",
        expected_rows=9,
    )
    association_rows, sources["residual_associations"] = _read_bound_csv(
        args.bridge_dir,
        bridge_manifest,
        key="residual_associations.csv",
        filename="residual_associations.csv",
        expected_rows=9,
    )

    sensitivity_path, sensitivity_manifest = _load_summary_manifest(
        args.sensitivity_dir,
        expected_protocol_sha=parents["sensitivity_protocol"]["sha256"],
        context="sensitivity",
    )
    if (
        sensitivity_manifest.get("coverage", {}).get("matched_rows") != 60
        or sensitivity_manifest.get("no_pooling") is not True
    ):
        raise ValueError("Corrected publication execution-sensitivity coverage is incomplete")
    sources["sensitivity_manifest"] = _file_record(sensitivity_path)
    sensitivity_rows, sources["optimizer_minus_adamw_sensitivity"] = _read_bound_csv(
        args.sensitivity_dir,
        sensitivity_manifest,
        key="optimizer_minus_adamw_sensitivity.csv",
        filename="optimizer_minus_adamw_sensitivity.csv",
        expected_rows=10,
    )
    ranking_rows, sources["stage_optimizer_rankings"] = _read_bound_csv(
        args.sensitivity_dir,
        sensitivity_manifest,
        key="stage_optimizer_rankings.csv",
        filename="stage_optimizer_rankings.csv",
        expected_rows=5,
    )

    sensitivity, rankings = _sensitivity_tables(sensitivity_rows, ranking_rows)
    return {
        "primary": _validate_contrasts(primary_rows, secondary=False),
        "secondary": _validate_contrasts(secondary_rows, secondary=True),
        "optimizer_stage": _validate_optimizer_stage(optimizer_stage_rows),
        "validation_selected": _selected_validation_rows(validation_rows, selection),
        "systems": _system_summary(system_rows),
        "final_geometry": _final_geometry(geometry_rows),
        "bridge": _bridge_table(bridge_summary_rows, association_rows),
        "sensitivity": sensitivity,
        "rankings": rankings,
        "sources": sources,
    }


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" if index else "---" for index in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _ci(row: dict[str, Any]) -> str:
    return f"{row['mean']:+.4f} [{row['lower']:+.4f}, {row['upper']:+.4f}]"


def _correlation_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def build_conclusion(evidence: dict[str, Any]) -> str:
    primary = {(row["treatment"], row["baseline"]): row for row in evidence["primary"]}
    secondary = {(row["treatment"], row["baseline"]): row for row in evidence["secondary"]}
    supported = [row["label"] for row in evidence["bridge"] if row["supported"]]
    supported_text = ", ".join(supported) if supported else "none of the nine frozen features"
    ranking = evidence["rankings"][-1]
    return (
        "Across all four predeclared rates, Muon versus AdamW is "
        f"{primary[('muon', 'adamw')]['support']} at {_ci(primary[('muon', 'adamw')])} nDCG@10, "
        "and NorMuon versus AdamW is "
        f"{primary[('normuon', 'adamw')]['support']} at {_ci(primary[('normuon', 'adamw')])}. "
        "The independently padded validation-selected comparisons are "
        f"{secondary[('muon', 'adamw')]['support']} for Muon and "
        f"{secondary[('normuon', 'adamw')]['support']} for NorMuon. "
        f"The final-stage optimizer ordering is {ranking['corrected_order']} versus the historical "
        f"{ranking['historical_order']}. Under the frozen held-out bridge rule, predictively useful "
        f"geometry features are: {supported_text}. These are one-seed, pinned-grid results; the "
        "execution comparison is sensitivity evidence, and the geometry bridge is not causal mediation."
    )


def render_markdown(evidence: dict[str, Any]) -> str:
    primary_table = _markdown_table(
        ["Contrast", "Mean and simultaneous 95% CI", "Decision"],
        [
            [
                f"{LABELS[row['treatment']]} − {LABELS[row['baseline']]}",
                _ci(row),
                str(row["support"]),
            ]
            for row in evidence["primary"]
        ],
    )
    secondary_table = _markdown_table(
        ["Contrast", "Selected runs", "Mean and simultaneous 95% CI", "Decision"],
        [
            [
                f"{LABELS[row['treatment']]} − {LABELS[row['baseline']]}",
                f"`{row['treatment_run_id']}` vs `{row['baseline_run_id']}`",
                _ci(row),
                str(row["support"]),
            ]
            for row in evidence["secondary"]
        ],
    )
    selected_table = _markdown_table(
        ["Optimizer", "Selected LR", "Validation loss", "Positive margin", "Run"],
        [
            [
                LABELS[row["optimizer"]],
                f"{row['learning_rate_value']:.0e}",
                f"{row['loss']:.6f}",
                f"{row['margin']:.6f}",
                f"`{row['run_id']}`",
            ]
            for row in evidence["validation_selected"]
        ],
    )
    dynamics_table = _markdown_table(
        ["Stage", "Progress", "Optimizer", "Four-rate mean", "Four-rate median"],
        [
            [
                str(row["stage"]),
                f"{float(row['progress_fraction']):.0%}",
                LABELS[row["optimizer"]],
                f"{row['mean']:.4f}",
                f"{row['median']:.4f}",
            ]
            for row in evidence["optimizer_stage"]
        ],
    )
    systems_table = _markdown_table(
        [
            "Optimizer",
            "Hours",
            "Samples/s",
            "Peak alloc. GiB",
            "Checkpoint GiB",
            "Optimizer state GiB",
        ],
        [
            [
                LABELS[row["optimizer"]],
                f"{row['mean_wall_time_hours']:.3f}",
                f"{row['mean_samples_per_second']:.2f}",
                f"{row['mean_peak_allocated_gib']:.2f}",
                f"{row['mean_checkpoint_gib']:.3f}",
                f"{row['mean_optimizer_state_gib']:.3f}",
            ]
            for row in evidence["systems"]
        ],
    )
    geometry_table = _markdown_table(
        [
            "Optimizer",
            "Segment/weight",
            "Stable-rank frac.",
            "Effective-rank frac.",
            "Row CV",
            "Top-1% row energy",
            "Cumulative/weight",
        ],
        [
            [
                LABELS[row["optimizer"]],
                f"{row['saved_segment_to_weight_ratio']:.6f}",
                f"{row['saved_segment_stable_rank_fraction_parameter_weighted']:.4f}",
                f"{row['saved_segment_sketch_effective_rank_fraction_parameter_weighted']:.4f}",
                f"{row['saved_segment_row_cv_parameter_weighted']:.4f}",
                f"{row['saved_segment_top_1pct_row_energy_parameter_weighted']:.4f}",
                f"{row['cumulative_displacement_to_weight_ratio']:.6f}",
            ]
            for row in evidence["final_geometry"]
        ],
    )
    bridge_table = _markdown_table(
        [
            "Frozen feature",
            "Baseline RMSE",
            "+ feature RMSE",
            "Reduction",
            "Improved folds",
            "Supported",
            "Residual Pearson",
            "Residual Spearman",
        ],
        [
            [
                row["label"],
                f"{row['baseline_rmse']:.5f}",
                f"{row['feature_rmse']:.5f}",
                f"{row['reduction']:+.5f}",
                f"{row['folds_improved']}/4",
                "yes" if row["supported"] else "no",
                _correlation_text(row["pearson"]),
                _correlation_text(row["spearman"]),
            ]
            for row in evidence["bridge"]
        ],
    )
    sensitivity_table = _markdown_table(
        ["Stage", "Optimizer − AdamW", "Historical", "Corrected", "Contrast shift"],
        [
            [
                str(row["stage"]),
                LABELS[row["optimizer"]],
                f"{row['historical']:+.4f}",
                f"{row['corrected']:+.4f}",
                f"{row['shift']:+.4f}",
            ]
            for row in evidence["sensitivity"]
        ],
    )
    ranking_table = _markdown_table(
        ["Stage", "Historical order", "Corrected order", "Changed"],
        [
            [
                str(row["stage"]),
                row["historical_order"],
                row["corrected_order"],
                "yes" if row["changed"] else "no",
            ]
            for row in evidence["rankings"]
        ],
    )
    return "\n\n".join(
        (
            "This corrective matrix uses independent padding for training and every checkpoint reload. "
            "The primary result averages all four predeclared rates within optimizer; the "
            "validation-selected recipe comparison is secondary.",
            "### Primary four-rate retrieval inference\n\n" + primary_table,
            "### Independently padded validation selection\n\n"
            + selected_table
            + "\n\n"
            + secondary_table,
            "### Five-stage retrieval dynamics\n\n" + dynamics_table,
            "### Systems results\n\n" + systems_table,
            "### Final-stage hidden-matrix geometry\n\n" + geometry_table,
            "### Does geometry predict retrieval beyond optimizer, stage, and dose?\n\n"
            + bridge_table
            + "\n\nA feature is supported only if pooled held-out RMSE decreases and at least "
            "three of four leave-dose-index-out folds improve. Residual correlations are descriptive.",
            "### Historical versus corrected execution sensitivity\n\n"
            + sensitivity_table
            + "\n\n"
            + ranking_table
            + "\n\nThe two executions are not pooled. These shifts diagnose execution-path "
            "sensitivity and are not a randomized causal estimate of packing.",
            "### Corrected conclusion\n\n" + build_conclusion(evidence),
        )
    )


def _tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def render_latex(evidence: dict[str, Any]) -> str:
    primary_rows = "\n".join(
        f"{_tex_escape(LABELS[row['treatment']] + ' - ' + LABELS[row['baseline']])} & "
        f"{row['mean']:+.4f} & [{row['lower']:+.4f}, {row['upper']:+.4f}] & "
        f"{_tex_escape(str(row['support']))} \\\\"
        for row in evidence["primary"]
    )
    bridge_rows = "\n".join(
        f"{_tex_escape(row['label'])} & {row['reduction']:+.5f} & {row['folds_improved']}/4 & "
        f"{'yes' if row['supported'] else 'no'} & {_correlation_text(row['spearman'])} \\\\"
        for row in evidence["bridge"]
    )
    sensitivity_final = [row for row in evidence["sensitivity"] if int(row["stage"]) == 5]
    sensitivity_rows = "\n".join(
        f"{_tex_escape(LABELS[row['optimizer']] + ' - AdamW')} & {row['historical']:+.4f} & "
        f"{row['corrected']:+.4f} & {row['shift']:+.4f} \\\\"
        for row in sensitivity_final
    )
    conclusion = _tex_escape(build_conclusion(evidence))
    return (
        "% Generated by embed-optim-render-dense-no-packing; do not edit manually.\n"
        "\\newcommand{\\CorrectedGeometryBridgeTable}{%\n"
        "\\begin{table*}[t]\n\\centering\n\\small\n"
        "\\begin{tabular}{lrrrl}\n\\toprule\nFrozen geometry feature & $\\Delta$RMSE & Folds & Supported & Residual $\\rho$ \\\\n\\midrule\n"
        + bridge_rows
        + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Four-fold leave-dose-index-out geometry-to-retrieval bridge. Positive "
        "$\\Delta$RMSE means the feature lowers pooled held-out error.}\n"
        "\\label{tab:corrected-bridge}\n\\end{table*}%\n}\n"
        "\\newcommand{\\CorrectedExecutionSensitivityTable}{%\n"
        "\\begin{table}[t]\n\\centering\n\\small\n"
        "\\begin{tabular}{lrrr}\n\\toprule\nContrast & Historical & Corrected & Shift \\\\n\\midrule\n"
        + sensitivity_rows
        + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Final-stage execution-path sensitivity. The executions are not pooled and "
        "the shift is not a causal packing estimate.}\n"
        "\\label{tab:corrected-sensitivity}\n\\end{table}%\n}\n"
        "\\section{Corrected Independently Padded Replication}\n"
        "\\label{sec:corrected-no-packing}\n"
        "The corrective matrix disables flattened input execution during training and every "
        "checkpoint reload. The primary estimand averages all four frozen rates within optimizer.\n\n"
        "\\begin{table}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{lrrl}\n\\toprule\nContrast & Mean & 95\\% CI & Decision \\\\n\\midrule\n"
        + primary_rows
        + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Corrected final-stage retrieval inference over 14 decontaminated BEIR tasks "
        "with simultaneous intervals.}\n"
        "\\label{tab:corrected-primary}\n\\end{table}\n\n"
        "\\paragraph{Corrected conclusion.} " + conclusion + "\n"
    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    protocol_path = args.protocol.resolve()
    protocol = _load_publication_protocol(protocol_path, repository)
    evidence = load_publication_evidence(args, protocol)
    block = render_markdown(evidence)
    standalone = "# Corrected independently padded DenseOn replication\n\n" + block + "\n"
    latex = render_latex(evidence)
    output_dir = args.output_dir.resolve()
    markdown_path = output_dir / "corrected_dense_results.md"
    _atomic_text(markdown_path, standalone)
    latex_path = args.latex_output.resolve()
    _atomic_text(latex_path, latex)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": "corrected_dense_no_packing_publication",
        "coverage": {
            "training_runs": 12,
            "checkpoints": 60,
            "beir_task_units": 840,
            "primary_contrasts": 3,
            "secondary_contrasts": 3,
            "optimizer_stage_rows": 15,
            "geometry_features": 9,
            "sensitivity_contrasts": 10,
        },
        "claim_boundary": protocol["claim_boundary"],
        "protocol": {**_file_record(protocol_path)},
        "sources": evidence["sources"],
        "outputs": {
            "standalone_markdown": _file_record(markdown_path),
            "paper_latex": _file_record(latex_path),
        },
        "conclusion": build_conclusion(evidence),
    }
    _atomic_json(output_dir / "summary_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/dense_no_packing_publication_protocol.json"),
    )
    parser.add_argument(
        "--outcomes-dir", type=Path, default=Path("reports/dense-no-packing-outcomes")
    )
    parser.add_argument(
        "--geometry-dir", type=Path, default=Path("reports/dense-no-packing-weight-space")
    )
    parser.add_argument(
        "--bridge-dir", type=Path, default=Path("reports/dense-no-packing-retrieval-bridge")
    )
    parser.add_argument(
        "--sensitivity-dir", type=Path, default=Path("reports/dense-no-packing-sensitivity")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/dense-no-packing-publication")
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=Path("paper/generated/corrected-no-packing.tex"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(build_report(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
