from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .probe_matrix import _requested_probe_identity, probe_job_complete
from .probes import resolve_probe_spec_path
from .scope import ALL_FAMILIES, normalize_families, resolve_scope
from .short_branch_evaluation import (
    _audit_counts,
    _load_branch_configs,
    build_short_branch_probe_jobs,
    build_short_branch_validation_jobs,
)
from .short_branch_summary import expected_short_branch_probe_metrics
from .validation_data import load_validation_spec
from .validation_matrix import validation_job_complete

FAMILIES = ("dense", "late")
ALGORITHMS = ("adamw", "muon", "normuon")
CHALLENGERS = ("muon", "normuon")
SCALED_CONDITIONS = {
    "adamw": "adamw-descent-1e-03",
    "muon": "muon-descent-1e-03",
    "normuon": "normuon-descent-1e-03",
}

DISCOVERY_ANCHOR_FIELDS = [
    "family",
    "anchor",
    "algorithm",
    "samples",
    "relative_scale",
    "mean_delta_contrastive_loss",
    "p95_delta_contrastive_loss",
    "p99_delta_contrastive_loss",
    "loss_improvement_fraction",
    "mean_delta_positive_margin",
    "p05_delta_positive_margin",
    "margin_improvement_fraction",
]

DISCOVERY_CONTRAST_FIELDS = [
    "family",
    "challenger",
    "reference",
    "anchors",
    "relative_scale",
    "mean_margin_contrast_mean",
    "mean_margin_contrast_median",
    "mean_margin_challenger_wins",
    "mean_margin_ties",
    "mean_margin_challenger_losses",
    "mean_margin_leave_one_out_negative_fraction",
    "p05_margin_contrast_mean",
    "p05_margin_contrast_median",
    "p05_margin_challenger_wins",
    "p05_margin_ties",
    "p05_margin_challenger_losses",
    "p05_margin_leave_one_out_positive_or_zero_fraction",
    "p95_loss_contrast_mean",
    "p95_loss_contrast_median",
    "p95_loss_challenger_wins",
    "p95_loss_ties",
    "p95_loss_challenger_losses",
    "p95_loss_leave_one_out_negative_fraction",
    "p99_loss_contrast_mean",
    "p99_loss_contrast_median",
    "p99_loss_challenger_wins",
    "p99_loss_ties",
    "p99_loss_challenger_losses",
    "p99_loss_leave_one_out_negative_fraction",
    "mean_tail_tradeoff_observed",
]

DISCOVERY_CROSS_TAIL_FIELDS = [
    "family",
    "anchor",
    "challenger",
    "reference",
    "samples",
    "tail_fraction",
    "tail_size",
    "adam_loss_change_mean_on_adam_tail",
    "challenger_loss_change_mean_on_adam_tail",
    "challenger_minus_adam_on_adam_tail_loss_mean",
    "adam_tail_challenger_win_fraction",
    "adam_loss_change_mean_on_challenger_tail",
    "challenger_loss_change_mean_on_challenger_tail",
    "challenger_minus_adam_on_challenger_tail_loss_mean",
    "challenger_tail_challenger_win_fraction",
    "tail_intersection",
    "tail_union",
    "tail_jaccard",
    "adam_tail_baseline_margin_percentile_median",
]

DISCOVERY_CROSS_TAIL_SUMMARY_FIELDS = [
    "family",
    "challenger",
    "reference",
    "anchors",
    "tail_fraction",
    "tail_size",
    "adam_tail_contrast_mean",
    "adam_tail_contrast_median",
    "adam_tail_anchor_wins",
    "adam_tail_leave_one_out_negative_fraction",
    "challenger_tail_contrast_mean",
    "challenger_tail_contrast_median",
    "challenger_tail_anchor_wins",
    "challenger_tail_leave_one_out_negative_fraction",
    "adam_tail_challenger_sample_win_fraction_mean",
    "challenger_tail_challenger_sample_win_fraction_mean",
    "tail_intersection_mean",
    "tail_jaccard_mean",
    "adam_tail_baseline_margin_percentile_median_mean",
    "dual_selected_tail_advantage",
    "tail_identity_regime",
]

SHORT_BRANCH_FIELDS = [
    "family",
    "seed",
    "operator",
    "run_id",
    "stage",
    "fraction",
    "step",
    "validation_samples",
    "validation_loss_mean",
    "validation_loss_p95",
    "validation_loss_p99",
    "validation_margin_mean",
    "validation_margin_p05",
    "unseen_samples",
    "unseen_margin_mean",
    "unseen_margin_p05",
    "unseen_pretrained_top1_agreement",
    "unseen_pretrained_score_drift_rms",
]

SHORT_BRANCH_CONTRAST_FIELDS = [
    "family",
    "seed",
    "stage",
    "fraction",
    "challenger",
    "reference",
    "delta_validation_loss_mean",
    "delta_validation_loss_p95",
    "delta_validation_loss_p99",
    "delta_validation_margin_mean",
    "delta_validation_margin_p05",
    "delta_unseen_margin_mean",
    "delta_unseen_margin_p05",
    "delta_unseen_pretrained_top1_agreement",
    "delta_unseen_pretrained_score_drift_rms",
]

SHORT_BRANCH_FINAL_FIELDS = [
    "family",
    "challenger",
    "reference",
    "stage",
    "seeds",
    "median_delta_validation_loss_mean",
    "median_delta_validation_loss_p95",
    "median_delta_validation_loss_p99",
    "validation_loss_p95_seed_wins",
    "median_delta_validation_margin_mean",
    "median_delta_validation_margin_p05",
    "median_delta_unseen_margin_mean",
    "median_delta_unseen_margin_p05",
    "unseen_margin_p05_seed_wins",
    "median_delta_unseen_pretrained_top1_agreement",
    "median_delta_unseen_pretrained_score_drift_rms",
    "tail_stability_decision",
]


def resolve_tail_stability_protocol(path: str | Path, prefix: Path | None = None) -> Path:
    path = Path(path)
    if path.is_file() or path.is_absolute() or path.parent != Path("configs"):
        return path
    prefix = Path(sys.prefix) if prefix is None else prefix
    installed = prefix / "share" / "embedding-optimizer-study" / "configs" / path.name
    return installed if installed.is_file() else path


def load_tail_stability_protocol(path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = resolve_tail_stability_protocol(path).resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "analysis_status",
        "frozen_at_utc",
        "objective",
        "source_inputs",
        "discovery_diagnostic",
        "secondary_discovery_diagnostic",
        "short_branch_confirmation",
        "freeze_context",
        "amendments",
        "claim_boundary",
    }
    if (
        set(protocol) != expected_fields
        or protocol.get("schema_version") != SCHEMA_VERSION
        or protocol.get("analysis_status")
        != "post_hoc_discovery_with_prospective_short_branch_confirmation"
        or not isinstance(protocol.get("frozen_at_utc"), str)
        or not isinstance(protocol.get("claim_boundary"), str)
    ):
        raise ValueError(f"Unsupported tail-stability protocol: {path}")
    root = path.parent.parent
    source = protocol["source_inputs"]
    pairs = (
        ("functional_intervention_spec", "functional_intervention_spec_sha256"),
        ("functional_intervention_summary", "functional_intervention_summary_sha256"),
        ("local_global_reversal_protocol", "local_global_reversal_protocol_sha256"),
        ("local_global_reversal_summary", "local_global_reversal_summary_sha256"),
        ("short_branch_protocol", "short_branch_protocol_sha256"),
        ("short_branch_matrix_manifest", "short_branch_matrix_manifest_sha256"),
        ("validation_spec", "validation_spec_sha256"),
        ("unseen_probe_spec", "unseen_probe_spec_sha256"),
    )
    for path_key, digest_key in pairs:
        declared = (root / source[path_key]).resolve()
        if not declared.is_file() or _sha256(declared) != source[digest_key]:
            raise ValueError(f"Frozen tail-stability source changed: {declared}")

    discovery = protocol["discovery_diagnostic"]
    if (
        discovery.get("anchors") != 20
        or discovery.get("anchors_per_family") != 10
        or discovery.get("examples_per_anchor") != 224
        or discovery.get("condition_direction") != "descent"
        or float(discovery.get("relative_scale", math.nan)) != 0.001
        or discovery.get("algorithms") != list(ALGORITHMS)
        or discovery.get("reference") != "adamw"
        or discovery.get("challengers") != list(CHALLENGERS)
        or discovery.get("quantile_definition")
        != "linear interpolation at index (n-1)q after stable ascending sort"
    ):
        raise ValueError("Discovery tail-stability contract differs from the frozen design")
    confirmation = protocol["short_branch_confirmation"]
    if (
        confirmation.get("expected_seeds") != [314159, 271828, 161803]
        or confirmation.get("expected_runs") != 18
        or confirmation.get("expected_checkpoints") != 90
        or confirmation.get("stages") != [1, 2, 3, 4, 5]
        or confirmation.get("results_available_at_freeze") is not False
    ):
        raise ValueError("Short-branch tail confirmation differs from the frozen design")
    freeze = protocol["freeze_context"]
    if (
        freeze.get("preliminary_tail_diagnostic_inspected") is not True
        or freeze.get("short_branch_results_available") is not False
        or freeze.get("spectral_transplant_results_available") is not False
    ):
        raise ValueError("Tail-stability freeze disclosure is incomplete")
    secondary = protocol["secondary_discovery_diagnostic"]
    if (
        secondary.get("status") != "post_hoc_after_cross_tail_inspection"
        or float(secondary.get("tail_fraction", math.nan)) != 0.05
        or secondary.get("tail_size_per_anchor") != 12
    ):
        raise ValueError("Secondary cross-tail diagnostic differs from its amended design")
    amendments = protocol["amendments"]
    if (
        not isinstance(amendments, list)
        or len(amendments) != 1
        or amendments[0].get("previous_protocol_sha256")
        != "b932fecb7a6bb0f509fe4f1697d48edebc7dfd4f0ad7a1d7d1ddcc8468800c53"
        or amendments[0].get("cross_tail_discovery_results_visible") is not True
        or amendments[0].get("short_branch_results_available") is not False
        or amendments[0].get("spectral_transplant_results_available") is not False
        or amendments[0].get("prospective_confirmation_rule_changed") is not False
    ):
        raise ValueError("Cross-tail protocol amendment disclosure is incomplete")
    return path, protocol


def _finite(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value for {label}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite value for {label}")
    return parsed


def _quantile(values: list[float], quantile: float) -> float:
    if not values or not 0 <= quantile <= 1 or not all(math.isfinite(value) for value in values):
        raise ValueError("Quantiles require a finite non-empty sample and q in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _midrank_percentiles(values: dict[int, float]) -> dict[int, float]:
    if len(values) < 2 or not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Percentile midranks require at least two finite values")
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    output = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        percentile = ((start + end - 1) / 2) / (len(ordered) - 1)
        for sample_id, _ in ordered[start:end]:
            output[sample_id] = percentile
        start = end
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty JSONL source: {path}")
    return rows


def _declared_path(root: Path, item: Any, *, verify_hash: bool = True) -> Path:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise ValueError("Declared file identity is malformed")
    path = Path(item["path"])
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if (
        not path.is_file()
        or path.stat().st_size != item.get("bytes")
        or (verify_hash and _sha256(path) != item.get("sha256"))
    ):
        raise ValueError(f"Declared file differs from its identity: {path}")
    return path


def _identity(path: Path, root: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path if root is None else path.relative_to(root.resolve())),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows or any(list(row) != fields for row in rows):
        raise ValueError(f"Cannot write empty or inconsistent table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def discovery_anchor_tail_rows(
    protocol_path: Path,
    protocol: dict[str, Any],
    families: tuple[str, ...] = ALL_FAMILIES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    families = normalize_families(families)
    root = protocol_path.parent.parent
    summary_path = (root / protocol["source_inputs"]["functional_intervention_summary"]).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sources = summary.get("sources")
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("status") != "complete"
        or summary.get("complete") is not True
        or summary.get("anchors") != 20
        or not isinstance(sources, list)
        or len(sources) != 20
    ):
        raise ValueError("Functional-intervention summary is incomplete")
    expected_samples = int(protocol["discovery_diagnostic"]["examples_per_anchor"])
    relative_scale = float(protocol["discovery_diagnostic"]["relative_scale"])
    tail_fraction = float(protocol["secondary_discovery_diagnostic"]["tail_fraction"])
    expected_tail_size = int(protocol["secondary_discovery_diagnostic"]["tail_size_per_anchor"])
    output = []
    cross_tail_output = []
    verified_sources = []
    labels = set()
    for source in sources:
        label = source.get("label")
        if not isinstance(label, str) or label in labels:
            raise ValueError("Functional-intervention anchor labels are invalid")
        labels.add(label)
        family = label.split("/", 1)[0]
        if family not in FAMILIES:
            raise ValueError(f"Unexpected functional-intervention family: {label}")
        sample_path = _declared_path(summary_path.parent, source.get("sample_metrics"))
        manifest_path = _declared_path(summary_path.parent, source.get("manifest"))
        records = _read_jsonl(sample_path)
        by_condition: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        for record in records:
            condition = record.get("condition")
            sample_id = int(record.get("sample_id"))
            if not isinstance(condition, str) or sample_id in by_condition[condition]:
                raise ValueError(f"Duplicate functional sample in {label}")
            by_condition[condition][sample_id] = record
        baseline = by_condition.get("baseline", {})
        if len(baseline) != expected_samples:
            raise ValueError(f"Functional baseline coverage differs in {label}")
        sample_ids = sorted(baseline)
        loss_deltas_by_algorithm: dict[str, dict[int, float]] = {}
        for algorithm in ALGORITHMS:
            condition = SCALED_CONDITIONS[algorithm]
            indexed = by_condition.get(condition, {})
            if set(indexed) != set(sample_ids):
                raise ValueError(f"Functional condition coverage differs in {label}/{condition}")
            loss_deltas = []
            margin_deltas = []
            indexed_loss_deltas = {}
            for sample_id in sample_ids:
                current = indexed[sample_id]
                base = baseline[sample_id]
                loss_delta = _finite(
                    current.get("contrastive_loss"), f"loss/{label}/{condition}"
                ) - _finite(base.get("contrastive_loss"), f"baseline loss/{label}")
                loss_deltas.append(loss_delta)
                indexed_loss_deltas[sample_id] = loss_delta
                margin_deltas.append(
                    _finite(current.get("positive_margin"), f"margin/{label}/{condition}")
                    - _finite(base.get("positive_margin"), f"baseline margin/{label}")
                )
            output.append(
                {
                    "family": family,
                    "anchor": label,
                    "algorithm": algorithm,
                    "samples": len(sample_ids),
                    "relative_scale": relative_scale,
                    "mean_delta_contrastive_loss": statistics.fmean(loss_deltas),
                    "p95_delta_contrastive_loss": _quantile(loss_deltas, 0.95),
                    "p99_delta_contrastive_loss": _quantile(loss_deltas, 0.99),
                    "loss_improvement_fraction": statistics.fmean(
                        value < 0 for value in loss_deltas
                    ),
                    "mean_delta_positive_margin": statistics.fmean(margin_deltas),
                    "p05_delta_positive_margin": _quantile(margin_deltas, 0.05),
                    "margin_improvement_fraction": statistics.fmean(
                        value > 0 for value in margin_deltas
                    ),
                }
            )
            loss_deltas_by_algorithm[algorithm] = indexed_loss_deltas
        tail_size = math.ceil(tail_fraction * len(sample_ids))
        if tail_size != expected_tail_size:
            raise ValueError(f"Cross-tail selection size differs in {label}: {tail_size}")
        baseline_margin_percentiles = _midrank_percentiles(
            {
                sample_id: _finite(
                    baseline[sample_id].get("positive_margin"),
                    f"baseline margin percentile/{label}",
                )
                for sample_id in sample_ids
            }
        )
        adam_tail = sorted(
            sample_ids,
            key=lambda sample_id: (-loss_deltas_by_algorithm["adamw"][sample_id], sample_id),
        )[:tail_size]
        for challenger in CHALLENGERS:
            challenger_tail = sorted(
                sample_ids,
                key=lambda sample_id: (
                    -loss_deltas_by_algorithm[challenger][sample_id],
                    sample_id,
                ),
            )[:tail_size]
            adam_tail_contrasts = [
                loss_deltas_by_algorithm[challenger][sample_id]
                - loss_deltas_by_algorithm["adamw"][sample_id]
                for sample_id in adam_tail
            ]
            challenger_tail_contrasts = [
                loss_deltas_by_algorithm[challenger][sample_id]
                - loss_deltas_by_algorithm["adamw"][sample_id]
                for sample_id in challenger_tail
            ]
            intersection = len(set(adam_tail) & set(challenger_tail))
            union = len(set(adam_tail) | set(challenger_tail))
            cross_tail_output.append(
                {
                    "family": family,
                    "anchor": label,
                    "challenger": challenger,
                    "reference": "adamw",
                    "samples": len(sample_ids),
                    "tail_fraction": tail_fraction,
                    "tail_size": tail_size,
                    "adam_loss_change_mean_on_adam_tail": statistics.fmean(
                        loss_deltas_by_algorithm["adamw"][sample_id] for sample_id in adam_tail
                    ),
                    "challenger_loss_change_mean_on_adam_tail": statistics.fmean(
                        loss_deltas_by_algorithm[challenger][sample_id] for sample_id in adam_tail
                    ),
                    "challenger_minus_adam_on_adam_tail_loss_mean": statistics.fmean(
                        adam_tail_contrasts
                    ),
                    "adam_tail_challenger_win_fraction": statistics.fmean(
                        value < 0 for value in adam_tail_contrasts
                    ),
                    "adam_loss_change_mean_on_challenger_tail": statistics.fmean(
                        loss_deltas_by_algorithm["adamw"][sample_id]
                        for sample_id in challenger_tail
                    ),
                    "challenger_loss_change_mean_on_challenger_tail": statistics.fmean(
                        loss_deltas_by_algorithm[challenger][sample_id]
                        for sample_id in challenger_tail
                    ),
                    "challenger_minus_adam_on_challenger_tail_loss_mean": statistics.fmean(
                        challenger_tail_contrasts
                    ),
                    "challenger_tail_challenger_win_fraction": statistics.fmean(
                        value < 0 for value in challenger_tail_contrasts
                    ),
                    "tail_intersection": intersection,
                    "tail_union": union,
                    "tail_jaccard": intersection / union,
                    "adam_tail_baseline_margin_percentile_median": statistics.median(
                        baseline_margin_percentiles[sample_id] for sample_id in adam_tail
                    ),
                }
            )
        verified_sources.append(
            {
                "label": label,
                "manifest": _identity(manifest_path),
                "sample_metrics": _identity(sample_path),
            }
        )
    expected = int(protocol["discovery_diagnostic"]["anchors"]) * len(ALGORITHMS)
    if len(output) != expected:
        raise AssertionError(f"Built {len(output)} discovery tail rows, expected {expected}")
    expected_cross_tail = int(protocol["discovery_diagnostic"]["anchors"]) * len(CHALLENGERS)
    if len(cross_tail_output) != expected_cross_tail:
        raise AssertionError(
            f"Built {len(cross_tail_output)} cross-tail rows, expected {expected_cross_tail}"
        )
    return (
        [row for row in output if row["family"] in families],
        [row for row in cross_tail_output if row["family"] in families],
        [source for source in verified_sources if source["label"].split("/", 1)[0] in families],
    )


def _sign_counts(values: list[float], *, beneficial: str) -> tuple[int, int, int]:
    tolerance = 1e-12
    ties = sum(abs(value) <= tolerance for value in values)
    if beneficial == "positive":
        wins = sum(value > tolerance for value in values)
    elif beneficial == "negative":
        wins = sum(value < -tolerance for value in values)
    else:
        raise ValueError(f"Unknown beneficial sign: {beneficial}")
    return wins, ties, len(values) - wins - ties


def _leave_one_out_fraction(values: list[float], *, expected: str) -> float:
    if len(values) < 2:
        raise ValueError("Leave-one-out sensitivity requires at least two values")
    means = [statistics.fmean(values[:index] + values[index + 1 :]) for index in range(len(values))]
    if expected == "negative":
        return statistics.fmean(value < 0 for value in means)
    if expected == "positive_or_zero":
        return statistics.fmean(value >= 0 for value in means)
    raise ValueError(f"Unknown leave-one-out direction: {expected}")


def discovery_family_contrasts(
    rows: list[dict[str, Any]], families: tuple[str, ...] = ALL_FAMILIES
) -> list[dict[str, Any]]:
    families = normalize_families(families)
    indexed = {(str(row["family"]), str(row["anchor"]), str(row["algorithm"])): row for row in rows}
    expected = 10 * len(families) * len(ALGORITHMS)
    if len(indexed) != expected:
        raise ValueError(f"Discovery tail contrasts require {expected} unique anchor/operator rows")
    output = []
    for family in families:
        anchors = sorted(
            {anchor for current_family, anchor, _ in indexed if current_family == family}
        )
        if len(anchors) != 10:
            raise ValueError(f"{family}: tail diagnostic requires ten anchors")
        for challenger in CHALLENGERS:
            contrasts: dict[str, list[float]] = {}
            for metric in (
                "mean_delta_positive_margin",
                "p05_delta_positive_margin",
                "p95_delta_contrastive_loss",
                "p99_delta_contrastive_loss",
            ):
                contrasts[metric] = [
                    float(indexed[(family, anchor, challenger)][metric])
                    - float(indexed[(family, anchor, "adamw")][metric])
                    for anchor in anchors
                ]
            mean_margin = contrasts["mean_delta_positive_margin"]
            p05_margin = contrasts["p05_delta_positive_margin"]
            p95_loss = contrasts["p95_delta_contrastive_loss"]
            p99_loss = contrasts["p99_delta_contrastive_loss"]
            mean_counts = _sign_counts(mean_margin, beneficial="positive")
            p05_counts = _sign_counts(p05_margin, beneficial="positive")
            p95_counts = _sign_counts(p95_loss, beneficial="negative")
            p99_counts = _sign_counts(p99_loss, beneficial="negative")
            output.append(
                {
                    "family": family,
                    "challenger": challenger,
                    "reference": "adamw",
                    "anchors": len(anchors),
                    "relative_scale": 0.001,
                    "mean_margin_contrast_mean": statistics.fmean(mean_margin),
                    "mean_margin_contrast_median": statistics.median(mean_margin),
                    "mean_margin_challenger_wins": mean_counts[0],
                    "mean_margin_ties": mean_counts[1],
                    "mean_margin_challenger_losses": mean_counts[2],
                    "mean_margin_leave_one_out_negative_fraction": _leave_one_out_fraction(
                        mean_margin, expected="negative"
                    ),
                    "p05_margin_contrast_mean": statistics.fmean(p05_margin),
                    "p05_margin_contrast_median": statistics.median(p05_margin),
                    "p05_margin_challenger_wins": p05_counts[0],
                    "p05_margin_ties": p05_counts[1],
                    "p05_margin_challenger_losses": p05_counts[2],
                    "p05_margin_leave_one_out_positive_or_zero_fraction": _leave_one_out_fraction(
                        p05_margin, expected="positive_or_zero"
                    ),
                    "p95_loss_contrast_mean": statistics.fmean(p95_loss),
                    "p95_loss_contrast_median": statistics.median(p95_loss),
                    "p95_loss_challenger_wins": p95_counts[0],
                    "p95_loss_ties": p95_counts[1],
                    "p95_loss_challenger_losses": p95_counts[2],
                    "p95_loss_leave_one_out_negative_fraction": _leave_one_out_fraction(
                        p95_loss, expected="negative"
                    ),
                    "p99_loss_contrast_mean": statistics.fmean(p99_loss),
                    "p99_loss_contrast_median": statistics.median(p99_loss),
                    "p99_loss_challenger_wins": p99_counts[0],
                    "p99_loss_ties": p99_counts[1],
                    "p99_loss_challenger_losses": p99_counts[2],
                    "p99_loss_leave_one_out_negative_fraction": _leave_one_out_fraction(
                        p99_loss, expected="negative"
                    ),
                    "mean_tail_tradeoff_observed": statistics.fmean(mean_margin) < 0
                    and statistics.median(p95_loss) < 0
                    and statistics.median(p05_margin) >= 0,
                }
            )
    return output


def discovery_cross_tail_summary(
    rows: list[dict[str, Any]], families: tuple[str, ...] = ALL_FAMILIES
) -> list[dict[str, Any]]:
    families = normalize_families(families)
    indexed = {
        (str(row["family"]), str(row["anchor"]), str(row["challenger"])): row for row in rows
    }
    expected = 10 * len(families) * len(CHALLENGERS)
    if len(indexed) != expected:
        raise ValueError(f"Cross-tail summary requires {expected} unique anchor/challenger rows")
    output = []
    for family in families:
        for challenger in CHALLENGERS:
            members = [
                row
                for (current_family, _, current_challenger), row in sorted(indexed.items())
                if current_family == family and current_challenger == challenger
            ]
            if len(members) != 10:
                raise ValueError(f"{family}/{challenger}: cross-tail summary requires ten anchors")
            adam_tail = [
                float(row["challenger_minus_adam_on_adam_tail_loss_mean"]) for row in members
            ]
            challenger_tail = [
                float(row["challenger_minus_adam_on_challenger_tail_loss_mean"]) for row in members
            ]
            dual_advantage = (
                statistics.fmean(adam_tail) < 0 and statistics.fmean(challenger_tail) < 0
            )
            if dual_advantage:
                regime = "shared-tail severity suppression"
            elif statistics.fmean(adam_tail) < 0 < statistics.fmean(challenger_tail):
                regime = "tail redistribution"
            else:
                regime = "mixed"
            output.append(
                {
                    "family": family,
                    "challenger": challenger,
                    "reference": "adamw",
                    "anchors": len(members),
                    "tail_fraction": float(members[0]["tail_fraction"]),
                    "tail_size": int(members[0]["tail_size"]),
                    "adam_tail_contrast_mean": statistics.fmean(adam_tail),
                    "adam_tail_contrast_median": statistics.median(adam_tail),
                    "adam_tail_anchor_wins": sum(value < 0 for value in adam_tail),
                    "adam_tail_leave_one_out_negative_fraction": _leave_one_out_fraction(
                        adam_tail, expected="negative"
                    ),
                    "challenger_tail_contrast_mean": statistics.fmean(challenger_tail),
                    "challenger_tail_contrast_median": statistics.median(challenger_tail),
                    "challenger_tail_anchor_wins": sum(value < 0 for value in challenger_tail),
                    "challenger_tail_leave_one_out_negative_fraction": _leave_one_out_fraction(
                        challenger_tail, expected="negative"
                    ),
                    "adam_tail_challenger_sample_win_fraction_mean": statistics.fmean(
                        float(row["adam_tail_challenger_win_fraction"]) for row in members
                    ),
                    "challenger_tail_challenger_sample_win_fraction_mean": statistics.fmean(
                        float(row["challenger_tail_challenger_win_fraction"]) for row in members
                    ),
                    "tail_intersection_mean": statistics.fmean(
                        int(row["tail_intersection"]) for row in members
                    ),
                    "tail_jaccard_mean": statistics.fmean(
                        float(row["tail_jaccard"]) for row in members
                    ),
                    "adam_tail_baseline_margin_percentile_median_mean": statistics.fmean(
                        float(row["adam_tail_baseline_margin_percentile_median"]) for row in members
                    ),
                    "dual_selected_tail_advantage": dual_advantage,
                    "tail_identity_regime": regime,
                }
            )
    return output


def _short_branch_tail_rows(
    protocol_path: Path,
    protocol: dict[str, Any],
    *,
    experiment_matrix: Path,
    matrix_dir: Path | None,
    results_root: Path,
    families: tuple[str, ...] = ALL_FAMILIES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    families = normalize_families(families)
    root = protocol_path.parent.parent
    report_manifest_path = (root / "reports/short-branch/summary_manifest.json").resolve()
    if not report_manifest_path.is_file():
        return [], [], "short-branch summary manifest is not available"
    report_manifest = json.loads(report_manifest_path.read_text(encoding="utf-8"))
    if report_manifest.get("complete") is not True:
        return [], [], "short-branch summary manifest is incomplete"
    short_protocol = (root / protocol["source_inputs"]["short_branch_protocol"]).resolve()
    _, short_spec, configs, _ = _load_branch_configs(
        short_protocol,
        experiment_matrix=experiment_matrix,
        matrix_dir=matrix_dir,
        audit_matrices=True,
        families=families,
    )
    validation_spec_path, _ = load_validation_spec(
        root / protocol["source_inputs"]["validation_spec"]
    )
    validation_jobs = build_short_branch_validation_jobs(configs, results_root / "query-disjoint")
    unseen_probe = Path(short_spec["evaluation"]["unseen_retrieval_probe"]).resolve()
    unseen_spec = resolve_probe_spec_path(
        root / protocol["source_inputs"]["unseen_probe_spec"]
    ).resolve()
    probe_identity = _requested_probe_identity(unseen_probe, unseen_spec)
    probe_jobs = build_short_branch_probe_jobs(
        configs,
        {family: Path(".") for family in families},
        results_root / "unseen-representation",
        probe_identity,
    )
    counts = _audit_counts(validation_jobs, probe_jobs, validation_spec_path)
    if (
        counts["validation_complete"] != counts["validation_expected"]
        or counts["unseen_probe_complete"] != counts["unseen_probe_expected"]
    ):
        raise ValueError(
            f"Complete short-branch summary disagrees with the audited evaluation matrix: {counts}"
        )

    rows_by_identity: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    sources = [{"short_branch_summary": _identity(report_manifest_path)}]
    for job in validation_jobs:
        if not validation_job_complete(job, validation_spec_path, verify_hashes=True):
            raise ValueError(f"Short-branch validation job failed audit: {job.label}")
        schedule = json.loads(
            (job.config.output_dir / "checkpoint_schedule.json").read_text(encoding="utf-8")
        )["steps"]
        stage = [int(value) for value in schedule].index(job.step) + 1
        operator = (
            "adamw" if job.config.optimizer.name == "hybrid_adamw" else job.config.optimizer.name
        )
        identity = (job.config.model_family, int(job.seed), operator, stage)
        if identity in rows_by_identity:
            raise ValueError(f"Duplicate short-branch validation identity: {identity}")
        manifest_path = job.output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sample_path = _declared_path(
            job.output_dir, manifest.get("outputs", {}).get("sample_metrics")
        )
        samples = _read_jsonl(sample_path)
        expected_samples = int(manifest.get("sample_records", -1))
        if len(samples) != expected_samples or expected_samples != 4096:
            raise ValueError(f"Short-branch validation sample coverage differs: {job.label}")
        losses = [
            _finite(row.get("contrastive_loss"), f"validation loss/{job.label}") for row in samples
        ]
        margins = [
            _finite(row.get("positive_margin"), f"validation margin/{job.label}") for row in samples
        ]
        rows_by_identity[identity] = {
            "family": identity[0],
            "seed": identity[1],
            "operator": identity[2],
            "run_id": job.config.run_id,
            "stage": stage,
            "fraction": float(job.config.checkpoint_fractions[stage - 1]),
            "step": job.step,
            "validation_samples": len(samples),
            "validation_loss_mean": statistics.fmean(losses),
            "validation_loss_p95": _quantile(losses, 0.95),
            "validation_loss_p99": _quantile(losses, 0.99),
            "validation_margin_mean": statistics.fmean(margins),
            "validation_margin_p05": _quantile(margins, 0.05),
        }
        sources.append(
            {
                "label": job.label,
                "validation_manifest": _identity(manifest_path),
                "validation_samples": _identity(sample_path),
            }
        )

    expected_probe = expected_short_branch_probe_metrics(configs, probe_jobs)
    for expected in expected_probe:
        if expected.job.kind != "checkpoint":
            continue
        if not probe_job_complete(expected.job):
            raise ValueError(f"Short-branch unseen probe failed audit: {expected.job.label}")
        identity = (
            expected.job.family,
            int(expected.seed),
            str(expected.optimizer),
            int(expected.stage),
        )
        if identity not in rows_by_identity:
            raise ValueError(f"Short-branch validation/probe identities disagree: {identity}")
        payload = json.loads(expected.job.metrics.read_text(encoding="utf-8"))
        score = payload.get("metrics", {}).get("score_geometry", {})
        margin = score.get("positive_hardest_negative_margin", {})
        reference = score.get("reference_ranking", {})
        row = rows_by_identity[identity]
        row.update(
            {
                "unseen_samples": int(score.get("samples", -1)),
                "unseen_margin_mean": _finite(margin.get("mean"), f"unseen mean/{identity}"),
                "unseen_margin_p05": _finite(margin.get("p05"), f"unseen p05/{identity}"),
                "unseen_pretrained_top1_agreement": _finite(
                    reference.get("top1_agreement"), f"top1 agreement/{identity}"
                ),
                "unseen_pretrained_score_drift_rms": _finite(
                    reference.get("score_drift_rms"), f"score drift/{identity}"
                ),
            }
        )
        if row["unseen_samples"] != 224:
            raise ValueError(f"Short-branch unseen sample coverage differs: {identity}")
        sources.append(
            {
                "label": expected.job.label,
                "unseen_metrics": _identity(expected.job.metrics),
            }
        )
    rows = [rows_by_identity[key] for key in sorted(rows_by_identity)]
    expected_rows = len(families) * 3 * len(ALGORITHMS) * 5
    if len(rows) != expected_rows or any(list(row) != SHORT_BRANCH_FIELDS for row in rows):
        raise ValueError(f"Short-branch tail matrix does not contain {expected_rows} complete rows")
    return rows, sources, None


def short_branch_contrasts(
    rows: list[dict[str, Any]],
    families: tuple[str, ...] = ALL_FAMILIES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    families = normalize_families(families)
    indexed = {
        (str(row["family"]), int(row["seed"]), int(row["stage"]), str(row["operator"])): row
        for row in rows
    }
    expected_rows = len(families) * 3 * 5 * len(ALGORITHMS)
    if len(indexed) != expected_rows:
        raise ValueError(f"Short-branch tail contrasts require {expected_rows} unique rows")
    contrast_metrics = (
        "validation_loss_mean",
        "validation_loss_p95",
        "validation_loss_p99",
        "validation_margin_mean",
        "validation_margin_p05",
        "unseen_margin_mean",
        "unseen_margin_p05",
        "unseen_pretrained_top1_agreement",
        "unseen_pretrained_score_drift_rms",
    )
    contrasts = []
    seeds = sorted({identity[1] for identity in indexed})
    for family in families:
        for seed in seeds:
            for stage in range(1, 6):
                reference = indexed[(family, seed, stage, "adamw")]
                for challenger in CHALLENGERS:
                    treatment = indexed[(family, seed, stage, challenger)]
                    contrasts.append(
                        {
                            "family": family,
                            "seed": seed,
                            "stage": stage,
                            "fraction": treatment["fraction"],
                            "challenger": challenger,
                            "reference": "adamw",
                            **{
                                f"delta_{metric}": float(treatment[metric])
                                - float(reference[metric])
                                for metric in contrast_metrics
                            },
                        }
                    )
    expected_contrasts = len(families) * 3 * 5 * len(CHALLENGERS)
    if len(contrasts) != expected_contrasts or any(
        list(row) != SHORT_BRANCH_CONTRAST_FIELDS for row in contrasts
    ):
        raise AssertionError("Short-branch tail contrast cardinality changed")
    final = []
    for family in families:
        for challenger in CHALLENGERS:
            members = [
                row
                for row in contrasts
                if row["family"] == family and row["challenger"] == challenger and row["stage"] == 5
            ]
            if len(members) != 3:
                raise ValueError("Final tail confirmation requires exactly three seeds")

            def median(metric: str) -> float:
                return statistics.median(float(row[f"delta_{metric}"]) for row in members)

            loss_p95 = median("validation_loss_p95")
            margin_p05 = median("unseen_margin_p05")
            if loss_p95 < 0 and margin_p05 > 0:
                decision = "supported"
            elif (loss_p95 < 0) != (margin_p05 > 0):
                decision = "mixed"
            else:
                decision = "not-supported"
            final.append(
                {
                    "family": family,
                    "challenger": challenger,
                    "reference": "adamw",
                    "stage": 5,
                    "seeds": len(members),
                    "median_delta_validation_loss_mean": median("validation_loss_mean"),
                    "median_delta_validation_loss_p95": loss_p95,
                    "median_delta_validation_loss_p99": median("validation_loss_p99"),
                    "validation_loss_p95_seed_wins": sum(
                        float(row["delta_validation_loss_p95"]) < 0 for row in members
                    ),
                    "median_delta_validation_margin_mean": median("validation_margin_mean"),
                    "median_delta_validation_margin_p05": median("validation_margin_p05"),
                    "median_delta_unseen_margin_mean": median("unseen_margin_mean"),
                    "median_delta_unseen_margin_p05": margin_p05,
                    "unseen_margin_p05_seed_wins": sum(
                        float(row["delta_unseen_margin_p05"]) > 0 for row in members
                    ),
                    "median_delta_unseen_pretrained_top1_agreement": median(
                        "unseen_pretrained_top1_agreement"
                    ),
                    "median_delta_unseen_pretrained_score_drift_rms": median(
                        "unseen_pretrained_score_drift_rms"
                    ),
                    "tail_stability_decision": decision,
                }
            )
    return contrasts, final


def _render_readme(
    discovery: list[dict[str, Any]],
    cross_tail: list[dict[str, Any]],
    final: list[dict[str, Any]],
    *,
    pending_reason: str | None,
    claim_boundary: str,
    families: tuple[str, ...] = ALL_FAMILIES,
) -> str:
    lines = [
        "# Mean improvement versus tail stability",
        "",
        "This analysis is post hoc for the completed discovery intervention and prospectively frozen for the three-seed shared-start branches.",
        "",
        "## Same-state discovery diagnostic",
        "",
        "Every row compares a per-tensor Frobenius-matched `1e-3` virtual step with AdamW. Negative mean-margin contrasts favor AdamW on average; negative loss-tail contrasts and positive margin-tail contrasts favor the challenger on the worst queries.",
        "",
        "| Family | Challenger | Mean margin Δ | p05 margin Δ | p95 loss Δ | p99 loss Δ | p99 anchor wins | Trade-off |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in discovery:
        lines.append(
            f"| {row['family']} | {row['challenger']} | {row['mean_margin_contrast_mean']:+.3e} | "
            f"{row['p05_margin_contrast_mean']:+.3e} | {row['p95_loss_contrast_mean']:+.3e} | "
            f"{row['p99_loss_contrast_mean']:+.3e} | {row['p99_loss_challenger_wins']}/10 | "
            f"{'yes' if row['mean_tail_tradeoff_observed'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "The result is a mean–tail trade-off, not a claim that Muon has a better average local step. Muon-family directions produce smaller average margin gains while reducing severe per-query regressions. Because this pattern was found after inspecting the discovery intervention, it is explanatory rather than confirmatory.",
            "",
            "## Which queries occupy the bad tail?",
            "",
            "This secondary diagnostic was added after its preliminary values were visible. It symmetrically selects each operator's worst 5% loss-change set (12/224 queries). A negative contrast on both selected sets indicates severity suppression on a shared fragile-query tail; an advantage only on AdamW's selected set indicates that the operator mainly changes which queries occupy its tail.",
            "",
            "| Family | Challenger | Δ on AdamW tail | Δ on challenger tail | tail Jaccard | AdamW-tail baseline-margin percentile | Regime |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in cross_tail:
        lines.append(
            f"| {row['family']} | {row['challenger']} | "
            f"{row['adam_tail_contrast_mean']:+.3e} | "
            f"{row['challenger_tail_contrast_mean']:+.3e} | "
            f"{row['tail_jaccard_mean']:.3f} | "
            f"{row['adam_tail_baseline_margin_percentile_median_mean']:.3f} | "
            f"{row['tail_identity_regime']} |"
        )
    architecture_readout = (
        "Late interaction shows a largely shared fragile-query set whose regression severity is "
        "reduced even when the challenger defines the tail. Dense retrieval shows much lower tail "
        "overlap and reverses sign on the challenger-selected set, which is evidence of tail "
        "redistribution rather than uniform query-wise dominance."
        if set(families) == set(ALL_FAMILIES)
        else "Within DenseOn, low tail overlap and sign reversal on the challenger-selected set "
        "indicate tail redistribution rather than uniform query-wise dominance."
    )
    lines.extend(["", architecture_readout, "", "## Prospective shared-start confirmation", ""])
    if final:
        lines.extend(
            [
                "| Family | Challenger | validation loss p95 Δ | unseen margin p05 Δ | loss-tail seed wins | margin-tail seed wins | Decision |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in final:
            lines.append(
                f"| {row['family']} | {row['challenger']} | "
                f"{row['median_delta_validation_loss_p95']:+.3e} | "
                f"{row['median_delta_unseen_margin_p05']:+.3e} | "
                f"{row['validation_loss_p95_seed_wins']}/3 | "
                f"{row['unseen_margin_p05_seed_wins']}/3 | {row['tail_stability_decision']} |"
            )
    else:
        lines.append(f"Pending: {pending_reason or 'short-branch outcomes are not available'}.")
    lines.extend(["", f"> Claim boundary: {claim_boundary}", ""])
    return "\n".join(lines)


def build_tail_stability_report(
    protocol_path: str | Path = "configs/tail_stability_analysis.json",
    *,
    output_dir: str | Path = "reports/tail-stability",
    experiment_matrix: str | Path = "configs/experiment.yaml",
    matrix_dir: str | Path | None = None,
    short_branch_results_root: str | Path | None = None,
    require_short_branch: bool = False,
    families: tuple[str, ...] = ALL_FAMILIES,
    scope_amendment: str | Path | None = None,
) -> dict[str, Any]:
    families, scope = resolve_scope(families, scope_amendment)
    protocol_path, protocol = load_tail_stability_protocol(protocol_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    discovery_rows, cross_tail_rows, discovery_sources = discovery_anchor_tail_rows(
        protocol_path, protocol, families
    )
    discovery_contrasts = discovery_family_contrasts(discovery_rows, families)
    cross_tail_summary = discovery_cross_tail_summary(cross_tail_rows, families)
    short_root = Path(
        short_branch_results_root or protocol["short_branch_confirmation"]["results_root"]
    ).resolve()
    short_rows, short_sources, pending_reason = _short_branch_tail_rows(
        protocol_path,
        protocol,
        experiment_matrix=Path(experiment_matrix).resolve(),
        matrix_dir=None if matrix_dir is None else Path(matrix_dir).resolve(),
        results_root=short_root,
        families=families,
    )
    if require_short_branch and pending_reason is not None:
        raise RuntimeError(f"Short-branch tail confirmation is not ready: {pending_reason}")
    short_contrasts: list[dict[str, Any]] = []
    final_summary: list[dict[str, Any]] = []
    if short_rows:
        short_contrasts, final_summary = short_branch_contrasts(short_rows, families)

    table_specs: list[tuple[str, list[str], list[dict[str, Any]]]] = [
        ("discovery_anchor_tail", DISCOVERY_ANCHOR_FIELDS, discovery_rows),
        ("discovery_family_contrasts", DISCOVERY_CONTRAST_FIELDS, discovery_contrasts),
        ("discovery_cross_tail", DISCOVERY_CROSS_TAIL_FIELDS, cross_tail_rows),
        (
            "discovery_cross_tail_summary",
            DISCOVERY_CROSS_TAIL_SUMMARY_FIELDS,
            cross_tail_summary,
        ),
    ]
    if short_rows:
        table_specs.extend(
            [
                ("short_branch_checkpoint_tail", SHORT_BRANCH_FIELDS, short_rows),
                (
                    "short_branch_checkpoint_contrasts",
                    SHORT_BRANCH_CONTRAST_FIELDS,
                    short_contrasts,
                ),
                ("short_branch_final_summary", SHORT_BRANCH_FINAL_FIELDS, final_summary),
            ]
        )
    outputs = {}
    for name, fields, rows in table_specs:
        path = output_dir / f"{name}.csv"
        _write_csv(path, fields, rows)
        outputs[name] = {**_identity(path, output_dir), "rows": len(rows)}
    readme_path = output_dir / "README.md"
    _write_text(
        readme_path,
        _render_readme(
            discovery_contrasts,
            cross_tail_summary,
            final_summary,
            pending_reason=pending_reason,
            claim_boundary=protocol["claim_boundary"],
            families=families,
        ),
    )
    outputs["readme"] = _identity(readme_path, output_dir)
    complete = pending_reason is None and bool(short_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if complete else "discovery-complete-confirmation-pending",
        "complete": complete,
        "discovery_complete": True,
        "short_branch_confirmation_complete": complete,
        "analysis_status": protocol["analysis_status"],
        "families": list(families),
        "scope_amendment": scope,
        "protocol": _identity(protocol_path),
        "discovery_anchors": len(discovery_rows) // len(ALGORITHMS),
        "discovery_anchor_operator_rows": len(discovery_rows),
        "discovery_contrasts": len(discovery_contrasts),
        "discovery_cross_tail_rows": len(cross_tail_rows),
        "discovery_cross_tail_summaries": len(cross_tail_summary),
        "short_branch_checkpoint_rows": len(short_rows),
        "short_branch_contrast_rows": len(short_contrasts),
        "short_branch_final_rows": len(final_summary),
        "pending_reason": pending_reason,
        "discovery_sources": discovery_sources,
        "short_branch_sources": short_sources,
        "outputs": outputs,
        "claim_boundary": protocol["claim_boundary"],
    }
    _atomic_json(output_dir / "summary_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and summarize the frozen mean-versus-tail stability analysis"
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/tail_stability_analysis.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/tail-stability"))
    parser.add_argument("--experiment-matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--matrix-dir", type=Path)
    parser.add_argument("--short-branch-results-root", type=Path)
    parser.add_argument(
        "--families", nargs="+", choices=("dense", "late"), default=["dense", "late"]
    )
    parser.add_argument("--scope-amendment", type=Path)
    parser.add_argument("--require-short-branch", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_tail_stability_report(
        args.protocol,
        output_dir=args.output_dir,
        experiment_matrix=args.experiment_matrix,
        matrix_dir=args.matrix_dir,
        short_branch_results_root=args.short_branch_results_root,
        require_short_branch=args.require_short_branch,
        families=tuple(args.families),
        scope_amendment=args.scope_amendment,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
