from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common_state_matrix import (
    _load_protocol,
    _resolve_reference,
    build_common_state_jobs,
    resolve_common_state_spec,
)
from .config import ModelFamily, load_matrix, resolve_matrix_path
from .functional_intervention import (
    intervention_conditions,
    load_intervention_protocol,
)
from .functional_intervention_matrix import (
    build_functional_intervention_jobs,
    functional_intervention_job_complete,
)
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .scope import ALL_FAMILIES, resolve_scope

METRICS = (
    "contrastive_loss",
    "positive_score",
    "hardest_negative_score",
    "positive_margin",
    "reciprocal_rank",
    "top1_accuracy",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(record)
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"Rows have inconsistent columns: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_identity(path: Path, root: Path, *, rows: int | None = None) -> dict[str, Any]:
    identity = {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if rows is not None:
        identity["rows"] = rows
    return identity


def _finite_number(value: Any, label: str) -> float:
    value = float(value)
    if not __import__("math").isfinite(value):
        raise ValueError(f"Non-finite {label}")
    return value


def _anchor_effects(
    label: str,
    family: str,
    records: list[dict[str, Any]],
    expected_conditions: list[str],
    expected_samples: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, float]]]:
    by_condition: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        condition = record.get("condition")
        sample_id = int(record.get("sample_id"))
        if condition not in expected_conditions or sample_id in by_condition[condition]:
            raise ValueError(f"Duplicate or unexpected condition/sample in {label}")
        for metric in METRICS:
            _finite_number(record.get(metric), f"{label}/{condition}/{sample_id}/{metric}")
        by_condition[condition][sample_id] = record
    if set(by_condition) != set(expected_conditions):
        raise ValueError(f"Condition coverage differs from the frozen protocol in {label}")
    baseline = by_condition["baseline"]
    if len(baseline) != expected_samples:
        raise ValueError(f"Baseline sample count differs in {label}")
    sample_ids = set(baseline)
    if any(set(rows) != sample_ids for rows in by_condition.values()):
        raise ValueError(f"Conditions do not contain the same paired samples in {label}")

    effects = []
    indexed: dict[tuple[str, int], dict[str, float]] = {}
    for condition in expected_conditions[1:]:
        rows = by_condition[condition]
        exemplar = next(iter(rows.values()))
        effect = {
            "family": family,
            "anchor": label,
            "condition": condition,
            "algorithm": exemplar["algorithm"],
            "direction": exemplar["direction"],
            "relative_scale": float(exemplar["relative_scale"]),
            "samples": len(rows),
        }
        for metric in METRICS:
            values = [float(rows[sample_id][metric]) for sample_id in sorted(sample_ids)]
            base_values = [float(baseline[sample_id][metric]) for sample_id in sorted(sample_ids)]
            deltas = [value - base for value, base in zip(values, base_values, strict=True)]
            effect[metric] = statistics.fmean(values)
            effect[f"delta_{metric}"] = statistics.fmean(deltas)
        effect["loss_improvement_rate"] = statistics.fmean(
            float(rows[sample_id]["contrastive_loss"])
            < float(baseline[sample_id]["contrastive_loss"])
            for sample_id in sample_ids
        )
        effects.append(effect)
        for sample_id in sample_ids:
            indexed[(condition, sample_id)] = {
                metric: float(rows[sample_id][metric]) - float(baseline[sample_id][metric])
                for metric in METRICS
            }
    return effects, indexed


def _optimizer_contrasts(
    label: str,
    family: str,
    conditions: list[Any],
    indexed: dict[tuple[str, int], dict[str, float]],
) -> list[dict[str, Any]]:
    by_key = {
        (condition.algorithm, condition.direction, condition.relative_scale): condition.condition
        for condition in conditions
        if condition.algorithm is not None
    }
    sample_ids = sorted({sample_id for _, sample_id in indexed})
    rows = []
    for challenger in ("muon", "normuon"):
        for direction in ("descent", "sign_reversal"):
            scales = sorted(
                scale
                for algorithm, candidate_direction, scale in by_key
                if algorithm == challenger and candidate_direction == direction
            )
            for scale in scales:
                adam_condition = by_key[("adamw", direction, scale)]
                challenger_condition = by_key[(challenger, direction, scale)]
                row = {
                    "family": family,
                    "anchor": label,
                    "challenger": challenger,
                    "reference": "adamw",
                    "direction": direction,
                    "relative_scale": scale,
                    "samples": len(sample_ids),
                }
                for metric in METRICS:
                    paired = [
                        indexed[(challenger_condition, sample_id)][metric]
                        - indexed[(adam_condition, sample_id)][metric]
                        for sample_id in sample_ids
                    ]
                    row[f"delta_delta_{metric}"] = statistics.fmean(paired)
                rows.append(row)
    return rows


def _family_summary(anchor_effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in anchor_effects:
        key = (
            row["family"],
            row["algorithm"],
            row["direction"],
            row["relative_scale"],
        )
        grouped[key].append(row)
    summary = []
    for key, rows in sorted(grouped.items()):
        family, algorithm, direction, scale = key
        output = {
            "family": family,
            "algorithm": algorithm,
            "direction": direction,
            "relative_scale": scale,
            "anchors": len(rows),
        }
        for metric in METRICS:
            values = [float(row[f"delta_{metric}"]) for row in rows]
            output[f"mean_anchor_delta_{metric}"] = statistics.fmean(values)
            output[f"median_anchor_delta_{metric}"] = statistics.median(values)
        output["anchors_with_lower_loss_fraction"] = statistics.fmean(
            float(row["delta_contrastive_loss"]) < 0 for row in rows
        )
        output["mean_sample_loss_improvement_rate"] = statistics.fmean(
            float(row["loss_improvement_rate"]) for row in rows
        )
        summary.append(output)
    return summary


def summarize_functional_interventions(
    jobs: list[Any],
    output_dir: str | Path,
    *,
    intervention_spec: str | Path,
    families: tuple[str, ...] = ALL_FAMILIES,
    scope_amendment: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    spec_path, spec = load_intervention_protocol(intervention_spec)
    conditions = intervention_conditions(spec)
    condition_names = [condition.condition for condition in conditions]
    families, scope = resolve_scope(families, scope_amendment)
    expected_per_family, remainder = divmod(
        int(spec["common_state"]["expected_anchors"]), len(ALL_FAMILIES)
    )
    if remainder:
        raise ValueError("Frozen intervention anchor count is not divisible by its family count")
    expected_anchors = expected_per_family * len(families)
    if len(jobs) != expected_anchors:
        raise ValueError(
            f"Intervention summary requires {expected_anchors} anchors for the requested scope"
        )
    if {job.common_state.family for job in jobs} != set(families):
        raise ValueError("Intervention jobs do not match the requested family scope")
    incomplete = [
        job.label
        for job in jobs
        if not functional_intervention_job_complete(job, spec_path, verify_hashes=True)
    ]
    if incomplete:
        raise ValueError("Incomplete functional intervention jobs: " + ", ".join(incomplete))

    sources = []
    anchor_effects = []
    optimizer_contrasts = []
    for job in jobs:
        manifest_path = job.output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sample_path = job.output_dir / manifest["outputs"]["sample_metrics"]["path"]
        records = _read_jsonl(sample_path)
        effects, indexed = _anchor_effects(
            job.label,
            job.common_state.family,
            records,
            condition_names,
            spec["evaluation_probe"]["count"],
        )
        anchor_effects.extend(effects)
        optimizer_contrasts.extend(
            _optimizer_contrasts(job.label, job.common_state.family, conditions, indexed)
        )
        sources.append(
            {
                "label": job.label,
                "manifest": {
                    "path": str(manifest_path),
                    "bytes": manifest_path.stat().st_size,
                    "sha256": _sha256(manifest_path),
                },
                "sample_metrics": {
                    "path": str(sample_path),
                    "bytes": sample_path.stat().st_size,
                    "sha256": _sha256(sample_path),
                },
            }
        )
    family_summary = _family_summary(anchor_effects)
    expected_effects = len(jobs) * (len(conditions) - 1)
    if len(anchor_effects) != expected_effects:
        raise AssertionError("Anchor effect cardinality changed")

    anchor_path = output_dir / "anchor_condition_effects.csv"
    contrast_path = output_dir / "optimizer_direction_contrasts.csv"
    summary_path = output_dir / "family_summary.csv"
    _write_csv(anchor_path, anchor_effects)
    _write_csv(contrast_path, optimizer_contrasts)
    _write_csv(summary_path, family_summary)
    outputs = {
        "anchor_condition_effects": _source_identity(
            anchor_path, output_dir, rows=len(anchor_effects)
        ),
        "optimizer_direction_contrasts": _source_identity(
            contrast_path, output_dir, rows=len(optimizer_contrasts)
        ),
        "family_summary": _source_identity(summary_path, output_dir, rows=len(family_summary)),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "status": "complete",
        "families": list(families),
        "scope_amendment": scope,
        "intervention_spec": {
            "path": str(spec_path),
            "bytes": spec_path.stat().st_size,
            "sha256": _sha256(spec_path),
        },
        "anchors": len(jobs),
        "conditions_per_anchor": len(conditions),
        "anchor_effect_records": len(anchor_effects),
        "optimizer_contrast_records": len(optimizer_contrasts),
        "family_summary_records": len(family_summary),
        "sources": sources,
        "outputs": outputs,
        "claim_boundary": spec["claim_boundary"],
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly summarize the scale-matched functional interventions"
    )
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--families", nargs="+", choices=("dense", "late"), default=["dense", "late"]
    )
    parser.add_argument("--scope-amendment", type=Path)
    parser.add_argument("--dense-reference-checkpoint", type=Path)
    parser.add_argument("--late-reference-checkpoint", type=Path)
    parser.add_argument("--common-state-root", type=Path, default=Path("results/common-state"))
    parser.add_argument("--result-root", type=Path, default=Path("results/functional-intervention"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/functional-intervention"))
    parser.add_argument(
        "--common-state-spec", type=Path, default=Path("configs/common_state_probe.json")
    )
    parser.add_argument(
        "--intervention-spec", type=Path, default=Path("configs/functional_intervention.json")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    spec_path, intervention = load_intervention_protocol(args.intervention_spec)
    common_spec_path = resolve_common_state_spec(args.common_state_spec).resolve()
    if _sha256(common_spec_path) != intervention["common_state"]["spec_sha256"]:
        raise ValueError("Common-state spec differs from the intervention lock")
    common_spec, common_anchor = _load_protocol(common_spec_path)
    matrix_path = resolve_matrix_path(args.matrix).resolve()
    all_configs = load_matrix(matrix_path)
    families, _ = resolve_scope(args.families, args.scope_amendment)
    configs = [config for config in all_configs if config.model_family in families]
    if {config.model_family for config in configs} != set(families):
        raise ValueError("Training matrix does not cover every requested model family")
    if intervention["common_state"]["expected_anchors"] != common_anchor["expected_total_anchors"]:
        raise ValueError("Functional intervention and common-state anchor locks differ")
    by_family = {config.model_family: config for config in configs}
    references: dict[ModelFamily, Path] = {}
    for family in families:
        explicit = (
            args.dense_reference_checkpoint if family == "dense" else args.late_reference_checkpoint
        )
        references[family] = _resolve_reference(by_family[family], explicit)
    common_jobs = build_common_state_jobs(configs, references, common_spec, args.common_state_root)
    jobs = build_functional_intervention_jobs(common_jobs, args.result_root)
    manifest = summarize_functional_interventions(
        jobs,
        args.output_dir,
        intervention_spec=spec_path,
        families=families,
        scope_amendment=args.scope_amendment,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
