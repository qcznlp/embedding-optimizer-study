from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import load_matrix, resolve_matrix_path
from .functional_intervention_summary import _read_jsonl, _write_csv
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .validation_data import load_validation_spec
from .validation_evaluation import METRICS
from .validation_matrix import (
    ValidationJob,
    build_validation_jobs,
    validation_job_complete,
)


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Validation summary encountered empty or non-finite values")
    return statistics.fmean(values)


def _paired_lr_effects(
    jobs: list[ValidationJob],
    samples: dict[str, dict[int, dict[str, Any]]],
    winners: dict[tuple[str, str], ValidationJob],
) -> list[dict[str, Any]]:
    rows = []
    grouped: dict[tuple[str, str], list[ValidationJob]] = defaultdict(list)
    for job in jobs:
        grouped[(job.config.model_family, job.config.optimizer.name)].append(job)
    for key, candidates in sorted(grouped.items()):
        family, optimizer = key
        winner = winners[key]
        reference = samples[winner.label]
        for candidate in sorted(candidates, key=lambda item: item.config.optimizer.lr):
            if candidate.label == winner.label:
                continue
            observed = samples[candidate.label]
            if set(observed) != set(reference):
                raise ValueError(f"Validation samples differ for {candidate.label}")
            row = {
                "family": family,
                "optimizer": optimizer,
                "candidate_run_id": candidate.config.run_id,
                "candidate_lr": candidate.config.optimizer.lr,
                "selected_run_id": winner.config.run_id,
                "selected_lr": winner.config.optimizer.lr,
                "samples": len(reference),
            }
            for metric in METRICS:
                deltas = [
                    float(observed[sample_id][metric]) - float(reference[sample_id][metric])
                    for sample_id in sorted(reference)
                ]
                row[f"candidate_minus_selected_{metric}"] = _mean(deltas)
                row[f"paired_se_{metric}"] = (
                    statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
                )
            rows.append(row)
    return rows


def select_recipes(
    jobs: list[ValidationJob], run_rows: list[dict[str, Any]]
) -> tuple[dict[tuple[str, str], ValidationJob], list[dict[str, Any]]]:
    metric_by_label = {f"{row['family']}/{row['run_id']}": row for row in run_rows}
    grouped_jobs: dict[tuple[str, str], list[ValidationJob]] = defaultdict(list)
    for job in jobs:
        grouped_jobs[(job.config.model_family, job.config.optimizer.name)].append(job)
    if len(grouped_jobs) != 6 or any(len(values) != 4 for values in grouped_jobs.values()):
        raise ValueError("Recipe selection requires four rates per optimizer and family")
    if {job.label for job in jobs} != set(metric_by_label):
        raise ValueError("Recipe validation metrics do not cover the exact job matrix")
    winners: dict[tuple[str, str], ValidationJob] = {}
    selected = []
    for key, candidates in sorted(grouped_jobs.items()):
        winner = min(
            candidates,
            key=lambda job: (
                float(metric_by_label[job.label]["contrastive_loss"]),
                -float(metric_by_label[job.label]["positive_margin"]),
                float(job.config.optimizer.lr),
            ),
        )
        winners[key] = winner
        selected.append(
            {
                "family": winner.config.model_family,
                "optimizer": winner.config.optimizer.name,
                "run_id": winner.config.run_id,
                "learning_rate": winner.config.optimizer.lr,
                "validation_contrastive_loss": metric_by_label[winner.label]["contrastive_loss"],
                "validation_positive_margin": metric_by_label[winner.label]["positive_margin"],
                "optimizer_config": winner.config.as_dict()["optimizer"],
                "model_name": winner.config.model_name,
                "model_revision": winner.config.model_revision,
            }
        )
    return winners, selected


def summarize_validation(
    jobs: list[ValidationJob],
    output_dir: str | Path,
    *,
    validation_spec: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    spec_path, spec = load_validation_spec(validation_spec)
    if len(jobs) != spec["evaluation"]["expected_jobs"]:
        raise ValueError("Recipe selection requires all 24 frozen validation jobs")
    incomplete = [
        job.label for job in jobs if not validation_job_complete(job, spec_path, verify_hashes=True)
    ]
    if incomplete:
        raise ValueError("Incomplete recipe-validation jobs: " + ", ".join(incomplete))

    run_rows = []
    group_rows = []
    samples: dict[str, dict[int, dict[str, Any]]] = {}
    sources = []
    reference_samples: set[tuple[int, str]] | None = None
    for job in jobs:
        manifest_path = job.output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sample_path = job.output_dir / manifest["outputs"]["sample_metrics"]["path"]
        group_path = job.output_dir / manifest["outputs"]["group_metrics"]["path"]
        sample_records = _read_jsonl(sample_path)
        sample_index = {int(row["sample_id"]): row for row in sample_records}
        if len(sample_index) != spec["evaluation"]["expected_sample_records_per_job"]:
            raise ValueError(f"Duplicate or missing validation samples in {job.label}")
        signature = {(sample_id, str(row["group"])) for sample_id, row in sample_index.items()}
        if reference_samples is None:
            reference_samples = signature
        elif signature != reference_samples:
            raise ValueError(f"Validation sample identity differs for {job.label}")
        samples[job.label] = sample_index
        grouped = _read_jsonl(group_path)
        if len(grouped) != 8 or {row["group"] for row in grouped} != {
            "__all__",
            "fever",
            "fiqa",
            "hotpotqa",
            "msmarco",
            "nq",
            "squadv2",
            "trivia",
        }:
            raise ValueError(f"Validation group coverage differs for {job.label}")
        for row in grouped:
            group_rows.append(
                {
                    "family": job.config.model_family,
                    "optimizer": job.config.optimizer.name,
                    "run_id": job.config.run_id,
                    "learning_rate": job.config.optimizer.lr,
                    **{key: row[key] for key in ("group", "samples", *METRICS)},
                }
            )
        overall = next(row for row in grouped if row["group"] == "__all__")
        run_rows.append(
            {
                "family": job.config.model_family,
                "optimizer": job.config.optimizer.name,
                "run_id": job.config.run_id,
                "learning_rate": job.config.optimizer.lr,
                "checkpoint": str(job.checkpoint),
                "samples": overall["samples"],
                **{metric: overall[metric] for metric in METRICS},
            }
        )
        sources.append(
            {
                "label": job.label,
                "manifest": _identity(manifest_path),
                "sample_metrics": _identity(sample_path),
                "group_metrics": _identity(group_path),
            }
        )

    winners, selected = select_recipes(jobs, run_rows)
    effects = _paired_lr_effects(jobs, samples, winners)

    run_path = output_dir / "run_metrics.csv"
    group_path = output_dir / "source_metrics.csv"
    effect_path = output_dir / "paired_lr_effects.csv"
    selection_path = output_dir / "recipe_selection.json"
    _write_csv(run_path, sorted(run_rows, key=lambda row: (row["family"], row["run_id"])))
    _write_csv(
        group_path,
        sorted(group_rows, key=lambda row: (row["family"], row["run_id"], row["group"])),
    )
    _write_csv(
        effect_path,
        sorted(effects, key=lambda row: (row["family"], row["optimizer"], row["candidate_lr"])),
    )
    selection_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "selection_rule": spec["recipe_selection"],
        "validation_spec": {
            "path": str(spec_path),
            "sha256": _sha256(spec_path),
        },
        "selected": selected,
    }
    _atomic_json(selection_path, selection_payload)
    outputs = {
        "run_metrics": _identity(run_path),
        "source_metrics": _identity(group_path),
        "paired_lr_effects": _identity(effect_path),
        "recipe_selection": _identity(selection_path),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "validation_spec": {
            "path": str(spec_path),
            "bytes": spec_path.stat().st_size,
            "sha256": _sha256(spec_path),
        },
        "jobs": len(jobs),
        "samples_per_job": spec["evaluation"]["expected_sample_records_per_job"],
        "selected_recipes": len(selected),
        "paired_lr_effects": len(effects),
        "sources": sources,
        "outputs": outputs,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select confirmatory recipes using only query-disjoint validation metrics"
    )
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--validation-spec", type=Path, default=Path("configs/validation_probe.json")
    )
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/recipe-validation"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    spec_path, spec = load_validation_spec(args.validation_spec)
    configs = load_matrix(resolve_matrix_path(args.matrix).resolve())
    result_root = (
        args.result_root.resolve()
        if args.result_root is not None
        else Path(spec["evaluation"]["output_root"]).resolve()
    )
    jobs = build_validation_jobs(configs, result_root)
    manifest = summarize_validation(jobs, args.output_dir, validation_spec=spec_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
