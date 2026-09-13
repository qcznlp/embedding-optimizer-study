"""Strict outcome summaries for the corrected Dense no-packing experiment."""

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

import numpy as np

from .aggregate import (
    audit_dataset_artifacts,
    audit_training_artifacts,
    collect_evaluations,
    collect_system_metrics,
)
from .config import RunConfig, load_matrix, resolve_matrix_path
from .corrected_beir_evaluation import audit_requested_results
from .corrected_input_execution import PADDED_DENSE_RECEIPT
from .corrected_validation_matrix import build_jobs, job_complete
from .decontamination import DECONTAMINATED_TASK_NAMES
from .geometry import _atomic_json, _sha256
from .validation_data import load_validation_spec

SCHEMA_VERSION = 1
OPTIMIZERS = ("adamw", "muon", "normuon")
CONTRASTS = (("muon", "adamw"), ("normuon", "adamw"), ("normuon", "muon"))


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"Refusing to write empty corrected table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "path": str(path.resolve()),
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_matrix(configs: list[RunConfig]) -> None:
    grouped: dict[str, list[RunConfig]] = defaultdict(list)
    for config in configs:
        grouped[config.optimizer.name].append(config)
    if (
        len(configs) != 12
        or set(grouped) != set(OPTIMIZERS)
        or any(len(grouped[name]) != 4 for name in OPTIMIZERS)
        or any(config.model_family != "dense" for config in configs)
        or any(config.dense_can_flatten_inputs for config in configs)
        or any(len(config.checkpoint_fractions) != 5 for config in configs)
    ):
        raise ValueError("Corrected outcome summary requires the frozen 12-run padded Dense matrix")


def _deep_audit_training(configs: list[RunConfig]) -> dict[str, Any]:
    dataset = audit_dataset_artifacts(configs)
    if not dataset["complete"]:
        raise RuntimeError("Corrected outcome training-data audit failed")
    training = audit_training_artifacts(
        configs,
        deep=True,
        expected_dataset_fingerprint=dataset.get("training_view_fingerprint"),
    )
    if not training["complete"]:
        raise RuntimeError(
            "Corrected outcome training audit failed: " + "; ".join(training["errors"][:5])
        )
    return {"dataset": dataset, "training": training}


def paired_max_t_intervals(
    effects: dict[tuple[str, str], list[float]],
    *,
    samples: int = 50_000,
    seed: int = 20260903,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Return common-resample, fixed-SE max-T intervals over the three contrasts."""

    if tuple(effects) != CONTRASTS or samples < 1:
        raise ValueError("max-T requires the three predeclared ordered contrasts")
    matrix = np.asarray([effects[key] for key in CONTRASTS], dtype=np.float64).T
    if matrix.shape != (len(DECONTAMINATED_TASK_NAMES), len(CONTRASTS)):
        raise ValueError(f"max-T effect matrix has unexpected shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("max-T effect matrix is non-finite")
    points = matrix.mean(axis=0)
    standard_errors = matrix.std(axis=0, ddof=1) / math.sqrt(matrix.shape[0])
    if np.any(standard_errors <= 0) or not np.isfinite(standard_errors).all():
        raise ValueError("max-T requires positive finite across-task standard errors")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, matrix.shape[0], size=(samples, matrix.shape[0]))
    bootstraps = matrix[indices].mean(axis=1)
    statistics_t = np.abs((bootstraps - points) / standard_errors)
    critical = float(np.quantile(statistics_t.max(axis=1), 0.95, method="linear"))
    nominal = np.quantile(bootstraps, (0.025, 0.975), axis=0, method="linear")
    output = {}
    for index, key in enumerate(CONTRASTS):
        lower = float(points[index] - critical * standard_errors[index])
        upper = float(points[index] + critical * standard_errors[index])
        support = "positive" if lower > 0 else "negative" if upper < 0 else "inconclusive"
        output[key] = {
            "mean_delta_ndcg_at_10": float(points[index]),
            "across_task_standard_error": float(standard_errors[index]),
            "nominal_bootstrap_ci_95_lower": float(nominal[0, index]),
            "nominal_bootstrap_ci_95_upper": float(nominal[1, index]),
            "simultaneous_ci_95_lower": lower,
            "simultaneous_ci_95_upper": upper,
            "simultaneous_critical_value": critical,
            "support": support,
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
            "tasks": matrix.shape[0],
        }
    return output


def _index_score_rows(rows: list[dict[str, Any]], configs: list[RunConfig]) -> dict:
    expected = {
        (config.run_id, stage, task)
        for config in configs
        for stage in range(1, 6)
        for task in DECONTAMINATED_TASK_NAMES
    }
    indexed = {}
    config_by_id = {config.run_id: config for config in configs}
    for row in rows:
        run_id = str(row["run_id"])
        stage = int(row["stage"])
        task = str(row["task"])
        identity = (run_id, stage, task)
        value = float(row["ndcg_at_10"])
        config = config_by_id.get(run_id)
        if (
            config is None
            or row.get("model_family") != "dense"
            or row.get("optimizer") != config.optimizer.name
            or float(row.get("learning_rate")) != config.optimizer.lr
            or identity in indexed
            or not math.isfinite(value)
        ):
            raise ValueError(f"Invalid corrected evaluation row: {identity}")
        indexed[identity] = {**row, "ndcg_at_10": value}
    if set(indexed) != expected:
        raise ValueError(
            f"Corrected score coverage differs: missing={len(expected - set(indexed))}, "
            f"unexpected={len(set(indexed) - expected)}"
        )
    return indexed


def _task_effects(
    indexed: dict,
    configs: list[RunConfig],
    selections: dict[str, str] | None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[float]]]:
    by_optimizer = {
        optimizer: [config for config in configs if config.optimizer.name == optimizer]
        for optimizer in OPTIMIZERS
    }
    task_rows = []
    effects = {contrast: [] for contrast in CONTRASTS}
    for task in DECONTAMINATED_TASK_NAMES:
        values = {}
        for optimizer in OPTIMIZERS:
            candidates = by_optimizer[optimizer]
            if selections is not None:
                candidates = [
                    config for config in candidates if config.run_id == selections[optimizer]
                ]
                if len(candidates) != 1:
                    raise ValueError(f"Invalid selected recipe for {optimizer}")
            values[optimizer] = statistics.fmean(
                indexed[(config.run_id, 5, task)]["ndcg_at_10"] for config in candidates
            )
        row = {"task": task, **{f"{name}_ndcg_at_10": values[name] for name in OPTIMIZERS}}
        for treatment, baseline in CONTRASTS:
            delta = values[treatment] - values[baseline]
            row[f"{treatment}_minus_{baseline}"] = delta
            effects[(treatment, baseline)].append(delta)
        task_rows.append(row)
    return task_rows, effects


def summarize_score_rows(
    rows: list[dict[str, Any]],
    configs: list[RunConfig],
    selections: dict[str, str],
    *,
    bootstrap_samples: int = 50_000,
    bootstrap_seed: int = 20260903,
) -> dict[str, list[dict[str, Any]]]:
    """Build the locked primary, secondary, and five-stage score tables."""

    _validate_matrix(configs)
    indexed = _index_score_rows(rows, configs)
    primary_tasks, primary_effects = _task_effects(indexed, configs, None)
    secondary_tasks, secondary_effects = _task_effects(indexed, configs, selections)
    primary_intervals = paired_max_t_intervals(
        primary_effects, samples=bootstrap_samples, seed=bootstrap_seed
    )
    secondary_intervals = paired_max_t_intervals(
        secondary_effects, samples=bootstrap_samples, seed=bootstrap_seed
    )
    primary = [
        {"treatment": treatment, "baseline": baseline, **primary_intervals[(treatment, baseline)]}
        for treatment, baseline in CONTRASTS
    ]
    secondary = [
        {
            "treatment": treatment,
            "baseline": baseline,
            "treatment_run_id": selections[treatment],
            "baseline_run_id": selections[baseline],
            **secondary_intervals[(treatment, baseline)],
        }
        for treatment, baseline in CONTRASTS
    ]
    run_stage = []
    for config in configs:
        for stage in range(1, 6):
            values = [
                indexed[(config.run_id, stage, task)]["ndcg_at_10"]
                for task in DECONTAMINATED_TASK_NAMES
            ]
            run_stage.append(
                {
                    "run_id": config.run_id,
                    "optimizer": config.optimizer.name,
                    "learning_rate": config.optimizer.lr,
                    "stage": stage,
                    "progress_fraction": stage / 5,
                    "tasks": len(values),
                    "mean_ndcg_at_10": statistics.fmean(values),
                    "median_ndcg_at_10": statistics.median(values),
                }
            )
    optimizer_stage = []
    for optimizer in OPTIMIZERS:
        for stage in range(1, 6):
            members = [
                row for row in run_stage if row["optimizer"] == optimizer and row["stage"] == stage
            ]
            optimizer_stage.append(
                {
                    "optimizer": optimizer,
                    "stage": stage,
                    "progress_fraction": stage / 5,
                    "learning_rates": len(members),
                    "mean_ndcg_at_10_across_rates": statistics.fmean(
                        float(row["mean_ndcg_at_10"]) for row in members
                    ),
                    "median_ndcg_at_10_across_rates": statistics.median(
                        float(row["mean_ndcg_at_10"]) for row in members
                    ),
                }
            )
    run_auc = []
    for config in configs:
        curve = sorted(
            (row for row in run_stage if row["run_id"] == config.run_id),
            key=lambda row: int(row["stage"]),
        )
        area = sum(
            (float(right["progress_fraction"]) - float(left["progress_fraction"]))
            * (float(left["mean_ndcg_at_10"]) + float(right["mean_ndcg_at_10"]))
            / 2
            for left, right in zip(curve, curve[1:])
        )
        run_auc.append(
            {
                "run_id": config.run_id,
                "optimizer": config.optimizer.name,
                "learning_rate": config.optimizer.lr,
                "observed_auc_20_to_100": area,
                "observed_mean_20_to_100": area / 0.8,
            }
        )
    return {
        "primary_task_effects": primary_tasks,
        "primary_summary": primary,
        "secondary_task_effects": secondary_tasks,
        "secondary_summary": secondary,
        "run_stage_scores": run_stage,
        "optimizer_stage_scores": optimizer_stage,
        "run_observed_auc": run_auc,
    }


def _validation_selection(
    configs: list[RunConfig],
    validation_root: Path,
    validation_spec: Path,
) -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]]]:
    spec_path, _ = load_validation_spec(validation_spec)
    jobs = build_jobs(configs, validation_root)
    incomplete = [job.label for job in jobs if not job_complete(job, spec_path, verify_hashes=True)]
    if incomplete:
        raise RuntimeError(f"Corrected validation is incomplete: {incomplete}")
    run_rows = []
    sources = []
    for job in jobs:
        manifest_path = job.output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_execution") != PADDED_DENSE_RECEIPT:
            raise ValueError(f"Validation is not independently padded: {manifest_path}")
        group_identity = manifest["outputs"]["group_metrics"]
        group_path = job.output_dir / group_identity["path"]
        if _sha256(group_path) != group_identity["sha256"]:
            raise ValueError(f"Validation group metrics differ from manifest: {group_path}")
        groups = [json.loads(line) for line in group_path.read_text().splitlines() if line.strip()]
        overall = [row for row in groups if row["group"] == "__all__"]
        if len(overall) != 1:
            raise ValueError(f"Validation has no unique overall row: {group_path}")
        run_rows.append(
            {
                "optimizer": job.config.optimizer.name,
                "run_id": job.config.run_id,
                "learning_rate": job.config.optimizer.lr,
                "contrastive_loss": float(overall[0]["contrastive_loss"]),
                "positive_margin": float(overall[0]["positive_margin"]),
            }
        )
        sources.append(
            {
                "run_id": job.config.run_id,
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": _sha256(manifest_path),
                "group_metrics_path": str(group_path.resolve()),
                "group_metrics_sha256": _sha256(group_path),
            }
        )
    selected = {}
    for optimizer in OPTIMIZERS:
        candidates = [row for row in run_rows if row["optimizer"] == optimizer]
        winner = min(
            candidates,
            key=lambda row: (float(row["contrastive_loss"]), float(row["learning_rate"])),
        )
        selected[optimizer] = str(winner["run_id"])
    return selected, run_rows, sources


def _load_outcome_protocol(path: Path, repository: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "prospective_corrected_outcome_implementation_lock":
        raise ValueError(f"Unexpected corrected outcome protocol status: {path}")
    for group in ("parent_bindings", "source_bindings"):
        for identity in payload[group].values():
            source = repository / identity["path"]
            if (
                not source.is_file()
                or _sha256(source) != identity["sha256"]
                or ("bytes" in identity and source.stat().st_size != int(identity["bytes"]))
            ):
                raise ValueError(f"Corrected outcome {group} mismatch: {source}")
    return payload


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    protocol_path = args.protocol.resolve()
    protocol = _load_outcome_protocol(protocol_path, repository)
    matrix_path = resolve_matrix_path(args.matrix).resolve()
    if _sha256(matrix_path) != protocol["parent_bindings"]["matrix"]["sha256"]:
        raise ValueError("Corrected outcome matrix differs from protocol")
    configs = load_matrix(matrix_path)
    _validate_matrix(configs)
    training_audit = _deep_audit_training(configs)
    checkpoints = [
        config.output_dir / f"checkpoint-{step}"
        for config in configs
        for step in json.loads(
            (config.output_dir / "checkpoint_schedule.json").read_text(encoding="utf-8")
        )["steps"]
    ]
    evaluation_audit = audit_requested_results(args.results_root.resolve(), checkpoints)
    if evaluation_audit["task_units"] != 840:
        raise RuntimeError("Corrected BEIR audit did not produce 840 task units")
    rows = collect_evaluations(args.results_root.resolve(), configs)
    selections, validation_rows, validation_sources = _validation_selection(
        configs, args.validation_root.resolve(), args.validation_spec.resolve()
    )
    tables = summarize_score_rows(
        rows,
        configs,
        selections,
        bootstrap_samples=int(protocol["inference"]["bootstrap_samples"]),
        bootstrap_seed=int(protocol["inference"]["bootstrap_seed"]),
    )
    output_dir = args.output_dir.resolve()
    outputs = {
        name: _atomic_csv(output_dir / f"{name}.csv", values) for name, values in tables.items()
    }
    outputs["validation_run_metrics"] = _atomic_csv(
        output_dir / "validation_run_metrics.csv", validation_rows
    )
    system_rows = collect_system_metrics(configs)
    if len(system_rows) != 12:
        raise RuntimeError(f"Expected 12 system rows, found {len(system_rows)}")
    outputs["system_metrics"] = _atomic_csv(output_dir / "system_metrics.csv", system_rows)
    selection_path = output_dir / "validation_recipe_selection.json"
    _atomic_json(
        selection_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "rule": (
                "minimum independently padded mean validation contrastive loss within optimizer; "
                "ties choose lower learning rate"
            ),
            "selected_run_ids": selections,
            "sources": validation_sources,
        },
    )
    outputs["validation_recipe_selection"] = {
        "path": str(selection_path),
        "bytes": selection_path.stat().st_size,
        "sha256": _sha256(selection_path),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": "corrected_dense_no_packing",
        "primary_estimand": protocol["inference"]["primary_estimand"],
        "secondary_estimand": protocol["inference"]["secondary_estimand"],
        "claim_boundary": protocol["claim_boundary"],
        "coverage": {
            "runs": 12,
            "checkpoints": 60,
            "tasks": 14,
            "task_units": 840,
        },
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256(protocol_path),
        },
        "training_audit": training_audit,
        "evaluation_audit": evaluation_audit,
        "outputs": outputs,
    }
    _atomic_json(output_dir / "summary_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/dense_no_packing_outcome_protocol.json"),
    )
    parser.add_argument(
        "--matrix", type=Path, default=Path("configs/dense_no_packing_retrain.yaml")
    )
    parser.add_argument("--results-root", type=Path, default=Path("results/dense-no-packing-beir"))
    parser.add_argument(
        "--validation-root", type=Path, default=Path("results/dense-no-packing-validation")
    )
    parser.add_argument(
        "--validation-spec", type=Path, default=Path("configs/validation_probe.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/dense-no-packing-outcomes")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(build_report(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
