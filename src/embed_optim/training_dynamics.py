from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .aggregate import collect_system_metrics, collect_training_history
from .config import RunConfig, load_matrix, resolve_matrix_path
from .confirmatory_summary import _atomic_csv
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["model_family"]), str(row["run_id"])


def stage_training_dynamics(
    configs: list[RunConfig], history_rows: list[dict[str, Any]], *, trailing_observations: int = 10
) -> list[dict[str, Any]]:
    if trailing_observations < 1:
        raise ValueError("trailing_observations must be positive")
    histories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in history_rows:
        histories[_identity(row)].append(row)
    output = []
    for config in configs:
        identity = (config.model_family, config.run_id)
        rows = sorted(histories.get(identity, []), key=lambda row: int(row["step"]))
        if (
            len(rows) != 391
            or len({int(row["step"]) for row in rows}) != 391
            or int(rows[-1]["step"]) != 3900
        ):
            raise ValueError(f"{identity}: expected 391 unique loss records through step 3900")
        schedule_path = config.output_dir / "checkpoint_schedule.json"
        steps = [
            int(step) for step in json.loads(schedule_path.read_text(encoding="utf-8"))["steps"]
        ]
        if len(steps) != 5 or steps != sorted(set(steps)):
            raise ValueError(f"{identity}: checkpoint schedule differs")
        for stage, checkpoint_step in enumerate(steps, start=1):
            available = [row for row in rows if int(row["step"]) <= checkpoint_step]
            window = available[-trailing_observations:]
            values = {
                "loss": [float(row["loss"]) for row in window],
                "grad_norm": [float(row["grad_norm"]) for row in window],
                "learning_rate": [float(row["learning_rate"]) for row in window],
                "epoch": [float(row["epoch"]) for row in window],
            }
            if len(window) != trailing_observations or not all(
                math.isfinite(value) for metric in values.values() for value in metric
            ):
                raise ValueError(f"{identity}/stage-{stage}: invalid trailing training window")
            output.append(
                {
                    "model_family": config.model_family,
                    "optimizer": config.optimizer.name,
                    "learning_rate": config.optimizer.lr,
                    "run_id": config.run_id,
                    "stage": stage,
                    "fraction": float(config.checkpoint_fractions[stage - 1]),
                    "checkpoint_step": checkpoint_step,
                    "observed_step": int(window[-1]["step"]),
                    "window_start_step": int(window[0]["step"]),
                    "window_observations": len(window),
                    "mean_loss": statistics.mean(values["loss"]),
                    "loss_standard_deviation": statistics.stdev(values["loss"]),
                    "median_grad_norm": statistics.median(values["grad_norm"]),
                    "end_learning_rate": values["learning_rate"][-1],
                    "end_epoch": values["epoch"][-1],
                }
            )
    if len(output) != 120:
        raise ValueError("Training-stage summary requires exactly 120 run/checkpoint rows")
    return output


def optimizer_system_summary(system_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {_identity(row): row for row in system_rows}
    if len(system_rows) != 24 or len(indexed) != 24:
        raise ValueError("System summary requires exactly 24 unique training runs")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in system_rows:
        grouped[(str(row["model_family"]), str(row["optimizer"]))].append(row)
    expected = {
        (family, optimizer)
        for family in ("dense", "late")
        for optimizer in ("adamw", "muon", "normuon")
    }
    if set(grouped) != expected or any(len(rows) != 4 for rows in grouped.values()):
        raise ValueError("System summary requires four learning-rate runs per family and optimizer")
    medians = {
        identity: {
            field: statistics.median(float(row[field]) for row in rows)
            for field in (
                "wall_time_hours",
                "samples_per_second",
                "steps_per_second",
                "peak_allocated_gib",
                "peak_reserved_gib",
                "checkpoint_gib",
                "optimizer_state_gib",
            )
        }
        for identity, rows in grouped.items()
    }
    output = []
    for identity, rows in sorted(grouped.items()):
        family, optimizer = identity
        baseline = medians[(family, "adamw")]
        summary = medians[identity]
        output.append(
            {
                "model_family": family,
                "optimizer": optimizer,
                "learning_rate_points": len(rows),
                "wall_time_hours_median": summary["wall_time_hours"],
                "wall_time_hours_min": min(float(row["wall_time_hours"]) for row in rows),
                "wall_time_hours_max": max(float(row["wall_time_hours"]) for row in rows),
                "samples_per_second_median": summary["samples_per_second"],
                "samples_per_second_min": min(float(row["samples_per_second"]) for row in rows),
                "samples_per_second_max": max(float(row["samples_per_second"]) for row in rows),
                "throughput_to_adamw_ratio": summary["samples_per_second"]
                / baseline["samples_per_second"],
                "steps_per_second_median": summary["steps_per_second"],
                "peak_allocated_gib_median": summary["peak_allocated_gib"],
                "peak_reserved_gib_median": summary["peak_reserved_gib"],
                "checkpoint_gib_median": summary["checkpoint_gib"],
                "optimizer_state_gib_median": summary["optimizer_state_gib"],
                "optimizer_state_to_adamw_ratio": summary["optimizer_state_gib"]
                / baseline["optimizer_state_gib"],
            }
        )
    return output


def _run_summary(
    system_rows: list[dict[str, Any]], stage_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    systems = {_identity(row): row for row in system_rows}
    final = {_identity(row): row for row in stage_rows if int(row["stage"]) == 5}
    if len(systems) != 24 or set(systems) != set(final):
        raise ValueError("System metrics and final-stage training dynamics do not align")
    return [
        {
            **systems[identity],
            "final_trailing_mean_loss": final[identity]["mean_loss"],
            "final_trailing_loss_standard_deviation": final[identity]["loss_standard_deviation"],
            "final_trailing_median_grad_norm": final[identity]["median_grad_norm"],
            "final_observed_step": final[identity]["observed_step"],
        }
        for identity in sorted(systems)
    ]


def _portable_path(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repository_root.resolve()))
    except ValueError:
        return str(resolved)


def _source_records(configs: list[RunConfig], repository_root: Path) -> list[dict[str, Any]]:
    records = []
    for config in configs:
        schedule = config.output_dir / "checkpoint_schedule.json"
        completion = config.output_dir / "completed.json"
        state = config.output_dir / "trainer_state_final.json"
        if not state.is_file():
            steps = json.loads(schedule.read_text(encoding="utf-8"))["steps"]
            state = config.output_dir / f"checkpoint-{int(steps[-1])}" / "trainer_state.json"
        inputs = (schedule, completion, state)
        if not all(path.is_file() for path in inputs):
            raise ValueError(f"{config.model_family}/{config.run_id}: training source is missing")
        payload = json.loads(completion.read_text(encoding="utf-8"))
        if (
            payload.get("model_family") != config.model_family
            or payload.get("run_id") != config.run_id
            or payload.get("global_step") != 3907
            or payload.get("dataset_rows") != 500_000
            or len(payload.get("checkpoints", [])) != 5
        ):
            raise ValueError(f"{config.model_family}/{config.run_id}: completion contract differs")
        records.append(
            {
                "model_family": config.model_family,
                "run_id": config.run_id,
                "inputs": [
                    {
                        "path": _portable_path(path, repository_root),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in inputs
                ],
            }
        )
    return records


def build_training_report(
    matrix: str | Path = "configs/experiment.yaml",
    output_dir: str | Path = "reports/training-dynamics",
    *,
    trailing_observations: int = 10,
) -> dict[str, Any]:
    matrix_path = resolve_matrix_path(matrix).resolve()
    repository_root = matrix_path.parent.parent
    configs = load_matrix(matrix_path)
    if len(configs) != 24:
        raise ValueError("Training report requires the complete 24-run discovery matrix")
    sources = _source_records(configs, repository_root)
    systems = collect_system_metrics(configs)
    histories = collect_training_history(configs)
    stages = stage_training_dynamics(
        configs, histories, trailing_observations=trailing_observations
    )
    runs = _run_summary(systems, stages)
    optimizer_summary = optimizer_system_summary(systems)
    output = Path(output_dir).resolve()
    tables = {
        "runs": _atomic_csv(output / "run_summary.csv", runs),
        "stages": _atomic_csv(output / "stage_dynamics.csv", stages),
        "optimizer_systems": _atomic_csv(
            output / "optimizer_system_summary.csv", optimizer_summary
        ),
    }
    for record in tables.values():
        record["path"] = _portable_path(Path(record["path"]), repository_root)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "matrix": {
            "path": _portable_path(matrix_path, repository_root),
            "sha256": _sha256(matrix_path),
        },
        "coverage": {
            "runs": len(runs),
            "checkpoints": len(stages),
            "optimizer_family_groups": len(optimizer_summary),
            "history_rows": len(histories),
            "trailing_observations": trailing_observations,
        },
        "sources": sources,
        "outputs": tables,
        "interpretation": (
            "The four learning rates are sweep points, not independent random seeds. Throughput "
            "and memory summaries are descriptive matched-hardware measurements; retrieval "
            "time-to-quality requires the complete checkpoint evaluation matrix."
        ),
    }
    _atomic_json(output / "summary_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly summarize complete discovery training and system dynamics"
    )
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/training-dynamics"))
    parser.add_argument("--trailing-observations", type=int, default=10)
    args = parser.parse_args(argv)
    if args.trailing_observations < 1:
        parser.error("--trailing-observations must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_training_report(
        args.matrix,
        args.output_dir,
        trailing_observations=args.trailing_observations,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
