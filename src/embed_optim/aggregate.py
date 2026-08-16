"""Aggregate checkpoint-level MTEB results and render the final study report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from .config import RunConfig, load_matrix
from .decontamination import DECONTAMINATED_TASK_NAMES

CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")
RESULTS_MARKERS = ("<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->")
SYSTEMS_MARKERS = ("<!-- SYSTEMS:BEGIN -->", "<!-- SYSTEMS:END -->")


def audit_training_artifacts(configs: list[RunConfig]) -> dict:
    """Verify that every planned run has five complete, resumable checkpoints."""

    errors: list[str] = []
    verified_runs = 0
    verified_checkpoints = 0
    for config in configs:
        label = f"{config.model_family}/{config.run_id}"
        output = config.output_dir
        schedule_path = output / "checkpoint_schedule.json"
        completed_path = output / "completed.json"
        final_state_path = output / "trainer_state_final.json"
        final_model_path = output / "final"
        if not schedule_path.is_file():
            errors.append(f"{label}: missing checkpoint_schedule.json")
            continue
        try:
            schedule = json.loads(schedule_path.read_text())
            steps = [int(step) for step in schedule["steps"]]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{label}: invalid checkpoint schedule ({error})")
            continue
        if len(steps) != 5 or steps != sorted(set(steps)):
            errors.append(
                f"{label}: expected five strictly increasing checkpoint steps, got {steps}"
            )
            continue
        completed: dict = {}
        if not completed_path.is_file():
            errors.append(f"{label}: missing completed.json")
        else:
            try:
                completed = json.loads(completed_path.read_text())
            except json.JSONDecodeError as error:
                errors.append(f"{label}: invalid completed.json ({error})")
            if completed and int(completed.get("global_step", -1)) != steps[-1]:
                errors.append(f"{label}: completion global_step does not match final checkpoint")
            if (
                completed
                and sorted(int(step) for step in completed.get("checkpoints", [])) != steps
            ):
                errors.append(f"{label}: completion checkpoint list does not match schedule")
        if not final_state_path.is_file():
            errors.append(f"{label}: missing trainer_state_final.json")
        else:
            try:
                final_step = int(json.loads(final_state_path.read_text())["global_step"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"{label}: invalid final Trainer state ({error})")
            else:
                if final_step != steps[-1]:
                    errors.append(f"{label}: final Trainer state step {final_step} != {steps[-1]}")
        if not final_model_path.is_dir() or not any(
            path.stat().st_size > 0 for path in final_model_path.rglob("*.safetensors")
        ):
            errors.append(f"{label}: missing final safetensors model")

        world_size = int(completed.get("system_metrics", {}).get("world_size", 4))
        run_checkpoint_errors = 0
        for step in steps:
            checkpoint = output / f"checkpoint-{step}"
            required = (
                checkpoint / "config.json",
                checkpoint / "optimizer.pt",
                checkpoint / "scheduler.pt",
                checkpoint / "trainer_state.json",
                checkpoint / "training_args.bin",
            )
            missing = [
                path.name for path in required if not path.is_file() or path.stat().st_size == 0
            ]
            if missing:
                errors.append(f"{label}/checkpoint-{step}: missing/empty {', '.join(missing)}")
                run_checkpoint_errors += 1
                continue
            if not any(path.stat().st_size > 0 for path in checkpoint.rglob("*.safetensors")):
                errors.append(f"{label}/checkpoint-{step}: missing/empty safetensors model")
                run_checkpoint_errors += 1
                continue
            rng_states = sorted(checkpoint.glob("rng_state_*.pth"))
            if len(rng_states) != world_size or any(
                path.stat().st_size == 0 for path in rng_states
            ):
                errors.append(
                    f"{label}/checkpoint-{step}: expected {world_size} non-empty rank RNG states, "
                    f"found {len(rng_states)}"
                )
                run_checkpoint_errors += 1
                continue
            try:
                checkpoint_step = int(json.loads(required[3].read_text())["global_step"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"{label}/checkpoint-{step}: invalid Trainer state ({error})")
                run_checkpoint_errors += 1
                continue
            if checkpoint_step != step:
                errors.append(f"{label}/checkpoint-{step}: Trainer state step is {checkpoint_step}")
                run_checkpoint_errors += 1
                continue
            verified_checkpoints += 1
        if run_checkpoint_errors == 0 and not any(
            error.startswith(f"{label}:") for error in errors
        ):
            verified_runs += 1

    return {
        "complete": not errors,
        "verified_runs": verified_runs,
        "expected_runs": len(configs),
        "verified_checkpoints": verified_checkpoints,
        "expected_checkpoints": len(configs) * 5,
        "errors": errors,
    }


def _contains_run_id(path: Path, run_id: str) -> bool:
    """Match a run directory exactly, avoiding muon/normuon substring collisions."""

    return any(part == run_id or part.startswith(f"{run_id}__checkpoint-") for part in path.parts)


def _run_for_result(path: Path, configs: list[RunConfig]) -> RunConfig | None:
    matches = [
        config
        for config in configs
        if config.model_family in path.parts and _contains_run_id(path, config.run_id)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _base_task_name(name: str) -> str:
    suffix = "Decontaminated"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def collect_evaluations(results_root: Path, configs: list[RunConfig]) -> list[dict]:
    indexed: dict[tuple, dict] = {}
    for path in results_root.rglob("*Decontaminated.json"):
        config = _run_for_result(path, configs)
        match = CHECKPOINT_PATTERN.search(str(path))
        if config is None or match is None:
            continue
        step = int(match.group(1))
        schedule_path = config.output_dir / "checkpoint_schedule.json"
        if not schedule_path.is_file():
            continue
        steps = sorted(json.loads(schedule_path.read_text())["steps"])
        if step not in steps:
            continue
        payload = json.loads(path.read_text())
        split_rows = [item for values in payload["scores"].values() for item in values]
        scores = [float(item.get("ndcg_at_10", item["main_score"])) for item in split_rows]
        if not scores or not all(math.isfinite(score) for score in scores):
            raise ValueError(f"Missing/non-finite nDCG@10 in {path}")
        task = _base_task_name(payload["task_name"])
        row = {
            "model_family": config.model_family,
            "optimizer": config.optimizer.name,
            "learning_rate": config.optimizer.lr,
            "aux_learning_rate": config.optimizer.aux_lr,
            "run_id": config.run_id,
            "stage": steps.index(step) + 1,
            "fraction": (steps.index(step) + 1) / 5,
            "checkpoint_step": step,
            "task": task,
            "ndcg_at_10": statistics.mean(scores),
            "subsets": len(scores),
            "result_path": str(path),
        }
        identity = (config.model_family, config.run_id, step, task)
        previous = indexed.get(identity)
        if previous and previous["ndcg_at_10"] != row["ndcg_at_10"]:
            raise ValueError(f"Conflicting duplicate evaluation for {identity}")
        indexed[identity] = row
    return sorted(indexed.values(), key=lambda row: tuple(str(value) for value in row.values()))


def collect_training_history(configs: list[RunConfig]) -> list[dict]:
    rows: list[dict] = []
    for config in configs:
        schedule_path = config.output_dir / "checkpoint_schedule.json"
        if not schedule_path.is_file():
            continue
        steps = sorted(json.loads(schedule_path.read_text())["steps"])
        state_path = config.output_dir / "trainer_state_final.json"
        if not state_path.is_file():
            state_path = config.output_dir / f"checkpoint-{steps[-1]}" / "trainer_state.json"
        if not state_path.is_file():
            continue
        for item in json.loads(state_path.read_text()).get("log_history", []):
            if "loss" not in item:
                continue
            rows.append(
                {
                    "model_family": config.model_family,
                    "optimizer": config.optimizer.name,
                    "learning_rate_config": config.optimizer.lr,
                    "run_id": config.run_id,
                    **item,
                }
            )
    return rows


def collect_system_metrics(configs: list[RunConfig]) -> list[dict]:
    rows = []
    for config in configs:
        path = config.output_dir / "completed.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        metrics = payload.get("system_metrics", {})
        adjustment_path = config.output_dir / "timing_adjustment.json"
        adjustment = json.loads(adjustment_path.read_text()) if adjustment_path.is_file() else {}
        segment_wall_time = metrics.get("wall_time_seconds_max_rank", 0)
        prior_wall_time = adjustment.get("prior_training_wall_time_seconds", 0)
        total_wall_time = segment_wall_time + prior_wall_time
        trainer = metrics.get("trainer", {})
        checkpoint_sizes = metrics.get("checkpoint_bytes", {})
        state_sizes = metrics.get("optimizer_state_bytes", {})
        rows.append(
            {
                "model_family": config.model_family,
                "optimizer": config.optimizer.name,
                "learning_rate": config.optimizer.lr,
                "run_id": config.run_id,
                "wall_time_hours": total_wall_time / 3600,
                "recorded_segment_wall_time_hours": segment_wall_time / 3600,
                "prior_training_wall_time_hours": prior_wall_time / 3600,
                "timing_adjustment_path": str(adjustment_path) if adjustment else None,
                "samples_per_second": payload.get("dataset_rows", 0) / total_wall_time
                if total_wall_time
                else None,
                "steps_per_second": payload.get("global_step", 0) / total_wall_time
                if total_wall_time
                else None,
                "trainer_reported_samples_per_second": trainer.get("train_samples_per_second"),
                "trainer_reported_steps_per_second": trainer.get("train_steps_per_second"),
                "peak_allocated_gib": metrics.get("peak_allocated_bytes_max_rank", 0) / 2**30,
                "peak_reserved_gib": metrics.get("peak_reserved_bytes_max_rank", 0) / 2**30,
                "checkpoint_gib": max(checkpoint_sizes.values(), default=0) / 2**30,
                "optimizer_state_gib": max(state_sizes.values(), default=0) / 2**30,
                "gpu_name": metrics.get("gpu_name"),
                "world_size": metrics.get("world_size"),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_summaries(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["model_family"],
            row["optimizer"],
            row["learning_rate"],
            row["run_id"],
            row["stage"],
            row["fraction"],
            row["checkpoint_step"],
        )
        groups[key].append(row)
    output = []
    for key, values in sorted(groups.items()):
        output.append(
            {
                "model_family": key[0],
                "optimizer": key[1],
                "learning_rate": key[2],
                "run_id": key[3],
                "stage": key[4],
                "fraction": key[5],
                "checkpoint_step": key[6],
                "mean_ndcg_at_10": statistics.mean(row["ndcg_at_10"] for row in values),
                "tasks_completed": len(values),
            }
        )
    return output


def _optimizer_summaries(summary: list[dict]) -> tuple[list[dict], list[dict]]:
    final = [row for row in summary if row["stage"] == 5]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in final:
        grouped[(row["model_family"], row["optimizer"])].append(row)

    optimizer_rows, best_dynamics = [], []
    for (family, optimizer), values in sorted(grouped.items()):
        values = sorted(values, key=lambda row: row["learning_rate"])
        scores = [row["mean_ndcg_at_10"] for row in values]
        best = max(values, key=lambda row: row["mean_ndcg_at_10"])
        curves = [row for row in summary if row["run_id"] == best["run_id"]]
        curves = sorted(curves, key=lambda row: row["stage"])
        best_dynamics.extend(curves)
        optimizer_rows.append(
            {
                "model_family": family,
                "optimizer": optimizer,
                "configurations": len(values),
                "best_run_id": best["run_id"],
                "best_learning_rate": best["learning_rate"],
                "best_final_ndcg_at_10": best["mean_ndcg_at_10"],
                "final_mean_across_lrs": statistics.mean(scores),
                "final_median_across_lrs": statistics.median(scores),
                "final_population_std_across_lrs": statistics.pstdev(scores),
                "final_min_across_lrs": min(scores),
                "final_max_across_lrs": max(scores),
                "best_config_mean_five_stage_ndcg_at_10": statistics.mean(
                    row["mean_ndcg_at_10"] for row in curves
                ),
            }
        )
    return optimizer_rows, best_dynamics


def _task_comparison(rows: list[dict], optimizer_rows: list[dict]) -> list[dict]:
    best_runs = {
        (row["model_family"], row["optimizer"]): row["best_run_id"] for row in optimizer_rows
    }
    lookup = {
        (row["model_family"], row["run_id"], row["stage"], row["task"]): row["ndcg_at_10"]
        for row in rows
    }
    output = []
    for family in ("dense", "late"):
        for task in DECONTAMINATED_TASK_NAMES:
            values = {
                optimizer: lookup[(family, best_runs[(family, optimizer)], 5, task)]
                for optimizer in ("adamw", "muon", "normuon")
            }
            output.append(
                {
                    "model_family": family,
                    "task": task,
                    **values,
                    "muon_minus_adamw": values["muon"] - values["adamw"],
                    "normuon_minus_adamw": values["normuon"] - values["adamw"],
                }
            )
    return output


def _system_summaries(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_family"], row["optimizer"])].append(row)
    output = []
    for (family, optimizer), values in sorted(grouped.items()):
        numeric = (
            "wall_time_hours",
            "samples_per_second",
            "steps_per_second",
            "peak_allocated_gib",
            "peak_reserved_gib",
            "checkpoint_gib",
            "optimizer_state_gib",
        )
        output.append(
            {
                "model_family": family,
                "optimizer": optimizer,
                "runs": len(values),
                **{
                    f"median_{key}": statistics.median(
                        row[key] for row in values if row[key] is not None
                    )
                    for key in numeric
                },
                "gpu_name": values[0]["gpu_name"],
                "world_size": values[0]["world_size"],
            }
        )
    return output


def _plot(summary: list[dict], output_dir: Path) -> None:
    if not summary:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame(summary)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for family in sorted(frame.model_family.unique()):
        subset = frame[frame.model_family == family]
        fig, axis = plt.subplots(figsize=(8, 5))
        for optimizer, values in subset.groupby("optimizer"):
            grouped = values.groupby("fraction")["mean_ndcg_at_10"]
            means = grouped.mean()
            std = grouped.std(ddof=0).fillna(0)
            axis.plot(means.index, means, marker="o", label=optimizer)
            axis.fill_between(means.index, means - std, means + std, alpha=0.15)
        axis.set(xlabel="Training fraction", ylabel="Mean decontaminated BEIR nDCG@10")
        axis.set_title(f"{family.capitalize()} training dynamics (mean ± LR-config SD)")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"{family}-training-dynamics.png", dpi=180)
        plt.close(fig)

        final = subset[subset.stage == 5]
        fig, axis = plt.subplots(figsize=(8, 5))
        for optimizer, values in final.groupby("optimizer"):
            ordered = values.sort_values("learning_rate")
            axis.semilogx(
                ordered.learning_rate,
                ordered.mean_ndcg_at_10,
                marker="o",
                label=optimizer,
            )
        axis.set(xlabel="Hidden-matrix learning rate", ylabel="Final mean nDCG@10")
        axis.set_title(f"{family.capitalize()} learning-rate sensitivity")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"{family}-lr-sensitivity.png", dpi=180)
        plt.close(fig)


def _markdown_table(headers: list[str], values: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if i < 2 else "---:" for i in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in values)
    return "\n".join(lines)


def _format_lr(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def _render_results(
    optimizer_rows: list[dict], best_dynamics: list[dict], task_rows: list[dict]
) -> str:
    final_table = _markdown_table(
        ["Family", "Optimizer", "Best LR", "Best final", "4-LR mean", "SD", "Range"],
        [
            [
                row["model_family"],
                row["optimizer"],
                _format_lr(row["best_learning_rate"]),
                f"{row['best_final_ndcg_at_10']:.4f}",
                f"{row['final_mean_across_lrs']:.4f}",
                f"{row['final_population_std_across_lrs']:.4f}",
                f"{row['final_min_across_lrs']:.4f}–{row['final_max_across_lrs']:.4f}",
            ]
            for row in optimizer_rows
        ],
    )

    dynamics_lookup = defaultdict(dict)
    for row in best_dynamics:
        dynamics_lookup[(row["model_family"], row["optimizer"])][row["stage"]] = row[
            "mean_ndcg_at_10"
        ]
    dynamics_table = _markdown_table(
        ["Family", "Optimizer", "20%", "40%", "60%", "80%", "100%"],
        [
            [family, optimizer, *[f"{stages[stage]:.4f}" for stage in range(1, 6)]]
            for (family, optimizer), stages in sorted(dynamics_lookup.items())
        ],
    )

    winners = []
    for family in ("dense", "late"):
        candidates = [row for row in optimizer_rows if row["model_family"] == family]
        best_tuned = max(candidates, key=lambda row: row["best_final_ndcg_at_10"])
        robust = max(candidates, key=lambda row: row["final_mean_across_lrs"])
        winners.append(
            f"- **{family.capitalize()}:** best tuned final score is "
            f"{best_tuned['optimizer']} at {_format_lr(best_tuned['best_learning_rate'])} "
            f"({best_tuned['best_final_ndcg_at_10']:.4f}); the highest four-LR mean is "
            f"{robust['optimizer']} ({robust['final_mean_across_lrs']:.4f})."
        )

    per_task_sections = []
    for family in ("dense", "late"):
        values = [row for row in task_rows if row["model_family"] == family]
        per_task_sections.append(f"#### {family.capitalize()} best-config task scores\n")
        per_task_sections.append(
            _markdown_table(
                ["Task", "AdamW", "Muon", "NorMuon", "Muon − AdamW", "NorMuon − AdamW"],
                [
                    [
                        row["task"],
                        f"{row['adamw']:.4f}",
                        f"{row['muon']:.4f}",
                        f"{row['normuon']:.4f}",
                        f"{row['muon_minus_adamw']:+.4f}",
                        f"{row['normuon_minus_adamw']:+.4f}",
                    ]
                    for row in values
                ],
            )
        )

    return "\n\n".join(
        [
            "All 1,680 planned task/checkpoint evaluations completed. Scores below are the "
            "unweighted mean nDCG@10 across the 14 tasks.",
            "### Final quality and learning-rate robustness\n\n" + final_table,
            "\n".join(winners),
            "![Dense training dynamics](../reports/figures/dense-training-dynamics.png)\n\n"
            "![Late-interaction training dynamics](../reports/figures/late-training-dynamics.png)",
            "### Dynamics of each optimizer's best final configuration\n\n" + dynamics_table,
            "![Dense learning-rate sensitivity](../reports/figures/dense-lr-sensitivity.png)\n\n"
            "![Late-interaction learning-rate sensitivity](../reports/figures/late-lr-sensitivity.png)",
            "### Per-task final scores for the best configuration of each optimizer",
            *per_task_sections,
            "The best-LR comparisons are selected on this same benchmark suite and should therefore "
            "be read as controlled exploratory results, not as an unbiased model-selection estimate. "
            "The four-LR mean, spread, and complete per-task rows are included to expose sensitivity "
            "rather than reporting only the winning point.",
        ]
    )


def _render_systems(rows: list[dict]) -> str:
    table = _markdown_table(
        [
            "Family",
            "Optimizer",
            "Median hours",
            "Samples/s",
            "Peak allocated GiB",
            "Optimizer state GiB",
            "Checkpoint GiB",
        ],
        [
            [
                row["model_family"],
                row["optimizer"],
                f"{row['median_wall_time_hours']:.2f}",
                f"{row['median_samples_per_second']:.2f}",
                f"{row['median_peak_allocated_gib']:.2f}",
                f"{row['median_optimizer_state_gib']:.2f}",
                f"{row['median_checkpoint_gib']:.2f}",
            ]
            for row in rows
        ],
    )
    gpu = rows[0]["gpu_name"] if rows else "unknown GPU"
    world_size = rows[0]["world_size"] if rows else 4
    return (
        f"Every run used {world_size} × {gpu}. Values are medians over the four learning-rate "
        "configurations for that optimizer and family; CUDA memory is the maximum per rank, not "
        "the sum across ranks.\n\n"
        + table
        + "\n\nThe recorded wall time includes training and five full checkpoint writes. Peak CUDA memory "
        "comes from PyTorch allocator counters inside each training process, so the independent "
        "utilization guard process is excluded. For checkpoint-resumed runs, throughput is recomputed "
        "from the sum of non-overlapping useful training segments rather than Trainer's resume-local "
        "runtime; the segment adjustment and original Trainer fields remain in the audit table. "
        "Exact per-run measurements are in "
        "`reports/system_metrics.csv`."
    )


def _replace_marked(text: str, markers: tuple[str, str], content: str) -> str:
    begin, end = markers
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"Expected exactly one marker pair {markers}")
    before, remainder = text.split(begin)
    _, after = remainder.split(end)
    return f"{before}{begin}\n\n{content}\n\n{end}{after}"


def render_blog(
    blog_path: Path,
    optimizer_rows: list[dict],
    best_dynamics: list[dict],
    task_rows: list[dict],
    system_rows: list[dict],
) -> None:
    text = blog_path.read_text()
    text = _replace_marked(
        text, RESULTS_MARKERS, _render_results(optimizer_rows, best_dynamics, task_rows)
    )
    text = _replace_marked(text, SYSTEMS_MARKERS, _render_systems(system_rows))
    text = text.replace(
        "**Experiment status:** training matrix in progress. This document already records the frozen protocol;\n"
        "the results sections are populated only from the checked-in aggregation artifacts after coverage reaches\n"
        "1,680/1,680.",
        "**Experiment status:** complete — 24/24 training runs and 1,680/1,680 checkpoint/task evaluations.",
    )
    blog_path.write_text(text)


def _coverage(
    rows: list[dict], summary: list[dict], configs: list[RunConfig], training_audit: dict
) -> dict:
    observed = {(row["model_family"], row["run_id"], row["stage"], row["task"]) for row in rows}
    expected = {
        (config.model_family, config.run_id, stage, task)
        for config in configs
        for stage in range(1, 6)
        for task in DECONTAMINATED_TASK_NAMES
    }
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    evaluation_complete = not missing and not unexpected
    return {
        "complete": evaluation_complete and training_audit["complete"],
        "training_complete": training_audit["complete"],
        "evaluation_complete": evaluation_complete,
        "verified_training_runs": training_audit["verified_runs"],
        "expected_training_runs": training_audit["expected_runs"],
        "verified_training_checkpoints": training_audit["verified_checkpoints"],
        "expected_training_checkpoints": training_audit["expected_checkpoints"],
        "observed_results": len(observed),
        "expected_results": len(expected),
        "observed_checkpoint_summaries": len(summary),
        "expected_checkpoint_summaries": len(configs) * 5,
        "missing": ["/".join(map(str, item)) for item in missing],
        "unexpected": ["/".join(map(str, item)) for item in unexpected],
        "training_errors": training_audit["errors"],
    }


def aggregate(args: argparse.Namespace) -> None:
    configs = load_matrix(args.matrix)
    rows = collect_evaluations(Path(args.results_root), configs)
    summary = _checkpoint_summaries(rows)
    optimizer_rows, best_dynamics = _optimizer_summaries(summary)
    system_metrics = collect_system_metrics(configs)
    system_rows = _system_summaries(system_metrics)
    training_audit = audit_training_artifacts(configs)
    coverage = _coverage(rows, summary, configs, training_audit)
    task_rows = _task_comparison(rows, optimizer_rows) if coverage["complete"] else []

    output = Path(args.output_dir)
    _write_csv(output / "evaluation_long.csv", rows)
    _write_csv(output / "checkpoint_summary.csv", summary)
    _write_csv(output / "optimizer_summary.csv", optimizer_rows)
    _write_csv(output / "best_config_dynamics.csv", best_dynamics)
    _write_csv(output / "best_config_task_comparison.csv", task_rows)
    _write_csv(output / "training_history.csv", collect_training_history(configs))
    _write_csv(output / "system_metrics.csv", system_metrics)
    _write_csv(output / "system_summary.csv", system_rows)
    (output / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    _plot(summary, output)
    print(json.dumps({key: value for key, value in coverage.items() if key != "missing"}, indent=2))

    if coverage["complete"] and not args.no_render_blog:
        render_blog(Path(args.blog), optimizer_rows, best_dynamics, task_rows, system_rows)
    if args.strict and not coverage["complete"]:
        raise RuntimeError("Evaluation matrix is incomplete; see coverage.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="configs/experiment.yaml")
    parser.add_argument("--results-root", default="results/decontaminated-beir")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--blog", default="docs/blog.md")
    parser.add_argument("--no-render-blog", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    aggregate(parse_args(argv))


if __name__ == "__main__":
    main()
