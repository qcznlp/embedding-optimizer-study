"""Evaluate all five checkpoints on the pinned decontaminated BEIR suite."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .config import RunConfig, load_matrix
from .decontamination import DECONTAMINATED_TASK_NAMES


def checkpoint_paths(config: RunConfig, stages: list[int] | None = None) -> list[Path]:
    schedule_path = config.output_dir / "checkpoint_schedule.json"
    if not schedule_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint schedule: {schedule_path}")
    steps = sorted(json.loads(schedule_path.read_text())["steps"])
    if len(steps) != 5:
        raise RuntimeError(f"Expected five checkpoint steps in {schedule_path}, got {steps}")
    selected = range(1, 6) if not stages else stages
    paths = [config.output_dir / f"checkpoint-{steps[stage - 1]}" for stage in selected]
    missing = [path for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    return [path.resolve() for path in paths]


def _selected_models(args: argparse.Namespace) -> dict[str, list[Path]]:
    configs = [
        config
        for config in load_matrix(args.matrix)
        if config.model_family in args.families
        and (not args.run_ids or config.run_id in args.run_ids)
    ]
    return {
        family: [
            checkpoint
            for config in configs
            if config.model_family == family
            for checkpoint in checkpoint_paths(config, args.stages)
        ]
        for family in args.families
    }


def run_evaluation(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    models = _selected_models(args)
    results = Path(args.results_root).resolve()
    log_dir = Path(args.log_dir).resolve()
    results.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[str, subprocess.Popen, object]] = []

    if dense_models := models.get("dense"):
        command = [
            sys.executable,
            str(repo / "scripts/eval/dense_parallel.py"),
            "--gpus",
            args.gpus_a,
            "--results_folder",
            str(results / "dense"),
            "--models",
            *(str(path) for path in dense_models),
            "--tasks",
            *args.tasks,
            "--bf16",
            "--fa2",
            "--local",
            "--decontaminated",
        ]
        handle = (log_dir / "dense-evaluation.log").open("a")
        processes.append(
            (
                "dense",
                subprocess.Popen(command, cwd=repo, stdout=handle, stderr=subprocess.STDOUT),
                handle,
            )
        )

    if late_models := models.get("late"):
        command = [
            "accelerate",
            "launch",
            "--num_processes",
            str(len(args.gpus_b.split(","))),
            "--main_process_port",
            str(args.late_port),
            str(repo / "scripts/eval/late_interaction.py"),
            "--models",
            *(str(path) for path in late_models),
            "--tasks",
            *args.tasks,
            "--results_folder",
            str(results / "late"),
            "--fa2",
            "--decontaminated",
        ]
        handle = (log_dir / "late-evaluation.log").open("a")
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus_b}
        processes.append(
            (
                "late",
                subprocess.Popen(
                    command,
                    cwd=repo,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                ),
                handle,
            )
        )

    failures = 0
    for family, process, handle in processes:
        return_code = process.wait()
        handle.close()
        print(f"{family} evaluation exited {return_code}", flush=True)
        failures += return_code != 0
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="configs/experiment.yaml")
    parser.add_argument(
        "--families", nargs="+", choices=["dense", "late"], default=["dense", "late"]
    )
    parser.add_argument("--run-ids", nargs="*", default=[])
    parser.add_argument("--stages", nargs="*", type=int, choices=range(1, 6))
    parser.add_argument("--tasks", nargs="+", default=list(DECONTAMINATED_TASK_NAMES))
    parser.add_argument("--gpus-a", default="0,1,2,3")
    parser.add_argument("--gpus-b", default="4,5,6,7")
    parser.add_argument("--late-port", type=int, default=29620)
    parser.add_argument("--results-root", default="results/decontaminated-beir")
    parser.add_argument("--log-dir", default="logs/evaluation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    failures = run_evaluation(parse_args(argv))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
