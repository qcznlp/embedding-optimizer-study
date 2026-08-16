"""Evaluate all five checkpoints on the pinned decontaminated BEIR suite."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .config import RunConfig, load_matrix
from .decontamination import DECONTAMINATED_TASK_NAMES


@dataclass
class EvaluationProcess:
    family: str
    process: subprocess.Popen
    handle: object
    model: Path | None = None


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


def _late_command(
    repo: Path,
    model: Path,
    args: argparse.Namespace,
    results: Path,
    port: int,
    num_processes: int,
) -> list[str]:
    return [
        "accelerate",
        "launch",
        "--num_processes",
        str(num_processes),
        "--main_process_port",
        str(port),
        str(repo / "scripts/eval/late_interaction.py"),
        "--models",
        str(model),
        "--tasks",
        *args.tasks,
        "--results_folder",
        str(results / "late"),
        "--fa2",
        "--decontaminated",
    ]


def _launch_late(
    repo: Path,
    model: Path,
    args: argparse.Namespace,
    results: Path,
    log_dir: Path,
    pool: str,
    gpus: str,
    port: int,
) -> EvaluationProcess:
    handle = (log_dir / f"late-evaluation-{pool}.log").open("a")
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus}
    process = subprocess.Popen(
        _late_command(
            repo,
            model,
            args,
            results,
            port,
            len([gpu for gpu in gpus.split(",") if gpu.strip()]),
        ),
        cwd=repo,
        env=environment,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    print(f"late pool-{pool} started {model}", flush=True)
    return EvaluationProcess("late", process, handle, model)


def run_evaluation(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    models = _selected_models(args)
    results = Path(args.results_root).resolve()
    log_dir = Path(args.log_dir).resolve()
    results.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    dense_job: EvaluationProcess | None = None

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
        dense_job = EvaluationProcess(
            "dense",
            subprocess.Popen(command, cwd=repo, stdout=handle, stderr=subprocess.STDOUT),
            handle,
        )

    late_queue = list(models.get("late", []))
    late_jobs: dict[str, EvaluationProcess] = {}
    pools = {
        "a": (args.gpus_a, args.late_port_a),
        "b": (args.gpus_b, args.late_port),
    }
    failures = 0
    while dense_job is not None or late_queue or late_jobs:
        if dense_job is not None and dense_job.process.poll() is not None:
            return_code = dense_job.process.returncode
            dense_job.handle.close()
            print(f"dense evaluation exited {return_code}", flush=True)
            failures += return_code != 0
            dense_job = None

        for pool, job in list(late_jobs.items()):
            if job.process.poll() is None:
                continue
            return_code = job.process.returncode
            job.handle.close()
            print(f"late pool-{pool} {job.model} exited {return_code}", flush=True)
            failures += return_code != 0
            del late_jobs[pool]

        for pool, (gpus, port) in pools.items():
            if not late_queue or pool in late_jobs or (pool == "a" and dense_job is not None):
                continue
            late_jobs[pool] = _launch_late(
                repo,
                late_queue.pop(0),
                args,
                results,
                log_dir,
                pool,
                gpus,
                port,
            )
        if dense_job is not None or late_queue or late_jobs:
            time.sleep(1)
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
    parser.add_argument("--late-port-a", type=int, default=29610)
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
