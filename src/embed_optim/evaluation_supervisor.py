from __future__ import annotations

import argparse
import subprocess
import sys
import time
from typing import Callable

from .config import RunConfig, load_matrix, resolve_matrix_path
from .matrix import _run_is_complete


def _evaluation_command(args: argparse.Namespace) -> list[str]:
    worker_python = args.worker_python or args.python
    return [
        args.python,
        "-m",
        "embed_optim.evaluate_matrix",
        "--matrix",
        str(resolve_matrix_path(args.matrix).resolve()),
        "--families",
        "dense",
        "late",
        "--gpus-a",
        args.gpus_a,
        "--gpus-b",
        args.gpus_b,
        "--late-port-a",
        str(args.late_port_a),
        "--late-port",
        str(args.late_port),
        "--results-root",
        args.results_root,
        "--log-dir",
        args.log_dir,
        "--worker-python",
        worker_python,
    ]


def _aggregate_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        "-m",
        "embed_optim.aggregate",
        "--matrix",
        str(resolve_matrix_path(args.matrix).resolve()),
        "--results-root",
        args.results_root,
        "--output-dir",
        args.output_dir,
        "--blog",
        args.blog,
        "--no-render-blog",
        "--strict",
    ]


def _remaining_training_runs(configs: list[RunConfig]) -> list[RunConfig]:
    return [config for config in configs if not _run_is_complete(config)]


def supervise(
    args: argparse.Namespace,
    *,
    run_command: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    """Wait for training, then resume evaluation until strict coverage is complete."""

    configs = load_matrix(args.matrix)
    if not configs:
        raise ValueError("The experiment matrix contains no training runs")

    previous_remaining: int | None = None
    while remaining := _remaining_training_runs(configs):
        if len(remaining) != previous_remaining:
            print(
                f"Waiting for training: {len(configs) - len(remaining)}/{len(configs)} runs complete",
                flush=True,
            )
            previous_remaining = len(remaining)
        sleeper(args.training_poll_seconds)

    print(f"All {len(configs)} training runs are complete; starting evaluation", flush=True)
    launches = 0
    while True:
        if args.max_launches and launches >= args.max_launches:
            print(
                f"Stopping after {launches} evaluation launches without strict coverage",
                file=sys.stderr,
                flush=True,
            )
            return 1

        launches += 1
        print(f"Evaluation launch {launches} started", flush=True)
        evaluation = run_command(_evaluation_command(args), check=False)
        print(f"Evaluation launch {launches} exited {evaluation.returncode}", flush=True)

        coverage = run_command(_aggregate_command(args), check=False)
        print(f"Strict coverage audit exited {coverage.returncode}", flush=True)
        if coverage.returncode == 0:
            print("Strict evaluation coverage is complete", flush=True)
            return 0

        if not args.max_launches or launches < args.max_launches:
            sleeper(args.restart_delay)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the full training matrix, then restart resumable evaluation until strict "
            "coverage is complete"
        )
    )
    parser.add_argument("--matrix", default="configs/experiment.yaml")
    parser.add_argument("--gpus-a", default="0,1,2,3")
    parser.add_argument("--gpus-b", default="4,5,6,7")
    parser.add_argument("--late-port-a", type=int, default=29610)
    parser.add_argument("--late-port", type=int, default=29620)
    parser.add_argument("--results-root", default="results/decontaminated-beir")
    parser.add_argument("--log-dir", default="logs/evaluation")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--blog", default="docs/blog.md")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--worker-python",
        default=None,
        help="Python executable for evaluator workers (default: --python)",
    )
    parser.add_argument("--training-poll-seconds", type=float, default=60.0)
    parser.add_argument("--restart-delay", type=float, default=60.0)
    parser.add_argument(
        "--max-launches",
        type=int,
        default=0,
        help="Maximum evaluator launches, where zero retries without a limit",
    )
    args = parser.parse_args(argv)
    if args.training_poll_seconds <= 0 or args.restart_delay < 0 or args.max_launches < 0:
        parser.error("poll/restart intervals and max launches must be non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(supervise(parse_args(argv)))


if __name__ == "__main__":
    main()
