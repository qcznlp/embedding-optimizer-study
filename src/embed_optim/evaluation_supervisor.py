from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from .config import RunConfig, load_matrix, resolve_matrix_path
from .matrix import _run_is_complete


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _ancestor_pids(pid: int) -> set[int]:
    ancestors: set[int] = set()
    current = pid
    while current > 1:
        try:
            parent_line = next(
                line
                for line in Path(f"/proc/{current}/status").read_text().splitlines()
                if line.startswith("PPid:")
            )
            parent = int(parent_line.split(":", 1)[1].strip())
        except (FileNotFoundError, ProcessLookupError, StopIteration, ValueError):
            break
        if parent <= 1 or parent in ancestors:
            break
        ancestors.add(parent)
        current = parent
    return ancestors


def _matching_command_pids(fragment: str) -> list[int]:
    """Return live external PIDs whose argv contains ``fragment``.

    The supervisor's own argv and wrapper ancestors contain every requested
    fragment, so they are excluded explicitly. A process disappearing during
    the scan is benign.
    """

    own_pid = os.getpid()
    excluded = {own_pid, *_ancestor_pids(own_pid)}
    matches = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        pid = int(path.parent.name)
        if pid in excluded:
            continue
        try:
            command = path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if fragment in command:
            matches.append(pid)
    return sorted(matches)


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


def _aggregate_command(args: argparse.Namespace, *, render_blog: bool = False) -> list[str]:
    command = [
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
        "--strict",
    ]
    if not render_blog:
        command.append("--no-render-blog")
    return command


def _wandb_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        "-m",
        "embed_optim.wandb_sync",
        "--matrix",
        str(resolve_matrix_path(args.matrix).resolve()),
    ]


def _remaining_training_runs(configs: list[RunConfig]) -> list[RunConfig]:
    return [config for config in configs if not _run_is_complete(config)]


def supervise(
    args: argparse.Namespace,
    *,
    run_command: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    pid_exists: Callable[[int], bool] = _pid_exists,
    matching_command_pids: Callable[[str], list[int]] = _matching_command_pids,
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
    if args.wait_for_pids or args.wait_for_commands:
        print(
            "Waiting for adopted evaluators: "
            f"pids={args.wait_for_pids}, command_fragments={args.wait_for_commands}",
            flush=True,
        )
        while True:
            live = [pid for pid in args.wait_for_pids if pid_exists(pid)]
            command_matches = {
                fragment: matching_command_pids(fragment) for fragment in args.wait_for_commands
            }
            command_matches = {fragment: pids for fragment, pids in command_matches.items() if pids}
            if not live and not command_matches:
                break
            print(
                f"Adopted evaluators still alive: pids={live}, command_matches={command_matches}",
                flush=True,
            )
            sleeper(args.wait_poll_seconds)

    cycles = 0
    coverage_complete = False
    while True:
        if args.max_launches and cycles >= args.max_launches:
            print(
                f"Stopping after {cycles} recovery cycles without complete finalization",
                file=sys.stderr,
                flush=True,
            )
            return 1

        cycles += 1
        if not coverage_complete:
            print(f"Evaluation recovery cycle {cycles} started", flush=True)
            evaluation = run_command(_evaluation_command(args), check=False)
            print(f"Evaluation coordinator exited {evaluation.returncode}", flush=True)

            coverage = run_command(_aggregate_command(args), check=False)
            print(f"Strict coverage audit exited {coverage.returncode}", flush=True)
            coverage_complete = coverage.returncode == 0
            if not coverage_complete:
                if not args.max_launches or cycles < args.max_launches:
                    sleeper(args.restart_delay)
                continue
            print("Strict evaluation coverage is complete", flush=True)
            if args.evaluation_only:
                print("Evaluation-only recovery is complete", flush=True)
                return 0

        if args.skip_wandb_sync:
            wandb_return_code = 0
        else:
            wandb = run_command(_wandb_command(args), check=False)
            wandb_return_code = wandb.returncode
            print(f"Canonical W&B sync exited {wandb_return_code}", flush=True)

        if wandb_return_code != 0:
            print("Deferring final report render until canonical W&B sync succeeds", flush=True)
            if not args.max_launches or cycles < args.max_launches:
                sleeper(args.restart_delay)
            continue

        final_report = run_command(_aggregate_command(args, render_blog=True), check=False)
        print(f"Final report and blog render exited {final_report.returncode}", flush=True)
        if final_report.returncode == 0:
            print("Evaluation, W&B histories, reports, and blog are complete", flush=True)
            return 0

        if not args.max_launches or cycles < args.max_launches:
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
    parser.add_argument(
        "--wait-for-pid",
        dest="wait_for_pids",
        action="append",
        type=int,
        default=[],
        help=(
            "Adopt an already-running evaluator or dispatcher by waiting for its PID before "
            "launching recovery; repeat for multiple independent coordinators"
        ),
    )
    parser.add_argument(
        "--wait-for-command",
        dest="wait_for_commands",
        action="append",
        default=[],
        help=(
            "Also wait while any external process argv contains this fragment; repeat to adopt "
            "orphan workers whose coordinator PID may change"
        ),
    )
    parser.add_argument("--wait-poll-seconds", type=float, default=30.0)
    parser.add_argument("--restart-delay", type=float, default=60.0)
    parser.add_argument(
        "--skip-wandb-sync",
        action="store_true",
        help="Skip canonical W&B publication while retaining strict evaluation and blog gates",
    )
    parser.add_argument(
        "--max-launches",
        type=int,
        default=0,
        help="Maximum evaluator launches, where zero retries without a limit",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help=(
            "Stop after strict evaluation coverage succeeds; use when a separate post-evaluation "
            "pipeline owns W&B publication and report rendering"
        ),
    )
    args = parser.parse_args(argv)
    if any(pid <= 0 for pid in args.wait_for_pids):
        parser.error("--wait-for-pid must be positive")
    if any(not fragment.strip() for fragment in args.wait_for_commands):
        parser.error("--wait-for-command must not be empty")
    if (
        args.training_poll_seconds <= 0
        or args.wait_poll_seconds <= 0
        or args.restart_delay < 0
        or args.max_launches < 0
    ):
        parser.error("poll/restart intervals and max launches must be non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(supervise(parse_args(argv)))


if __name__ == "__main__":
    main()
