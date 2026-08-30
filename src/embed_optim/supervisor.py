from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
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


def _selected_configs(args: argparse.Namespace) -> list[RunConfig]:
    configs = [
        config for config in load_matrix(args.matrix) if config.model_family in args.families
    ]
    if args.run_ids:
        requested = set(args.run_ids)
        configs = [config for config in configs if config.run_id in requested]
    if not configs:
        raise ValueError("No experiment configurations matched the requested filters")
    return configs


def _matrix_command(args: argparse.Namespace, families: list[str] | None = None) -> list[str]:
    families = list(args.families if families is None else families)
    command = [
        args.python,
        "-m",
        "embed_optim.matrix",
        "--matrix",
        str(resolve_matrix_path(args.matrix).resolve()),
        "--families",
        *families,
        "--gpus-a",
        args.gpus_a,
        "--gpus-b",
        args.gpus_b,
        "--port-a",
        str(args.port_a),
        "--port-b",
        str(args.port_b),
        "--log-dir",
        args.log_dir,
    ]
    if args.run_ids:
        command.extend(["--run-ids", *args.run_ids])
    return command


def supervise(
    args: argparse.Namespace,
    *,
    run_command: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    pid_exists: Callable[[int], bool] = _pid_exists,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    """Restart the matrix orchestrator until every selected run is structurally complete."""

    configs = _selected_configs(args)
    if args.wait_for_pid is not None:
        print(f"Waiting for adopted matrix PID {args.wait_for_pid}", flush=True)
        while pid_exists(args.wait_for_pid):
            sleeper(args.poll_seconds)

    launches = 0
    while True:
        incomplete = [config for config in configs if not _run_is_complete(config)]
        if not incomplete:
            print(f"All {len(configs)} selected runs are complete", flush=True)
            return 0
        if args.max_launches and launches >= args.max_launches:
            print(
                f"Stopping after {launches} launches with {len(incomplete)} runs incomplete",
                file=sys.stderr,
                flush=True,
            )
            return 1

        launches += 1
        labels = ", ".join(f"{config.model_family}/{config.run_id}" for config in incomplete)
        print(
            f"Matrix launch {launches}; {len(incomplete)}/{len(configs)} incomplete: {labels}",
            flush=True,
        )
        launch_families = list(args.families)
        if args.sequential_families:
            launch_families = [
                next(
                    family
                    for family in args.families
                    if any(config.model_family == family for config in incomplete)
                )
            ]
        command = _matrix_command(args, launch_families)
        result = run_command(command, check=False)
        remaining = sum(not _run_is_complete(config) for config in configs)
        print(
            f"Matrix launch {launches} exited {result.returncode}; {remaining} runs remain",
            flush=True,
        )
        if remaining and (not args.max_launches or launches < args.max_launches):
            sleeper(args.restart_delay)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restart the training matrix orchestrator until selected runs are complete"
    )
    parser.add_argument("--matrix", default="configs/experiment.yaml")
    parser.add_argument("--families", nargs="+", choices=("dense", "late"), default=["dense"])
    parser.add_argument("--run-ids", nargs="*", default=[])
    parser.add_argument("--gpus-a", default="0,1,2,3")
    parser.add_argument("--gpus-b", default="4,5,6,7")
    parser.add_argument("--port-a", type=int, default=29510)
    parser.add_argument("--port-b", type=int, default=29520)
    parser.add_argument("--log-dir", default="logs/training")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--restart-delay", type=float, default=30.0)
    parser.add_argument("--max-launches", type=int, default=0)
    parser.add_argument("--sequential-families", action="store_true")
    args = parser.parse_args(argv)
    if args.wait_for_pid is not None and args.wait_for_pid <= 0:
        parser.error("--wait-for-pid must be positive")
    if args.poll_seconds <= 0 or args.restart_delay < 0 or args.max_launches < 0:
        parser.error("poll/restart intervals and max launches must be non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(supervise(parse_args(argv)))


if __name__ == "__main__":
    main()
