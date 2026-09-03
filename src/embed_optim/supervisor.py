from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
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


def _write_state(args: argparse.Namespace, **fields: object) -> None:
    state_file = getattr(args, "state_file", None)
    if state_file is None:
        return
    path = Path(state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "scope": "training_matrix_supervisor",
        "observed_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        **fields,
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def supervise(
    args: argparse.Namespace,
    *,
    run_command: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    pid_exists: Callable[[int], bool] = _pid_exists,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    """Restart the matrix orchestrator until every selected run is structurally complete."""

    configs = _selected_configs(args)
    selected_runs = [f"{config.model_family}/{config.run_id}" for config in configs]
    wait_pids = list(getattr(args, "wait_for_pids", []))
    if args.wait_for_pid is not None:
        wait_pids.append(args.wait_for_pid)
    wait_pids = list(dict.fromkeys(wait_pids))
    if wait_pids:
        print(
            "Waiting for adopted training PIDs " + ",".join(str(pid) for pid in wait_pids),
            flush=True,
        )
        remaining_pids = set(wait_pids)
        _write_state(
            args,
            phase="waiting_for_adopted_training",
            selected_runs=selected_runs,
            adopted_training_pids=wait_pids,
            remaining_adopted_training_pids=sorted(remaining_pids),
            launches=0,
        )
        while remaining_pids:
            remaining_pids = {pid for pid in remaining_pids if pid_exists(pid)}
            _write_state(
                args,
                phase="waiting_for_adopted_training",
                selected_runs=selected_runs,
                adopted_training_pids=wait_pids,
                remaining_adopted_training_pids=sorted(remaining_pids),
                launches=0,
            )
            if not remaining_pids:
                break
            sleeper(args.poll_seconds)

    launches = 0
    while True:
        incomplete = [config for config in configs if not _run_is_complete(config)]
        if not incomplete:
            print(f"All {len(configs)} selected runs are complete", flush=True)
            _write_state(
                args,
                phase="complete",
                selected_runs=selected_runs,
                remaining_runs=[],
                launches=launches,
            )
            return 0
        if args.max_launches and launches >= args.max_launches:
            print(
                f"Stopping after {launches} launches with {len(incomplete)} runs incomplete",
                file=sys.stderr,
                flush=True,
            )
            _write_state(
                args,
                phase="launch_limit_reached",
                selected_runs=selected_runs,
                remaining_runs=[f"{config.model_family}/{config.run_id}" for config in incomplete],
                launches=launches,
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
        _write_state(
            args,
            phase="matrix_running",
            selected_runs=selected_runs,
            remaining_runs=labels.split(", "),
            launches=launches,
            command=command,
        )
        result = run_command(command, check=False)
        remaining = sum(not _run_is_complete(config) for config in configs)
        print(
            f"Matrix launch {launches} exited {result.returncode}; {remaining} runs remain",
            flush=True,
        )
        _write_state(
            args,
            phase="matrix_exited",
            selected_runs=selected_runs,
            remaining_runs=[
                f"{config.model_family}/{config.run_id}"
                for config in configs
                if not _run_is_complete(config)
            ],
            launches=launches,
            last_matrix_return_code=result.returncode,
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
    parser.add_argument(
        "--state-file",
        type=Path,
        help="Atomically publish supervisor phase and remaining work as JSON",
    )
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument(
        "--wait-for-pids",
        nargs="+",
        type=int,
        default=[],
        help="Wait for every explicitly adopted training PID before launching the matrix",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--restart-delay", type=float, default=30.0)
    parser.add_argument("--max-launches", type=int, default=0)
    parser.add_argument("--sequential-families", action="store_true")
    args = parser.parse_args(argv)
    if args.wait_for_pid is not None and args.wait_for_pid <= 0:
        parser.error("--wait-for-pid must be positive")
    if any(pid <= 0 for pid in args.wait_for_pids):
        parser.error("--wait-for-pids values must be positive")
    if args.poll_seconds <= 0 or args.restart_delay < 0 or args.max_launches < 0:
        parser.error("poll/restart intervals and max launches must be non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(supervise(parse_args(argv)))


if __name__ == "__main__":
    main()
