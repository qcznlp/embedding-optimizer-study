"""Artifact-only guard for the corrected matrix's next-pair handoff.

The first recovery supervisor predates a durable controller heartbeat.  This guard never inspects
or signals processes.  It waits for the two adopted runs to become deeply complete, gives the
existing matrix controller an explicit grace period to create either successor log/output, and
only then invokes the frozen recovery supervisor if no successor artifact appears.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from .config import RunConfig, load_matrix, resolve_matrix_path
from .matrix import _run_is_complete

DEFAULT_PROTOCOL = Path("configs/dense_no_packing_matrix_handoff_guard.json")
DEFAULT_LOG_DIR = Path("logs/dense-no-packing-handoff-guard")


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path, repository: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(repository).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@contextmanager
def _exclusive_lease(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A corrected matrix handoff guard is already active") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"started_at={_timestamp()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield


def _load_protocol(path: Path, repository: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise ValueError("Invalid corrected matrix handoff protocol")
    if (
        protocol.get("process_inspection") is not False
        or protocol.get("signals_processes") is not False
    ):
        raise ValueError("Handoff guard must explicitly forbid process inspection and signaling")
    expected = protocol.get("sources")
    if not isinstance(expected, list):
        raise ValueError("Handoff guard protocol has no source identities")
    observed = [_identity(repository / item["path"], repository) for item in expected]
    if observed != expected:
        raise RuntimeError("Corrected matrix handoff source contract changed")
    return protocol


def _selected_configs(matrix: Path) -> dict[str, RunConfig]:
    configs = load_matrix(matrix)
    if (
        len(configs) != 12
        or any(config.model_family != "dense" for config in configs)
        or any(config.dense_can_flatten_inputs for config in configs)
    ):
        raise ValueError("Handoff guard requires the frozen 12-run padded Dense matrix")
    return {config.run_id: config for config in configs}


def _successor_artifacts(config: RunConfig, training_log_dir: Path) -> list[str]:
    artifacts: list[str] = []
    log_path = training_log_dir / f"{config.model_family}-{config.run_id}.log"
    if log_path.exists():
        artifacts.append(str(log_path))
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        artifacts.append(str(config.output_dir))
    return artifacts


def _supervisor_command(protocol: dict[str, Any], repository: Path) -> tuple[str, ...]:
    runtime = protocol["runtime"]
    command = [
        str(runtime["python"]),
        "-m",
        "embed_optim.supervisor",
        "--matrix",
        str((repository / protocol["matrix"]).resolve()),
        "--families",
        "dense",
        "--gpus-a",
        str(runtime["gpus_a"]),
        "--gpus-b",
        str(runtime["gpus_b"]),
        "--port-a",
        str(runtime["port_a"]),
        "--port-b",
        str(runtime["port_b"]),
        "--log-dir",
        str(runtime["training_log_dir"]),
        "--state-file",
        str((repository / runtime["recovery_state_file"]).resolve()),
        "--poll-seconds",
        str(runtime["supervisor_poll_seconds"]),
        "--restart-delay",
        str(runtime["restart_delay_seconds"]),
        "--max-launches",
        "0",
    ]
    return tuple(command)


def run_guard(
    args: argparse.Namespace,
    *,
    run_command: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    repository = args.workdir.resolve()
    protocol_path = args.protocol.resolve()
    log_dir = args.log_dir.resolve()
    state_path = log_dir / "state.json"
    with _exclusive_lease(log_dir / "controller.lease"):
        protocol = _load_protocol(protocol_path, repository)
        matrix = resolve_matrix_path(repository / protocol["matrix"]).resolve()
        configs = _selected_configs(matrix)
        current_ids = list(protocol["current_run_ids"])
        successor_ids = list(protocol["successor_run_ids"])
        if set(current_ids) & set(successor_ids) or any(
            run_id not in configs for run_id in current_ids + successor_ids
        ):
            raise ValueError("Handoff guard run IDs are missing or overlapping")
        poll_seconds = float(protocol["runtime"]["guard_poll_seconds"])
        grace_seconds = float(protocol["runtime"]["successor_grace_seconds"])
        training_log_dir = repository / protocol["runtime"]["training_log_dir"]

        while True:
            protocol = _load_protocol(protocol_path, repository)
            complete = [run_id for run_id in current_ids if _run_is_complete(configs[run_id])]
            _atomic_json(
                state_path,
                {
                    "schema_version": 1,
                    "status": "waiting_for_current_runs",
                    "observed_at_utc": _timestamp(),
                    "current_run_ids": current_ids,
                    "complete_current_run_ids": complete,
                    "successor_run_ids": successor_ids,
                },
            )
            if len(complete) == len(current_ids):
                break
            sleeper(poll_seconds)

        grace_started = clock()
        while clock() - grace_started < grace_seconds:
            protocol = _load_protocol(protocol_path, repository)
            observed = {
                run_id: _successor_artifacts(configs[run_id], training_log_dir)
                for run_id in successor_ids
            }
            if any(observed.values()):
                _atomic_json(
                    state_path,
                    {
                        "schema_version": 1,
                        "status": "yielded_to_existing_matrix",
                        "complete": True,
                        "observed_at_utc": _timestamp(),
                        "successor_artifacts": observed,
                        "takeover_launched": False,
                    },
                )
                return 0
            _atomic_json(
                state_path,
                {
                    "schema_version": 1,
                    "status": "successor_grace_period",
                    "observed_at_utc": _timestamp(),
                    "grace_elapsed_seconds": clock() - grace_started,
                    "grace_seconds": grace_seconds,
                    "successor_artifacts": observed,
                },
            )
            sleeper(min(poll_seconds, max(0.0, grace_seconds - (clock() - grace_started))))

        observed = {
            run_id: _successor_artifacts(configs[run_id], training_log_dir)
            for run_id in successor_ids
        }
        if any(observed.values()):
            raise RuntimeError(
                "Successor artifacts appeared at the takeover boundary; refusing race"
            )
        command = _supervisor_command(protocol, repository)
        _atomic_json(
            state_path,
            {
                "schema_version": 1,
                "status": "takeover_running",
                "observed_at_utc": _timestamp(),
                "successor_artifacts": observed,
                "takeover_launched": True,
                "command": list(command),
            },
        )
        result = run_command(command, cwd=repository, check=False)
        complete = all(_run_is_complete(config) for config in configs.values())
        _atomic_json(
            state_path,
            {
                "schema_version": 1,
                "status": "complete" if result.returncode == 0 and complete else "takeover_failed",
                "complete": result.returncode == 0 and complete,
                "observed_at_utc": _timestamp(),
                "takeover_launched": True,
                "return_code": result.returncode,
                "all_runs_complete": complete,
            },
        )
        return 0 if result.returncode == 0 and complete else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_guard(parse_args(argv)))


if __name__ == "__main__":
    main()
