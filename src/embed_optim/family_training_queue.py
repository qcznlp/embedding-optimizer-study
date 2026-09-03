from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import RunConfig, load_matrix, resolve_matrix_path
from .matrix import _run_is_complete


@dataclass(frozen=True)
class QueueJob:
    phase: str
    matrix: Path
    config: RunConfig

    @property
    def identity(self) -> str:
        return f"{self.phase}/{self.matrix.stem}/{self.config.model_family}/{self.config.run_id}"


@dataclass(frozen=True)
class CommandOutcome:
    return_code: int
    timed_out: bool = False
    error: str | None = None
    process_group_cleaned: bool = False


class QueueTermination(SystemExit):
    """Convert catchable termination signals into a cleanup-aware exit."""

    def __init__(self, signum: int) -> None:
        super().__init__(128 + signum)
        self.signum = signum


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _repository(path: Path) -> Path:
    candidates: list[Path] = []
    for anchor in (path.resolve().parent, Path.cwd().resolve()):
        candidates.extend((anchor, *anchor.parents))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "embed_optim"
        ).is_dir():
            return candidate
    raise ValueError(
        f"Cannot locate the embedding-optimizer-study repository for queue plan {path}; "
        "run from the repository or place the plan below it"
    )


def _bound_path(repository: Path, binding: dict[str, Any]) -> Path:
    path = (repository / str(binding.get("path", ""))).resolve()
    if not path.is_file() or _sha256(path) != binding.get("sha256"):
        raise ValueError(f"Frozen queue source differs: {path}")
    return path


def load_queue_plan(path: str | Path) -> tuple[Path, dict[str, Any], dict[str, list[QueueJob]]]:
    resolved = resolve_matrix_path(path).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "frozen_before_dense_confirmatory_or_short_branch_training"
        or payload.get("family") != "dense"
        or set(payload.get("pools") or {}) != {"a", "b"}
    ):
        raise ValueError("Dense training queue does not match its frozen schema")
    repository = _repository(resolved)
    amendment = (repository / payload["scope_amendment"]["path"]).resolve()
    amendment_payload = json.loads(amendment.read_text(encoding="utf-8"))
    if amendment_payload.get(
        "status"
    ) != "user_directed_post_hoc_scope_amendment" or amendment_payload.get("active_scope", {}).get(
        "families"
    ) != ["dense"]:
        raise ValueError("Dense training queue is not bound to the active scope amendment")
    bindings = payload.get("source_bindings") or []
    bound = {_bound_path(repository, item) for item in bindings}
    if len(bound) != 6:
        raise ValueError("Dense training queue must bind all six generated matrices")

    jobs_by_pool: dict[str, list[QueueJob]] = {}
    all_jobs: list[QueueJob] = []
    for pool, records in payload["pools"].items():
        jobs: list[QueueJob] = []
        for record in records:
            phase = str(record.get("phase"))
            if phase not in {"confirmatory", "short-branch"}:
                raise ValueError(f"Unknown dense queue phase: {phase}")
            matrix = (repository / str(record.get("matrix", ""))).resolve()
            if matrix not in bound:
                raise ValueError(f"Queue job uses an unbound matrix: {matrix}")
            selected = [
                config
                for config in load_matrix(matrix)
                if config.model_family == "dense" and config.run_id == record.get("run_id")
            ]
            if len(selected) != 1:
                raise ValueError(f"Queue job does not select one DenseOn run: {record}")
            config = selected[0]
            if not Path(config.output_root).is_absolute():
                config = replace(
                    config,
                    output_root=str((repository / config.output_root).resolve()),
                )
            job = QueueJob(phase, matrix, config)
            jobs.append(job)
            all_jobs.append(job)
        jobs_by_pool[pool] = jobs

    identities = [job.identity for job in all_jobs]
    output_dirs = [job.config.output_dir.resolve() for job in all_jobs]
    expected = payload.get("expected") or {}
    phase_counts = {
        phase: sum(job.phase == phase for job in all_jobs)
        for phase in ("confirmatory", "short-branch")
    }
    if (
        len(identities) != expected.get("total_runs")
        or len(set(identities)) != len(identities)
        or len(set(output_dirs)) != len(output_dirs)
        or phase_counts["confirmatory"] != expected.get("confirmatory_runs")
        or phase_counts["short-branch"] != expected.get("short_branch_runs")
    ):
        raise ValueError("Dense training queue coverage differs from the frozen 9+9 design")
    return resolved, payload, jobs_by_pool


def _pid_command(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, ProcessLookupError):
        return None


def _wait_for_process(pid: int | None, fragment: str | None, poll_seconds: float) -> None:
    if pid is None:
        return
    while True:
        command = _pid_command(pid)
        if command is None or (fragment is not None and fragment not in command):
            return
        print(f"waiting for prerequisite pid {pid}: {command}", flush=True)
        time.sleep(poll_seconds)


@contextmanager
def _exclusive_pool_lease(path: Path, metadata: dict[str, Any]) -> Iterator[int]:
    """Hold a non-blocking process lease for one queue pool."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            raise RuntimeError(
                f"Dense queue pool lease is already held: {path} ({owner})"
            ) from error

        lease = {
            "schema_version": 1,
            "acquired_at": _timestamp(),
            "pid": os.getpid(),
            **metadata,
        }
        handle.seek(0)
        handle.truncate()
        json.dump(lease, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield handle.fileno()
    finally:
        # Do not explicitly LOCK_UN: a matrix child receives a duplicate of this
        # open file description. Closing only our copy keeps the lease held if
        # the queue coordinator is killed while that child is still alive.
        handle.close()


def _termination_signal(signum: int, _frame: object) -> None:
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    raise QueueTermination(signum)


def _terminate_and_reap_process_group(
    process: subprocess.Popen,
    *,
    termination_grace_seconds: float,
) -> None:
    """Best-effort TERM/KILL of the whole child session, followed by reap."""

    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + termination_grace_seconds
    try:
        process.wait(timeout=termination_grace_seconds)
    except (subprocess.TimeoutExpired, KeyboardInterrupt, QueueTermination):
        pass

    # The direct matrix parent may already have been reaped while torchrun or
    # workers remain in its session. Probe the PGID itself through the grace
    # window instead of treating a completed parent wait as tree completion.
    group_exists = True
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            group_exists = False
            break
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    if not group_exists:
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    # The direct parent may already be reaped, so its repeated wait cannot
    # synchronize with orphaned workers. Allow SIGKILL delivery to close their
    # inherited lease descriptors before returning to the queue lease scope.
    kill_deadline = time.monotonic() + max(1.0, termination_grace_seconds)
    while time.monotonic() < kill_deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    while True:
        try:
            process.wait()
            return
        except (KeyboardInterrupt, QueueTermination):
            continue
        except ChildProcessError:
            return


def _deep_completion_audit(config: RunConfig) -> dict[str, Any]:
    """Fail closed unless terminal artifacts and every checkpoint pass deep audit."""

    if not _run_is_complete(config):
        return {
            "complete": False,
            "verified_checkpoints": 0,
            "expected_checkpoints": 5,
            "problems": ["terminal completion contract is incomplete or inconsistent"],
        }
    try:
        from .checkpoint_watch import audit_checkpoint_integrity

        return audit_checkpoint_integrity(config, world_size=4)
    except Exception as error:  # noqa: BLE001
        return {
            "complete": False,
            "verified_checkpoints": 0,
            "expected_checkpoints": 5,
            "problems": [f"deep checkpoint audit failed ({type(error).__name__}: {error})"],
        }


def _quarantine_invalid_completion(
    job: QueueJob,
    audit: dict[str, Any],
) -> Path | None:
    """Atomically preserve a bad completed output so a clean rerun cannot shallow-skip it."""

    output = job.config.output_dir.resolve()
    if not output.exists() and not output.is_symlink():
        return None
    if output.is_symlink():
        raise RuntimeError(f"Refusing to quarantine symlinked training output: {output}")
    recovery_root = output.parent / ".invalid-completed-runs"
    recovery_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = recovery_root / f"{output.name}.{stamp}.pid{os.getpid()}"
    suffix = 0
    while destination.exists():
        suffix += 1
        destination = recovery_root / f"{output.name}.{stamp}.pid{os.getpid()}.{suffix}"
    os.replace(output, destination)
    _atomic_json(
        destination.with_name(f"{destination.name}.queue-recovery.json"),
        {
            "schema_version": 1,
            "quarantined_at": _timestamp(),
            "job": job.identity,
            "original_output_dir": str(output),
            "quarantined_output_dir": str(destination),
            "audit": audit,
        },
    )
    return destination


def _run_command_with_watchdog(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    termination_grace_seconds: float,
    lease_fd: int | None = None,
) -> CommandOutcome:
    """Run a matrix in its own process group and bound its total wall time."""

    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            start_new_session=True,
            pass_fds=(() if lease_fd is None else (lease_fd,)),
        )
    except OSError as error:
        return CommandOutcome(126, error=f"{type(error).__name__}: {error}")
    handled_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {signum: signal.getsignal(signum) for signum in handled_signals}
    for signum in handled_signals:
        signal.signal(signum, _termination_signal)
    try:
        try:
            return_code = process.wait(timeout=timeout_seconds)
            if return_code != 0:
                _terminate_and_reap_process_group(
                    process,
                    termination_grace_seconds=termination_grace_seconds,
                )
                return CommandOutcome(return_code, process_group_cleaned=True)
            return CommandOutcome(return_code)
        except subprocess.TimeoutExpired:
            print(
                f"training watchdog expired after {timeout_seconds:.0f}s; "
                "terminating process group",
                file=sys.stderr,
                flush=True,
            )
            _terminate_and_reap_process_group(
                process,
                termination_grace_seconds=termination_grace_seconds,
            )
            return CommandOutcome(124, timed_out=True, process_group_cleaned=True)
    except BaseException:
        # Ignore a repeated termination signal while cleanup is in progress;
        # restore the caller's handlers after the child has been reaped.
        for signum in handled_signals:
            signal.signal(signum, signal.SIG_IGN)
        _terminate_and_reap_process_group(
            process,
            termination_grace_seconds=termination_grace_seconds,
        )
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _job_log_dir(repository: Path, job: QueueJob) -> Path:
    seed = job.matrix.stem
    parent = "confirmatory-training" if job.phase == "confirmatory" else "short-branch-training"
    return repository / "logs" / parent / seed


def run_pool(args: argparse.Namespace) -> int:
    plan_path, _, jobs_by_pool = load_queue_plan(args.plan)
    repository = _repository(plan_path)
    jobs = jobs_by_pool[args.pool]
    ledger_path = args.ledger or (
        repository / "logs" / "dense-only-runtime" / f"training-queue-{args.pool}.json"
    )
    ledger_path = ledger_path.resolve()
    plan_identity = {"path": str(plan_path), "sha256": _sha256(plan_path)}
    lease_path = repository / "logs" / "dense-only-runtime" / f"training-queue-{args.pool}.lease"
    with _exclusive_pool_lease(
        lease_path,
        {"pool": args.pool, "plan": plan_identity, "ledger": str(ledger_path)},
    ) as lease_fd:
        if ledger_path.is_file():
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            if (
                ledger.get("schema_version") != 1
                or ledger.get("plan") != plan_identity
                or ledger.get("pool") != args.pool
                or ledger.get("family") != "dense"
                or ledger.get("gpus") != args.gpus
            ):
                raise ValueError(f"Existing queue ledger has a different identity: {ledger_path}")
        else:
            ledger = {
                "schema_version": 1,
                "started_at": _timestamp(),
                "plan": plan_identity,
                "pool": args.pool,
                "family": "dense",
                "gpus": args.gpus,
                "jobs": [],
            }

        # Reset the aggregate state before waiting or inspecting any job. If this
        # process is interrupted, consumers see an incomplete queue rather than a
        # sticky success bit left by an earlier invocation.
        previous_finished_at = ledger.pop("finished_at", None)
        if previous_finished_at is not None:
            ledger["previous_finished_at"] = previous_finished_at
        ledger["complete"] = False
        ledger["active_attempt"] = {"pid": os.getpid(), "started_at": _timestamp()}
        _atomic_json(ledger_path, ledger)

        _wait_for_process(args.wait_pid, args.wait_command_fragment, args.poll_seconds)
        for index, job in enumerate(jobs, start=1):
            record = next(
                (item for item in ledger["jobs"] if item["identity"] == job.identity),
                None,
            )
            if record is None:
                record = {
                    "index": index,
                    "identity": job.identity,
                    "matrix": str(job.matrix),
                    "output_dir": str(job.config.output_dir.resolve()),
                    "attempts": [],
                    "complete": False,
                }
                ledger["jobs"].append(record)

            recorded_complete = bool(record.get("complete"))
            terminal_complete = _run_is_complete(job.config)
            if recorded_complete or terminal_complete:
                preflight_audit = _deep_completion_audit(job.config)
                record["last_integrity_audit"] = {
                    "audited_at": _timestamp(),
                    "complete": preflight_audit.get("complete") is True,
                    "verified_checkpoints": preflight_audit.get("verified_checkpoints", 0),
                    "expected_checkpoints": preflight_audit.get("expected_checkpoints", 5),
                    "problems": list(preflight_audit.get("problems", [])),
                }
                if preflight_audit.get("complete") is True:
                    record["complete"] = True
                    _atomic_json(ledger_path, ledger)
                    print(
                        f"queue pool-{args.pool}: deeply verified {job.identity}",
                        flush=True,
                    )
                    continue

                quarantined = _quarantine_invalid_completion(job, preflight_audit)
                invalidation = {
                    "invalidated_at": _timestamp(),
                    "problems": list(preflight_audit.get("problems", [])),
                    "quarantined_output_dir": str(quarantined) if quarantined else None,
                }
                record.setdefault("invalidations", []).append(invalidation)
                record["complete"] = False
                _atomic_json(ledger_path, ledger)
                print(
                    f"queue pool-{args.pool}: invalidated completed artifact {job.identity}; "
                    "starting a clean recoverable rerun",
                    file=sys.stderr,
                    flush=True,
                )

            command = [
                args.python,
                "-m",
                "embed_optim.matrix",
                "--matrix",
                str(job.matrix),
                "--families",
                "dense",
                "--run-ids",
                job.config.run_id,
                "--gpus-a",
                args.gpus,
                "--gpus-b",
                args.gpus,
                "--port-a",
                str(args.port),
                "--port-b",
                str(args.port + 1),
                "--log-dir",
                str(_job_log_dir(repository, job)),
                "--max-retries",
                str(args.max_retries),
            ]
            print(
                f"queue pool-{args.pool} job {index}/{len(jobs)}: {' '.join(command)}",
                flush=True,
            )
            started = _timestamp()
            outcome = _run_command_with_watchdog(
                command,
                cwd=repository,
                timeout_seconds=args.job_timeout_seconds,
                termination_grace_seconds=args.termination_grace_seconds,
                lease_fd=lease_fd,
            )
            postflight_audit = (
                _deep_completion_audit(job.config)
                if outcome.return_code == 0
                else {
                    "complete": False,
                    "verified_checkpoints": 0,
                    "expected_checkpoints": 5,
                    "problems": ["training command did not exit successfully"],
                }
            )
            attempt = {
                "started_at": started,
                "finished_at": _timestamp(),
                "return_code": outcome.return_code,
                "timed_out": outcome.timed_out,
                "process_group_cleaned": outcome.process_group_cleaned,
                "watchdog_timeout_seconds": args.job_timeout_seconds,
                "error": outcome.error,
                "integrity_audit": postflight_audit,
            }
            record["attempts"].append(attempt)
            record["complete"] = (
                outcome.return_code == 0 and postflight_audit.get("complete") is True
            )
            if not record["complete"] and outcome.return_code == 0 and _run_is_complete(job.config):
                quarantined = _quarantine_invalid_completion(job, postflight_audit)
                record.setdefault("invalidations", []).append(
                    {
                        "invalidated_at": _timestamp(),
                        "problems": list(postflight_audit.get("problems", [])),
                        "quarantined_output_dir": str(quarantined) if quarantined else None,
                    }
                )
            _atomic_json(ledger_path, ledger)
            if not record["complete"]:
                ledger["failed_job"] = job.identity
                ledger["finished_at"] = _timestamp()
                ledger.pop("active_attempt", None)
                _atomic_json(ledger_path, ledger)
                return 1

        ledger.pop("failed_job", None)
        ledger.pop("active_attempt", None)
        ledger["complete"] = True
        ledger["finished_at"] = _timestamp()
        _atomic_json(ledger_path, ledger)
        print(f"Dense training queue pool-{args.pool} complete", flush=True)
        return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one resumable GPU pool from a frozen single-family training queue"
    )
    parser.add_argument("--plan", type=Path, default=Path("configs/dense_training_queue.json"))
    parser.add_argument("--pool", choices=("a", "b"), required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--job-timeout-seconds",
        type=float,
        default=24 * 60 * 60,
        help="Total wall-time watchdog per matrix command (default: 86400 / 24 hours)",
    )
    parser.add_argument(
        "--termination-grace-seconds",
        type=float,
        default=120.0,
        help="Grace after watchdog SIGTERM before SIGKILL (default: 120)",
    )
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--wait-command-fragment")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args(argv)
    if (
        args.port <= 0
        or args.port >= 65535
        or args.max_retries < 0
        or any(
            not math.isfinite(value) or value <= 0
            for value in (
                args.poll_seconds,
                args.job_timeout_seconds,
                args.termination_grace_seconds,
            )
        )
    ):
        parser.error(
            "--port must be in [1, 65534], timeout/poll/grace values must be positive, "
            "and --max-retries must be non-negative"
        )
    if args.wait_command_fragment and args.wait_pid is None:
        parser.error("--wait-command-fragment requires --wait-pid")
    gpu_tokens = [token.strip() for token in args.gpus.split(",")]
    if len(gpu_tokens) != 4 or any(
        re.fullmatch(r"(?:0|[1-9][0-9]*)", token) is None for token in gpu_tokens
    ):
        parser.error("--gpus must contain exactly four canonical non-negative integers")
    gpu_ids = [int(token) for token in gpu_tokens]
    if len(set(gpu_ids)) != len(gpu_ids):
        parser.error("--gpus must identify four unique devices")
    args.gpus = ",".join(str(gpu) for gpu in gpu_ids)
    return args


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_pool(parse_args(argv)))


if __name__ == "__main__":
    main()
