from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    candidate = path.resolve().parent.parent
    if not (candidate / "pyproject.toml").is_file():
        raise ValueError(f"Queue plan must live under the repository configs directory: {path}")
    return candidate


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
            job = QueueJob(phase, matrix, selected[0])
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
            "complete": False,
            "started_at": _timestamp(),
            "plan": plan_identity,
            "pool": args.pool,
            "family": "dense",
            "gpus": args.gpus,
            "jobs": [],
        }
        _atomic_json(ledger_path, ledger)

    _wait_for_process(args.wait_pid, args.wait_command_fragment, args.poll_seconds)
    completed = {record["identity"] for record in ledger["jobs"] if record.get("complete")}
    for index, job in enumerate(jobs, start=1):
        if job.identity in completed and _run_is_complete(job.config):
            print(f"queue pool-{args.pool}: already complete {job.identity}", flush=True)
            continue
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
        completed_process = subprocess.run(command, cwd=repository, check=False)
        attempt = {
            "started_at": started,
            "finished_at": _timestamp(),
            "return_code": completed_process.returncode,
        }
        record["attempts"].append(attempt)
        record["complete"] = completed_process.returncode == 0 and _run_is_complete(job.config)
        _atomic_json(ledger_path, ledger)
        if not record["complete"]:
            ledger["failed_job"] = job.identity
            ledger["finished_at"] = _timestamp()
            _atomic_json(ledger_path, ledger)
            return 1

    ledger.pop("failed_job", None)
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
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--wait-command-fragment")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args(argv)
    if args.port <= 0 or args.max_retries < 0 or args.poll_seconds <= 0:
        parser.error("--port/--poll-seconds must be positive and --max-retries non-negative")
    if args.wait_command_fragment and args.wait_pid is None:
        parser.error("--wait-command-fragment requires --wait-pid")
    if len([gpu for gpu in args.gpus.split(",") if gpu.strip()]) != 4:
        parser.error("--gpus must identify exactly four devices")
    return args


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_pool(parse_args(argv)))


if __name__ == "__main__":
    main()
