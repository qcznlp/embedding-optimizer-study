"""Start canonical Dense confirmatory evaluation on a proven-idle training pool."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import load_matrix, resolve_matrix_path
from .confirmatory_data import load_confirmatory_protocol
from .confirmatory_evaluation import (
    _matrix_paths,
    audit_confirmatory_evaluations,
    audit_confirmatory_training,
)
from .confirmatory_matrix import audit_confirmatory_matrices
from .evaluate_matrix import (
    _evaluation_source_manifest,
    _record_evaluation_inputs,
    checkpoint_paths,
)
from .family_training_queue import QueueJob, load_queue_plan
from .gpu_lease import validate_disjoint_gpu_pools
from .matrix import _run_is_complete
from .scope import resolve_scope


class ConditionNotReady(RuntimeError):
    """A safe, transient reason not to launch the handoff yet."""


@dataclass(frozen=True)
class HandoffDecision:
    idle_pool: str
    active_pool: str
    gpu_tokens: str
    remaining_identity: str
    job_statuses: tuple[dict[str, Any], ...]
    queue_ledgers: dict[str, dict[str, Any]]
    active_provenance: dict[str, Any]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_argv(pid: int) -> list[str] | None:
    """Read only an explicitly supplied queue PID or one of its direct children."""

    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    values = [item.decode(errors="replace") for item in raw.split(b"\0") if item]
    return values or None


def _read_cwd(pid: int) -> Path:
    try:
        return Path(os.readlink(f"/proc/{pid}/cwd")).resolve()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError) as error:
        raise ConditionNotReady(f"Cannot resolve working directory for queue PID {pid}") from error


def _read_children(pid: int) -> list[int]:
    try:
        content = Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError) as error:
        raise ConditionNotReady(f"Cannot resolve children for queue PID {pid}") from error
    try:
        return [int(value) for value in content.split()]
    except ValueError as error:
        raise RuntimeError(f"Invalid child PID ledger for queue PID {pid}") from error


def _is_module(argv: list[str], module: str) -> bool:
    return any(argv[index : index + 2] == ["-m", module] for index in range(len(argv) - 1))


def _option_values(argv: list[str], name: str) -> list[str] | None:
    positions = [index for index, value in enumerate(argv) if value == name]
    if len(positions) > 1:
        raise RuntimeError(f"Process command repeats {name}")
    if not positions:
        return None
    start = positions[0] + 1
    end = start
    while end < len(argv) and not argv[end].startswith("--"):
        end += 1
    if start == end:
        raise RuntimeError(f"Process command has no value for {name}")
    return argv[start:end]


def _one_option(argv: list[str], name: str, *, default: str | None = None) -> str:
    values = _option_values(argv, name)
    if values is None:
        if default is None:
            raise RuntimeError(f"Process command omits {name}")
        return default
    if len(values) != 1:
        raise RuntimeError(f"Process command must give one value for {name}")
    return values[0]


def _resolved_process_path(pid: int, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else _read_cwd(pid) / path).resolve()


def _load_queue_ledger(
    path: Path,
    *,
    pool: str,
    gpus: str,
    plan_path: Path,
    jobs: list[QueueJob],
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConditionNotReady(f"Queue pool-{pool} ledger is not readable: {path}") from error
    plan_identity = {"path": str(plan_path), "sha256": _sha256(plan_path)}
    if (
        payload.get("schema_version") != 1
        or payload.get("plan") != plan_identity
        or payload.get("pool") != pool
        or payload.get("family") != "dense"
        or payload.get("gpus") != gpus
        or not isinstance(payload.get("jobs"), list)
        or not isinstance(payload.get("complete"), bool)
    ):
        raise RuntimeError(f"Queue pool-{pool} ledger identity differs from the frozen queue")
    records = payload["jobs"]
    identities = [record.get("identity") for record in records]
    expected_prefix = [job.identity for job in jobs[: len(records)]]
    if identities != expected_prefix or len(identities) != len(set(identities)):
        raise RuntimeError(f"Queue pool-{pool} ledger is not an exact frozen-order prefix")
    for index, (record, job) in enumerate(zip(records, jobs, strict=False), start=1):
        if (
            record.get("index") != index
            or Path(str(record.get("matrix"))).resolve() != job.matrix.resolve()
            or Path(str(record.get("output_dir"))).resolve() != job.config.output_dir.resolve()
            or not isinstance(record.get("complete"), bool)
            or not isinstance(record.get("attempts"), list)
        ):
            raise RuntimeError(f"Queue pool-{pool} record {index} has invalid provenance")
        if record["complete"]:
            tail_attempt = record["attempts"][-1] if record["attempts"] else {}
            integrity = record.get("last_integrity_audit") or {}
            successful_attempt = tail_attempt.get("return_code") == 0
            successful_reaudit = (
                integrity.get("complete") is True
                and integrity.get("verified_checkpoints")
                == integrity.get("expected_checkpoints")
                == 5
                and not integrity.get("problems")
            )
            if not successful_attempt and not successful_reaudit:
                raise RuntimeError(
                    f"Queue pool-{pool} completed record {index} lacks successful provenance"
                )
    if payload["complete"] and (
        len(records) != len(jobs) or any(record.get("complete") is not True for record in records)
    ):
        raise RuntimeError(f"Queue pool-{pool} claims completion without every frozen job")
    return payload


def _verify_active_queue(
    *,
    pid: int,
    pool: str,
    gpus: str,
    plan_path: Path,
    ledger_path: Path,
    remaining: QueueJob,
    read_argv: Callable[[int], list[str] | None],
    read_children: Callable[[int], list[int]],
) -> dict[str, Any]:
    queue_argv = read_argv(pid)
    if queue_argv is None:
        raise ConditionNotReady(f"Active queue pool-{pool} PID {pid} is not alive")
    if not _is_module(queue_argv, "embed_optim.family_training_queue"):
        raise RuntimeError(f"Active queue pool-{pool} PID is not the queue coordinator")
    if _one_option(queue_argv, "--pool") != pool or _one_option(queue_argv, "--gpus") != gpus:
        raise RuntimeError(f"Active queue pool-{pool} command has different pool/GPU provenance")
    declared_plan = _one_option(queue_argv, "--plan", default="configs/dense_training_queue.json")
    if _resolved_process_path(pid, declared_plan) != plan_path:
        raise RuntimeError(f"Active queue pool-{pool} command uses a different plan")
    declared_ledger = _one_option(queue_argv, "--ledger", default=str(ledger_path))
    if _resolved_process_path(pid, declared_ledger) != ledger_path:
        raise RuntimeError(f"Active queue pool-{pool} command uses a different ledger")

    children = read_children(pid)
    if len(children) != 1:
        raise ConditionNotReady(
            f"Active queue pool-{pool} must have exactly one direct training child; got {children}"
        )
    child_pid = children[0]
    child_argv = read_argv(child_pid)
    if child_argv is None:
        raise ConditionNotReady(f"Active queue pool-{pool} child {child_pid} is not readable")
    if not _is_module(child_argv, "embed_optim.matrix"):
        raise RuntimeError(f"Active queue pool-{pool} child is not a matrix trainer")
    families = _option_values(child_argv, "--families")
    run_ids = _option_values(child_argv, "--run-ids")
    if families != ["dense"] or run_ids != [remaining.config.run_id]:
        raise RuntimeError(
            f"Active queue pool-{pool} child does not select the remaining Dense run"
        )
    if (
        _resolved_process_path(child_pid, _one_option(child_argv, "--matrix"))
        != remaining.matrix.resolve()
        or _one_option(child_argv, "--gpus-a") != gpus
        or _one_option(child_argv, "--gpus-b") != gpus
    ):
        raise RuntimeError(f"Active queue pool-{pool} child has different matrix/GPU provenance")
    return {
        "queue_pid": pid,
        "queue_argv": queue_argv,
        "trainer_pid": child_pid,
        "trainer_argv": child_argv,
        "identity": remaining.identity,
    }


def inspect_handoff_condition(
    args: argparse.Namespace,
    *,
    run_is_complete: Callable[[Any], bool] = _run_is_complete,
    read_argv: Callable[[int], list[str] | None] = _read_argv,
    read_children: Callable[[int], list[int]] = _read_children,
) -> HandoffDecision:
    """Prove one active frozen job and one fully completed, disjoint pool."""

    plan_path, payload, jobs_by_pool = load_queue_plan(args.plan)
    repository = plan_path.parent.parent
    expected_scope = (repository / payload["scope_amendment"]["path"]).resolve()
    if args.scope_amendment.resolve() != expected_scope:
        raise RuntimeError("Handoff scope amendment differs from the frozen queue")
    families, _ = resolve_scope(("dense",), args.scope_amendment)
    if families != ("dense",):
        raise AssertionError("Dense evaluation handoff resolved a non-Dense scope")
    pools = validate_disjoint_gpu_pools(args.gpus_a, args.gpus_b)
    gpu_strings = {name: ",".join(tokens) for name, tokens in pools.items()}
    ledger_paths = {"a": args.ledger_a.resolve(), "b": args.ledger_b.resolve()}
    ledgers = {
        pool: _load_queue_ledger(
            ledger_paths[pool],
            pool=pool,
            gpus=gpu_strings[pool],
            plan_path=plan_path,
            jobs=jobs_by_pool[pool],
        )
        for pool in ("a", "b")
    }
    records = {
        pool: {record["identity"]: record for record in ledgers[pool]["jobs"]}
        for pool in ("a", "b")
    }
    statuses: list[dict[str, Any]] = []
    remaining: list[tuple[str, QueueJob]] = []
    for pool in ("a", "b"):
        for job in jobs_by_pool[pool]:
            artifact_complete = bool(run_is_complete(job.config))
            record = records[pool].get(job.identity)
            ledger_complete = record is not None and record.get("complete") is True
            if artifact_complete != ledger_complete:
                raise ConditionNotReady(
                    f"Queue pool-{pool} has not durably reconciled {job.identity}"
                )
            statuses.append(
                {
                    "pool": pool,
                    "identity": job.identity,
                    "phase": job.phase,
                    "artifact_complete": artifact_complete,
                    "ledger_complete": ledger_complete,
                }
            )
            if not artifact_complete:
                remaining.append((pool, job))
    if len(statuses) != 18 or len({item["identity"] for item in statuses}) != 18:
        raise RuntimeError("Handoff did not account for exactly 18 unique frozen queue jobs")
    if len(remaining) != 1:
        raise ConditionNotReady(f"Frozen Dense queue has {len(remaining)} unfinished jobs, not one")

    active_pool, remaining_job = remaining[0]
    idle_pool = "b" if active_pool == "a" else "a"
    if remaining_job != jobs_by_pool[active_pool][-1] or remaining_job.phase != "short-branch":
        raise RuntimeError("The only unfinished job is not the frozen pool's final short branch")
    if ledgers[idle_pool].get("complete") is not True:
        raise ConditionNotReady(f"Candidate idle queue pool-{idle_pool} is not complete")
    if ledgers[active_pool].get("complete") is not False:
        raise RuntimeError(f"Active queue pool-{active_pool} incorrectly claims completion")
    if set(pools[idle_pool]) & set(pools[active_pool]):
        raise RuntimeError("Candidate idle pool overlaps the final training run's pool")

    queue_pids = {"a": args.queue_pid_a, "b": args.queue_pid_b}
    idle_argv = read_argv(queue_pids[idle_pool])
    if idle_argv is not None:
        raise ConditionNotReady(
            f"Candidate idle queue pool-{idle_pool} PID {queue_pids[idle_pool]} is still alive"
        )
    active = _verify_active_queue(
        pid=queue_pids[active_pool],
        pool=active_pool,
        gpus=gpu_strings[active_pool],
        plan_path=plan_path,
        ledger_path=ledger_paths[active_pool],
        remaining=remaining_job,
        read_argv=read_argv,
        read_children=read_children,
    )
    ledger_receipts = {
        pool: {
            "path": str(ledger_paths[pool]),
            "sha256": _sha256(ledger_paths[pool]),
            "complete": ledgers[pool]["complete"],
            "completed_jobs": sum(
                record.get("complete") is True for record in ledgers[pool]["jobs"]
            ),
        }
        for pool in ("a", "b")
    }
    return HandoffDecision(
        idle_pool=idle_pool,
        active_pool=active_pool,
        gpu_tokens=gpu_strings[idle_pool],
        remaining_identity=remaining_job.identity,
        job_statuses=tuple(statuses),
        queue_ledgers=ledger_receipts,
        active_provenance=active,
    )


def _deep_audit_confirmatory(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path, protocol = load_confirmatory_protocol(args.protocol)
    matrix_dir = Path(args.matrix_dir or protocol["training"]["matrix_output_dir"]).resolve()
    matrix_audit = audit_confirmatory_matrices(
        protocol_path,
        experiment_matrix=args.experiment_matrix,
        validation_spec=args.validation_spec,
        output_dir=matrix_dir,
    )
    per_seed: dict[str, Any] = {}
    for seed, matrix_path in _matrix_paths(protocol, matrix_dir).items():
        configs = [config for config in load_matrix(matrix_path) if config.model_family == "dense"]
        audit = audit_confirmatory_training(
            protocol_path,
            seed,
            configs,
            families=("dense",),
        )
        if (
            audit.get("complete") is not True
            or audit.get("errors")
            or audit.get("verified_runs") != 3
            or audit.get("verified_checkpoints") != 15
        ):
            details = "; ".join(str(item) for item in audit.get("errors", [])[:10])
            raise RuntimeError(f"Seed {seed} Dense confirmatory deep audit failed: {details}")
        seed_root = args.results_root.resolve() / f"seed{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        _record_evaluation_inputs(
            seed_root,
            {
                "dense": [
                    checkpoint for config in configs for checkpoint in checkpoint_paths(config, [5])
                ]
            },
        )
        per_seed[str(seed)] = {
            "matrix": str(matrix_path),
            "matrix_sha256": _sha256(matrix_path),
            "verified_runs": audit["verified_runs"],
            "verified_checkpoints": audit["verified_checkpoints"],
            "evaluation_inputs": {
                "path": str(seed_root / "evaluation_inputs.json"),
                "sha256": _sha256(seed_root / "evaluation_inputs.json"),
            },
        }
    return {
        "status": "complete",
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "matrix_manifest_sha256": matrix_audit["manifest_sha256"],
        "verified_runs": sum(item["verified_runs"] for item in per_seed.values()),
        "verified_checkpoints": sum(item["verified_checkpoints"] for item in per_seed.values()),
        "per_seed": per_seed,
    }


def _evaluation_command(args: argparse.Namespace, decision: HandoffDecision) -> list[str]:
    protocol_path, protocol = load_confirmatory_protocol(args.protocol)
    matrix_dir = Path(args.matrix_dir or protocol["training"]["matrix_output_dir"]).resolve()
    return [
        args.python,
        "-m",
        "embed_optim.confirmatory_evaluation",
        "--protocol",
        str(protocol_path),
        "--experiment-matrix",
        str(args.experiment_matrix.resolve()),
        "--validation-spec",
        str(args.validation_spec.resolve()),
        "--matrix-dir",
        str(matrix_dir),
        "--results-root",
        str(args.results_root.resolve()),
        "--families",
        "dense",
        "--scope-amendment",
        str(args.scope_amendment.resolve()),
        "--log-dir",
        str(args.evaluation_log_dir.resolve()),
        "--gpus-a",
        decision.gpu_tokens,
        "--gpus-b",
        decision.gpu_tokens,
        "--worker-python",
        args.worker_python,
        "--gpu-lock-dir",
        str(args.gpu_lock_dir.resolve()),
        "--gpu-lock-timeout-seconds",
        str(args.gpu_lock_timeout_seconds),
        "--receipt",
        str(args.receipt.resolve()),
    ]


def _run_with_timeout(command: list[str], args: argparse.Namespace) -> int:
    args.process_log.parent.mkdir(parents=True, exist_ok=True)
    with args.process_log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=args.workdir,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=args.evaluation_timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=args.termination_grace_seconds)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise TimeoutError(
                f"Dense evaluation handoff exceeded {args.evaluation_timeout_seconds}s"
            ) from None


def _result_receipt(args: argparse.Namespace) -> dict[str, Any]:
    audit = audit_confirmatory_evaluations(
        args.protocol,
        experiment_matrix=args.experiment_matrix,
        validation_spec=args.validation_spec,
        matrix_dir=args.matrix_dir,
        results_root=args.results_root,
        families=("dense",),
        scope_amendment=args.scope_amendment,
    )
    if audit.get("complete") is not True or audit.get("valid_units") != 126:
        raise RuntimeError("Dense confirmatory handoff did not produce 126 canonical final units")
    manifests = []
    expected_sources = _evaluation_source_manifest(args.workdir)
    for seed in (314159, 271828, 161803):
        for name in ("evaluation_runtime.json", "evaluation_inputs.json"):
            path = args.results_root.resolve() / f"seed{seed}" / name
            if not path.is_file():
                raise RuntimeError(f"Missing canonical evaluation provenance: {path}")
            if name == "evaluation_runtime.json":
                runtime = json.loads(path.read_text(encoding="utf-8"))
                if (
                    runtime.get("schema_version") != 2
                    or runtime.get("source_files") != expected_sources
                ):
                    raise RuntimeError(f"Canonical evaluation runtime provenance differs: {path}")
            manifests.append({"path": str(path), "sha256": _sha256(path)})
    return {
        "valid_units": audit["valid_units"],
        "expected_units": audit["expected_units"],
        "receipt": {"path": str(args.receipt.resolve()), "sha256": _sha256(args.receipt)},
        "manifests": manifests,
    }


def run_handoff(args: argparse.Namespace) -> int:
    plan_path, _, _ = load_queue_plan(args.plan)
    resolve_scope(("dense",), args.scope_amendment)
    ledger_path = args.handoff_ledger.resolve()
    lock_path = args.supervisor_lock.resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as supervisor_lock:
        try:
            fcntl.flock(supervisor_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another Dense evaluation handoff supervisor holds the lock"
            ) from error
        previous = None
        if ledger_path.is_file():
            previous = json.loads(ledger_path.read_text(encoding="utf-8"))
            if previous.get("status") == "complete":
                _deep_audit_confirmatory(args)
                _result_receipt(args)
                print("Dense confirmatory handoff is already complete", flush=True)
                return 0
            if not args.resume:
                raise RuntimeError("Incomplete handoff ledger exists; pass --resume to retry")
        ledger: dict[str, Any] = {
            "schema_version": 1,
            "status": "waiting-condition",
            "started_at": _timestamp(),
            "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
            "scope_amendment": {
                "path": str(args.scope_amendment.resolve()),
                "sha256": _sha256(args.scope_amendment.resolve()),
            },
            "families": ["dense"],
            "complete": False,
            "pool_gpus": {"a": args.gpus_a, "b": args.gpus_b},
            "queue_pids": {"a": args.queue_pid_a, "b": args.queue_pid_b},
            "attempts": list(previous.get("attempts", [])) if previous else [],
        }
        _atomic_json(ledger_path, ledger)
        try:
            deadline = time.monotonic() + args.condition_timeout_seconds
            last_reason = None
            while True:
                try:
                    decision = inspect_handoff_condition(args)
                    break
                except ConditionNotReady as error:
                    reason = str(error)
                    if reason != last_reason:
                        print(reason, flush=True)
                        ledger["last_not_ready"] = {"at": _timestamp(), "reason": reason}
                        _atomic_json(ledger_path, ledger)
                        last_reason = reason
                    if time.monotonic() >= deadline:
                        ledger.update(
                            status="condition-timeout",
                            complete=False,
                            finished_at=_timestamp(),
                            error=reason,
                        )
                        _atomic_json(ledger_path, ledger)
                        return 2
                    time.sleep(min(args.poll_seconds, max(0.0, deadline - time.monotonic())))

            ledger.update(
                status="deep-audit",
                handoff_condition={
                    "observed_at": _timestamp(),
                    "idle_pool": decision.idle_pool,
                    "active_pool": decision.active_pool,
                    "gpu_tokens": decision.gpu_tokens,
                    "remaining_identity": decision.remaining_identity,
                    "job_statuses": list(decision.job_statuses),
                    "queue_ledgers": decision.queue_ledgers,
                    "active_provenance": decision.active_provenance,
                },
            )
            _atomic_json(ledger_path, ledger)
            audit = _deep_audit_confirmatory(args)
            command = _evaluation_command(args, decision)
            attempt = {
                "started_at": _timestamp(),
                "command": command,
                "idle_pool": decision.idle_pool,
                "gpu_tokens": decision.gpu_tokens,
                "deep_audit": audit,
            }
            ledger["status"] = "evaluating"
            ledger["attempts"].append(attempt)
            _atomic_json(ledger_path, ledger)
            return_code = _run_with_timeout(command, args)
            attempt.update(finished_at=_timestamp(), return_code=return_code)
            if return_code:
                raise RuntimeError(f"Dense confirmatory evaluator exited {return_code}")
            result = _result_receipt(args)
            ledger.update(
                status="complete",
                complete=True,
                finished_at=_timestamp(),
                result=result,
            )
            _atomic_json(ledger_path, ledger)
            print("Dense confirmatory early evaluation handoff complete", flush=True)
            return 0
        except BaseException as error:
            ledger.update(
                status="error",
                complete=False,
                finished_at=_timestamp(),
                error=f"{type(error).__name__}: {error}",
            )
            _atomic_json(ledger_path, ledger)
            raise
        finally:
            fcntl.flock(supervisor_lock.fileno(), fcntl.LOCK_UN)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/dense_training_queue.json"))
    parser.add_argument(
        "--scope-amendment",
        type=Path,
        default=Path("configs/dense_scope_amendment.json"),
    )
    parser.add_argument("--ledger-a", type=Path, required=True)
    parser.add_argument("--ledger-b", type=Path, required=True)
    parser.add_argument("--queue-pid-a", type=int, required=True)
    parser.add_argument("--queue-pid-b", type=int, required=True)
    parser.add_argument("--gpus-a", default="0,1,2,3")
    parser.add_argument("--gpus-b", default="4,5,6,7")
    parser.add_argument("--protocol", type=Path, default=Path("configs/confirmatory_protocol.json"))
    parser.add_argument("--experiment-matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--validation-spec", type=Path, default=Path("configs/validation_probe.json")
    )
    parser.add_argument("--matrix-dir", type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results/confirmatory-beir"))
    parser.add_argument(
        "--evaluation-log-dir", type=Path, default=Path("logs/confirmatory-evaluation")
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("reports/confirmatory/evaluation-receipt.json"),
    )
    parser.add_argument(
        "--handoff-ledger",
        type=Path,
        default=Path("logs/dense-only-runtime/early-evaluation-handoff.json"),
    )
    parser.add_argument(
        "--supervisor-lock",
        type=Path,
        default=Path("logs/dense-only-runtime/early-evaluation-handoff.lock"),
    )
    parser.add_argument(
        "--gpu-lock-dir",
        type=Path,
        default=Path("logs/dense-only-runtime/gpu-leases"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--worker-python", default=sys.executable)
    parser.add_argument("--condition-timeout-seconds", type=float, default=172_800.0)
    parser.add_argument("--evaluation-timeout-seconds", type=float, default=86_400.0)
    parser.add_argument("--gpu-lock-timeout-seconds", type=float, default=86_400.0)
    parser.add_argument("--termination-grace-seconds", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--process-log", type=Path, default=Path("logs/dense-eval-handoff.log"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        validate_disjoint_gpu_pools(args.gpus_a, args.gpus_b)
    except ValueError as error:
        parser.error(str(error))
    if (
        args.queue_pid_a <= 1
        or args.queue_pid_b <= 1
        or args.queue_pid_a == args.queue_pid_b
        or min(
            args.condition_timeout_seconds,
            args.evaluation_timeout_seconds,
            args.gpu_lock_timeout_seconds,
            args.termination_grace_seconds,
            args.poll_seconds,
        )
        <= 0
    ):
        parser.error("PIDs must be distinct and positive; timeout/poll values must be positive")
    args.workdir = args.workdir.resolve()
    if not (args.workdir / "pyproject.toml").is_file():
        parser.error("--workdir must be the repository root")
    for name in (
        "plan",
        "scope_amendment",
        "protocol",
        "experiment_matrix",
        "validation_spec",
    ):
        value = getattr(args, name)
        setattr(args, name, resolve_matrix_path(value).resolve())
    if args.matrix_dir is not None:
        args.matrix_dir = args.matrix_dir.resolve()
    return args


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_handoff(parse_args(argv)))


if __name__ == "__main__":
    main()
