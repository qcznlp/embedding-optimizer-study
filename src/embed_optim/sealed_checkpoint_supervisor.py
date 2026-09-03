"""Continuously back up sealed checkpoints from the corrected Dense campaign.

The corrected completion controller uploads an entire run after all five stages
are complete.  This independent, CPU/network-only supervisor closes the
durability window for checkpoints produced earlier in a run.  It never marks a
run scientifically complete and never launches or controls training.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from huggingface_hub import HfApi

from .config import RunConfig, load_matrix
from .corrected_checkpoint_backup import DEFAULT_PREFIX, DEFAULT_REPO
from .incremental_checkpoint_backup import (
    DEFAULT_RECEIPT_ROOT,
    REQUIRED_CHECKPOINT_FILES,
    backup_checkpoint,
    validate_sealed_checkpoint,
)

DEFAULT_LOG_DIR = Path("logs/dense-no-packing-sealed-backup")
DEFAULT_FULL_RECEIPT_ROOT = Path("reports/dense-no-packing/checkpoint-backup")
DEFAULT_COMPLETION_LEDGER = Path("logs/dense-no-packing-finalization/pipeline-ledger.json")


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
    temporary.replace(path)


def _file_identity(path: Path, workdir: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        label = str(resolved.relative_to(workdir.resolve()))
    except ValueError:
        label = str(resolved)
    return {
        "path": label,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _selected_configs(matrix: Path) -> list[RunConfig]:
    configs = load_matrix(matrix)
    if (
        len(configs) != 12
        or len({config.run_id for config in configs}) != 12
        or any(config.model_family != "dense" for config in configs)
        or any(config.dense_can_flatten_inputs for config in configs)
        or any(len(config.checkpoint_fractions) != 5 for config in configs)
        or {config.optimizer.name for config in configs} != {"adamw", "muon", "normuon"}
    ):
        raise ValueError("Sealed-checkpoint backup requires the 12-run padded Dense matrix")
    return configs


def _checkpoint_schedule(config: RunConfig) -> list[int] | None:
    path = config.output_dir / "checkpoint_schedule.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        steps = [int(value) for value in payload["steps"]]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid checkpoint schedule {path}: {error}") from error
    if len(steps) != 5 or steps != sorted(set(steps)) or steps[0] <= 0:
        raise RuntimeError(f"Invalid checkpoint schedule {path}: expected five increasing steps")
    if payload.get("max_steps") is not None and int(payload["max_steps"]) != steps[-1]:
        raise RuntimeError(f"Invalid checkpoint schedule {path}: max_steps differs from final step")
    return steps


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"
    if not isinstance(payload, dict):
        return None, "receipt is not a JSON object"
    return payload, None


def _inventory_is_complete(payload: dict[str, Any], *, require_digest: bool) -> bool:
    inventory = payload.get("inventory")
    if not isinstance(inventory, dict) or inventory.get("complete") is not True:
        return False
    try:
        counts_match = int(inventory["local_files"]) == int(inventory["remote_files"])
        bytes_match = int(inventory["local_bytes"]) == int(inventory["remote_bytes"])
    except (KeyError, TypeError, ValueError):
        return False
    if not counts_match or not bytes_match:
        return False
    mismatch_fields = ["missing", "extra", "size_mismatch"]
    if require_digest:
        mismatch_fields.append("digest_mismatch")
    return all(inventory.get(field) == [] for field in mismatch_fields)


def _full_run_coverage(
    config: RunConfig,
    *,
    repo_id: str,
    remote_prefix: str,
    receipt_root: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    path = receipt_root / f"{config.run_id}.json"
    receipt, error = _read_json(path)
    if error is not None or receipt is None:
        return None, error
    expected_prefix = f"{remote_prefix.rstrip('/')}/{config.run_id}"
    valid = (
        receipt.get("schema_version") == 1
        and receipt.get("status") == "complete"
        and receipt.get("run_id") == config.run_id
        and receipt.get("repo_id") == repo_id
        and receipt.get("repo_type") == "model"
        and receipt.get("remote_prefix") == expected_prefix
        and bool(receipt.get("commit_oid"))
        and bool(receipt.get("commit_url"))
        and _inventory_is_complete(receipt, require_digest=False)
    )
    if not valid:
        return None, f"invalid full-run receipt: {path}"
    return {
        "kind": "full_run_receipt",
        "receipt": str(path),
        "commit_oid": receipt["commit_oid"],
        "remote_prefix": expected_prefix,
    }, None


def _incremental_coverage(
    config: RunConfig,
    step: int,
    *,
    matrix: Path,
    repo_id: str,
    remote_prefix: str,
    receipt_root: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    path = receipt_root / f"{config.run_id}-checkpoint-{step}.json"
    receipt, error = _read_json(path)
    if error is not None or receipt is None:
        return None, error
    expected_prefix = f"{remote_prefix.rstrip('/')}/{config.run_id}/checkpoint-{step}"
    matrix_record = receipt.get("matrix") or {}
    try:
        checkpoint_step_matches = int(receipt.get("checkpoint_step", -1)) == step
        required_files_match = set(receipt.get("required_files") or []) == set(
            REQUIRED_CHECKPOINT_FILES
        )
    except (TypeError, ValueError):
        checkpoint_step_matches = False
        required_files_match = False
    valid = (
        receipt.get("schema_version") == 1
        and receipt.get("status") == "complete"
        and receipt.get("scientific_completion") is False
        and receipt.get("run_id") == config.run_id
        and checkpoint_step_matches
        and receipt.get("repo_id") == repo_id
        and receipt.get("repo_type") == "model"
        and receipt.get("remote_prefix") == expected_prefix
        and required_files_match
        and matrix_record.get("sha256") == _sha256(matrix)
        and bool(receipt.get("commit_oid"))
        and bool(receipt.get("commit_url"))
        and _inventory_is_complete(receipt, require_digest=True)
    )
    if not valid:
        return None, f"invalid incremental receipt: {path}"
    return {
        "kind": "incremental_checkpoint_receipt",
        "receipt": str(path),
        "commit_oid": receipt["commit_oid"],
        "remote_prefix": expected_prefix,
        "files": receipt["inventory"]["remote_files"],
        "bytes": receipt["inventory"]["remote_bytes"],
    }, None


def _checkpoint_age_seconds(path: Path, now_epoch: float) -> float:
    newest = max(
        (item.stat().st_mtime for item in path.rglob("*") if item.is_file()),
        default=path.stat().st_mtime,
    )
    return max(0.0, now_epoch - newest)


def _completion_backup_active(path: Path, run_id: str) -> bool:
    ledger, _ = _read_json(path)
    if ledger is None:
        return False
    backup = (ledger.get("backups") or {}).get(run_id) or {}
    return ledger.get("active_run") == run_id or (
        backup.get("complete") is not True and bool(backup.get("attempts"))
    )


def _contract(args: argparse.Namespace, workdir: Path) -> dict[str, Any]:
    source = Path(__file__).resolve()
    incremental_source = source.with_name("incremental_checkpoint_backup.py")
    body = {
        "schema_version": 1,
        "sources": [
            _file_identity(source, workdir),
            _file_identity(incremental_source, workdir),
            _file_identity(args.matrix, workdir),
        ],
        "arguments": {
            "matrix": str(args.matrix.resolve()),
            "repo_id": str(args.repo_id),
            "remote_prefix": str(args.remote_prefix),
            "receipt_root": str(args.receipt_root.resolve()),
            "full_receipt_root": str(args.full_receipt_root.resolve()),
            "completion_ledger": str(args.completion_ledger.resolve()),
            "stability_seconds": float(args.stability_seconds),
            "final_grace_seconds": float(args.final_grace_seconds),
        },
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return {**body, "sha256": hashlib.sha256(encoded).hexdigest()}


def _new_state(contract: dict[str, Any], expected: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": "corrected_dense_no_packing_checkpoint_durability",
        "status": "watching",
        "complete": False,
        "scientific_completion": False,
        "started_at_utc": _utc_now(),
        "observed_at_utc": _utc_now(),
        "contract": contract,
        "expected_checkpoints": expected,
        "covered_checkpoints": 0,
        "runs": {},
    }


def _load_state(path: Path, contract: dict[str, Any], expected: int) -> dict[str, Any]:
    if not path.is_file():
        return _new_state(contract, expected)
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != 1 or state.get("contract") != contract:
        raise RuntimeError("Sealed-checkpoint supervisor contract differs from its state")
    if int(state.get("expected_checkpoints", -1)) != expected:
        raise RuntimeError("Sealed-checkpoint supervisor expected count changed")
    return state


def _process_cycle(
    configs: list[RunConfig],
    args: argparse.Namespace,
    state: dict[str, Any],
    *,
    api: HfApi,
    backup: Callable[..., dict[str, Any]] = backup_checkpoint,
    now_epoch: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Inspect one artifact snapshot and upload each newly sealed checkpoint."""

    now_epoch = time.time() if now_epoch is None else now_epoch
    previous_runs = state.get("runs") or {}
    runs: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    covered = 0
    failures = 0

    for config in configs:
        previous = previous_runs.get(config.run_id) or {}
        run_record: dict[str, Any] = {
            "optimizer": config.optimizer.name,
            "schedule_ready": False,
            "checkpoints": {},
        }
        runs[config.run_id] = run_record
        try:
            schedule = _checkpoint_schedule(config)
        except RuntimeError as error:
            run_record["status"] = "invalid_schedule"
            run_record["error"] = str(error)
            failures += 1
            continue
        if schedule is None:
            run_record["status"] = "waiting_for_schedule"
            continue
        run_record["schedule_ready"] = True
        run_record["expected_steps"] = schedule

        full_coverage, full_error = _full_run_coverage(
            config,
            repo_id=args.repo_id,
            remote_prefix=args.remote_prefix,
            receipt_root=args.full_receipt_root,
        )
        if full_coverage is not None:
            run_record["status"] = "covered_by_full_run_receipt"
            run_record["covered_checkpoints"] = 5
            for step in schedule:
                run_record["checkpoints"][str(step)] = {
                    "status": "covered",
                    "coverage": full_coverage,
                }
                covered += 1
            continue
        if full_error is not None:
            run_record["full_receipt_warning"] = full_error

        for step in schedule:
            previous_checkpoint = (previous.get("checkpoints") or {}).get(str(step)) or {}
            record: dict[str, Any] = {
                "step": step,
                "attempts": int(previous_checkpoint.get("attempts", 0)),
            }
            run_record["checkpoints"][str(step)] = record
            coverage, receipt_error = _incremental_coverage(
                config,
                step,
                matrix=args.matrix,
                repo_id=args.repo_id,
                remote_prefix=args.remote_prefix,
                receipt_root=args.receipt_root,
            )
            if coverage is not None:
                record.update({"status": "covered", "coverage": coverage})
                covered += 1
                continue
            if receipt_error is not None:
                record.update({"status": "invalid_receipt", "error": receipt_error})
                failures += 1
                continue

            checkpoint = config.output_dir / f"checkpoint-{step}"
            if not checkpoint.is_dir():
                record["status"] = "waiting_for_checkpoint"
                continue
            try:
                validate_sealed_checkpoint(config, step)
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
                record.update(
                    {
                        "status": "waiting_for_checkpoint_seal",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            if step == schedule[-1]:
                try:
                    age = _checkpoint_age_seconds(checkpoint, now_epoch)
                except OSError as error:
                    record.update(
                        {
                            "status": "waiting_for_checkpoint_stability",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    continue
                record["age_seconds"] = round(age, 3)
                if age < args.final_grace_seconds:
                    record["status"] = "waiting_for_full_run_backup_grace"
                    continue
                if _completion_backup_active(args.completion_ledger, config.run_id):
                    record["status"] = "waiting_for_active_full_run_backup"
                    continue

            record["attempts"] += 1
            try:
                receipt = backup(
                    api,
                    config,
                    step,
                    matrix_path=args.matrix,
                    repo_id=args.repo_id,
                    remote_prefix=args.remote_prefix,
                    receipt_root=args.receipt_root,
                    stability_seconds=args.stability_seconds,
                    audit_only=False,
                )
            except Exception as error:  # noqa: BLE001 - a daemon must retain retry state
                record.update(
                    {
                        "status": "retry_wait",
                        "error": f"{type(error).__name__}: {error}",
                        "attempted_at_utc": _utc_now(),
                    }
                )
                failures += 1
                events.append(
                    {
                        "event": "checkpoint_backup_failed",
                        "run_id": config.run_id,
                        "step": step,
                        "error": record["error"],
                    }
                )
                continue
            coverage = {
                "kind": "incremental_checkpoint_receipt",
                "receipt": str(args.receipt_root / f"{config.run_id}-checkpoint-{step}.json"),
                "commit_oid": receipt["commit_oid"],
                "remote_prefix": receipt["remote_prefix"],
                "files": receipt["inventory"]["remote_files"],
                "bytes": receipt["inventory"]["remote_bytes"],
            }
            record.update(
                {
                    "status": "covered",
                    "coverage": coverage,
                    "completed_at_utc": _utc_now(),
                }
            )
            covered += 1
            events.append(
                {
                    "event": "checkpoint_backed_up",
                    "run_id": config.run_id,
                    "step": step,
                    "commit_oid": receipt["commit_oid"],
                    "files": receipt["inventory"]["remote_files"],
                    "bytes": receipt["inventory"]["remote_bytes"],
                }
            )

        run_covered = sum(
            item.get("status") == "covered" for item in run_record["checkpoints"].values()
        )
        run_record["covered_checkpoints"] = run_covered
        run_record["status"] = "covered" if run_covered == 5 else "watching"

    expected = len(configs) * 5
    state.update(
        {
            "observed_at_utc": _utc_now(),
            "expected_checkpoints": expected,
            "covered_checkpoints": covered,
            "pending_checkpoints": expected - covered,
            "cycle_failures": failures,
            "runs": runs,
            "complete": covered == expected,
            "status": "complete" if covered == expected else "watching",
            "scientific_completion": False,
        }
    )
    if covered == expected and "finished_at_utc" not in state:
        state["finished_at_utc"] = _utc_now()
    return state, events


@contextmanager
def _exclusive_lease(path: Path) -> Iterator[int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A sealed-checkpoint backup supervisor is already active") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started_at={_utc_now()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield handle.fileno()


def run_supervisor(
    args: argparse.Namespace,
    *,
    api: HfApi | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    workdir = args.workdir.resolve()
    os.chdir(workdir)
    args.matrix = args.matrix.resolve()
    args.receipt_root = args.receipt_root.resolve()
    args.full_receipt_root = args.full_receipt_root.resolve()
    args.completion_ledger = args.completion_ledger.resolve()
    args.state = args.state.resolve()
    args.lease = args.lease.resolve()
    configs = _selected_configs(args.matrix)
    contract = _contract(args, workdir)
    state = _load_state(args.state, contract, len(configs) * 5)
    api = HfApi() if api is None else api

    with _exclusive_lease(args.lease):
        while state.get("complete") is not True:
            if _contract(args, workdir) != contract:
                raise RuntimeError("Sealed-checkpoint supervisor source or matrix changed")
            state, events = _process_cycle(configs, args, state, api=api)
            _atomic_json(args.state, state)
            for event in events:
                print(json.dumps(event, sort_keys=True), flush=True)
            if args.once or state.get("complete") is True:
                break
            sleeper(args.poll_seconds)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously upload sealed corrected Dense checkpoints"
    )
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--matrix", type=Path, default=Path("configs/dense_no_packing_retrain.yaml")
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--remote-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--receipt-root", type=Path, default=DEFAULT_RECEIPT_ROOT)
    parser.add_argument("--full-receipt-root", type=Path, default=DEFAULT_FULL_RECEIPT_ROOT)
    parser.add_argument("--completion-ledger", type=Path, default=DEFAULT_COMPLETION_LEDGER)
    parser.add_argument("--state", type=Path, default=DEFAULT_LOG_DIR / "state.json")
    parser.add_argument("--lease", type=Path, default=DEFAULT_LOG_DIR / "supervisor.lease")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--stability-seconds", type=float, default=2.0)
    parser.add_argument("--final-grace-seconds", type=float, default=180.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or args.stability_seconds < 0 or args.final_grace_seconds < 0:
        parser.error("Polling must be positive and timing controls must be non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_supervisor(parse_args(argv)))


if __name__ == "__main__":
    main()
