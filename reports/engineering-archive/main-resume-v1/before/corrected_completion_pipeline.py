"""Resume-safe operational controller for the corrected Dense campaign.

This controller makes no scientific choices.  It waits for the frozen 12-run
matrix to become deeply complete, backs up each run as soon as it is complete,
and then invokes the already source-bound evaluation and analysis entrypoints
in their declared dependency order.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from .config import RunConfig, load_matrix, resolve_matrix_path
from .matrix import _run_is_complete


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: tuple[str, ...]


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


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


def _file_identity(path: Path, repository: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        label = str(resolved.relative_to(repository))
    except ValueError:
        label = str(resolved)
    return {
        "path": label,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _module(python: str, name: str, *arguments: str) -> tuple[str, ...]:
    return (python, "-m", name, *arguments)


def _selected_configs(matrix: Path) -> list[RunConfig]:
    configs = load_matrix(matrix)
    if (
        len(configs) != 12
        or any(config.model_family != "dense" for config in configs)
        or any(config.dense_can_flatten_inputs for config in configs)
        or {config.optimizer.name for config in configs} != {"adamw", "muon", "normuon"}
        or len({config.run_id for config in configs}) != 12
        or any(len(config.checkpoint_fractions) != 5 for config in configs)
    ):
        raise ValueError("Corrected completion requires the frozen 12-run padded Dense matrix")
    return configs


def pipeline_steps(args: argparse.Namespace, repository: Path) -> list[PipelineStep]:
    python = str(args.python)
    matrix = str(args.matrix.resolve())
    gpus = str(args.gpus)
    return [
        PipelineStep(
            "training-progress-receipt",
            _module(
                python,
                "embed_optim.corrected_progress",
                "--matrix",
                matrix,
                "--log-dir",
                str(args.training_log_dir.resolve()),
                "--output",
                str((repository / "CURRENT_PROGRESS.json").resolve()),
            ),
        ),
        PipelineStep(
            "checkpoint-backup-audit",
            _module(
                python,
                "embed_optim.corrected_checkpoint_backup",
                "--matrix",
                matrix,
                "--repo-id",
                str(args.checkpoint_repo),
                "--remote-prefix",
                str(args.checkpoint_prefix),
                "--audit-only",
            ),
        ),
        PipelineStep(
            "padded-validation",
            _module(
                python,
                "embed_optim.corrected_validation_matrix",
                "--matrix",
                matrix,
                "--gpus",
                gpus,
            ),
        ),
        PipelineStep(
            "padded-validation-audit",
            _module(
                python,
                "embed_optim.corrected_validation_matrix",
                "--matrix",
                matrix,
                "--gpus",
                gpus,
                "--audit-only",
                "--verify-hashes",
            ),
        ),
        PipelineStep(
            "decontaminated-beir",
            _module(
                python,
                "embed_optim.corrected_beir_evaluation",
                "--matrix",
                matrix,
                "--gpus",
                gpus,
            ),
        ),
        PipelineStep(
            "decontaminated-beir-audit",
            _module(
                python,
                "embed_optim.corrected_beir_evaluation",
                "--matrix",
                matrix,
                "--gpus",
                gpus,
                "--audit-only",
            ),
        ),
        PipelineStep(
            "weight-space",
            _module(
                python,
                "embed_optim.corrected_geometry_matrix",
                "--matrix",
                matrix,
                "--local-files-only",
            ),
        ),
        PipelineStep(
            "outcome-summary",
            _module(python, "embed_optim.corrected_outcome_summary", "--matrix", matrix),
        ),
        PipelineStep(
            "retrieval-bridge",
            _module(python, "embed_optim.corrected_retrieval_bridge"),
        ),
        PipelineStep(
            "execution-sensitivity",
            _module(python, "embed_optim.corrected_execution_sensitivity"),
        ),
        PipelineStep(
            "publication-render",
            _module(python, "embed_optim.corrected_publication"),
        ),
        PipelineStep(
            "paper-release",
            ("make", "-C", str((repository / "paper").resolve()), "release", f"PYTHON={python}"),
        ),
        PipelineStep(
            "paper-audit",
            _module(
                python,
                "embed_optim.paper_audit",
                "--strict",
                "--families",
                "dense",
                "--scope-amendment",
                str((repository / "configs/dense_scope_amendment.json").resolve()),
            ),
        ),
        PipelineStep(
            "portable-evidence-audit",
            (
                python,
                str((repository / "scripts/portable_evidence.py").resolve()),
                "--audit-only",
            ),
        ),
        PipelineStep("tests", _module(python, "pytest", "-q")),
        PipelineStep("ruff-check", _module(python, "ruff", "check", "src", "tests", "scripts")),
        PipelineStep(
            "ruff-format-check",
            _module(python, "ruff", "format", "--check", "src", "tests", "scripts"),
        ),
    ]


def _contract(
    args: argparse.Namespace,
    repository: Path,
    steps: list[PipelineStep],
) -> dict[str, Any]:
    protocol_names = (
        "dense_no_packing_execution_protocol.json",
        "dense_no_packing_evaluation_protocol.json",
        "dense_no_packing_analysis_protocol.json",
        "dense_no_packing_outcome_protocol.json",
        "dense_no_packing_bridge_implementation_protocol_v2.json",
        "dense_no_packing_sensitivity_implementation_protocol.json",
        "dense_no_packing_publication_protocol.json",
    )
    sources = [
        Path(__file__).resolve(),
        (repository / "src/embed_optim/corrected_checkpoint_backup.py").resolve(),
        args.matrix.resolve(),
    ]
    sources.extend((repository / "configs" / name).resolve() for name in protocol_names)
    body = {
        "schema_version": 1,
        "sources": [_file_identity(path, repository) for path in sources],
        "steps": [
            {"index": index, "name": step.name, "command": list(step.command)}
            for index, step in enumerate(steps, start=1)
        ],
        "arguments": {
            "matrix": str(args.matrix.resolve()),
            "python": str(args.python),
            "gpus": str(args.gpus),
            "training_log_dir": str(args.training_log_dir.resolve()),
            "checkpoint_repo": str(args.checkpoint_repo),
            "checkpoint_prefix": str(args.checkpoint_prefix),
        },
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return {**body, "sha256": hashlib.sha256(encoded).hexdigest()}


@contextmanager
def _exclusive_lease(path: Path) -> Iterator[int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A corrected completion controller is already active") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started_at={_timestamp()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield handle.fileno()


def _new_ledger(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": "corrected_dense_no_packing_completion",
        "status": "waiting_for_training",
        "complete": False,
        "started_at_utc": _timestamp(),
        "observed_at_utc": _timestamp(),
        "contract": contract,
        "complete_runs": [],
        "backups": {},
        "steps": [],
    }


def _load_ledger(path: Path, contract: dict[str, Any], *, resume: bool) -> dict[str, Any]:
    if not path.is_file():
        return _new_ledger(contract)
    if not resume:
        raise FileExistsError(f"Corrected completion ledger already exists: {path}")
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if ledger.get("contract") != contract:
        raise RuntimeError("Corrected completion contract differs from the existing ledger")
    if ledger.get("complete") is True:
        return ledger
    ledger.pop("failed_step", None)
    ledger["steps"] = []
    for backup in (ledger.get("backups") or {}).values():
        if backup.get("complete") is True:
            backup["complete"] = False
            backup["audit_only"] = True
    ledger["status"] = "waiting_for_training"
    ledger["observed_at_utc"] = _timestamp()
    return ledger


def _assert_contract_unchanged(
    args: argparse.Namespace,
    repository: Path,
    steps: list[PipelineStep],
    expected: dict[str, Any],
) -> None:
    if _contract(args, repository, steps) != expected:
        raise RuntimeError("Corrected completion source or command contract changed while running")


def _run_attempt(
    command: tuple[str, ...],
    *,
    repository: Path,
    log_path: Path,
    lease_fd: int,
    run_command: Callable[..., subprocess.CompletedProcess[Any]],
) -> tuple[int | None, str | None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return_code = None
    execution_error = None
    with log_path.open("w", encoding="utf-8") as handle:
        try:
            result = run_command(
                command,
                cwd=repository,
                stdout=handle,
                stderr=subprocess.STDOUT,
                pass_fds=(lease_fd,),
                check=False,
            )
            return_code = result.returncode
        except OSError as error:
            execution_error = f"{type(error).__name__}: {error}"
            handle.write(f"controller execution error: {execution_error}\n")
            handle.flush()
            os.fsync(handle.fileno())
    return return_code, execution_error


def _run_until_success(
    *,
    label: str,
    command: tuple[str, ...],
    record: dict[str, Any],
    args: argparse.Namespace,
    repository: Path,
    log_dir: Path,
    ledger: dict[str, Any],
    ledger_path: Path,
    steps: list[PipelineStep],
    contract: dict[str, Any],
    lease_fd: int,
    run_command: Callable[..., subprocess.CompletedProcess[Any]],
    sleeper: Callable[[float], None],
) -> None:
    attempts = record.setdefault("attempts", [])
    while record.get("complete") is not True:
        _assert_contract_unchanged(args, repository, steps, contract)
        attempt = len(attempts) + 1
        log_path = log_dir / f"{label}.attempt-{attempt}.log"
        started = _timestamp()
        return_code, execution_error = _run_attempt(
            command,
            repository=repository,
            log_path=log_path,
            lease_fd=lease_fd,
            run_command=run_command,
        )
        attempt_record: dict[str, Any] = {
            "attempt": attempt,
            "started_at_utc": started,
            "finished_at_utc": _timestamp(),
            "return_code": return_code,
            "log": _file_identity(log_path, repository),
        }
        if execution_error is not None:
            attempt_record["execution_error"] = execution_error
        attempts.append(attempt_record)
        ledger["observed_at_utc"] = _timestamp()
        if return_code == 0:
            ledger.pop("failed_step", None)
            record["complete"] = True
            record["finished_at_utc"] = _timestamp()
            _atomic_json(ledger_path, ledger)
            return
        ledger["status"] = "retry_wait"
        ledger["failed_step"] = label
        _atomic_json(ledger_path, ledger)
        if args.max_attempts and attempt >= args.max_attempts:
            raise RuntimeError(f"Corrected completion step exhausted retries: {label}")
        sleeper(args.retry_delay)


def _backup_command(
    args: argparse.Namespace, config: RunConfig, *, audit_only: bool
) -> tuple[str, ...]:
    command = list(
        _module(
            str(args.python),
            "embed_optim.corrected_checkpoint_backup",
            "--matrix",
            str(args.matrix.resolve()),
            "--repo-id",
            str(args.checkpoint_repo),
            "--remote-prefix",
            str(args.checkpoint_prefix),
            "--run-ids",
            config.run_id,
        )
    )
    if audit_only:
        command.append("--audit-only")
    return tuple(command)


def run_pipeline(
    args: argparse.Namespace,
    *,
    run_command: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    repository = args.workdir.resolve()
    args.matrix = resolve_matrix_path(args.matrix).resolve()
    configs = _selected_configs(args.matrix)
    steps = pipeline_steps(args, repository)
    contract = _contract(args, repository, steps)
    log_dir = (repository / args.log_dir).resolve()
    ledger_path = log_dir / "pipeline-ledger.json"
    with _exclusive_lease(log_dir / "controller.lease") as lease_fd:
        ledger = _load_ledger(ledger_path, contract, resume=args.resume)
        if ledger.get("complete") is True:
            return 0
        _atomic_json(ledger_path, ledger)

        while True:
            _assert_contract_unchanged(args, repository, steps, contract)
            complete = [config for config in configs if _run_is_complete(config)]
            ledger["complete_runs"] = [config.run_id for config in complete]
            ledger["training_runs_complete"] = len(complete)
            ledger["training_runs_expected"] = len(configs)
            ledger["status"] = "waiting_for_training"
            ledger["observed_at_utc"] = _timestamp()
            _atomic_json(ledger_path, ledger)

            for config in complete:
                backup = ledger["backups"].setdefault(
                    config.run_id,
                    {
                        "run_id": config.run_id,
                        "command": list(_backup_command(args, config, audit_only=False)),
                        "attempts": [],
                        "complete": False,
                    },
                )
                if backup.get("complete") is True:
                    continue
                command = _backup_command(
                    args,
                    config,
                    audit_only=bool(backup.get("audit_only")),
                )
                backup["command"] = list(command)
                ledger["status"] = "checkpoint_backup"
                ledger["active_run"] = config.run_id
                _atomic_json(ledger_path, ledger)
                _run_until_success(
                    label=f"backup-{config.run_id}",
                    command=command,
                    record=backup,
                    args=args,
                    repository=repository,
                    log_dir=log_dir,
                    ledger=ledger,
                    ledger_path=ledger_path,
                    steps=steps,
                    contract=contract,
                    lease_fd=lease_fd,
                    run_command=run_command,
                    sleeper=sleeper,
                )
                backup.pop("audit_only", None)
                ledger.pop("active_run", None)
                _atomic_json(ledger_path, ledger)

            if len(complete) == len(configs):
                break
            sleeper(args.poll_seconds)

        ledger["status"] = "finalizing"
        ledger["steps"] = []
        _atomic_json(ledger_path, ledger)
        for index, step in enumerate(steps, start=1):
            record: dict[str, Any] = {
                "index": index,
                "name": step.name,
                "command": list(step.command),
                "attempts": [],
                "complete": False,
            }
            ledger["steps"].append(record)
            ledger["active_step"] = step.name
            ledger["status"] = "finalizing"
            _atomic_json(ledger_path, ledger)
            _run_until_success(
                label=f"{index:02d}-{step.name}",
                command=step.command,
                record=record,
                args=args,
                repository=repository,
                log_dir=log_dir,
                ledger=ledger,
                ledger_path=ledger_path,
                steps=steps,
                contract=contract,
                lease_fd=lease_fd,
                run_command=run_command,
                sleeper=sleeper,
            )

        ledger.pop("active_step", None)
        ledger.pop("failed_step", None)
        ledger["status"] = "complete"
        ledger["complete"] = True
        ledger["finished_at_utc"] = _timestamp()
        ledger["observed_at_utc"] = _timestamp()
        _atomic_json(ledger_path, ledger)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Back up and finalize the corrected padded Dense experiment"
    )
    parser.add_argument(
        "--matrix", type=Path, default=Path("configs/dense_no_packing_retrain.yaml")
    )
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--training-log-dir", type=Path, default=Path("logs/dense-no-packing-v1"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/dense-no-packing-finalization"))
    parser.add_argument("--checkpoint-repo", default="qcz/embedding-optimizer-study-checkpoints")
    parser.add_argument("--checkpoint-prefix", default="corrected-dense-no-packing-v1/dense")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--retry-delay", type=float, default=300.0)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Attempts per operation; zero retries indefinitely",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        parser.error("--gpus must contain unique GPU IDs")
    if args.poll_seconds <= 0 or args.retry_delay < 0 or args.max_attempts < 0:
        parser.error("Polling must be positive and retry controls must be non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_pipeline(parse_args(argv)))


if __name__ == "__main__":
    main()
