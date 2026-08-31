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

from .dense_completion_pipeline import (
    CORE_STEP_NAMES,
    VALIDATION_STEP_NAMES,
    _args_from_pipeline_arguments,
    _assert_step_contract_unchanged,
    _repository_contract_sources,
    _step_contract,
    _validate_training_inputs,
)
from .dense_completion_pipeline import (
    pipeline_steps as completion_pipeline_steps,
)
from .scope import resolve_scope, scope_amendments_equal


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_time_ticks: int
    command: str


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _module(python: str, module: str, *arguments: str) -> tuple[str, ...]:
    return python, "-m", module, *arguments


def _under_workdir(path: Path, workdir: Path) -> Path:
    return path.resolve() if path.is_absolute() else (workdir / path).resolve()


def pipeline_steps(args: argparse.Namespace) -> list[PipelineStep]:
    workdir = args.workdir.resolve()
    scope = str(_under_workdir(args.scope_amendment, workdir))
    dense_scope = ("--families", "dense", "--scope-amendment", scope)
    causal_protocol = "configs/causal_chain_analysis.json"
    lint_targets = ("src", "tests", "scripts/eval")
    steps = [
        PipelineStep(
            "temporal-predictors-fresh-audit",
            _module(
                args.python,
                "embed_optim.temporal_short_branch_predictors",
                "--protocol",
                "configs/short_branch_protocol.json",
                "--analysis-protocol",
                causal_protocol,
                *dense_scope,
                "--experiment-matrix",
                "configs/experiment.yaml",
                "--output-csv",
                "reports/short-branch/temporal_mechanism_predictors.csv",
                "--manifest",
                "reports/short-branch/temporal_mechanism_predictors.manifest.json",
                "--cache-dir",
                "reports/short-branch/temporal-predictor-cache",
                "--audit",
            ),
        ),
        PipelineStep(
            "temporal-short-branch-fresh-audit",
            _module(
                args.python,
                "embed_optim.temporal_short_branch",
                "--protocol",
                causal_protocol,
                "--scope-amendment",
                scope,
                "--predictor-csv",
                "reports/short-branch/temporal_mechanism_predictors.csv",
                "--predictor-manifest",
                "reports/short-branch/temporal_mechanism_predictors.manifest.json",
                "--outcome-csv",
                "reports/tail-stability/short_branch_checkpoint_tail.csv",
                "--outcome-manifest",
                "reports/tail-stability/summary_manifest.json",
                "--output-dir",
                "reports/temporal-short-branch",
                "--audit",
            ),
        ),
        PipelineStep(
            "dose-band-fresh-audit",
            _module(
                args.python,
                "embed_optim.dose_band_analysis",
                "--protocol",
                causal_protocol,
                "--audit",
            ),
        ),
        PipelineStep(
            "discovery-report",
            _module(
                args.python,
                "embed_optim.aggregate",
                *dense_scope,
                "--strict",
            ),
        ),
        PipelineStep(
            "retrieval-dynamics",
            _module(
                args.python,
                "embed_optim.retrieval_dynamics",
                *dense_scope,
            ),
        ),
        PipelineStep(
            "mechanism-report",
            _module(
                args.python,
                "embed_optim.mechanism_report",
                *dense_scope,
            ),
        ),
        PipelineStep(
            "outcome-report",
            _module(
                args.python,
                "embed_optim.outcome_report",
                *dense_scope,
            ),
        ),
        PipelineStep(
            "paper-results",
            _module(
                args.python,
                "embed_optim.paper_results",
                *dense_scope,
            ),
        ),
        PipelineStep(
            "paper-audit-strict",
            _module(
                args.python,
                "embed_optim.paper_audit",
                "--strict",
                *dense_scope,
            ),
        ),
        PipelineStep("tests", _module(args.python, "pytest", "-q")),
        PipelineStep(
            "ruff-check",
            _module(args.python, "ruff", "check", *lint_targets),
        ),
        PipelineStep(
            "ruff-format-check",
            _module(args.python, "ruff", "format", "--check", *lint_targets),
        ),
        PipelineStep("paper-build", ("make", "-C", "paper", "clean", "all")),
        PipelineStep("distribution-build", ("uv", "build")),
        PipelineStep(
            "distribution-audit",
            _module(args.python, "embed_optim.distribution_audit"),
        ),
    ]
    if args.include_wandb:
        steps.append(
            PipelineStep(
                "wandb-sync-dense",
                _module(
                    args.python,
                    "embed_optim.wandb_sync",
                    *dense_scope,
                ),
            )
        )
    return steps


def _read_process_identity(pid: int) -> ProcessIdentity | None:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return None
    closing_parenthesis = stat.rfind(")")
    if closing_parenthesis < 0:
        raise RuntimeError(f"Cannot parse process identity for PID {pid}")
    fields_after_command = stat[closing_parenthesis + 2 :].split()
    if len(fields_after_command) <= 19:
        raise RuntimeError(f"Cannot parse process start time for PID {pid}")
    return ProcessIdentity(
        pid=pid,
        start_time_ticks=int(fields_after_command[19]),
        command=command,
    )


def _wait_for_completion(args: argparse.Namespace) -> None:
    if args.wait_pid is None:
        return
    initial = _read_process_identity(args.wait_pid)
    if initial is None:
        return
    if args.wait_command_fragment not in initial.command:
        raise RuntimeError(
            f"PID {args.wait_pid} is not the requested Dense completion pipeline: "
            f"{initial.command!r}"
        )
    print(
        f"waiting for Dense completion pipeline PID {args.wait_pid} "
        f"(start={initial.start_time_ticks})",
        flush=True,
    )
    while True:
        current = _read_process_identity(args.wait_pid)
        if current is None:
            return
        if current.start_time_ticks != initial.start_time_ticks:
            raise RuntimeError(f"PID {args.wait_pid} was reused while waiting for Dense completion")
        if args.wait_command_fragment not in current.command:
            raise RuntimeError(
                f"PID {args.wait_pid} changed identity while waiting for Dense completion"
            )
        time.sleep(args.poll_seconds)


def _read_completion_ledger(
    path: Path,
    *,
    expected_scope: dict[str, Any],
    repository: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read Dense completion ledger: {path}") from error
    steps = payload.get("steps")
    step_names = (
        tuple(step.get("name") for step in steps)
        if isinstance(steps, list) and all(isinstance(step, dict) for step in steps)
        else ()
    )
    valid_step_names = step_names in {
        CORE_STEP_NAMES,
        CORE_STEP_NAMES + VALIDATION_STEP_NAMES,
    }
    training_plan = payload.get("training_plan")
    training_ledgers = payload.get("training_ledgers")
    step_contract = payload.get("step_contract")
    completion_binding = payload.get("input_binding")
    pipeline_arguments = payload.get("pipeline_arguments")
    if (
        payload.get("schema_version") != 1
        or payload.get("complete") is not True
        or payload.get("families") != ["dense"]
        or not scope_amendments_equal(payload.get("scope_amendment"), expected_scope, repository)
        or not isinstance(steps, list)
        or not steps
        or not valid_step_names
        or any(
            not isinstance(step, dict)
            or step.get("index") != index
            or not isinstance(step.get("command"), list)
            or not all(isinstance(token, str) for token in step["command"])
            or step.get("complete") is not True
            for index, step in enumerate(steps, start=1)
        )
        or "failed_step" in payload
        or not isinstance(training_plan, dict)
        or not isinstance(training_plan.get("path"), str)
        or not isinstance(training_ledgers, list)
        or len(training_ledgers) != 2
        or any(
            not isinstance(record, dict) or not isinstance(record.get("path"), str)
            for record in training_ledgers
        )
        or not isinstance(step_contract, dict)
        or not isinstance(completion_binding, dict)
        or not isinstance(pipeline_arguments, dict)
    ):
        raise RuntimeError(
            "Dense completion ledger is not a complete, clean, scope-matched Dense-only run: "
            f"{path}"
        )

    try:
        canonical_args = _args_from_pipeline_arguments(pipeline_arguments)
    except RuntimeError as error:
        raise RuntimeError(
            f"Dense completion ledger has invalid canonical arguments: {path}"
        ) from error
    if (
        canonical_args.workdir.resolve() != repository.resolve()
        or canonical_args.scope_amendment.resolve()
        != (repository / expected_scope["path"]).resolve()
    ):
        raise RuntimeError(f"Dense completion ledger canonical paths differ: {path}")
    canonical_steps = completion_pipeline_steps(canonical_args)
    expected_contract = _step_contract(
        canonical_steps,
        implementation_paths=_repository_contract_sources(repository),
    )
    try:
        current_training_inputs = _validate_training_inputs(
            workdir=repository,
            scope=expected_scope,
            training_plan=Path(training_plan["path"]),
            training_ledgers=[Path(record["path"]) for record in training_ledgers],
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"Dense completion ledger upstream provenance is no longer valid: {path}"
        ) from error
    expected_binding = {
        "scope_amendment": expected_scope,
        "training_plan": current_training_inputs["training_plan"],
        "training_ledgers": current_training_inputs["training_ledgers"],
        "step_contract_sha256": expected_contract["sha256"],
        "pipeline_arguments": pipeline_arguments,
    }
    if (
        step_contract != expected_contract
        or training_plan != current_training_inputs["training_plan"]
        or training_ledgers != current_training_inputs["training_ledgers"]
        or completion_binding != expected_binding
        or any(step.get("input_binding") != expected_binding for step in steps)
    ):
        raise RuntimeError(
            "Dense completion ledger provenance or step contract differs from current inputs: "
            f"{path}"
        )
    source = {
        "path": str(path),
        "bytes": len(raw),
        "sha256": _sha256_bytes(raw),
    }
    return payload, source


def _matching_completed_prefix(
    previous: dict[str, Any],
    steps: list[PipelineStep],
    input_binding: dict[str, Any] | None = None,
) -> int:
    completed = 0
    for old, current in zip(previous.get("steps") or [], steps, strict=False):
        if (
            old.get("name") != current.name
            or old.get("command") != list(current.command)
            or old.get("complete") is not True
            or (input_binding is not None and old.get("input_binding") != input_binding)
        ):
            break
        completed += 1
    return completed


def _validate_previous_ledger(
    previous: dict[str, Any],
    *,
    scope: dict[str, Any],
    repository: Path,
) -> None:
    if (
        previous.get("schema_version") != 1
        or previous.get("families") != ["dense"]
        or not scope_amendments_equal(previous.get("scope_amendment"), scope, repository)
    ):
        raise ValueError("Dense finalization ledger is bound to a different scope")


def _finalization_input_binding(
    *,
    scope: dict[str, Any],
    completion_source: dict[str, Any],
    step_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scope_amendment": scope,
        "completion_ledger": completion_source,
        "step_contract_sha256": step_contract["sha256"],
    }


def _assert_completion_unchanged(
    path: Path,
    *,
    scope: dict[str, Any],
    repository: Path,
    expected_source: dict[str, Any],
) -> None:
    _, current_source = _read_completion_ledger(
        path,
        expected_scope=scope,
        repository=repository,
    )
    if current_source != expected_source:
        raise RuntimeError("Dense completion provenance changed while finalization was running")


def run_pipeline(args: argparse.Namespace) -> int:
    workdir = args.workdir.resolve()
    scope_path = _under_workdir(args.scope_amendment, workdir)
    families, scope = resolve_scope(["dense"], scope_path)
    if families != ("dense",) or scope is None:
        raise AssertionError("Dense finalization pipeline received a non-dense scope")

    _wait_for_completion(args)
    completion_path = _under_workdir(args.completion_ledger, workdir)
    _, completion_source = _read_completion_ledger(
        completion_path,
        expected_scope=scope,
        repository=workdir,
    )
    steps = pipeline_steps(args)
    implementation_paths = _repository_contract_sources(workdir)
    step_contract = _step_contract(steps, implementation_paths=implementation_paths)
    input_binding = _finalization_input_binding(
        scope=scope,
        completion_source=completion_source,
        step_contract=step_contract,
    )
    log_dir = _under_workdir(args.log_dir, workdir)
    ledger_path = log_dir / "pipeline-ledger.json"

    previous: dict[str, Any] | None = None
    completed_prefix = 0
    if ledger_path.is_file():
        previous = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise FileExistsError(f"Dense finalization ledger already exists: {ledger_path}")
        _validate_previous_ledger(
            previous,
            scope=scope,
            repository=workdir,
        )
        # Never trust prior report/build success bits. Re-run the complete
        # orchestration so strict renderers and audits revalidate all outputs.
        completed_prefix = 0

    now = _timestamp()
    ledger: dict[str, Any] = {
        "schema_version": 1,
        "complete": False,
        "started_at": previous.get("started_at", now) if previous else now,
        "families": ["dense"],
        "scope_amendment": scope,
        "completion_ledger": completion_source,
        "step_contract": step_contract,
        "input_binding": input_binding,
        "steps": list(previous.get("steps", [])[:completed_prefix]) if previous else [],
    }
    _atomic_json(ledger_path, ledger)

    for index, step in enumerate(steps[completed_prefix:], start=completed_prefix + 1):
        _assert_completion_unchanged(
            completion_path,
            scope=scope,
            repository=workdir,
            expected_source=completion_source,
        )
        _assert_step_contract_unchanged(
            steps,
            step_contract,
            implementation_paths=implementation_paths,
        )
        record: dict[str, Any] = {
            "index": index,
            "name": step.name,
            "command": list(step.command),
            "input_binding": input_binding,
            "attempts": [],
            "complete": False,
        }
        ledger["steps"].append(record)
        _atomic_json(ledger_path, ledger)
        for attempt in range(1, args.step_retries + 2):
            log_path = log_dir / f"{index:02d}-{step.name}.attempt-{attempt}.log"
            print(f"Dense finalization step {index}/{len(steps)}: {step.name}", flush=True)
            started = _timestamp()
            return_code: int | None = None
            execution_error: str | None = None
            with log_path.open("w", encoding="utf-8") as handle:
                try:
                    result = subprocess.run(
                        step.command,
                        cwd=workdir,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                    return_code = result.returncode
                except OSError as error:
                    execution_error = f"{type(error).__name__}: {error}"
                    handle.write(f"pipeline execution error: {execution_error}\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            attempt_record: dict[str, Any] = {
                "attempt": attempt,
                "started_at": started,
                "finished_at": _timestamp(),
                "return_code": return_code,
                "log": {
                    "path": str(log_path.resolve()),
                    "bytes": log_path.stat().st_size,
                    "sha256": _sha256(log_path),
                },
            }
            if execution_error is not None:
                attempt_record["execution_error"] = execution_error
            record["attempts"].append(attempt_record)
            if return_code == 0:
                _assert_completion_unchanged(
                    completion_path,
                    scope=scope,
                    repository=workdir,
                    expected_source=completion_source,
                )
                _assert_step_contract_unchanged(
                    steps,
                    step_contract,
                    implementation_paths=implementation_paths,
                )
                record["complete"] = True
                record["finished_at"] = _timestamp()
                _atomic_json(ledger_path, ledger)
                break
            _atomic_json(ledger_path, ledger)
            if attempt <= args.step_retries:
                time.sleep(args.retry_delay)
        if record["complete"] is not True:
            ledger["failed_step"] = step.name
            ledger["finished_at"] = _timestamp()
            _atomic_json(ledger_path, ledger)
            return 1

    ledger.pop("failed_step", None)
    ledger["complete"] = True
    ledger["finished_at"] = _timestamp()
    _atomic_json(ledger_path, ledger)
    print("Dense-only final artifacts and release audits complete", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume-safe DenseOn-only final artifact and release pipeline"
    )
    parser.add_argument(
        "--scope-amendment",
        type=Path,
        default=Path("configs/dense_scope_amendment.json"),
    )
    parser.add_argument(
        "--completion-ledger",
        type=Path,
        default=Path("logs/dense-completion-pipeline/pipeline-ledger.json"),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/dense-finalization-pipeline"),
    )
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument(
        "--wait-command-fragment",
        default="embed_optim.dense_completion_pipeline",
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--step-retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--include-wandb",
        action="store_true",
        help="Append the external Dense-only W&B synchronization step",
    )
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or args.retry_delay < 0 or args.step_retries < 0:
        parser.error("Polling/retry values must be non-negative and polling must be positive")
    if args.wait_pid is not None and args.wait_pid <= 0:
        parser.error("--wait-pid must be positive")
    if args.wait_pid is not None and not args.wait_command_fragment:
        parser.error("--wait-command-fragment cannot be empty when --wait-pid is set")
    return args


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_pipeline(parse_args(argv)))


if __name__ == "__main__":
    main()
