from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dense_completion_pipeline import (
    _assert_step_contract_unchanged,
    _exclusive_controller_lease,
    _repository_contract_sources,
    _step_contract,
)
from .geometry import _atomic_json, _sha256
from .scope import resolve_scope, scope_amendments_equal

UPSTREAM_FINALIZATION_STEP_NAMES = (
    "temporal-predictors-fresh-audit",
    "temporal-short-branch-fresh-audit",
    "discovery-report",
    "dose-band-fresh-audit",
    "retrieval-dynamics",
    "mechanism-report",
    "outcome-report",
    "paper-results",
    "paper-audit-strict",
    "tests",
    "ruff-check",
    "ruff-format-check",
    "paper-build",
    "paper-audit-post-build-strict",
    "wandb-audit-dense-sources",
    "wandb-sync-dense",
    "distribution-build",
    "distribution-audit",
)

RELEASE_STEP_NAMES = (
    "candidate-data-resume",
    "candidate-data-audit",
    "candidate-evaluation",
    "candidate-summary",
    "candidate-summary-audit",
    "discovery-report",
    "retrieval-dynamics",
    "mechanism-report",
    "outcome-report",
    "paper-results",
    "candidate-publication",
    "candidate-publication-audit",
    "paper-audit-strict",
    "tests",
    "ruff-check",
    "ruff-format-check",
    "paper-build",
    "candidate-publication-post-build-audit",
    "paper-audit-post-build-strict",
    "distribution-build",
    "distribution-audit",
)


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: tuple[str, ...]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _module(python: str, module: str, *arguments: str) -> tuple[str, ...]:
    return python, "-m", module, *arguments


def _under_workdir(path: Path, workdir: Path) -> Path:
    return path.resolve() if path.is_absolute() else (workdir / path).resolve()


def _source_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _declared_data_output(protocol_path: Path) -> Path:
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        declared = protocol["evaluation"]["data_output"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(
            f"Cannot resolve candidate-breadth data output from {protocol_path}"
        ) from error
    if not isinstance(declared, str) or not declared:
        raise RuntimeError("Candidate-breadth protocol data output is malformed")
    return (protocol_path.parent.parent / declared).resolve()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_source_record(
    record: Any,
    *,
    repository: Path,
    label: str,
) -> dict[str, Any]:
    if (
        not isinstance(record, dict)
        or set(record) != {"path", "bytes", "sha256"}
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("bytes"), int)
        or record["bytes"] < 0
        or not isinstance(record.get("sha256"), str)
        or len(record["sha256"]) != 64
    ):
        raise RuntimeError(f"Malformed {label} source record")
    declared = Path(record["path"])
    path = declared.resolve() if declared.is_absolute() else (repository / declared).resolve()
    if not path.is_file() or _source_identity(path) != {**record, "path": str(path)}:
        raise RuntimeError(f"{label} source changed: {path}")
    return {**record, "path": str(path)}


def _validate_historical_step_contract(contract: Any) -> list[dict[str, Any]]:
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise RuntimeError("Upstream finalization step contract is malformed")
    steps = contract.get("steps")
    sources = contract.get("implementation_sources")
    if (
        not isinstance(steps, list)
        or not isinstance(sources, list)
        or not sources
        or tuple(step.get("name") for step in steps if isinstance(step, dict))
        != UPSTREAM_FINALIZATION_STEP_NAMES
    ):
        raise RuntimeError("Upstream finalization step contract has unexpected coverage")
    for index, step in enumerate(steps, start=1):
        if (
            not isinstance(step, dict)
            or set(step) != {"index", "name", "command"}
            or step.get("index") != index
            or not isinstance(step.get("command"), list)
            or not step["command"]
            or not all(isinstance(token, str) for token in step["command"])
        ):
            raise RuntimeError("Upstream finalization step contract is malformed")
    for source in sources:
        if (
            not isinstance(source, dict)
            or set(source) != {"path", "bytes", "sha256"}
            or not isinstance(source.get("path"), str)
            or not isinstance(source.get("bytes"), int)
            or source["bytes"] < 0
            or not isinstance(source.get("sha256"), str)
            or len(source["sha256"]) != 64
        ):
            raise RuntimeError("Upstream finalization implementation source record is malformed")
    body = {"steps": steps, "implementation_sources": sources}
    if contract.get("sha256") != _canonical_hash(body):
        raise RuntimeError("Upstream finalization step contract hash differs")
    return steps


def _read_finalization_ledger(
    path: Path,
    *,
    expected_scope: dict[str, Any],
    repository: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read upstream Dense finalization ledger: {path}") from error
    records = payload.get("steps")
    contract_steps = _validate_historical_step_contract(payload.get("step_contract"))
    if (
        payload.get("schema_version") != 1
        or payload.get("complete") is not True
        or payload.get("families") != ["dense"]
        or not scope_amendments_equal(payload.get("scope_amendment"), expected_scope, repository)
        or "failed_step" in payload
        or not isinstance(payload.get("finished_at"), str)
        or not isinstance(records, list)
        or len(records) != len(UPSTREAM_FINALIZATION_STEP_NAMES)
    ):
        raise RuntimeError(
            "Upstream Dense finalization ledger is not complete, clean, and scope matched"
        )

    completion_source = _validate_source_record(
        payload.get("completion_ledger"),
        repository=repository,
        label="upstream completion ledger",
    )
    step_contract = payload["step_contract"]
    expected_binding = {
        "scope_amendment": payload["scope_amendment"],
        "completion_ledger": completion_source,
        "step_contract_sha256": step_contract["sha256"],
    }
    if payload.get("input_binding") != expected_binding:
        raise RuntimeError("Upstream finalization input binding differs")

    for index, (record, contract_step) in enumerate(
        zip(records, contract_steps, strict=True), start=1
    ):
        attempts = record.get("attempts") if isinstance(record, dict) else None
        if (
            not isinstance(record, dict)
            or record.get("index") != index
            or record.get("name") != UPSTREAM_FINALIZATION_STEP_NAMES[index - 1]
            or record.get("command") != contract_step["command"]
            or record.get("input_binding") != expected_binding
            or record.get("complete") is not True
            or not isinstance(attempts, list)
            or not attempts
            or not isinstance(attempts[-1], dict)
            or attempts[-1].get("return_code") != 0
        ):
            raise RuntimeError(f"Upstream finalization step {index} is invalid")
        for attempt_index, attempt in enumerate(attempts, start=1):
            if (
                not isinstance(attempt, dict)
                or attempt.get("attempt") != attempt_index
                or not isinstance(attempt.get("started_at"), str)
                or not isinstance(attempt.get("finished_at"), str)
                or not isinstance(attempt.get("return_code"), int)
            ):
                raise RuntimeError(
                    f"Upstream finalization step {index} attempt record is malformed"
                )
            _validate_source_record(
                attempt.get("log"),
                repository=repository,
                label=f"upstream finalization step {index} log",
            )

    source = {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return payload, source


def pipeline_steps(args: argparse.Namespace) -> list[PipelineStep]:
    workdir = args.workdir.resolve()
    protocol = str(_under_workdir(args.protocol, workdir))
    data_output = str(_under_workdir(args.data_output, workdir))
    summary_dir = str(_under_workdir(args.summary_dir, workdir))
    blog = str(_under_workdir(args.blog, workdir))
    paper = str(_under_workdir(args.paper, workdir))
    manifest = str(_under_workdir(args.publication_manifest, workdir))
    scope = str(_under_workdir(args.scope_amendment, workdir))
    dense_scope = ("--families", "dense", "--scope-amendment", scope)
    lint_targets = ("src", "tests", "scripts/eval")
    publication_command = _module(
        args.python,
        "embed_optim.candidate_breadth_publication",
        "--protocol",
        protocol,
        "--summary-dir",
        summary_dir,
        "--blog",
        blog,
        "--paper",
        paper,
        "--manifest",
        manifest,
    )
    paper_audit = _module(
        args.python,
        "embed_optim.paper_audit",
        "--strict",
        *dense_scope,
    )
    steps = [
        PipelineStep(
            "candidate-data-resume",
            _module(
                args.python,
                "embed_optim.candidate_breadth_data",
                "--protocol",
                protocol,
                "--output",
                data_output,
                "--resume",
            ),
        ),
        PipelineStep(
            "candidate-data-audit",
            _module(
                args.python,
                "embed_optim.candidate_breadth_data",
                "--protocol",
                protocol,
                "--output",
                data_output,
                "--audit-only",
            ),
        ),
        PipelineStep(
            "candidate-evaluation",
            _module(
                args.python,
                "embed_optim.candidate_breadth_matrix",
                "--protocol",
                protocol,
                "--gpus",
                args.gpus,
                "--python",
                args.python,
                "--retries",
                str(args.worker_retries),
            ),
        ),
        PipelineStep(
            "candidate-summary",
            _module(
                args.python,
                "embed_optim.candidate_breadth_summary",
                "--protocol",
                protocol,
                "--output-dir",
                summary_dir,
            ),
        ),
        PipelineStep(
            "candidate-summary-audit",
            _module(
                args.python,
                "embed_optim.candidate_breadth_summary",
                "--protocol",
                protocol,
                "--output-dir",
                summary_dir,
                "--audit-only",
            ),
        ),
        PipelineStep(
            "discovery-report",
            _module(args.python, "embed_optim.aggregate", *dense_scope, "--strict"),
        ),
        PipelineStep(
            "retrieval-dynamics",
            _module(args.python, "embed_optim.retrieval_dynamics", *dense_scope),
        ),
        PipelineStep(
            "mechanism-report",
            _module(args.python, "embed_optim.mechanism_report", *dense_scope),
        ),
        PipelineStep(
            "outcome-report",
            _module(args.python, "embed_optim.outcome_report", *dense_scope),
        ),
        PipelineStep(
            "paper-results",
            _module(args.python, "embed_optim.paper_results", *dense_scope),
        ),
        PipelineStep("candidate-publication", publication_command),
        PipelineStep("candidate-publication-audit", publication_command + ("--audit-only",)),
        PipelineStep("paper-audit-strict", paper_audit),
        PipelineStep("tests", _module(args.python, "pytest", "-q")),
        PipelineStep("ruff-check", _module(args.python, "ruff", "check", *lint_targets)),
        PipelineStep(
            "ruff-format-check",
            _module(args.python, "ruff", "format", "--check", *lint_targets),
        ),
        PipelineStep("paper-build", ("make", "-C", "paper", "release", f"PYTHON={args.python}")),
        PipelineStep(
            "candidate-publication-post-build-audit",
            publication_command + ("--audit-only",),
        ),
        PipelineStep("paper-audit-post-build-strict", paper_audit),
        PipelineStep("distribution-build", ("uv", "build")),
        PipelineStep(
            "distribution-audit",
            _module(args.python, "embed_optim.distribution_audit"),
        ),
    ]
    if tuple(step.name for step in steps) != RELEASE_STEP_NAMES:
        raise AssertionError("Candidate-breadth release step contract changed")
    return steps


def _assert_repository_step_contract_unchanged(
    steps: list[PipelineStep],
    expected: dict[str, Any],
    *,
    repository: Path,
) -> None:
    _assert_step_contract_unchanged(
        steps,
        expected,
        implementation_paths=_repository_contract_sources(repository),
    )


def _release_input_binding(
    *,
    scope: dict[str, Any],
    upstream_source: dict[str, Any],
    protocol_source: dict[str, Any],
    step_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scope_amendment": scope,
        "upstream_finalization_ledger": upstream_source,
        "candidate_protocol": protocol_source,
        "step_contract_sha256": step_contract["sha256"],
    }


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
        raise ValueError("Candidate-breadth release ledger is bound to a different scope")


def _assert_frozen_inputs_unchanged(
    *,
    upstream_path: Path,
    upstream_source: dict[str, Any],
    protocol_path: Path,
    protocol_source: dict[str, Any],
    scope: dict[str, Any],
    repository: Path,
) -> None:
    _, current_upstream = _read_finalization_ledger(
        upstream_path,
        expected_scope=scope,
        repository=repository,
    )
    if current_upstream != upstream_source:
        raise RuntimeError("Upstream Dense finalization ledger changed during post-hoc release")
    if _source_identity(protocol_path) != protocol_source:
        raise RuntimeError("Candidate-breadth protocol changed during post-hoc release")


def run_pipeline(args: argparse.Namespace) -> int:
    workdir = args.workdir.resolve()
    scope_path = _under_workdir(args.scope_amendment, workdir)
    families, scope = resolve_scope(["dense"], scope_path)
    if families != ("dense",) or scope is None:
        raise AssertionError("Candidate-breadth release received a non-dense scope")
    protocol_path = _under_workdir(args.protocol, workdir)
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    configured_data_output = _under_workdir(args.data_output, workdir)
    declared_data_output = _declared_data_output(protocol_path)
    if configured_data_output != declared_data_output:
        raise ValueError(
            f"--data-output must match the immutable candidate protocol: {declared_data_output}"
        )
    upstream_path = _under_workdir(args.upstream_finalization_ledger, workdir)
    steps = pipeline_steps(args)
    step_contract = _step_contract(
        steps,
        implementation_paths=_repository_contract_sources(workdir),
    )
    log_dir = _under_workdir(args.log_dir, workdir)
    with _exclusive_controller_lease(
        log_dir / "controller.lease",
        controller="candidate-breadth-release",
        workdir=workdir,
        step_contract_sha256=step_contract["sha256"],
    ) as lease_fd:
        _, upstream_source = _read_finalization_ledger(
            upstream_path,
            expected_scope=scope,
            repository=workdir,
        )
        protocol_source = _source_identity(protocol_path)
        _assert_repository_step_contract_unchanged(
            steps,
            step_contract,
            repository=workdir,
        )
        input_binding = _release_input_binding(
            scope=scope,
            upstream_source=upstream_source,
            protocol_source=protocol_source,
            step_contract=step_contract,
        )
        ledger_path = log_dir / "pipeline-ledger.json"
        previous: dict[str, Any] | None = None
        if ledger_path.is_file():
            previous = json.loads(ledger_path.read_text(encoding="utf-8"))
            if not args.resume:
                raise FileExistsError(
                    f"Candidate-breadth release ledger already exists: {ledger_path}"
                )
            _validate_previous_ledger(previous, scope=scope, repository=workdir)

        now = _timestamp()
        ledger: dict[str, Any] = {
            "schema_version": 1,
            "complete": False,
            "started_at": previous.get("started_at", now) if previous else now,
            "families": ["dense"],
            "scope_amendment": scope,
            "upstream_finalization_ledger": upstream_source,
            "candidate_protocol": protocol_source,
            "step_contract": step_contract,
            "input_binding": input_binding,
            "steps": [],
        }
        _atomic_json(ledger_path, ledger)

        for index, step in enumerate(steps, start=1):
            _assert_frozen_inputs_unchanged(
                upstream_path=upstream_path,
                upstream_source=upstream_source,
                protocol_path=protocol_path,
                protocol_source=protocol_source,
                scope=scope,
                repository=workdir,
            )
            _assert_repository_step_contract_unchanged(
                steps,
                step_contract,
                repository=workdir,
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
                print(
                    f"Candidate-breadth release step {index}/{len(steps)}: {step.name}",
                    flush=True,
                )
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
                            pass_fds=(lease_fd,),
                            check=False,
                        )
                        return_code = result.returncode
                    except OSError as error:
                        execution_error = f"{type(error).__name__}: {error}"
                        handle.write(f"pipeline execution error: {execution_error}\n")
                        handle.flush()
                attempt_record: dict[str, Any] = {
                    "attempt": attempt,
                    "started_at": started,
                    "finished_at": _timestamp(),
                    "return_code": return_code,
                    "log": _source_identity(log_path),
                }
                if execution_error is not None:
                    attempt_record["execution_error"] = execution_error
                record["attempts"].append(attempt_record)
                if return_code == 0:
                    _assert_frozen_inputs_unchanged(
                        upstream_path=upstream_path,
                        upstream_source=upstream_source,
                        protocol_path=protocol_path,
                        protocol_source=protocol_source,
                        scope=scope,
                        repository=workdir,
                    )
                    _assert_repository_step_contract_unchanged(
                        steps,
                        step_contract,
                        repository=workdir,
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
        print("Dense candidate-breadth publication release complete", flush=True)
        return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the post-hoc candidate-breadth diagnostic and publication release after "
            "a completed Dense finalization"
        )
    )
    parser.add_argument(
        "--scope-amendment",
        type=Path,
        default=Path("configs/dense_scope_amendment.json"),
    )
    parser.add_argument(
        "--upstream-finalization-ledger",
        type=Path,
        default=Path("logs/dense-finalization-pipeline/pipeline-ledger.json"),
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/candidate_breadth_probe.json")
    )
    parser.add_argument(
        "--data-output", type=Path, default=Path("data/candidate-breadth-224-seed20260901")
    )
    parser.add_argument("--summary-dir", type=Path, default=Path("reports/candidate-breadth"))
    parser.add_argument("--blog", type=Path, default=Path("docs/blog.md"))
    parser.add_argument("--paper", type=Path, default=Path("paper/generated/candidate-breadth.tex"))
    parser.add_argument(
        "--publication-manifest",
        type=Path,
        default=Path("reports/candidate-breadth/publication_manifest.json"),
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs/candidate-breadth-release"))
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--worker-retries", type=int, default=1)
    parser.add_argument("--step-retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        gpus = [int(value.strip()) for value in args.gpus.split(",")]
    except ValueError:
        parser.error("--gpus must be a comma-separated list of non-negative integers")
    if not gpus or any(value < 0 for value in gpus) or len(gpus) != len(set(gpus)):
        parser.error("--gpus must contain unique non-negative integers")
    if args.worker_retries < 0 or args.step_retries < 0 or args.retry_delay < 0:
        parser.error("Retry counts and delay must be non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_pipeline(parse_args(argv)))


if __name__ == "__main__":
    main()
