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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .family_training_queue import load_queue_plan
from .scope import resolve_scope, scope_amendments_equal

CORE_STEP_NAMES = (
    "hybrid-training-audit",
    "confirmatory-training-audit-seed-314159",
    "confirmatory-training-audit-seed-271828",
    "confirmatory-training-audit-seed-161803",
    "short-branch-training-audit-seed-314159",
    "short-branch-training-audit-seed-271828",
    "short-branch-training-audit-seed-161803",
    "hybrid-adamw-evaluation",
    "hybrid-adamw-summary",
    "hybrid-adamw-dynamics-evaluation",
    "confirmatory-evaluation",
    "confirmatory-evaluation-audit",
    "confirmatory-summary",
    "confirmatory-dynamics-evaluation",
    "dense-retrieval-dynamics-audit",
    "dense-retrieval-dynamics-summary-build",
    "dense-retrieval-dynamics-summary-audit",
    "short-branch-training-audit",
    "short-branch-evaluation",
    "short-branch-evaluation-audit",
    "short-branch-summary",
    "temporal-short-branch-predictors",
    "temporal-short-branch-predictors-audit",
    "tail-stability-summary",
    "temporal-short-branch-analysis",
    "temporal-short-branch-audit",
    "spectral-transplant-matrix",
    "spectral-transplant-audit",
    "spectral-transplant-summary",
    "dose-band-analysis",
    "dose-band-audit",
)
VALIDATION_STEP_NAMES = (
    "tests",
    "ruff-check",
    "ruff-format-check",
    "distribution-build",
)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _under_workdir(path: Path, workdir: Path) -> Path:
    return path.resolve() if path.is_absolute() else (workdir / path).resolve()


def _source_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    raw = resolved.read_bytes()
    return {
        "path": str(resolved),
        "bytes": len(raw),
        "sha256": _sha256_bytes(raw),
    }


def _repository_contract_sources(repository: Path) -> tuple[Path, ...]:
    root = repository.resolve()
    sources: set[Path] = set()
    for directory, pattern in (
        (root / "src" / "embed_optim", "*.py"),
        (root / "scripts" / "eval", "*.py"),
        (root / "tests", "*.py"),
        (root / "configs", "*.json"),
        (root / "configs", "*.yaml"),
    ):
        if directory.is_dir():
            sources.update(path.resolve() for path in directory.rglob(pattern) if path.is_file())
    # Keep generated paper fragments out of this set: paper_results legitimately
    # rewrites paper/results.tex and paper/generated/*.tex during finalization.
    # Only immutable templates and vendored style inputs belong to the contract.
    for relative in (
        "pyproject.toml",
        "uv.lock",
        "paper/main.tex",
        "paper/references.bib",
        "paper/Makefile",
        "paper/vendor/acl.sty",
        "paper/vendor/acl_natbib.bst",
    ):
        path = (root / relative).resolve()
        if path.is_file():
            sources.add(path)
    if not sources:
        raise RuntimeError(f"Cannot build repository source contract under {root}")
    return tuple(sorted(sources, key=str))


def _step_contract(
    steps: list[PipelineStep],
    *,
    implementation_paths: tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    records = [
        {"index": index, "name": step.name, "command": list(step.command)}
        for index, step in enumerate(steps, start=1)
    ]
    sources = [
        _source_identity(path)
        for path in sorted(
            implementation_paths or (Path(__file__).resolve(),),
            key=lambda path: str(path.resolve()),
        )
    ]
    contract = {"steps": records, "implementation_sources": sources}
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        **contract,
        "sha256": _sha256_bytes(encoded),
    }


def _assert_step_contract_unchanged(
    steps: list[PipelineStep],
    expected: dict[str, Any],
    *,
    implementation_paths: tuple[Path, ...] | None = None,
) -> None:
    if _step_contract(steps, implementation_paths=implementation_paths) != expected:
        raise RuntimeError("Pipeline step contract changed while the pipeline was running")


def _completion_input_binding(
    *,
    scope: dict[str, Any],
    training_inputs: dict[str, Any],
    step_contract: dict[str, Any],
    pipeline_arguments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scope_amendment": scope,
        "training_plan": training_inputs["training_plan"],
        "training_ledgers": training_inputs["training_ledgers"],
        "step_contract_sha256": step_contract["sha256"],
        "pipeline_arguments": pipeline_arguments,
    }


def _pipeline_arguments(
    args: argparse.Namespace, *, workdir: Path, scope_path: Path
) -> dict[str, Any]:
    return {
        "workdir": str(workdir.resolve()),
        "scope_amendment": str(scope_path.resolve()),
        "python": str(args.python),
        "gpus": str(args.gpus),
        "gpus_b": str(args.gpus_b),
        "worker_retries": int(args.worker_retries),
        "include_validation": bool(args.include_validation),
    }


def _args_from_pipeline_arguments(payload: dict[str, Any]) -> argparse.Namespace:
    required = {
        "workdir": str,
        "scope_amendment": str,
        "python": str,
        "gpus": str,
        "gpus_b": str,
        "worker_retries": int,
        "include_validation": bool,
    }
    if set(payload) != set(required) or any(
        not isinstance(payload.get(name), expected_type) for name, expected_type in required.items()
    ):
        raise RuntimeError("Dense completion pipeline arguments are malformed")
    return argparse.Namespace(
        workdir=Path(payload["workdir"]),
        scope_amendment=Path(payload["scope_amendment"]),
        python=payload["python"],
        gpus=payload["gpus"],
        gpus_b=payload["gpus_b"],
        worker_retries=payload["worker_retries"],
        include_validation=payload["include_validation"],
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def _exclusive_controller_lease(
    path: Path,
    *,
    controller: str,
    workdir: Path,
    step_contract_sha256: str,
) -> Iterator[int]:
    """Hold one non-blocking lease for the complete controller lifetime."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            raise RuntimeError(
                f"Dense {controller} controller lease is already held: {path} ({owner})"
            ) from error
        payload = {
            "schema_version": 1,
            "acquired_at": _timestamp(),
            "pid": os.getpid(),
            "controller": controller,
            "workdir": str(workdir.resolve()),
            "step_contract_sha256": step_contract_sha256,
        }
        handle.seek(0)
        handle.truncate()
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield handle.fileno()
    finally:
        handle.close()


def _assert_repository_step_contract_unchanged(
    steps: list[PipelineStep],
    expected: dict[str, Any],
    *,
    repository: Path,
) -> None:
    """Re-enumerate sources so additions and removals also fail closed."""

    _assert_step_contract_unchanged(
        steps,
        expected,
        implementation_paths=_repository_contract_sources(repository),
    )


def _module(python: str, module: str, *arguments: str) -> tuple[str, ...]:
    return python, "-m", module, *arguments


def pipeline_steps(args: argparse.Namespace) -> list[PipelineStep]:
    scope = str(args.scope_amendment.resolve())
    family_scope = ("--families", "dense", "--scope-amendment", scope)
    causal_chain_protocol = ("--protocol", "configs/causal_chain_analysis.json")
    all_gpus = args.gpus
    four_gpu_b = args.gpus_b
    predictor_device = f"cuda:{all_gpus.split(',', 1)[0].strip()}"
    worker_python = args.python
    steps: list[PipelineStep] = [
        PipelineStep(
            "hybrid-training-audit",
            _module(
                args.python,
                "embed_optim.checkpoint_watch",
                "--matrix",
                "configs/hybrid_adamw.yaml",
                "--families",
                "dense",
                "--state",
                "logs/hybrid-adamw-training/dense-checkpoint-audit.json",
                "--fail-on-problem",
            ),
        )
    ]
    for phase in ("confirmatory", "short-branch"):
        log_parent = f"logs/{phase}-training"
        matrix_parent = f"configs/generated/{phase}"
        for seed in (314159, 271828, 161803):
            steps.append(
                PipelineStep(
                    f"{phase}-training-audit-seed-{seed}",
                    _module(
                        args.python,
                        "embed_optim.checkpoint_watch",
                        "--matrix",
                        f"{matrix_parent}/seed{seed}.yaml",
                        "--families",
                        "dense",
                        "--state",
                        f"{log_parent}/seed{seed}/dense-checkpoint-audit.json",
                        "--fail-on-problem",
                    ),
                )
            )
    steps.extend(
        [
            PipelineStep(
                "hybrid-adamw-evaluation",
                _module(
                    args.python,
                    "embed_optim.hybrid_evaluation",
                    "--matrix",
                    "configs/hybrid_adamw.yaml",
                    *family_scope,
                    "--stages",
                    "5",
                    "--gpus-a",
                    all_gpus,
                    "--gpus-b",
                    four_gpu_b,
                    "--results-root",
                    "results/hybrid-adamw-beir",
                    "--log-dir",
                    "logs/hybrid-adamw-evaluation",
                    "--worker-python",
                    worker_python,
                ),
            ),
            PipelineStep(
                "hybrid-adamw-summary",
                _module(
                    args.python,
                    "embed_optim.hybrid_control",
                    *family_scope,
                ),
            ),
            PipelineStep(
                "hybrid-adamw-dynamics-evaluation",
                _module(
                    args.python,
                    "embed_optim.dense_retrieval_dynamics_evaluation",
                    "--contract",
                    "configs/dense_retrieval_dynamics_extension.json",
                    "--suite",
                    "hybrid",
                    "--gpus-a",
                    all_gpus,
                    "--gpus-b",
                    four_gpu_b,
                    "--worker-python",
                    worker_python,
                    "--receipt",
                    "reports/dense-retrieval-dynamics/hybrid-evaluation-receipt.json",
                ),
            ),
            PipelineStep(
                "confirmatory-evaluation",
                _module(
                    args.python,
                    "embed_optim.confirmatory_evaluation",
                    *family_scope,
                    "--gpus-a",
                    all_gpus,
                    "--gpus-b",
                    four_gpu_b,
                    "--worker-python",
                    worker_python,
                    "--receipt",
                    "reports/confirmatory/evaluation-receipt.json",
                ),
            ),
            PipelineStep(
                "confirmatory-evaluation-audit",
                _module(
                    args.python,
                    "embed_optim.confirmatory_evaluation",
                    *family_scope,
                    "--audit-only",
                    "--receipt",
                    "reports/confirmatory/evaluation-receipt.json",
                ),
            ),
            PipelineStep(
                "confirmatory-summary",
                _module(
                    args.python,
                    "embed_optim.confirmatory_summary",
                    *family_scope,
                ),
            ),
            PipelineStep(
                "confirmatory-dynamics-evaluation",
                _module(
                    args.python,
                    "embed_optim.dense_retrieval_dynamics_evaluation",
                    "--contract",
                    "configs/dense_retrieval_dynamics_extension.json",
                    "--suite",
                    "confirmatory",
                    "--gpus-a",
                    all_gpus,
                    "--gpus-b",
                    four_gpu_b,
                    "--worker-python",
                    worker_python,
                    "--receipt",
                    "reports/dense-retrieval-dynamics/confirmatory-evaluation-receipt.json",
                ),
            ),
            PipelineStep(
                "dense-retrieval-dynamics-audit",
                _module(
                    args.python,
                    "embed_optim.dense_retrieval_dynamics_evaluation",
                    "--contract",
                    "configs/dense_retrieval_dynamics_extension.json",
                    "--suite",
                    "all",
                    "--audit-only",
                    "--receipt",
                    "reports/dense-retrieval-dynamics/evaluation-receipt.json",
                ),
            ),
            PipelineStep(
                "dense-retrieval-dynamics-summary-build",
                _module(
                    args.python,
                    "embed_optim.dense_retrieval_dynamics_summary",
                    "--contract",
                    "configs/dense_retrieval_dynamics_extension.json",
                    "--output-dir",
                    "reports/dense-retrieval-dynamics",
                ),
            ),
            PipelineStep(
                "dense-retrieval-dynamics-summary-audit",
                _module(
                    args.python,
                    "embed_optim.dense_retrieval_dynamics_summary",
                    "--contract",
                    "configs/dense_retrieval_dynamics_extension.json",
                    "--output-dir",
                    "reports/dense-retrieval-dynamics",
                    "--audit-only",
                ),
            ),
            PipelineStep(
                "short-branch-training-audit",
                _module(
                    args.python,
                    "embed_optim.short_branch_evaluation",
                    *family_scope,
                    "--training-audit-only",
                ),
            ),
            PipelineStep(
                "short-branch-evaluation",
                _module(
                    args.python,
                    "embed_optim.short_branch_evaluation",
                    *family_scope,
                    "--gpus",
                    all_gpus,
                    "--max-retries",
                    str(args.worker_retries),
                ),
            ),
            PipelineStep(
                "short-branch-evaluation-audit",
                _module(
                    args.python,
                    "embed_optim.short_branch_evaluation",
                    *family_scope,
                    "--audit-only",
                    "--verify-hashes",
                ),
            ),
            PipelineStep(
                "short-branch-summary",
                _module(
                    args.python,
                    "embed_optim.short_branch_summary",
                    *family_scope,
                ),
            ),
            PipelineStep(
                "temporal-short-branch-predictors",
                _module(
                    args.python,
                    "embed_optim.temporal_short_branch_predictors",
                    *family_scope,
                    "--analysis-protocol",
                    "configs/causal_chain_analysis.json",
                    "--device",
                    predictor_device,
                ),
            ),
            PipelineStep(
                "temporal-short-branch-predictors-audit",
                _module(
                    args.python,
                    "embed_optim.temporal_short_branch_predictors",
                    *family_scope,
                    "--analysis-protocol",
                    "configs/causal_chain_analysis.json",
                    "--audit",
                ),
            ),
            PipelineStep(
                "tail-stability-summary",
                _module(
                    args.python,
                    "embed_optim.tail_stability",
                    *family_scope,
                    "--require-short-branch",
                ),
            ),
            PipelineStep(
                "temporal-short-branch-analysis",
                _module(
                    args.python,
                    "embed_optim.temporal_short_branch",
                    *causal_chain_protocol,
                ),
            ),
            PipelineStep(
                "temporal-short-branch-audit",
                _module(
                    args.python,
                    "embed_optim.temporal_short_branch",
                    *causal_chain_protocol,
                    "--scope-amendment",
                    scope,
                    "--audit",
                ),
            ),
            PipelineStep(
                "spectral-transplant-matrix",
                _module(
                    args.python,
                    "embed_optim.spectral_transplant_matrix",
                    *family_scope,
                    "--gpus",
                    all_gpus,
                    "--max-retries",
                    str(args.worker_retries),
                ),
            ),
            PipelineStep(
                "spectral-transplant-audit",
                _module(
                    args.python,
                    "embed_optim.spectral_transplant_matrix",
                    *family_scope,
                    "--audit-only",
                    "--verify-hashes",
                ),
            ),
            PipelineStep(
                "spectral-transplant-summary",
                _module(
                    args.python,
                    "embed_optim.spectral_transplant_summary",
                    *family_scope,
                ),
            ),
            PipelineStep(
                "dose-band-analysis",
                _module(
                    args.python,
                    "embed_optim.dose_band_analysis",
                    *causal_chain_protocol,
                ),
            ),
            PipelineStep(
                "dose-band-audit",
                _module(
                    args.python,
                    "embed_optim.dose_band_analysis",
                    *causal_chain_protocol,
                    "--audit",
                ),
            ),
        ]
    )
    if args.include_validation:
        steps.extend(
            [
                PipelineStep("tests", _module(args.python, "pytest", "-q")),
                PipelineStep("ruff-check", _module(args.python, "ruff", "check", ".")),
                PipelineStep(
                    "ruff-format-check",
                    _module(args.python, "ruff", "format", "--check", "."),
                ),
                PipelineStep("distribution-build", ("uv", "build")),
            ]
        )
    observed_names = tuple(step.name for step in steps)
    expected_names = CORE_STEP_NAMES + (VALIDATION_STEP_NAMES if args.include_validation else ())
    if observed_names != expected_names:
        raise AssertionError("Dense completion step contract changed")
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


def _wait_for_training(args: argparse.Namespace) -> None:
    pending: dict[int, ProcessIdentity] = {}
    for pid in args.wait_pids:
        identity = _read_process_identity(pid)
        if identity is None:
            continue
        if args.wait_command_fragment not in identity.command:
            raise RuntimeError(
                f"PID {pid} is not the requested Dense training queue: {identity.command!r}"
            )
        pending[pid] = identity
    while pending:
        finished: list[int] = []
        for pid in sorted(pending):
            current = _read_process_identity(pid)
            if current is None:
                finished.append(pid)
                continue
            initial = pending[pid]
            if current.start_time_ticks != initial.start_time_ticks:
                raise RuntimeError(f"PID {pid} was reused while waiting for Dense training")
            if args.wait_command_fragment not in current.command:
                raise RuntimeError(f"PID {pid} changed identity while waiting for Dense training")
        for pid in finished:
            pending.pop(pid)
        if pending:
            print(f"waiting for Dense training queues: {sorted(pending)}", flush=True)
            time.sleep(args.poll_seconds)


def _declared_path(path: object, workdir: Path) -> Path | None:
    if not isinstance(path, str) or not path:
        return None
    declared = Path(path)
    return declared.resolve() if declared.is_absolute() else (workdir / declared).resolve()


def _validate_training_inputs(
    *,
    workdir: Path,
    scope: dict[str, Any],
    training_plan: Path,
    training_ledgers: list[Path],
) -> dict[str, Any]:
    repository = workdir.resolve()
    plan_path = _under_workdir(training_plan, repository)
    ledger_paths = [_under_workdir(path, repository) for path in training_ledgers]
    if len(ledger_paths) != 2 or len(set(ledger_paths)) != 2:
        raise RuntimeError("Dense completion requires exactly two unique training-ledger paths")

    try:
        resolved_plan, plan_payload, jobs_by_pool = load_queue_plan(plan_path)
        plan_source = _source_identity(resolved_plan)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot validate frozen Dense training plan: {plan_path}") from error
    if resolved_plan != plan_path:
        raise RuntimeError("Dense training plan resolved to an unexpected path")

    declared_scope = _declared_path(
        (plan_payload.get("scope_amendment") or {}).get("path"), repository
    )
    expected_scope = _declared_path(scope.get("path"), repository)
    if (
        declared_scope is None
        or declared_scope != expected_scope
        or not expected_scope.is_file()
        or _sha256(expected_scope) != scope.get("sha256")
    ):
        raise RuntimeError("Dense training plan is bound to a different scope amendment")

    ledger_records: list[tuple[str, dict[str, Any]]] = []
    observed_pools: set[str] = set()
    for ledger_path in ledger_paths:
        try:
            raw = ledger_path.read_bytes()
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Cannot read Dense training queue ledger: {ledger_path}") from error
        pool = payload.get("pool")
        jobs = payload.get("jobs")
        observed_plan = payload.get("plan")
        observed_plan_path = (
            _declared_path(observed_plan.get("path"), repository)
            if isinstance(observed_plan, dict)
            else None
        )
        expected_jobs = jobs_by_pool.get(pool) if isinstance(pool, str) else None
        expected_identities = (
            [job.identity for job in expected_jobs] if expected_jobs is not None else []
        )
        observed_identities = (
            [record.get("identity") for record in jobs]
            if isinstance(jobs, list) and all(isinstance(record, dict) for record in jobs)
            else []
        )
        job_records_match = (
            isinstance(jobs, list)
            and expected_jobs is not None
            and len(jobs) == len(expected_jobs)
            and all(
                record.get("index") == index
                and _declared_path(record.get("matrix"), repository) == job.matrix.resolve()
                and _declared_path(record.get("output_dir"), repository)
                == job.config.output_dir.resolve()
                for index, (record, job) in enumerate(
                    zip(jobs, expected_jobs, strict=True),
                    start=1,
                )
            )
        )
        if (
            payload.get("schema_version") != 1
            or payload.get("complete") is not True
            or payload.get("family") != "dense"
            or pool not in {"a", "b"}
            or pool in observed_pools
            or "failed_job" in payload
            or not isinstance(observed_plan, dict)
            or observed_plan_path != plan_path
            or observed_plan.get("sha256") != plan_source["sha256"]
            or not isinstance(jobs, list)
            or len(jobs) != 9
            or observed_identities != expected_identities
            or not job_records_match
            or any(record.get("complete") is not True for record in jobs)
        ):
            raise RuntimeError(f"Dense training queue did not finish cleanly: {ledger_path}")
        observed_pools.add(pool)
        ledger_records.append(
            (
                pool,
                {
                    "pool": pool,
                    "path": str(ledger_path),
                    "bytes": len(raw),
                    "sha256": _sha256_bytes(raw),
                },
            )
        )

    if observed_pools != {"a", "b"}:
        raise RuntimeError("Dense completion requires exactly training pools a and b")
    return {
        "training_plan": plan_source,
        "training_ledgers": [record for _, record in sorted(ledger_records)],
    }


def _assert_training_inputs_unchanged(
    args: argparse.Namespace,
    *,
    scope: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    current = _validate_training_inputs(
        workdir=args.workdir.resolve(),
        scope=scope,
        training_plan=args.training_plan,
        training_ledgers=args.training_ledgers,
    )
    if current != expected:
        raise RuntimeError("Dense training input provenance changed while completion was running")


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


def run_pipeline(args: argparse.Namespace) -> int:
    workdir = args.workdir.resolve()
    scope_path = _under_workdir(args.scope_amendment, workdir)
    families, scope = resolve_scope(["dense"], scope_path)
    if families != ("dense",) or scope is None:
        raise AssertionError("Dense completion pipeline received a non-dense scope")
    steps = pipeline_steps(args)
    startup_sources = _repository_contract_sources(workdir)
    step_contract = _step_contract(steps, implementation_paths=startup_sources)
    log_dir = _under_workdir(args.log_dir, workdir)
    with _exclusive_controller_lease(
        log_dir / "controller.lease",
        controller="completion",
        workdir=workdir,
        step_contract_sha256=step_contract["sha256"],
    ) as lease_fd:
        _wait_for_training(args)
        _assert_repository_step_contract_unchanged(
            steps,
            step_contract,
            repository=workdir,
        )
        return _run_pipeline_after_wait(
            args,
            workdir=workdir,
            scope_path=scope_path,
            scope=scope,
            steps=steps,
            step_contract=step_contract,
            log_dir=log_dir,
            lease_fd=lease_fd,
        )


def _run_pipeline_after_wait(
    args: argparse.Namespace,
    *,
    workdir: Path,
    scope_path: Path,
    scope: dict[str, Any],
    steps: list[PipelineStep],
    step_contract: dict[str, Any],
    log_dir: Path,
    lease_fd: int,
) -> int:
    training_inputs = _validate_training_inputs(
        workdir=workdir,
        scope=scope,
        training_plan=args.training_plan,
        training_ledgers=args.training_ledgers,
    )
    _assert_repository_step_contract_unchanged(
        steps,
        step_contract,
        repository=workdir,
    )
    pipeline_arguments = _pipeline_arguments(args, workdir=workdir, scope_path=scope_path)
    input_binding = _completion_input_binding(
        scope=scope,
        training_inputs=training_inputs,
        step_contract=step_contract,
        pipeline_arguments=pipeline_arguments,
    )
    ledger_path = log_dir / "pipeline-ledger.json"
    previous = None
    completed_prefix = 0
    if ledger_path.is_file():
        previous = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise FileExistsError(f"Dense completion ledger already exists: {ledger_path}")
        if not scope_amendments_equal(previous.get("scope_amendment"), scope, workdir):
            raise ValueError("Dense completion ledger is bound to a different scope amendment")
        # Conservative recovery contract: orchestration steps are always rerun.
        # Their own strict caches may reuse content-addressed units, but this
        # coordinator never trusts an old success bit or output receipt.
        completed_prefix = 0
    now = _timestamp()
    ledger: dict[str, Any] = {
        "schema_version": 1,
        "complete": False,
        "started_at": previous.get("started_at", now) if previous else now,
        "families": ["dense"],
        "scope_amendment": scope,
        "training_plan": training_inputs["training_plan"],
        "training_ledgers": training_inputs["training_ledgers"],
        "step_contract": step_contract,
        "pipeline_arguments": pipeline_arguments,
        "input_binding": input_binding,
        "steps": list(previous.get("steps", [])[:completed_prefix]) if previous else [],
    }
    _atomic_json(ledger_path, ledger)
    for index, step in enumerate(steps[completed_prefix:], start=completed_prefix + 1):
        _assert_training_inputs_unchanged(args, scope=scope, expected=training_inputs)
        _assert_repository_step_contract_unchanged(
            steps,
            step_contract,
            repository=workdir,
        )
        record = {
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
            print(f"Dense pipeline step {index}/{len(steps)}: {step.name}", flush=True)
            started = _timestamp()
            with log_path.open("w", encoding="utf-8") as handle:
                result = subprocess.run(
                    step.command,
                    cwd=args.workdir,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    pass_fds=(lease_fd,),
                    check=False,
                )
            record["attempts"].append(
                {
                    "attempt": attempt,
                    "started_at": started,
                    "finished_at": _timestamp(),
                    "return_code": result.returncode,
                    "log": {
                        "path": str(log_path.resolve()),
                        "bytes": log_path.stat().st_size,
                        "sha256": _sha256(log_path),
                    },
                }
            )
            if result.returncode == 0:
                _assert_training_inputs_unchanged(
                    args,
                    scope=scope,
                    expected=training_inputs,
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
    print("Dense completion evaluation and mechanism pipeline complete", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume-safe DenseOn-only evaluation and mechanism completion pipeline"
    )
    parser.add_argument(
        "--scope-amendment",
        type=Path,
        default=Path("configs/dense_scope_amendment.json"),
    )
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--gpus-b", default="4,5,6,7")
    parser.add_argument("--wait-pids", type=int, nargs="*", default=[])
    parser.add_argument("--wait-command-fragment", default="embed_optim.family_training_queue")
    parser.add_argument(
        "--training-ledgers",
        type=Path,
        nargs="+",
        default=[
            Path("logs/dense-only-runtime/training-queue-a.json"),
            Path("logs/dense-only-runtime/training-queue-b.json"),
        ],
    )
    parser.add_argument(
        "--training-plan",
        type=Path,
        default=Path("configs/dense_training_queue.json"),
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs/dense-completion-pipeline"))
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--worker-retries", type=int, default=2)
    parser.add_argument("--step-retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-validation", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.poll_seconds <= 0
        or args.retry_delay < 0
        or args.worker_retries < 0
        or args.step_retries < 0
    ):
        parser.error("Polling/retry values must be non-negative and polling must be positive")
    if len(args.training_ledgers) != 2:
        parser.error("--training-ledgers requires exactly the pool-a and pool-b ledgers")
    resolved_ledgers = [
        _under_workdir(path, args.workdir.resolve()) for path in args.training_ledgers
    ]
    if len(set(resolved_ledgers)) != 2:
        parser.error("--training-ledgers paths must be unique")
    if len(args.wait_pids) != len(set(args.wait_pids)) or any(pid <= 0 for pid in args.wait_pids):
        parser.error("--wait-pids must contain unique positive process IDs")
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if len(gpu_ids) != 8 or len(set(gpu_ids)) != 8:
        parser.error("--gpus must identify eight unique devices")
    return args


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_pipeline(parse_args(argv)))


if __name__ == "__main__":
    main()
