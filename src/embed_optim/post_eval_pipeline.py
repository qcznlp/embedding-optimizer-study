from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .config import resolve_matrix_path
from .evaluation_supervisor import _matching_command_pids
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: tuple[str, ...]


class TransientProgressAuditError(ValueError):
    """The watcher could not audit its current snapshot while evaluators may still recover."""


class UnboundPipelineLedgerError(ValueError):
    """A legacy terminal ledger is valid except for recorded log identities."""


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _module(python: str, module: str, *arguments: str) -> tuple[str, ...]:
    return python, "-m", module, *arguments


def pipeline_steps(args: argparse.Namespace) -> list[PipelineStep]:
    matrix = str(resolve_matrix_path(args.matrix).resolve())
    shared = ("--matrix", matrix)
    steps = [
        PipelineStep(
            "strict-evaluation-audit",
            _module(
                args.python,
                "embed_optim.aggregate",
                *shared,
                "--results-root",
                args.results_root,
                "--output-dir",
                args.reports_dir,
                "--blog",
                args.blog,
                "--strict",
                "--no-render-blog",
            ),
        ),
    ]
    if not args.skip_wandb_sync:
        steps.append(
            PipelineStep(
                "canonical-wandb-sync",
                _module(args.python, "embed_optim.wandb_sync", *shared),
            )
        )
    steps.extend(
        [
            PipelineStep(
                "strict-blog-render",
                _module(
                    args.python,
                    "embed_optim.aggregate",
                    *shared,
                    "--results-root",
                    args.results_root,
                    "--output-dir",
                    args.reports_dir,
                    "--blog",
                    args.blog,
                    "--strict",
                ),
            ),
            PipelineStep(
                "weight-space-reaudit",
                _module(
                    args.python,
                    "embed_optim.geometry_summary",
                    *shared,
                    "--geometry-root",
                    "results/weight-space",
                    "--output-dir",
                    "reports/weight-space",
                    "--verify-inputs",
                ),
            ),
            PipelineStep(
                "training-dynamics-summary",
                _module(args.python, "embed_optim.training_dynamics", *shared),
            ),
            PipelineStep(
                "training-dynamics-plot",
                _module(args.python, "embed_optim.training_dynamics_plot"),
            ),
            PipelineStep(
                "retrieval-dynamics-summary",
                _module(args.python, "embed_optim.retrieval_dynamics", *shared),
            ),
            PipelineStep(
                "validation-data-audit",
                _module(args.python, "embed_optim.validation_data", "--audit-only"),
            ),
            PipelineStep(
                "recipe-validation-matrix",
                _module(
                    args.python,
                    "embed_optim.validation_matrix",
                    *shared,
                    "--gpus",
                    args.gpus,
                    "--max-retries",
                    str(args.worker_retries),
                ),
            ),
            PipelineStep(
                "recipe-validation-audit",
                _module(
                    args.python,
                    "embed_optim.validation_matrix",
                    *shared,
                    "--audit-only",
                    "--verify-hashes",
                ),
            ),
            PipelineStep(
                "recipe-validation-summary",
                _module(args.python, "embed_optim.validation_summary", *shared),
            ),
            PipelineStep(
                "confirmatory-data-preparation",
                _module(args.python, "embed_optim.confirmatory_data"),
            ),
            PipelineStep(
                "confirmatory-data-source-audit",
                _module(
                    args.python,
                    "embed_optim.confirmatory_data",
                    "--audit-only",
                    "--verify-source",
                ),
            ),
            PipelineStep(
                "confirmatory-matrix-generation",
                _module(args.python, "embed_optim.confirmatory_matrix"),
            ),
            PipelineStep(
                "common-state-matrix",
                _module(
                    args.python,
                    "embed_optim.common_state_matrix",
                    *shared,
                    "--gpus",
                    args.gpus,
                    "--max-retries",
                    str(args.worker_retries),
                ),
            ),
            PipelineStep(
                "common-state-audit",
                _module(
                    args.python,
                    "embed_optim.common_state_matrix",
                    *shared,
                    "--audit-only",
                    "--verify-hashes",
                ),
            ),
            PipelineStep(
                "common-state-summary",
                _module(
                    args.python,
                    "embed_optim.common_state_summary",
                    *shared,
                    "--result-root",
                    "results/common-state",
                    "--output-dir",
                    "reports/common-state",
                ),
            ),
            PipelineStep(
                "basis-sensitivity-analysis",
                _module(
                    args.python,
                    "embed_optim.basis_sensitivity",
                    "--device",
                    "cuda:0",
                ),
            ),
            PipelineStep(
                "basis-sensitivity-audit",
                _module(
                    args.python,
                    "embed_optim.basis_sensitivity",
                    "--audit-only",
                    "--verify-inputs",
                ),
            ),
            PipelineStep(
                "short-branch-matrix-generation",
                _module(args.python, "embed_optim.short_branch"),
            ),
            PipelineStep(
                "common-state-exact-spectra",
                _module(
                    args.python,
                    "embed_optim.common_state_spectra",
                    *shared,
                    "--gpus",
                    args.gpus,
                    "--max-retries",
                    str(args.worker_retries),
                ),
            ),
            PipelineStep(
                "common-state-spectra-audit",
                _module(
                    args.python,
                    "embed_optim.common_state_spectra",
                    *shared,
                    "--audit-only",
                    "--verify-hashes",
                ),
            ),
            PipelineStep(
                "common-state-spectra-plot",
                _module(args.python, "embed_optim.common_state_spectrum_plot"),
            ),
            PipelineStep(
                "functional-intervention-matrix",
                _module(
                    args.python,
                    "embed_optim.functional_intervention_matrix",
                    *shared,
                    "--gpus",
                    args.gpus,
                    "--max-retries",
                    str(args.worker_retries),
                ),
            ),
            PipelineStep(
                "functional-intervention-audit",
                _module(
                    args.python,
                    "embed_optim.functional_intervention_matrix",
                    *shared,
                    "--audit-only",
                    "--verify-hashes",
                ),
            ),
            PipelineStep(
                "functional-intervention-summary",
                _module(
                    args.python,
                    "embed_optim.functional_intervention_summary",
                    *shared,
                ),
            ),
        ]
    )
    probe_tiers = (
        (
            "training",
            "data/probes/training-1024-seed1729",
            "configs/representation_probe.json",
            "results/representation-space/training",
            "logs/representation-space/training",
        ),
        (
            "unseen",
            "data/probes/decontaminated-beir-224-seed4242",
            "configs/beir_representation_probe.json",
            "results/representation-space/decontaminated-beir",
            "logs/representation-space/decontaminated-beir",
        ),
    )
    for tier, probe, spec, output, log_dir in probe_tiers:
        for family, batch_size in (
            ("dense", args.dense_probe_batch_size),
            ("late", args.late_probe_batch_size),
        ):
            steps.append(
                PipelineStep(
                    f"{tier}-{family}-representation-matrix",
                    _module(
                        args.python,
                        "embed_optim.probe_matrix",
                        *shared,
                        "--families",
                        family,
                        "--probe",
                        probe,
                        "--probe-spec",
                        spec,
                        "--output-root",
                        output,
                        "--log-dir",
                        log_dir,
                        "--gpus",
                        args.gpus,
                        "--batch-size",
                        str(batch_size),
                        "--max-retries",
                        str(args.worker_retries),
                    ),
                )
            )
        steps.append(
            PipelineStep(
                f"{tier}-representation-summary",
                _module(
                    args.python,
                    "embed_optim.representation_summary",
                    *shared,
                    "--result-root",
                    output,
                    "--probe",
                    probe,
                    "--probe-spec",
                    spec,
                ),
            )
        )
    steps.extend(
        [
            PipelineStep(
                "representation-dynamics-plot",
                _module(args.python, "embed_optim.representation_plot"),
            ),
            PipelineStep(
                "late-token-dynamics-plot",
                (
                    args.python,
                    "-c",
                    "from embed_optim.representation_plot import late_main; late_main()",
                ),
            ),
            PipelineStep(
                "mechanism-bridge",
                _module(args.python, "embed_optim.mechanism_bridge", *shared),
            ),
            PipelineStep(
                "mechanism-blog-render",
                _module(args.python, "embed_optim.mechanism_report"),
            ),
        ]
    )
    training_pools = (
        "--gpus-a",
        args.gpus_a,
        "--gpus-b",
        args.gpus_b,
    )
    steps.extend(
        [
            PipelineStep(
                "hybrid-adamw-training",
                _module(
                    args.python,
                    "embed_optim.matrix",
                    "--matrix",
                    "configs/hybrid_adamw.yaml",
                    *training_pools,
                    "--port-a",
                    "29810",
                    "--port-b",
                    "29820",
                    "--log-dir",
                    "logs/hybrid-adamw-training",
                    "--max-retries",
                    str(args.worker_retries),
                ),
            ),
            PipelineStep(
                "hybrid-adamw-evaluation",
                _module(
                    args.python,
                    "embed_optim.hybrid_evaluation",
                    "--matrix",
                    "configs/hybrid_adamw.yaml",
                    "--stages",
                    "5",
                    *training_pools,
                    "--late-port-a",
                    "29830",
                    "--late-port",
                    "29840",
                    "--results-root",
                    "results/hybrid-adamw-beir",
                    "--log-dir",
                    "logs/hybrid-adamw-evaluation",
                    "--worker-python",
                    args.python,
                ),
            ),
            PipelineStep(
                "hybrid-adamw-summary",
                _module(args.python, "embed_optim.hybrid_control"),
            ),
        ]
    )
    confirmatory_seeds = (314159, 271828, 161803)
    for offset, seed in enumerate(confirmatory_seeds):
        steps.append(
            PipelineStep(
                f"confirmatory-training-seed-{seed}",
                _module(
                    args.python,
                    "embed_optim.matrix",
                    "--matrix",
                    f"configs/generated/confirmatory/seed{seed}.yaml",
                    *training_pools,
                    "--port-a",
                    str(29910 + 20 * offset),
                    "--port-b",
                    str(29920 + 20 * offset),
                    "--log-dir",
                    f"logs/confirmatory-training/seed{seed}",
                    "--max-retries",
                    str(args.worker_retries),
                ),
            )
        )
    steps.extend(
        [
            PipelineStep(
                "confirmatory-evaluation",
                _module(
                    args.python,
                    "embed_optim.confirmatory_evaluation",
                    *training_pools,
                    "--worker-python",
                    args.python,
                ),
            ),
            PipelineStep(
                "confirmatory-evaluation-audit",
                _module(
                    args.python,
                    "embed_optim.confirmatory_evaluation",
                    "--audit-only",
                ),
            ),
            PipelineStep(
                "confirmatory-summary",
                _module(args.python, "embed_optim.confirmatory_summary"),
            ),
        ]
    )
    for offset, seed in enumerate(confirmatory_seeds):
        steps.append(
            PipelineStep(
                f"short-branch-training-seed-{seed}",
                _module(
                    args.python,
                    "embed_optim.matrix",
                    "--matrix",
                    f"configs/generated/short-branch/seed{seed}.yaml",
                    *training_pools,
                    "--port-a",
                    str(30010 + 20 * offset),
                    "--port-b",
                    str(30020 + 20 * offset),
                    "--log-dir",
                    f"logs/short-branch-training/seed{seed}",
                    "--max-retries",
                    str(args.worker_retries),
                ),
            )
        )
    steps.extend(
        [
            PipelineStep(
                "short-branch-training-audit",
                _module(
                    args.python,
                    "embed_optim.short_branch_evaluation",
                    "--training-audit-only",
                ),
            ),
            PipelineStep(
                "short-branch-evaluation",
                _module(
                    args.python,
                    "embed_optim.short_branch_evaluation",
                    "--gpus",
                    args.gpus,
                    "--max-retries",
                    str(args.worker_retries),
                ),
            ),
            PipelineStep(
                "short-branch-evaluation-audit",
                _module(
                    args.python,
                    "embed_optim.short_branch_evaluation",
                    "--audit-only",
                    "--verify-hashes",
                ),
            ),
            PipelineStep(
                "short-branch-summary",
                _module(args.python, "embed_optim.short_branch_summary"),
            ),
            PipelineStep(
                "outcome-blog-render",
                _module(args.python, "embed_optim.outcome_report"),
            ),
            PipelineStep(
                "paper-results-render",
                _module(args.python, "embed_optim.paper_results"),
            ),
            PipelineStep(
                "paper-evidence-audit",
                _module(args.python, "embed_optim.paper_audit"),
            ),
            PipelineStep("paper-draft-build", ("make", "-C", "paper")),
        ]
    )
    if not args.skip_validation:
        steps.extend(
            [
                PipelineStep("tests", _module(args.python, "pytest", "-q")),
                PipelineStep("ruff-check", _module(args.python, "ruff", "check", ".")),
                PipelineStep(
                    "ruff-format-check", _module(args.python, "ruff", "format", "--check", ".")
                ),
                PipelineStep("distribution-build", ("uv", "build")),
                PipelineStep(
                    "distribution-audit",
                    _module(args.python, "embed_optim.distribution_audit"),
                ),
            ]
        )
    steps.append(
        PipelineStep(
            "paper-final-strict-audit",
            _module(args.python, "embed_optim.paper_audit", "--strict"),
        )
    )
    return steps


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _strict_progress(path: Path) -> tuple[bool, int, int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False, 0, 1680
    expected = payload.get("expected_units")
    valid = payload.get("valid_units")
    if payload.get("schema_version") == SCHEMA_VERSION and payload.get("error") is not None:
        raise TransientProgressAuditError(str(payload["error"]))
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or expected != 1680
        or isinstance(valid, bool)
        or not isinstance(valid, int)
        or not 0 <= valid <= expected
        or payload.get("unexpected_units") != 0
        or payload.get("error") is not None
    ):
        raise ValueError(f"Invalid strict evaluation progress: {path}")
    complete = (
        payload.get("complete") is True and valid == expected and payload.get("missing_units") == 0
    )
    return complete, valid, expected


def _write_ledger(path: Path, ledger: dict[str, Any]) -> None:
    _atomic_json(path, ledger)


def _successful_record(record: Any) -> bool:
    attempts = record.get("attempts") if isinstance(record, dict) else None
    return bool(
        isinstance(attempts, list)
        and attempts
        and isinstance(attempts[-1], dict)
        and attempts[-1].get("return_code") == 0
        and record.get("complete") is True
    )


def _with_log_identities(record: dict[str, Any]) -> dict[str, Any]:
    """Verify and bind every attempt log without trusting prior identity fields."""

    attempts = record.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError(f"Pipeline step has no auditable attempts: {record.get('name')}")
    verified = []
    for expected_attempt, attempt in enumerate(attempts, start=1):
        if (
            not isinstance(attempt, dict)
            or isinstance(attempt.get("attempt"), bool)
            or attempt.get("attempt") != expected_attempt
            or isinstance(attempt.get("return_code"), bool)
            or not isinstance(attempt.get("return_code"), int)
            or not isinstance(attempt.get("log_path"), str)
            or (
                "bytes" in attempt
                and (
                    isinstance(attempt.get("bytes"), bool)
                    or not isinstance(attempt.get("bytes"), int)
                    or attempt["bytes"] < 0
                )
            )
            or (
                "sha256" in attempt
                and (not isinstance(attempt.get("sha256"), str) or len(attempt["sha256"]) != 64)
            )
        ):
            raise ValueError(f"Pipeline attempt record is invalid: {record.get('name')}")
        path = Path(attempt["log_path"]).resolve()
        if not path.is_file():
            raise ValueError(f"Pipeline attempt log is missing: {path}")
        observed_bytes = path.stat().st_size
        observed_sha256 = _sha256(path)
        if ("bytes" in attempt and attempt.get("bytes") != observed_bytes) or (
            "sha256" in attempt and attempt.get("sha256") != observed_sha256
        ):
            raise ValueError(f"Pipeline attempt log identity differs: {path}")
        verified.append(
            {
                **attempt,
                "log_path": str(path),
                "bytes": observed_bytes,
                "sha256": observed_sha256,
            }
        )
    return {**record, "attempts": verified}


def _audit_pipeline_ledger_payload(
    payload: dict[str, Any],
    steps: list[PipelineStep],
) -> dict[str, Any]:
    records = payload.get("steps")
    if (
        isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("complete") is not True
        or not isinstance(payload.get("started_at"), str)
        or not isinstance(payload.get("finished_at"), str)
        or "failed_step" in payload
        or not isinstance(records, list)
        or len(records) != len(steps)
    ):
        raise ValueError("Post-evaluation pipeline ledger is not terminal and complete")
    for index, (step, record) in enumerate(zip(steps, records, strict=True), start=1):
        if (
            not isinstance(record, dict)
            or isinstance(record.get("index"), bool)
            or record.get("index") != index
            or record.get("name") != step.name
            or record.get("command") != list(step.command)
            or not _successful_record(record)
        ):
            raise ValueError(f"Pipeline ledger step {index} differs: {step.name}")
        verified_record = _with_log_identities(record)
        attempts = record["attempts"]
        if (
            not isinstance(record.get("finished_at"), str)
            or record["finished_at"] != attempts[-1].get("finished_at")
            or any(
                not isinstance(attempt.get("started_at"), str)
                or not isinstance(attempt.get("finished_at"), str)
                for attempt in attempts
            )
            or any(attempt["return_code"] == 0 for attempt in attempts[:-1])
        ):
            raise ValueError(f"Pipeline ledger step metadata is invalid: {step.name}")
        if verified_record != record:
            raise UnboundPipelineLedgerError(
                f"Pipeline ledger step is not content-bound: {step.name}"
            )
    history = payload.get("resume_history", [])
    resume_count = payload.get("resume_count", 0)
    if (
        not isinstance(history, list)
        or isinstance(resume_count, bool)
        or not isinstance(resume_count, int)
        or resume_count != len(history)
        or (history and not isinstance(payload.get("resumed_at"), str))
    ):
        raise ValueError("Pipeline resume history is invalid")
    for resume_index, item in enumerate(history, start=1):
        source = item.get("source") if isinstance(item, dict) else None
        completed_prefix = item.get("completed_prefix") if isinstance(item, dict) else None
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("path"), str)
            or isinstance(source.get("bytes"), bool)
            or not isinstance(source.get("bytes"), int)
            or source["bytes"] < 0
            or not isinstance(source.get("sha256"), str)
            or len(source["sha256"]) != 64
            or isinstance(completed_prefix, bool)
            or not isinstance(completed_prefix, int)
            or completed_prefix < 0
        ):
            raise ValueError("Pipeline resume source identity is invalid")
        path = Path(source["path"]).resolve()
        if (
            source["path"] != str(path)
            or not path.is_file()
            or path.stat().st_size != source["bytes"]
            or _sha256(path) != source["sha256"]
        ):
            raise ValueError(f"Pipeline resume source differs: {path}")
        try:
            archived = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError) as error:
            raise ValueError(f"Pipeline resume source is invalid: {path}") from error
        archived_history = archived.get("resume_history", [])
        archived_resume_count = archived.get("resume_count", 0)
        if (
            isinstance(archived.get("schema_version"), bool)
            or archived.get("schema_version") != SCHEMA_VERSION
            or archived.get("started_at") != payload["started_at"]
            or archived.get("progress") != payload.get("progress")
            or isinstance(archived_resume_count, bool)
            or not isinstance(archived_resume_count, int)
            or archived_resume_count != resume_index - 1
            or archived_history != history[: resume_index - 1]
            or item.get("failed_step") != archived.get("failed_step")
            or item.get("finished_at") != archived.get("finished_at")
            or not isinstance(archived.get("steps"), list)
            or completed_prefix > len(archived["steps"])
            or any(
                not _successful_record(record) for record in archived["steps"][:completed_prefix]
            )
        ):
            raise ValueError(f"Pipeline resume archive chain differs: {path}")
    return {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "steps": len(steps),
        "attempts": sum(len(record["attempts"]) for record in records),
        "resume_count": resume_count,
    }


def audit_pipeline_ledger(
    ledger_path: str | Path,
    steps: list[PipelineStep],
) -> dict[str, Any]:
    path = Path(ledger_path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Cannot audit invalid pipeline ledger: {path}") from error
    result = _audit_pipeline_ledger_payload(payload, steps)
    return {**result, "ledger": str(path), "sha256": _sha256(path)}


def _resume_prefix(
    ledger_path: Path,
    steps: list[PipelineStep],
    progress_path: Path,
) -> tuple[dict[str, Any], int]:
    try:
        previous = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Cannot resume invalid pipeline ledger: {ledger_path}") from error
    records = previous.get("steps")
    recorded_progress = previous.get("progress")
    recorded_resume_count = previous.get("resume_count", 0)
    resume_history = previous.get("resume_history", [])
    if (
        isinstance(previous.get("schema_version"), bool)
        or previous.get("schema_version") != SCHEMA_VERSION
        or not isinstance(records, list)
        or not isinstance(recorded_progress, str)
        or Path(recorded_progress).resolve() != progress_path
        or isinstance(recorded_resume_count, bool)
        or not isinstance(recorded_resume_count, int)
        or recorded_resume_count < 0
        or not isinstance(resume_history, list)
        or recorded_resume_count != len(resume_history)
    ):
        raise ValueError(f"Pipeline resume ledger differs from this handoff: {ledger_path}")
    prefix = 0
    for step, record in zip(steps, records, strict=False):
        if (
            not isinstance(record, dict)
            or record.get("name") != step.name
            or record.get("command") != list(step.command)
            or not _successful_record(record)
        ):
            break
        try:
            _with_log_identities(record)
        except ValueError:
            break
        prefix += 1
    return previous, prefix


def _archive_resume_source(
    ledger_path: Path,
    previous: dict[str, Any],
    resume_count: int,
) -> Path:
    archive = ledger_path.with_name(f"pipeline-ledger.before-resume-{resume_count}.json")
    if archive.exists():
        try:
            existing = json.loads(archive.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError) as error:
            raise ValueError(f"Invalid existing pipeline resume archive: {archive}") from error
        if existing != previous:
            raise ValueError(f"Refusing to overwrite different pipeline resume archive: {archive}")
    else:
        _atomic_json(archive, previous)
    return archive


def supervise_post_eval(
    args: argparse.Namespace,
    *,
    run_command: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
    pid_exists: Callable[[int], bool] = _pid_exists,
    matching_command_pids: Callable[[str], list[int]] = _matching_command_pids,
) -> int:
    all_steps = pipeline_steps(args)
    if args.audit_ledger_only:
        result = audit_pipeline_ledger(
            Path(args.log_dir) / "pipeline-ledger.json",
            all_steps,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.dry_run:
        print(
            json.dumps([{"name": step.name, "command": list(step.command)} for step in all_steps])
        )
        return 0

    progress_path = Path(args.progress).resolve()
    previous_valid: int | None = None
    previous_progress_error: str | None = None
    while True:
        try:
            complete, valid, expected = _strict_progress(progress_path)
        except TransientProgressAuditError as error:
            live = [pid for pid in args.wait_pids if pid_exists(pid)]
            command_matches = {
                fragment: matching_command_pids(fragment) for fragment in args.wait_for_commands
            }
            command_matches = {fragment: pids for fragment, pids in command_matches.items() if pids}
            if not live and not command_matches:
                raise
            message = str(error)
            if message != previous_progress_error:
                print(
                    "Strict coverage audit is temporarily unavailable while evaluators remain "
                    f"live: pids={live}, command_matches={command_matches}: {message}",
                    flush=True,
                )
                previous_progress_error = message
            sleeper(args.poll_seconds)
            continue
        previous_progress_error = None
        if valid != previous_valid:
            print(f"Waiting for strict BEIR coverage: {valid}/{expected}", flush=True)
            previous_valid = valid
        if complete:
            break
        sleeper(args.poll_seconds)

    while True:
        live = [pid for pid in args.wait_pids if pid_exists(pid)]
        command_matches = {
            fragment: matching_command_pids(fragment) for fragment in args.wait_for_commands
        }
        command_matches = {fragment: pids for fragment, pids in command_matches.items() if pids}
        if not live and not command_matches:
            break
        print(
            "Coverage is complete; waiting for evaluators to exit: "
            f"pids={live}, command_matches={command_matches}",
            flush=True,
        )
        sleeper(args.poll_seconds)
    if args.settle_seconds:
        print(f"Evaluator complete; settling for {args.settle_seconds:g}s", flush=True)
        sleeper(args.settle_seconds)

    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = log_dir / "pipeline-ledger.json"
    previous: dict[str, Any] | None = None
    completed_prefix = 0
    resume_count = 0
    archive: Path | None = None
    if args.resume:
        previous, completed_prefix = _resume_prefix(ledger_path, all_steps, progress_path)
        if previous.get("complete") is True and completed_prefix == len(all_steps):
            try:
                audit_pipeline_ledger(ledger_path, all_steps)
            except UnboundPipelineLedgerError:
                if previous.get("resume_history", []):
                    raise
                print(
                    "Migrating complete pre-hash pipeline ledger without rerunning steps",
                    flush=True,
                )
            else:
                print("Post-evaluation pipeline ledger is already complete", flush=True)
                return 0
        prefix_records = [
            _with_log_identities(record) for record in previous["steps"][:completed_prefix]
        ]
        resume_count = int(previous.get("resume_count", 0)) + 1
        archive = _archive_resume_source(ledger_path, previous, resume_count)
        print(
            f"Resuming post-evaluation pipeline after {completed_prefix}/{len(all_steps)} "
            f"matching completed steps",
            flush=True,
        )
    steps = all_steps[completed_prefix:]
    now = _timestamp()
    ledger: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "complete": False,
        "started_at": previous.get("started_at", now) if previous is not None else now,
        "progress": str(progress_path),
        "wait_pids": args.wait_pids,
        "wait_for_commands": args.wait_for_commands,
        "steps": prefix_records if previous is not None else [],
    }
    if previous is not None and archive is not None:
        ledger["resume_count"] = resume_count
        ledger["resumed_at"] = now
        ledger["resume_history"] = [
            *list(previous.get("resume_history", [])),
            {
                "source": {
                    "path": str(archive.resolve()),
                    "bytes": archive.stat().st_size,
                    "sha256": _sha256(archive),
                },
                "completed_prefix": completed_prefix,
                "failed_step": previous.get("failed_step"),
                "finished_at": previous.get("finished_at"),
            },
        ]
    _write_ledger(ledger_path, ledger)
    for index, step in enumerate(steps, start=completed_prefix + 1):
        record: dict[str, Any] = {
            "index": index,
            "name": step.name,
            "command": list(step.command),
            "attempts": [],
            "complete": False,
        }
        ledger["steps"].append(record)
        _write_ledger(ledger_path, ledger)
        for attempt in range(1, args.step_retries + 2):
            resume_suffix = f".resume-{resume_count}" if resume_count else ""
            log_path = log_dir / (f"{index:02d}-{step.name}{resume_suffix}.attempt-{attempt}.log")
            started = _timestamp()
            print(f"Pipeline step {index}/{len(all_steps)} started: {step.name}", flush=True)
            with log_path.open("w", encoding="utf-8") as handle:
                completed = run_command(
                    list(step.command),
                    cwd=args.workdir,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            attempt_record = {
                "attempt": attempt,
                "started_at": started,
                "finished_at": _timestamp(),
                "return_code": completed.returncode,
                "log_path": str(log_path.resolve()),
                "bytes": log_path.stat().st_size,
                "sha256": _sha256(log_path),
            }
            record["attempts"].append(attempt_record)
            if completed.returncode == 0:
                record["complete"] = True
                record["finished_at"] = attempt_record["finished_at"]
                _write_ledger(ledger_path, ledger)
                print(f"Pipeline step completed: {step.name}", flush=True)
                break
            _write_ledger(ledger_path, ledger)
            print(
                f"Pipeline step failed ({completed.returncode}): {step.name}; log={log_path}",
                file=sys.stderr,
                flush=True,
            )
            if attempt <= args.step_retries:
                sleeper(args.retry_delay)
        if record["complete"] is not True:
            ledger["failed_step"] = step.name
            ledger["finished_at"] = _timestamp()
            _write_ledger(ledger_path, ledger)
            return 1

    candidate = {
        **ledger,
        "complete": True,
        "finished_at": _timestamp(),
    }
    _audit_pipeline_ledger_payload(candidate, all_steps)
    _write_ledger(ledger_path, candidate)
    print("Post-evaluation mechanism and reporting pipeline is complete", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for strict BEIR coverage, then run all mechanism and reporting gates"
    )
    parser.add_argument("--matrix", default="configs/experiment.yaml")
    parser.add_argument("--progress", default="logs/evaluation/live-audit.json")
    parser.add_argument("--results-root", default="results/decontaminated-beir")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--blog", default="docs/blog.md")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--gpus-a", default="0,1,2,3")
    parser.add_argument("--gpus-b", default="4,5,6,7")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workdir", default=str(Path.cwd()))
    parser.add_argument("--log-dir", default="logs/post-eval-pipeline")
    parser.add_argument("--wait-pids", type=int, nargs="*", default=[])
    parser.add_argument(
        "--wait-for-command",
        dest="wait_for_commands",
        action="append",
        default=[],
        help=(
            "Also wait for external evaluator argv containing this fragment after strict "
            "coverage; repeat to cover replacement or orphan workers"
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--settle-seconds", type=float, default=60.0)
    parser.add_argument("--retry-delay", type=float, default=300.0)
    parser.add_argument("--step-retries", type=int, default=2)
    parser.add_argument("--worker-retries", type=int, default=2)
    parser.add_argument("--dense-probe-batch-size", type=int, default=32)
    parser.add_argument("--late-probe-batch-size", type=int, default=8)
    parser.add_argument("--skip-wandb-sync", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument(
        "--audit-ledger-only",
        action="store_true",
        help="Independently verify the terminal command plan, logs, and resume archive chain",
    )
    args = parser.parse_args(argv)
    if (
        args.poll_seconds <= 0
        or args.settle_seconds < 0
        or args.retry_delay < 0
        or args.step_retries < 0
        or args.worker_retries < 0
        or args.dense_probe_batch_size <= 0
        or args.late_probe_batch_size <= 0
        or any(pid <= 0 for pid in args.wait_pids)
        or any(not fragment.strip() for fragment in args.wait_for_commands)
    ):
        parser.error(
            "poll/retry intervals, retries, batch sizes, wait PIDs, or command fragments are "
            "invalid"
        )
    return args


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(supervise_post_eval(parse_args(argv)))


if __name__ == "__main__":
    main()
