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
from .geometry import SCHEMA_VERSION, _atomic_json


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: tuple[str, ...]


class TransientProgressAuditError(ValueError):
    """The watcher could not audit its current snapshot while evaluators may still recover."""


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
                ),
            ),
            PipelineStep(
                "hybrid-adamw-evaluation",
                _module(
                    args.python,
                    "embed_optim.evaluate_matrix",
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
                ),
            )
        )
    steps.extend(
        [
            PipelineStep(
                "short-branch-training-audit",
                _module(args.python, "embed_optim.short_branch", "--audit-only"),
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
                PipelineStep("distribution-build", _module(args.python, "build")),
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


def supervise_post_eval(
    args: argparse.Namespace,
    *,
    run_command: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
    pid_exists: Callable[[int], bool] = _pid_exists,
) -> int:
    steps = pipeline_steps(args)
    if args.dry_run:
        print(json.dumps([{"name": step.name, "command": list(step.command)} for step in steps]))
        return 0

    progress_path = Path(args.progress).resolve()
    previous_valid: int | None = None
    previous_progress_error: str | None = None
    while True:
        try:
            complete, valid, expected = _strict_progress(progress_path)
        except TransientProgressAuditError as error:
            live = [pid for pid in args.wait_pids if pid_exists(pid)]
            if not live:
                raise
            message = str(error)
            if message != previous_progress_error:
                print(
                    "Strict coverage audit is temporarily unavailable while evaluator PIDs "
                    f"remain live {live}: {message}",
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

    while live := [pid for pid in args.wait_pids if pid_exists(pid)]:
        print(f"Coverage is complete; waiting for evaluator PIDs to exit: {live}", flush=True)
        sleeper(args.poll_seconds)
    if args.settle_seconds:
        print(f"Evaluator complete; settling for {args.settle_seconds:g}s", flush=True)
        sleeper(args.settle_seconds)

    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = log_dir / "pipeline-ledger.json"
    ledger: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "complete": False,
        "started_at": _timestamp(),
        "progress": str(progress_path),
        "wait_pids": args.wait_pids,
        "steps": [],
    }
    _write_ledger(ledger_path, ledger)
    for index, step in enumerate(steps, start=1):
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
            log_path = log_dir / f"{index:02d}-{step.name}.attempt-{attempt}.log"
            started = _timestamp()
            print(f"Pipeline step {index}/{len(steps)} started: {step.name}", flush=True)
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
                "log_path": str(log_path),
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

    ledger["complete"] = True
    ledger["finished_at"] = _timestamp()
    _write_ledger(ledger_path, ledger)
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
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--settle-seconds", type=float, default=60.0)
    parser.add_argument("--retry-delay", type=float, default=300.0)
    parser.add_argument("--step-retries", type=int, default=2)
    parser.add_argument("--worker-retries", type=int, default=2)
    parser.add_argument("--dense-probe-batch-size", type=int, default=32)
    parser.add_argument("--late-probe-batch-size", type=int, default=8)
    parser.add_argument("--skip-wandb-sync", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
    ):
        parser.error("poll/retry intervals, retries, batch sizes, or wait PIDs are invalid")
    return args


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(supervise_post_eval(parse_args(argv)))


if __name__ == "__main__":
    main()
