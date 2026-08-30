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

from .scope import resolve_scope


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: tuple[str, ...]


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


def _module(python: str, module: str, *arguments: str) -> tuple[str, ...]:
    return python, "-m", module, *arguments


def pipeline_steps(args: argparse.Namespace) -> list[PipelineStep]:
    repository = args.workdir.resolve()
    scope = str(args.scope_amendment.resolve())
    family_scope = ("--families", "dense", "--scope-amendment", scope)
    all_gpus = args.gpus
    four_gpu_b = args.gpus_b
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
                "tail-stability-summary",
                _module(
                    args.python,
                    "embed_optim.tail_stability",
                    *family_scope,
                    "--require-short-branch",
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
    if repository != Path.cwd().resolve():
        raise ValueError("Dense completion pipeline must be launched from --workdir")
    return steps


def _pid_command(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, ProcessLookupError):
        return None


def _wait_for_training(args: argparse.Namespace) -> None:
    pending = set(args.wait_pids)
    while pending:
        finished = []
        for pid in sorted(pending):
            command = _pid_command(pid)
            if command is None or args.wait_command_fragment not in command:
                finished.append(pid)
        pending.difference_update(finished)
        if pending:
            print(f"waiting for Dense training queues: {sorted(pending)}", flush=True)
            time.sleep(args.poll_seconds)
    for path in args.training_ledgers:
        payload = json.loads(path.resolve().read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != 1
            or payload.get("complete") is not True
            or payload.get("family") != "dense"
            or len(payload.get("jobs") or []) != 9
            or any(record.get("complete") is not True for record in payload["jobs"])
        ):
            raise RuntimeError(f"Dense training queue did not finish cleanly: {path}")


def _matching_completed_prefix(previous: dict[str, Any], steps: list[PipelineStep]) -> int:
    completed = 0
    for old, current in zip(previous.get("steps") or [], steps, strict=False):
        if (
            old.get("name") != current.name
            or old.get("command") != list(current.command)
            or old.get("complete") is not True
        ):
            break
        completed += 1
    return completed


def run_pipeline(args: argparse.Namespace) -> int:
    families, scope = resolve_scope(["dense"], args.scope_amendment)
    if families != ("dense",):
        raise AssertionError("Dense completion pipeline received a non-dense scope")
    _wait_for_training(args)
    steps = pipeline_steps(args)
    log_dir = args.log_dir.resolve()
    ledger_path = log_dir / "pipeline-ledger.json"
    previous = None
    completed_prefix = 0
    if ledger_path.is_file():
        previous = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise FileExistsError(f"Dense completion ledger already exists: {ledger_path}")
        if previous.get("scope_amendment") != scope:
            raise ValueError("Dense completion ledger is bound to a different scope amendment")
        completed_prefix = _matching_completed_prefix(previous, steps)
        if previous.get("complete") is True and completed_prefix == len(steps):
            print("Dense completion pipeline is already complete", flush=True)
            return 0
    now = _timestamp()
    ledger: dict[str, Any] = {
        "schema_version": 1,
        "complete": False,
        "started_at": previous.get("started_at", now) if previous else now,
        "families": ["dense"],
        "scope_amendment": scope,
        "training_ledgers": [
            {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}
            for path in args.training_ledgers
        ],
        "steps": list(previous.get("steps", [])[:completed_prefix]) if previous else [],
    }
    _atomic_json(ledger_path, ledger)
    for index, step in enumerate(steps[completed_prefix:], start=completed_prefix + 1):
        record = {
            "index": index,
            "name": step.name,
            "command": list(step.command),
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
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if len(gpu_ids) != 8 or len(set(gpu_ids)) != 8:
        parser.error("--gpus must identify eight unique devices")
    return args


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_pipeline(parse_args(argv)))


if __name__ == "__main__":
    main()
