from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RunConfig, load_matrix, resolve_matrix_path
from .geometry import SCHEMA_VERSION, _sha256
from .validation_data import audit_validation_data, load_validation_spec
from .validation_evaluation import run_validation_evaluation


@dataclass(frozen=True)
class ValidationJob:
    config: RunConfig
    checkpoint: Path
    output_dir: Path

    @property
    def label(self) -> str:
        return f"{self.config.model_family}/{self.config.run_id}"


@dataclass
class RunningJob:
    job: ValidationJob
    process: subprocess.Popen
    log_handle: Any
    attempts: int


def _final_checkpoint(config: RunConfig) -> Path:
    schedule_path = config.output_dir / "checkpoint_schedule.json"
    try:
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        steps = [int(value) for value in schedule["steps"]]
        fractions = [float(value) for value in schedule["fractions"]]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid checkpoint schedule {schedule_path}: {error}") from error
    if len(steps) != 5 or steps != sorted(set(steps)) or fractions[-1] != 1.0:
        raise ValueError(f"Final-checkpoint contract changed under {config.output_dir}")
    checkpoint = (config.output_dir / f"checkpoint-{steps[-1]}").resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def build_validation_jobs(configs: list[RunConfig], output_root: str | Path) -> list[ValidationJob]:
    output_root = Path(output_root).resolve()
    jobs = [
        ValidationJob(
            config,
            _final_checkpoint(config),
            output_root / config.model_family / config.run_id,
        )
        for config in configs
    ]
    labels = [job.label for job in jobs]
    if len(labels) != 24 or len(set(labels)) != 24:
        raise ValueError("Recipe validation requires exactly 24 unique main-matrix runs")
    return jobs


def validation_job_complete(
    job: ValidationJob,
    validation_spec: str | Path,
    *,
    verify_hashes: bool = False,
) -> bool:
    try:
        spec_path, spec = load_validation_spec(validation_spec)
        manifest_path = job.output_dir / "manifest.json"
        if not manifest_path.is_file():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs = manifest["outputs"]
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("status") != "complete"
            or manifest.get("family") != job.config.model_family
            or Path(manifest["checkpoint"]["path"]).resolve() != job.checkpoint
            or manifest["validation_spec"]["sha256"] != _sha256(spec_path)
            or manifest.get("sample_records")
            != spec["evaluation"]["expected_sample_records_per_job"]
            or manifest.get("group_records") != 8
            or set(outputs) != {"sample_metrics", "group_metrics"}
        ):
            return False
        for item in outputs.values():
            path = job.output_dir / item["path"]
            if not path.is_file() or path.stat().st_size != item.get("bytes"):
                return False
            if verify_hashes and _sha256(path) != item.get("sha256"):
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _job_cli(job: ValidationJob, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "embed_optim.validation_matrix",
        "--worker",
        "--family",
        job.config.model_family,
        "--label",
        job.label,
        "--checkpoint",
        str(job.checkpoint),
        "--output-dir",
        str(job.output_dir),
        "--probe",
        str(args.probe.resolve()),
        "--validation-spec",
        str(args.validation_spec.resolve()),
        "--device",
        "cuda",
    ]


def _launch(job: ValidationJob, gpu: str, args: argparse.Namespace, attempts: int) -> RunningJob:
    log_path = args.log_dir.resolve() / f"{job.label.replace('/', '__')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    command = _job_cli(job, args)
    print(f"[{gpu}] {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=Path.cwd(),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"},
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return RunningJob(job, process, log_handle, attempts)


def run_matrix(jobs: list[ValidationJob], args: argparse.Namespace) -> int:
    pending = [
        job
        for job in jobs
        if not validation_job_complete(job, args.validation_spec, verify_hashes=args.verify_hashes)
    ]
    print(
        json.dumps(
            {
                "complete": len(jobs) - len(pending),
                "expected": len(jobs),
                "pending": len(pending),
                "verify_hashes": args.verify_hashes,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.audit_only:
        return len(pending)
    if args.dry_run:
        for job in pending:
            print(job.label)
        return 0
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError(f"--gpus must contain unique comma-separated IDs, got {args.gpus!r}")
    running: dict[str, RunningJob] = {}
    attempts: dict[str, int] = {}
    failures = 0
    while pending or running:
        for gpu, running_job in list(running.items()):
            return_code = running_job.process.poll()
            if return_code is None:
                continue
            running_job.log_handle.close()
            del running[gpu]
            if return_code == 0 and validation_job_complete(running_job.job, args.validation_spec):
                print(f"completed {running_job.job.label}", flush=True)
                continue
            if running_job.attempts <= args.max_retries:
                pending.insert(0, running_job.job)
                print(
                    f"retrying {running_job.job.label} after exit {return_code} "
                    f"(attempt {running_job.attempts})",
                    flush=True,
                )
            else:
                failures += 1
                print(
                    f"failed {running_job.job.label} after {running_job.attempts} attempts",
                    flush=True,
                )
                if args.fail_fast:
                    for other in running.values():
                        other.process.terminate()
                    return failures
        for gpu in gpus:
            if gpu in running or not pending:
                continue
            job = pending.pop(0)
            attempt = attempts.get(job.label, 0) + 1
            attempts[job.label] = attempt
            running[gpu] = _launch(job, gpu, args, attempt)
        if running:
            time.sleep(1)
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all 24 final checkpoints on query-disjoint validation data"
    )
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--probe", type=Path, default=Path("data/validation-4096-seed20260826"))
    parser.add_argument(
        "--validation-spec", type=Path, default=Path("configs/validation_probe.json")
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/recipe-validation"))
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-hashes", action="store_true")

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--family", choices=("dense", "late"), help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative")
    args.validation_spec, spec = load_validation_spec(args.validation_spec)
    args.probe = args.probe.resolve()
    if args.worker:
        required = (args.family, args.label, args.checkpoint, args.output_dir)
        if any(value is None for value in required):
            raise ValueError("Worker invocation is missing required fields")
        run_validation_evaluation(
            args.checkpoint,
            args.probe,
            args.output_dir,
            family=args.family,
            validation_spec=args.validation_spec,
            device=args.device,
            audit_probe=False,
        )
        return
    if not args.dry_run:
        audit_validation_data(
            args.probe,
            spec["source"]["training_data"],
            spec_path=args.validation_spec,
        )
    matrix_path = resolve_matrix_path(args.matrix).resolve()
    configs = load_matrix(matrix_path)
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else Path(spec["evaluation"]["output_root"]).resolve()
    )
    jobs = build_validation_jobs(configs, output_root)
    if len(jobs) != spec["evaluation"]["expected_jobs"]:
        raise ValueError("Validation job count differs from its frozen protocol")
    failures = run_matrix(jobs, args)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
