"""Run corrected padded validation for the 12-run Dense matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .aggregate import audit_dataset_artifacts, audit_training_artifacts
from .config import RunConfig, load_matrix, resolve_matrix_path
from .corrected_input_execution import PADDED_DENSE_RECEIPT
from .corrected_validation_evaluation import run_corrected_validation_evaluation
from .geometry import _sha256
from .validation_data import audit_validation_data, load_validation_spec
from .validation_matrix import _final_checkpoint


@dataclass(frozen=True)
class Job:
    config: RunConfig
    checkpoint: Path
    output_dir: Path

    @property
    def label(self) -> str:
        return f"dense/{self.config.run_id}"


@dataclass
class Running:
    job: Job
    process: subprocess.Popen
    handle: object
    attempt: int


def build_jobs(configs: list[RunConfig], output_root: str | Path) -> list[Job]:
    if (
        len(configs) != 12
        or any(config.model_family != "dense" for config in configs)
        or any(config.dense_can_flatten_inputs for config in configs)
    ):
        raise ValueError("Corrected validation requires exactly 12 padded Dense configurations")
    jobs = [
        Job(config, _final_checkpoint(config), Path(output_root).resolve() / config.run_id)
        for config in configs
    ]
    if len({job.label for job in jobs}) != 12:
        raise ValueError("Corrected validation run IDs are not unique")
    return jobs


def job_complete(job: Job, validation_spec: str | Path, verify_hashes: bool = False) -> bool:
    try:
        spec_path, spec = load_validation_spec(validation_spec)
        manifest = json.loads((job.output_dir / "manifest.json").read_text(encoding="utf-8"))
        outputs = manifest["outputs"]
        if (
            manifest.get("status") != "complete"
            or manifest.get("family") != "dense"
            or manifest.get("input_execution") != PADDED_DENSE_RECEIPT
            or Path(manifest["checkpoint"]["path"]).resolve() != job.checkpoint
            or manifest["validation_spec"]["sha256"] != _sha256(spec_path)
            or manifest.get("sample_records")
            != spec["evaluation"]["expected_sample_records_per_job"]
            or set(outputs) != {"sample_metrics", "group_metrics"}
        ):
            return False
        for identity in outputs.values():
            path = job.output_dir / identity["path"]
            if not path.is_file() or path.stat().st_size != identity.get("bytes"):
                return False
            if verify_hashes and _sha256(path) != identity.get("sha256"):
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _validate_training(configs: list[RunConfig]) -> None:
    dataset = audit_dataset_artifacts(configs)
    if not dataset["complete"]:
        raise RuntimeError("Corrected validation training-data audit failed")
    training = audit_training_artifacts(
        configs,
        deep=True,
        expected_dataset_fingerprint=dataset.get("training_view_fingerprint"),
    )
    if not training["complete"]:
        raise RuntimeError("Corrected validation checkpoint audit failed: " + "; ".join(training["errors"][:5]))


def _launch(job: Job, gpu: str, args: argparse.Namespace, attempt: int) -> Running:
    log_path = args.log_dir.resolve() / f"{job.config.run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "embed_optim.corrected_validation_matrix",
        "--worker",
        "--checkpoint",
        str(job.checkpoint),
        "--output-dir",
        str(job.output_dir),
        "--probe",
        str(args.probe.resolve()),
        "--validation-spec",
        str(args.validation_spec.resolve()),
    ]
    print(f"[{gpu}] {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=Path.cwd(),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"},
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    return Running(job, process, handle, attempt)


def run_jobs(jobs: list[Job], args: argparse.Namespace) -> int:
    pending = [job for job in jobs if not job_complete(job, args.validation_spec, args.verify_hashes)]
    print(json.dumps({"complete": len(jobs) - len(pending), "expected": len(jobs)}), flush=True)
    if args.audit_only:
        return len(pending)
    if args.dry_run:
        for job in pending:
            print(job.label)
        return 0
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must contain unique GPU IDs")
    running: dict[str, Running] = {}
    attempts: dict[str, int] = {}
    failures = 0
    while pending or running:
        for gpu, item in list(running.items()):
            code = item.process.poll()
            if code is None:
                continue
            item.handle.close()
            del running[gpu]
            if code == 0 and job_complete(item.job, args.validation_spec):
                print(f"completed {item.job.label}", flush=True)
            elif item.attempt <= args.max_retries:
                pending.insert(0, item.job)
                print(f"retrying {item.job.label} after exit {code}", flush=True)
            else:
                failures += 1
                print(f"failed {item.job.label}", flush=True)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=Path("configs/dense_no_packing_retrain.yaml"))
    parser.add_argument("--probe", type=Path, default=Path("data/validation-4096-seed20260826"))
    parser.add_argument("--validation-spec", type=Path, default=Path("configs/validation_probe.json"))
    parser.add_argument("--output-root", type=Path, default=Path("results/dense-no-packing-validation"))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/dense-no-packing-validation"))
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.validation_spec, spec = load_validation_spec(args.validation_spec)
    if args.worker:
        if args.checkpoint is None or args.output_dir is None:
            raise ValueError("Corrected validation worker is missing paths")
        run_corrected_validation_evaluation(
            args.checkpoint,
            args.probe,
            args.output_dir,
            validation_spec=args.validation_spec,
            audit_probe=False,
        )
        return
    audit_validation_data(args.probe.resolve(), spec["source"]["training_data"], spec_path=args.validation_spec)
    matrix = resolve_matrix_path(args.matrix).resolve()
    configs = load_matrix(matrix)
    jobs = build_jobs(configs, args.output_root)
    if not args.dry_run:
        _validate_training(configs)
    if failures := run_jobs(jobs, args):
        raise SystemExit(failures)


if __name__ == "__main__":
    main()
