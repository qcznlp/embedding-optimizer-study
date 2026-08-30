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

from .common_state_matrix import (
    CommonStateJob,
    _load_protocol,
    _resolve_reference,
    build_common_state_jobs,
    common_state_job_complete,
    resolve_common_state_spec,
)
from .config import ModelFamily, load_matrix, resolve_matrix_path
from .geometry import SCHEMA_VERSION, _sha256
from .spectral_transplant import (
    load_spectral_transplant_protocol,
    run_spectral_transplant_intervention,
    spectral_conditions,
)


@dataclass(frozen=True)
class SpectralTransplantJob:
    common_state: CommonStateJob
    output_dir: Path

    @property
    def label(self) -> str:
        return self.common_state.label


@dataclass
class RunningJob:
    job: SpectralTransplantJob
    process: subprocess.Popen
    log_handle: Any
    attempts: int


def build_spectral_transplant_jobs(
    common_jobs: list[CommonStateJob], output_root: str | Path
) -> list[SpectralTransplantJob]:
    root = Path(output_root).resolve()
    return [SpectralTransplantJob(job, root / job.label) for job in common_jobs]


def spectral_transplant_job_complete(
    job: SpectralTransplantJob,
    spectral_spec: str | Path,
    common_state_spec: str | Path,
    *,
    verify_hashes: bool = False,
) -> bool:
    try:
        spec_path, spec = load_spectral_transplant_protocol(spectral_spec)
        transformed_conditions = spectral_conditions(spec)
        common_state_spec = Path(common_state_spec).resolve()
        manifest_path = job.output_dir / "manifest.json"
        update_manifest_path = job.common_state.update_dir / "manifest.json"
        direction_manifest_path = job.output_dir / "directions" / "manifest.json"
        if (
            not manifest_path.is_file()
            or not update_manifest_path.is_file()
            or not direction_manifest_path.is_file()
            or not common_state_job_complete(
                job.common_state, common_state_spec, verify_hashes=verify_hashes
            )
        ):
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        direction_manifest = json.loads(direction_manifest_path.read_text(encoding="utf-8"))
        outputs = manifest.get("outputs", {})
        direction_outputs = direction_manifest.get("outputs", {})
        source_update_identity = {
            "path": str(update_manifest_path.resolve()),
            "bytes": update_manifest_path.stat().st_size,
            "sha256": _sha256(update_manifest_path),
        }
        direction_manifest_identity = {
            "path": str(direction_manifest_path.resolve()),
            "bytes": direction_manifest_path.stat().st_size,
            "sha256": _sha256(direction_manifest_path),
        }
        expected_condition_names = [
            "baseline",
            "adamw-native",
            "muon-native",
            *(condition.name for condition in transformed_conditions),
        ]
        declared_conditions = manifest.get("conditions")
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("status") != "complete"
            or manifest.get("family") != job.common_state.family
            or Path(manifest["checkpoint"]["path"]).resolve() != job.common_state.checkpoint
            or manifest.get("spectral_transplant_spec", {}).get("sha256") != _sha256(spec_path)
            or manifest.get("source_update_manifest") != source_update_identity
            or manifest.get("direction_manifest") != direction_manifest_identity
            or not isinstance(declared_conditions, list)
            or [item.get("condition") for item in declared_conditions] != expected_condition_names
            or manifest.get("sample_records")
            != spec["intervention"]["expected_sample_records_per_anchor"]
            or manifest.get("condition_records")
            != spec["intervention"]["expected_conditions_per_anchor"]
            or set(outputs) != {"sample_metrics", "condition_metrics"}
            or direction_manifest.get("schema_version") != SCHEMA_VERSION
            or direction_manifest.get("status") != "complete"
            or Path(direction_manifest["checkpoint"]["path"]).resolve()
            != job.common_state.checkpoint
            or direction_manifest.get("spectral_transplant_spec", {}).get("sha256")
            != _sha256(spec_path)
            or direction_manifest.get("source_update_manifest") != source_update_identity
            or direction_manifest.get("tensors")
            != spec["anchor_scope"]["expected_hidden_tensors_per_anchor"]
            or direction_manifest.get("parameters")
            != spec["anchor_scope"]["expected_hidden_parameters_per_anchor"]
            or direction_manifest.get("condition_records")
            != spec["anchor_scope"]["expected_hidden_tensors_per_anchor"]
            * len(transformed_conditions)
            or set(direction_outputs)
            != {"direction_metrics", *(condition.name for condition in transformed_conditions)}
        ):
            return False
        for root, declared in (
            (job.output_dir, outputs.values()),
            (job.output_dir / "directions", direction_outputs.values()),
        ):
            for item in declared:
                path = root / item["path"]
                if not path.is_file() or path.stat().st_size != item.get("bytes"):
                    return False
                if verify_hashes and _sha256(path) != item.get("sha256"):
                    return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_job(
    job: SpectralTransplantJob,
    *,
    spectral_spec: Path,
    common_state_spec: Path,
    device: str,
) -> None:
    run_spectral_transplant_intervention(
        job.common_state.checkpoint,
        job.common_state.update_dir,
        job.output_dir,
        family=job.common_state.family,
        spectral_spec=spectral_spec,
        device=device,
    )
    if not spectral_transplant_job_complete(
        job,
        spectral_spec,
        common_state_spec,
    ):
        raise RuntimeError(f"Spectral transplant failed its audit: {job.label}")


def _job_cli(job: SpectralTransplantJob, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "embed_optim.spectral_transplant_matrix",
        "--worker",
        "--family",
        job.common_state.family,
        "--label",
        job.label,
        "--checkpoint",
        str(job.common_state.checkpoint),
        "--update-dir",
        str(job.common_state.update_dir),
        "--output-dir",
        str(job.output_dir),
        "--spectral-spec",
        str(args.spectral_spec.resolve()),
        "--common-state-spec",
        str(args.common_state_spec.resolve()),
        "--device",
        "cuda",
    ]


def _launch(
    job: SpectralTransplantJob,
    gpu: str,
    args: argparse.Namespace,
    attempts: int,
) -> RunningJob:
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


def run_matrix(jobs: list[SpectralTransplantJob], args: argparse.Namespace) -> int:
    pending = [
        job
        for job in jobs
        if not spectral_transplant_job_complete(
            job,
            args.spectral_spec,
            args.common_state_spec,
            verify_hashes=args.verify_hashes,
        )
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
    unavailable = [
        job.label
        for job in pending
        if not common_state_job_complete(
            job.common_state,
            args.common_state_spec,
            verify_hashes=args.verify_hashes,
        )
    ]
    if unavailable:
        raise ValueError(
            "Spectral transplant requires completed common-state inputs: " + ", ".join(unavailable)
        )
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
            if return_code == 0 and spectral_transplant_job_complete(
                running_job.job,
                args.spectral_spec,
                args.common_state_spec,
            ):
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
        description="Run the frozen AdamW--Muon spectrum/basis transplant matrix"
    )
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--families", nargs="+", choices=("dense", "late"), default=["dense", "late"]
    )
    parser.add_argument("--dense-reference-checkpoint", type=Path)
    parser.add_argument("--late-reference-checkpoint", type=Path)
    parser.add_argument("--common-state-root", type=Path, default=Path("results/common-state"))
    parser.add_argument("--output-root", type=Path, default=Path("results/spectral-transplant"))
    parser.add_argument(
        "--common-state-spec", type=Path, default=Path("configs/common_state_probe.json")
    )
    parser.add_argument(
        "--spectral-spec",
        type=Path,
        default=Path("configs/spectral_transplant_intervention.json"),
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/spectral-transplant"))
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-hashes", action="store_true")

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--family", choices=("dense", "late"), help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--update-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative")
    args.spectral_spec, protocol = load_spectral_transplant_protocol(args.spectral_spec)
    args.common_state_spec = resolve_common_state_spec(args.common_state_spec).resolve()
    if _sha256(args.common_state_spec) != protocol["source_inputs"]["common_state_spec_sha256"]:
        raise ValueError("Common-state spec differs from the spectral-transplant lock")
    common_spec, common_anchor = _load_protocol(args.common_state_spec)
    if args.worker:
        required = (
            args.family,
            args.label,
            args.checkpoint,
            args.update_dir,
            args.output_dir,
        )
        if any(value is None for value in required):
            raise ValueError("Worker invocation is missing required job fields")
        partition = protocol["anchor_scope"]
        common_job = CommonStateJob(
            family=args.family,
            label=args.label,
            checkpoint=args.checkpoint.resolve(),
            gradient_dir=args.update_dir.resolve().parent / "gradients",
            update_dir=args.update_dir.resolve(),
            gradient_steps=int(common_spec["selection"]["gradient_steps"]),
            hidden_tensors=int(partition["expected_hidden_tensors_per_anchor"]),
            hidden_parameters=int(partition["expected_hidden_parameters_per_anchor"]),
        )
        run_job(
            SpectralTransplantJob(common_job, args.output_dir.resolve()),
            spectral_spec=args.spectral_spec,
            common_state_spec=args.common_state_spec,
            device=args.device,
        )
        return

    matrix_path = resolve_matrix_path(args.matrix).resolve()
    all_configs = load_matrix(matrix_path)
    configs = [config for config in all_configs if config.model_family in args.families]
    if not configs:
        raise ValueError("No matrix configurations matched the requested families")
    if not set(args.families).issubset(common_anchor["expected_families"]):
        raise ValueError("Requested families are outside the frozen common-state protocol")
    by_family = {config.model_family: config for config in configs}
    references: dict[ModelFamily, Path] = {}
    for family, config in by_family.items():
        explicit = (
            args.dense_reference_checkpoint if family == "dense" else args.late_reference_checkpoint
        )
        if args.dry_run and explicit is None:
            references[family] = Path(config.model_name).resolve()
        else:
            references[family] = _resolve_reference(config, explicit)
    common_jobs = build_common_state_jobs(
        configs,
        references,
        common_spec,
        args.common_state_root,
    )
    jobs = build_spectral_transplant_jobs(common_jobs, args.output_root)
    expected = protocol["anchor_scope"]["expected_total_anchors"]
    if len(jobs) != expected:
        raise AssertionError(f"Built {len(jobs)} spectral jobs, expected {expected}")
    failures = run_matrix(jobs, args)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
