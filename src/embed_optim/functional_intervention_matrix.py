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
from .functional_intervention import (
    load_intervention_protocol,
    run_functional_intervention,
)
from .geometry import SCHEMA_VERSION, _sha256


@dataclass(frozen=True)
class FunctionalInterventionJob:
    common_state: CommonStateJob
    output_dir: Path

    @property
    def label(self) -> str:
        return self.common_state.label


@dataclass
class RunningJob:
    job: FunctionalInterventionJob
    process: subprocess.Popen
    log_handle: Any
    attempts: int


def build_functional_intervention_jobs(
    common_jobs: list[CommonStateJob], output_root: str | Path
) -> list[FunctionalInterventionJob]:
    output_root = Path(output_root).resolve()
    return [FunctionalInterventionJob(job, output_root / job.label) for job in common_jobs]


def functional_intervention_job_complete(
    job: FunctionalInterventionJob,
    intervention_spec: str | Path,
    *,
    verify_hashes: bool = False,
) -> bool:
    try:
        spec_path, spec = load_intervention_protocol(intervention_spec)
        manifest_path = job.output_dir / "manifest.json"
        update_manifest_path = job.common_state.update_dir / "manifest.json"
        if not manifest_path.is_file() or not update_manifest_path.is_file():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs = manifest["outputs"]
        expected_outputs = {"sample_metrics", "condition_metrics"}
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("status") != "complete"
            or manifest.get("family") != job.common_state.family
            or Path(manifest["checkpoint"]["path"]).resolve() != job.common_state.checkpoint
            or manifest["intervention_spec"]["sha256"] != _sha256(spec_path)
            or manifest["common_state_updates"]["sha256"] != _sha256(update_manifest_path)
            or manifest.get("sample_records")
            != spec["intervention"]["expected_sample_records_per_anchor"]
            or manifest.get("condition_records")
            != spec["intervention"]["expected_conditions_per_anchor"]
            or set(outputs) != expected_outputs
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


def run_job(
    job: FunctionalInterventionJob,
    *,
    probe: Path,
    intervention_spec: Path,
    device: str,
) -> None:
    run_functional_intervention(
        job.common_state.checkpoint,
        job.common_state.update_dir,
        probe,
        job.output_dir,
        family=job.common_state.family,
        intervention_spec=intervention_spec,
        device=device,
    )
    if not functional_intervention_job_complete(job, intervention_spec):
        raise RuntimeError(f"Functional intervention failed its audit: {job.label}")


def _job_cli(job: FunctionalInterventionJob, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "embed_optim.functional_intervention_matrix",
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
        "--probe",
        str(args.probe.resolve()),
        "--intervention-spec",
        str(args.intervention_spec.resolve()),
        "--device",
        "cuda",
    ]


def _launch(
    job: FunctionalInterventionJob,
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


def run_matrix(jobs: list[FunctionalInterventionJob], args: argparse.Namespace) -> int:
    pending = [
        job
        for job in jobs
        if not functional_intervention_job_complete(
            job, args.intervention_spec, verify_hashes=args.verify_hashes
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
            job.common_state, args.common_state_spec, verify_hashes=args.verify_hashes
        )
    ]
    if unavailable:
        raise ValueError(
            "Functional interventions require completed common-state inputs: "
            + ", ".join(unavailable)
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
            if return_code == 0 and functional_intervention_job_complete(
                running_job.job, args.intervention_spec
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
        description="Run the frozen scale-matched functional-intervention matrix"
    )
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--families", nargs="+", choices=("dense", "late"), default=["dense", "late"]
    )
    parser.add_argument("--dense-reference-checkpoint", type=Path)
    parser.add_argument("--late-reference-checkpoint", type=Path)
    parser.add_argument("--common-state-root", type=Path, default=Path("results/common-state"))
    parser.add_argument("--output-root", type=Path, default=Path("results/functional-intervention"))
    parser.add_argument(
        "--probe", type=Path, default=Path("data/probes/decontaminated-beir-224-seed4242")
    )
    parser.add_argument(
        "--common-state-spec", type=Path, default=Path("configs/common_state_probe.json")
    )
    parser.add_argument(
        "--intervention-spec", type=Path, default=Path("configs/functional_intervention.json")
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/functional-intervention"))
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
    args.intervention_spec, intervention = load_intervention_protocol(args.intervention_spec)
    args.common_state_spec = resolve_common_state_spec(args.common_state_spec).resolve()
    if _sha256(args.common_state_spec) != intervention["common_state"]["spec_sha256"]:
        raise ValueError("Common-state spec differs from the functional-intervention lock")
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
        partition = intervention["common_state"]["expected_hidden_partition"]
        common_job = CommonStateJob(
            family=args.family,
            label=args.label,
            checkpoint=args.checkpoint.resolve(),
            gradient_dir=args.update_dir.resolve().parent / "gradients",
            update_dir=args.update_dir.resolve(),
            gradient_steps=int(common_spec["selection"]["gradient_steps"]),
            hidden_tensors=int(partition["tensors"]),
            hidden_parameters=int(partition["parameters"]),
        )
        run_job(
            FunctionalInterventionJob(common_job, args.output_dir.resolve()),
            probe=args.probe.resolve(),
            intervention_spec=args.intervention_spec,
            device=args.device,
        )
        return

    matrix_path = resolve_matrix_path(args.matrix).resolve()
    all_configs = load_matrix(matrix_path)
    configs = [config for config in all_configs if config.model_family in args.families]
    if not configs:
        raise ValueError("No matrix configurations matched the requested families")
    if not set(args.families).issubset(set(common_anchor["expected_families"])):
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
    common_jobs = build_common_state_jobs(configs, references, common_spec, args.common_state_root)
    jobs = build_functional_intervention_jobs(common_jobs, args.output_root)
    if len(jobs) != intervention["common_state"]["expected_anchors"]:
        raise ValueError("Functional intervention job count differs from its frozen protocol")
    failures = run_matrix(jobs, args)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
