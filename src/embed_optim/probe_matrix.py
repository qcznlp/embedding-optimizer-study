from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from huggingface_hub import snapshot_download

from .config import ModelFamily, RunConfig, load_matrix, resolve_matrix_path
from .geometry import _sha256
from .probe_export import export_probe
from .probes import resolve_probe_spec_path
from .representation_geometry import SCHEMA_VERSION, _export_manifest_identity, analyze_probe
from .scope import resolve_scope

JobKind = Literal["reference", "checkpoint"]


@dataclass(frozen=True)
class ProbeJob:
    kind: JobKind
    family: ModelFamily
    label: str
    checkpoint: Path
    export: Path
    metrics: Path
    reference_export: Path | None
    probe_manifest_sha256: str | None = None
    probe_spec_sha256: str | None = None


@dataclass
class RunningJob:
    job: ProbeJob
    process: subprocess.Popen
    log_handle: Any
    attempts: int


def _declared_checkpoint_steps(config: RunConfig) -> list[int]:
    schedule_path = config.output_dir / "checkpoint_schedule.json"
    try:
        payload = json.loads(schedule_path.read_text(encoding="utf-8"))
        steps = [int(value) for value in payload["steps"]]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid checkpoint schedule {schedule_path}: {error}") from error
    if len(steps) != 5 or steps != sorted(set(steps)):
        raise ValueError(
            f"Expected five increasing checkpoint steps in {schedule_path}, got {steps}"
        )
    missing = [step for step in steps if not (config.output_dir / f"checkpoint-{step}").is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoints for {config.model_family}/{config.run_id}: {missing}"
        )
    return steps


def build_probe_jobs(
    configs: list[RunConfig],
    references: dict[ModelFamily, Path],
    output_root: Path,
    probe_identity: tuple[str, str] | None = None,
) -> list[ProbeJob]:
    output_root = output_root.resolve()
    families = sorted({config.model_family for config in configs})
    missing_references = [family for family in families if family not in references]
    if missing_references:
        raise ValueError(f"Missing pretrained references for families: {missing_references}")

    jobs: list[ProbeJob] = []
    probe_manifest_sha256, probe_spec_sha256 = probe_identity or (None, None)
    for family in families:
        export = output_root / "exports" / family / "pretrained.npz"
        jobs.append(
            ProbeJob(
                kind="reference",
                family=family,
                label=f"{family}/pretrained",
                checkpoint=references[family].resolve(),
                export=export,
                metrics=output_root / "metrics" / family / "pretrained.json",
                reference_export=None,
                probe_manifest_sha256=probe_manifest_sha256,
                probe_spec_sha256=probe_spec_sha256,
            )
        )

    for config in sorted(configs, key=lambda item: (item.model_family, item.run_id)):
        reference_export = output_root / "exports" / config.model_family / "pretrained.npz"
        for step in _declared_checkpoint_steps(config):
            relative = Path(config.model_family) / config.run_id / f"checkpoint-{step}"
            jobs.append(
                ProbeJob(
                    kind="checkpoint",
                    family=config.model_family,
                    label=f"{config.model_family}/{config.run_id}/checkpoint-{step}",
                    checkpoint=(config.output_dir / f"checkpoint-{step}").resolve(),
                    export=output_root / "exports" / relative.with_suffix(".npz"),
                    metrics=output_root / "metrics" / relative.with_suffix(".json"),
                    reference_export=reference_export,
                    probe_manifest_sha256=probe_manifest_sha256,
                    probe_spec_sha256=probe_spec_sha256,
                )
            )
    return jobs


def _requested_probe_identity(probe: Path, probe_spec: Path) -> tuple[str, str]:
    manifest_path = probe.resolve() / "manifest.json"
    selection_path = probe.resolve() / "selection.jsonl"
    if not manifest_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError(f"Frozen probe is incomplete under {probe.resolve()}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("selection_sha256") != _sha256(selection_path):
        raise ValueError("Frozen probe selection ledger digest mismatch")
    manifest_sha256 = _sha256(manifest_path)
    resolved_spec = resolve_probe_spec_path(probe_spec).resolve()
    spec = json.loads(resolved_spec.read_text(encoding="utf-8"))
    expected = spec.get("expected")
    if not isinstance(expected, dict) or expected.get("manifest_sha256") != manifest_sha256:
        raise ValueError(
            f"Probe {manifest_path} is not bound to frozen specification {resolved_spec}"
        )
    return manifest_sha256, _sha256(resolved_spec)


def _valid_export(
    path: Path,
    family: ModelFamily,
    *,
    probe_manifest_sha256: str | None = None,
    probe_spec_sha256: str | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        source_sha256 = _sha256(path)
        with np.load(path, allow_pickle=False) as archive:
            array_metadata = {
                name: {"shape": list(archive[name].shape), "dtype": str(archive[name].dtype)}
                for name in sorted(archive.files)
            }
        _export_manifest_identity(
            path.resolve(),
            source_sha256=source_sha256,
            family=family,
            array_metadata=array_metadata,
            required=True,
        )
        manifest = json.loads(
            path.with_suffix(path.suffix + ".manifest.json").read_text(encoding="utf-8")
        )
        probe = manifest["probe"]
        if (
            probe_manifest_sha256 is not None
            and probe.get("manifest_sha256") != probe_manifest_sha256
        ):
            return False
        frozen_spec = probe.get("frozen_spec")
        if probe_spec_sha256 is not None and (
            not isinstance(frozen_spec, dict) or frozen_spec.get("sha256") != probe_spec_sha256
        ):
            return False
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def probe_job_complete(job: ProbeJob) -> bool:
    identity = {
        "probe_manifest_sha256": job.probe_manifest_sha256,
        "probe_spec_sha256": job.probe_spec_sha256,
    }
    if not _valid_export(job.export, job.family, **identity) or not job.metrics.is_file():
        return False
    if job.reference_export is not None and not _valid_export(
        job.reference_export, job.family, **identity
    ):
        return False
    try:
        payload = json.loads(job.metrics.read_text(encoding="utf-8"))
        current = payload["input"]
        export_manifest = current["export_manifest"]
        reference = current.get("reference")
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("family") != job.family
            or payload.get("label") != job.label
            or Path(current["path"]).resolve() != job.export.resolve()
            or current["sha256"] != _sha256(job.export)
            or export_manifest["sha256"]
            != _sha256(job.export.with_suffix(job.export.suffix + ".manifest.json"))
            or payload["parameters"].get("require_export_manifest") is not True
        ):
            return False
        if job.reference_export is None:
            return reference is None
        return (
            isinstance(reference, dict)
            and Path(reference["path"]).resolve() == job.reference_export.resolve()
            and reference["sha256"] == _sha256(job.reference_export)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_probe_job(
    job: ProbeJob,
    *,
    probe: Path,
    probe_spec: Path,
    batch_size: int,
    model_dtype: Literal["bfloat16", "float32"],
    storage_dtype: Literal["float16", "float32"],
    device: str,
    flash_attention: bool,
) -> None:
    if not _valid_export(
        job.export,
        job.family,
        probe_manifest_sha256=job.probe_manifest_sha256,
        probe_spec_sha256=job.probe_spec_sha256,
    ):
        export_probe(
            job.checkpoint,
            probe,
            job.export,
            family=job.family,
            batch_size=batch_size,
            model_dtype=model_dtype,
            storage_dtype=storage_dtype,
            device=device,
            flash_attention=flash_attention,
            overwrite=job.export.exists()
            or job.export.with_suffix(job.export.suffix + ".manifest.json").exists(),
            probe_spec=probe_spec,
        )
    analyze_probe(
        job.export,
        job.metrics,
        family=job.family,
        label=job.label,
        batch_size=max(1, min(batch_size, 16)),
        require_export_manifest=True,
        reference_source=job.reference_export,
    )
    if not probe_job_complete(job):
        raise RuntimeError(f"Probe job did not pass its completion audit: {job.label}")


def _job_cli(job: ProbeJob, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "embed_optim.probe_matrix",
        "--worker",
        "--kind",
        job.kind,
        "--family",
        job.family,
        "--label",
        job.label,
        "--checkpoint",
        str(job.checkpoint),
        "--export",
        str(job.export),
        "--metrics",
        str(job.metrics),
        "--probe",
        str(args.probe.resolve()),
        "--probe-spec",
        str(args.probe_spec.resolve()),
        "--batch-size",
        str(args.batch_size),
        "--model-dtype",
        args.model_dtype,
        "--storage-dtype",
        args.storage_dtype,
        "--device",
        "cuda:0",
        "--gpus",
        args.gpus,
        "--cpu-threads-per-worker",
        str(args.cpu_threads_per_worker),
    ]
    if job.reference_export is not None:
        command.extend(["--reference-export", str(job.reference_export)])
    if args.no_flash_attention:
        command.append("--no-flash-attention")
    return command


def _launch(job: ProbeJob, gpu: str, args: argparse.Namespace, attempts: int) -> RunningJob:
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
    return RunningJob(job=job, process=process, log_handle=log_handle, attempts=attempts)


def run_probe_matrix(jobs: list[ProbeJob], args: argparse.Namespace) -> int:
    pending = [job for job in jobs if not probe_job_complete(job)]
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
            if return_code == 0 and probe_job_complete(running_job.job):
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
                print(f"failed {running_job.job.label} after {running_job.attempts} attempts")
                if args.fail_fast:
                    for other in running.values():
                        other.process.terminate()
                    return failures

        for gpu in gpus:
            if gpu in running:
                continue
            ready_index = next(
                (
                    index
                    for index, job in enumerate(pending)
                    if job.reference_export is None
                    or _valid_export(
                        job.reference_export,
                        job.family,
                        probe_manifest_sha256=job.probe_manifest_sha256,
                        probe_spec_sha256=job.probe_spec_sha256,
                    )
                ),
                None,
            )
            if ready_index is None:
                continue
            job = pending.pop(ready_index)
            attempt = attempts.get(job.label, 0) + 1
            attempts[job.label] = attempt
            running[gpu] = _launch(job, gpu, args, attempt)

        if pending and not running:
            blocked = ", ".join(job.label for job in pending[:5])
            raise RuntimeError(
                f"No runnable probe jobs remain; missing reference outputs for {blocked}"
            )
        if running:
            time.sleep(1)
    return failures


def _resolve_reference(config: RunConfig, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    return Path(
        snapshot_download(
            repo_id=config.model_name,
            revision=config.model_revision,
        )
    ).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and analyze the fixed probe matrix")
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--families", nargs="+", choices=("dense", "late"), default=["dense"])
    parser.add_argument("--scope-amendment", type=Path)
    parser.add_argument("--run-ids", nargs="*", default=[])
    parser.add_argument("--dense-reference-checkpoint", type=Path)
    parser.add_argument("--late-reference-checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results/representation-space"))
    parser.add_argument("--probe", type=Path, default=Path("data/probes/training-1024-seed1729"))
    parser.add_argument(
        "--probe-spec", type=Path, default=Path("configs/representation_probe.json")
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/representation-space"))
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-flash-attention", action="store_true")
    parser.add_argument(
        "--cpu-threads-per-worker",
        type=int,
        default=0,
        help=(
            "CPU threads used by each probe worker; 0 divides available CPUs "
            "across the requested GPU workers"
        ),
    )

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--kind", choices=("reference", "checkpoint"), help=argparse.SUPPRESS)
    parser.add_argument("--family", choices=("dense", "late"), help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--export", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--metrics", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--reference-export", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_size <= 0 or args.max_retries < 0 or args.cpu_threads_per_worker < 0:
        raise ValueError(
            "--batch-size must be positive; --max-retries and "
            "--cpu-threads-per-worker must be non-negative"
        )
    if not args.worker:
        families, _ = resolve_scope(args.families, args.scope_amendment)
        args.families = list(families)
    if args.worker:
        requested_workers = len({value.strip() for value in args.gpus.split(",") if value.strip()})
        cpu_threads = args.cpu_threads_per_worker or max(
            1, (os.cpu_count() or 1) // max(1, requested_workers)
        )
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(1)
        required = (args.kind, args.family, args.label, args.checkpoint, args.export, args.metrics)
        if any(value is None for value in required):
            raise ValueError("Worker invocation is missing required job fields")
        probe_identity = _requested_probe_identity(args.probe, args.probe_spec)
        run_probe_job(
            ProbeJob(
                kind=args.kind,
                family=args.family,
                label=args.label,
                checkpoint=args.checkpoint.resolve(),
                export=args.export.resolve(),
                metrics=args.metrics.resolve(),
                reference_export=(
                    None if args.reference_export is None else args.reference_export.resolve()
                ),
                probe_manifest_sha256=probe_identity[0],
                probe_spec_sha256=probe_identity[1],
            ),
            probe=args.probe.resolve(),
            probe_spec=args.probe_spec.resolve(),
            batch_size=args.batch_size,
            model_dtype=args.model_dtype,
            storage_dtype=args.storage_dtype,
            device=args.device,
            flash_attention=not args.no_flash_attention,
        )
        return

    matrix_path = resolve_matrix_path(args.matrix).resolve()
    configs = [
        config
        for config in load_matrix(matrix_path)
        if config.model_family in args.families
        and (not args.run_ids or config.run_id in args.run_ids)
    ]
    if not configs:
        raise ValueError("No matrix configurations matched the requested filters")
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
    probe_identity = _requested_probe_identity(args.probe, args.probe_spec)
    jobs = build_probe_jobs(configs, references, args.output_root, probe_identity)
    failures = run_probe_matrix(jobs, args)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
