"""Evaluate all five factorial checkpoints on the frozen 224-query unseen probe."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from . import probe_export, probe_matrix
from .config import RunConfig, load_matrix
from .decontamination import DECONTAMINATED_TASK_NAMES
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .gpu_lease import acquire_gpu_lease, parse_gpu_tokens
from .probe_matrix import ProbeJob, _requested_probe_identity, _resolve_reference
from .representation_geometry import dense_probe_scores
from .state_operator_factorial import (
    MATRIX_ROOT,
    SCIENTIFIC_PROTOCOL,
    audit_branch_data,
    audit_factorial_matrices,
    load_factorial_protocol,
)
from .state_operator_factorial_contract import require_factorial_implementation
from .supplemental_training_audit import audit_derived_training_artifacts

PROBE_ROOT = Path("data/probes/decontaminated-beir-224-seed4242")
PROBE_SPEC = Path("configs/beir_representation_probe.json")
RESULTS_ROOT = Path("results/state-operator-factorial/probe")
LOG_ROOT = Path("logs/state-operator-factorial/probe")
GPU_LOCK_ROOT = Path("logs/dense-only-runtime/gpu-leases")
_base_probe_job_complete = probe_matrix.probe_job_complete


def load_all_configs(
    *,
    protocol_path: str | Path = SCIENTIFIC_PROTOCOL,
    matrix_root: str | Path = MATRIX_ROOT,
    calibration_root: str | Path | None = None,
) -> tuple[Path, dict[str, Any], list[RunConfig]]:
    resolved, protocol = load_factorial_protocol(protocol_path)
    kwargs: dict[str, Any] = {"matrix_root": matrix_root, "deep_data_audit": False}
    if calibration_root is not None:
        kwargs["calibration_root"] = calibration_root
    audit_factorial_matrices(resolved, **kwargs)
    root = Path(matrix_root).resolve()
    configs = [
        config
        for state in protocol["factorial_design"]["factors"]["weight_state"]
        for seed in protocol["branch_data"]["order_seeds"]
        for config in load_matrix(root / f"{state}-seed{seed}.yaml")
    ]
    identities = {(config.seed, config.run_id) for config in configs}
    if (
        len(configs) != protocol["factorial_design"]["expected_runs"]
        or len(identities) != len(configs)
        or any(config.model_family != "dense" for config in configs)
        or any(config.dense_can_flatten_inputs is not False for config in configs)
    ):
        raise ValueError("All-factorial matrix coverage differs")
    return resolved, protocol, configs


def _state_for_run(run_id: str) -> str:
    prefix, separator, _ = run_id.partition("__")
    if not separator or prefix not in {"adamw-state", "muon-state"}:
        raise ValueError(f"Invalid factorial run ID: {run_id}")
    return prefix.replace("-", "_")


def _checkpoint_steps(config: RunConfig, expected: list[int]) -> list[int]:
    schedule_path = config.output_dir / "checkpoint_schedule.json"
    try:
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        steps = [int(value) for value in schedule["steps"]]
        fractions = [float(value) for value in schedule["fractions"]]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid factorial checkpoint schedule: {schedule_path}") from error
    if steps != expected or fractions != list(config.checkpoint_fractions):
        raise ValueError(f"Factorial checkpoint schedule differs: {schedule_path}")
    if any(not (config.output_dir / f"checkpoint-{step}").is_dir() for step in steps):
        raise FileNotFoundError(f"Factorial checkpoint is missing under {config.output_dir}")
    return steps


def build_probe_jobs(
    configs: list[RunConfig],
    protocol: dict[str, Any],
    reference: Path,
    output_root: str | Path = RESULTS_ROOT,
    probe_identity: tuple[str, str] | None = None,
) -> list[ProbeJob]:
    root = Path(output_root).resolve()
    manifest_sha256, spec_sha256 = probe_identity or (None, None)
    reference_export = root / "exports/dense/pretrained.npz"
    jobs = [
        ProbeJob(
            kind="reference",
            family="dense",
            label="dense/pretrained",
            checkpoint=reference.resolve(),
            export=reference_export,
            metrics=root / "metrics/dense/pretrained.json",
            reference_export=None,
            probe_manifest_sha256=manifest_sha256,
            probe_spec_sha256=spec_sha256,
        )
    ]
    expected_steps = protocol["factorial_design"]["training"]["expected_checkpoint_steps"]
    for config in sorted(configs, key=lambda item: (item.seed, item.run_id)):
        state = _state_for_run(config.run_id)
        for step in _checkpoint_steps(config, expected_steps):
            relative = (
                Path("dense") / state / f"seed{config.seed}" / config.run_id / f"checkpoint-{step}"
            )
            jobs.append(
                ProbeJob(
                    kind="checkpoint",
                    family="dense",
                    label=str(relative),
                    checkpoint=(config.output_dir / f"checkpoint-{step}").resolve(),
                    export=root / "exports" / relative.with_suffix(".npz"),
                    metrics=root / "metrics" / relative.with_suffix(".json"),
                    reference_export=reference_export,
                    probe_manifest_sha256=manifest_sha256,
                    probe_spec_sha256=spec_sha256,
                )
            )
    expected = 1 + protocol["factorial_design"]["expected_runs"] * 5
    if len(jobs) != expected or len({job.label for job in jobs}) != expected:
        raise ValueError(f"Factorial probe requires {expected} unique jobs")
    return jobs


def _receipt_path(job: ProbeJob) -> Path:
    return job.export.with_suffix(job.export.suffix + ".padded-execution.json")


def _factorial_metric_path(job: ProbeJob) -> Path:
    return job.metrics.with_name(f"{job.metrics.stem}.factorial.json")


def _file_identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _job_request(
    job: ProbeJob, protocol_path: Path, probe: Path, probe_spec: Path, args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "scientific_protocol": _file_identity(protocol_path),
        "kind": job.kind,
        "label": job.label,
        "checkpoint": str(job.checkpoint.resolve()),
        "probe": str(probe.resolve()),
        "probe_manifest_sha256": job.probe_manifest_sha256,
        "probe_spec": _file_identity(probe_spec),
        "export": str(job.export.resolve()),
        "representation_metrics": str(job.metrics.resolve()),
        "factorial_metrics": (
            None if job.kind == "reference" else str(_factorial_metric_path(job).resolve())
        ),
        "reference_export": (
            None if job.reference_export is None else str(job.reference_export.resolve())
        ),
        "encoding": {
            "batch_size": args.batch_size,
            "model_dtype": args.model_dtype,
            "storage_dtype": args.storage_dtype,
            "device": "cuda:0",
            "flash_attention": not args.no_flash_attention,
        },
        "input_execution": {
            "mode": "independently_padded",
            "sentence_transformers_can_flatten_inputs": False,
        },
    }


@contextmanager
def _padded_probe_loader(receipt_path: Path, receipt: dict[str, Any]) -> Iterator[None]:
    original = probe_export._load_model

    def load(*args: Any, **kwargs: Any) -> Any:
        model = original(*args, **kwargs)
        family = kwargs.get("family", args[0] if args else None)
        if family != "dense" or not hasattr(model, "_first_module"):
            raise TypeError("Factorial probe requires a Dense SentenceTransformer model")
        first = model._first_module()
        if not hasattr(first, "can_flatten_inputs"):
            raise AttributeError("Dense transformer exposes no can_flatten_inputs control")
        first.can_flatten_inputs = False
        if bool(first.can_flatten_inputs):
            raise RuntimeError("Could not disable Dense flattened-input execution")
        receipt["status"] = "model_verified"
        receipt["observed_input_execution"] = receipt["request"]["input_execution"]
        _atomic_json(receipt_path, receipt)
        return model

    probe_export._load_model = load
    try:
        yield
    finally:
        probe_export._load_model = original


def _summary(scores: torch.Tensor, reference_scores: torch.Tensor) -> dict[str, float | int]:
    if scores.shape != reference_scores.shape or scores.ndim != 2 or scores.shape[1] != 8:
        raise ValueError("Factorial probe scores require matching [samples, 8] arrays")
    positive = scores[:, 0]
    hardest = scores[:, 1:].max(dim=1).values
    margins = positive - hardest
    ranks = 1 + (scores[:, 1:] >= positive[:, None]).sum(dim=1)
    losses = torch.nn.functional.cross_entropy(
        scores / 0.02,
        torch.zeros(scores.shape[0], dtype=torch.long),
        reduction="none",
    )
    values: dict[str, float | int] = {
        "samples": scores.shape[0],
        "contrastive_loss_mean": float(losses.mean().item()),
        "positive_margin_mean": float(margins.mean().item()),
        "positive_margin_p05": float(torch.quantile(margins, 0.05).item()),
        "mean_reciprocal_rank": float(ranks.float().reciprocal().mean().item()),
        "top1_accuracy": float(ranks.eq(1).float().mean().item()),
        "pretrained_top1_agreement": float(
            (scores.argmax(dim=1) == reference_scores.argmax(dim=1)).float().mean().item()
        ),
    }
    if not all(isinstance(value, int) or math.isfinite(value) for value in values.values()):
        raise ValueError("Factorial probe produced a non-finite metric")
    return values


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def _write_factorial_metrics(job: ProbeJob, protocol_path: Path) -> dict[str, Any]:
    if job.kind != "checkpoint" or job.reference_export is None:
        raise ValueError("Factorial metrics require a checkpoint and pretrained reference")
    with (
        np.load(job.export, allow_pickle=False) as current,
        np.load(job.reference_export, allow_pickle=False) as reference,
    ):
        sample_ids = current["sample_ids"].astype(np.int64, copy=True)
        groups = current["sample_groups"].astype(str, copy=True)
        if not np.array_equal(sample_ids, reference["sample_ids"]):
            raise ValueError("Current and pretrained probe sample identities differ")
        if not np.array_equal(groups, reference["sample_groups"].astype(str)):
            raise ValueError("Current and pretrained probe groups differ")
        scores = dense_probe_scores(
            torch.from_numpy(current["query_embeddings"]),
            torch.from_numpy(current["document_embeddings"]),
        )
        reference_scores = dense_probe_scores(
            torch.from_numpy(reference["query_embeddings"]),
            torch.from_numpy(reference["document_embeddings"]),
        )
    unique_groups = sorted(set(groups.tolist()))
    by_task = {}
    for group in unique_groups:
        indices = torch.from_numpy(np.flatnonzero(groups == group))
        by_task[group] = _summary(scores[indices], reference_scores[indices])
    if (
        len(sample_ids) != 224
        or unique_groups != sorted(DECONTAMINATED_TASK_NAMES)
        or any(item["samples"] != 16 for item in by_task.values())
    ):
        raise ValueError("Factorial probe lost its balanced 14-by-16 task coverage")
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "label": job.label,
        "scientific_protocol": _file_identity(protocol_path),
        "input": {
            "export": _file_identity(job.export),
            "export_manifest": _file_identity(
                job.export.with_suffix(job.export.suffix + ".manifest.json")
            ),
            "reference_export": _file_identity(job.reference_export),
            "representation_metrics": _file_identity(job.metrics),
        },
        "overall": _summary(scores, reference_scores),
        "by_task": by_task,
    }
    path = _factorial_metric_path(job)
    _atomic_json(path, output)
    return output


def padded_probe_job_complete(
    job: ProbeJob,
    protocol_path: Path,
    probe: Path,
    probe_spec: Path,
    args: argparse.Namespace,
) -> bool:
    if not _base_probe_job_complete(job):
        return False
    receipt_path = _receipt_path(job)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        request = _job_request(job, protocol_path, probe, probe_spec, args)
        if (
            receipt.get("schema_version") != SCHEMA_VERSION
            or receipt.get("status") != "complete"
            or receipt.get("request") != request
            or receipt.get("observed_input_execution") != request["input_execution"]
            or receipt.get("export") != _file_identity(job.export)
            or receipt.get("export_manifest")
            != _file_identity(job.export.with_suffix(job.export.suffix + ".manifest.json"))
            or receipt.get("representation_metrics") != _file_identity(job.metrics)
        ):
            return False
        if job.kind == "checkpoint":
            metric_path = _factorial_metric_path(job)
            payload = json.loads(metric_path.read_text(encoding="utf-8"))
            if (
                receipt.get("factorial_metrics") != _file_identity(metric_path)
                or payload.get("status") != "complete"
                or payload.get("label") != job.label
                or payload.get("input", {}).get("export") != _file_identity(job.export)
                or set(payload.get("by_task", {})) != set(DECONTAMINATED_TASK_NAMES)
                or not _all_finite(payload)
            ):
                return False
        elif receipt.get("factorial_metrics") is not None:
            return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_padded_probe_job(
    job: ProbeJob,
    protocol_path: Path,
    probe: Path,
    probe_spec: Path,
    args: argparse.Namespace,
) -> None:
    require_factorial_implementation()
    receipt_path = _receipt_path(job)
    request = _job_request(job, protocol_path, probe, probe_spec, args)
    export_manifest = job.export.with_suffix(job.export.suffix + ".manifest.json")
    if (
        job.export.exists() or export_manifest.exists() or job.metrics.exists()
    ) and not receipt_path.is_file():
        raise FileExistsError("Refusing untagged probe artifacts without padded execution receipt")
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("request") != request or receipt.get("status") not in {
            "in_progress",
            "model_verified",
            "complete",
        }:
            raise ValueError("Existing padded probe receipt differs from the requested job")
        if receipt.get("status") == "complete" and padded_probe_job_complete(
            job, protocol_path, probe, probe_spec, args
        ):
            return
        if (
            _base_probe_job_complete(job)
            and receipt.get("observed_input_execution") != request["input_execution"]
        ):
            raise RuntimeError("Completed probe artifacts lack an observed padded execution")
    else:
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "in_progress",
            "request": request,
        }
        _atomic_json(receipt_path, receipt)

    with _padded_probe_loader(receipt_path, receipt):
        probe_matrix.run_probe_job(
            job,
            probe=probe,
            probe_spec=probe_spec,
            batch_size=args.batch_size,
            model_dtype=args.model_dtype,
            storage_dtype=args.storage_dtype,
            device="cuda:0",
            flash_attention=not args.no_flash_attention,
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("observed_input_execution") != request["input_execution"]:
        raise RuntimeError("Probe worker completed without observing padded Dense execution")
    if job.kind == "checkpoint":
        _write_factorial_metrics(job, protocol_path)
        factorial_metrics = _file_identity(_factorial_metric_path(job))
    else:
        factorial_metrics = None
    receipt.update(
        {
            "status": "complete",
            "export": _file_identity(job.export),
            "export_manifest": _file_identity(export_manifest),
            "representation_metrics": _file_identity(job.metrics),
            "factorial_metrics": factorial_metrics,
        }
    )
    _atomic_json(receipt_path, receipt)
    if not padded_probe_job_complete(job, protocol_path, probe, probe_spec, args):
        raise RuntimeError(f"Factorial padded probe audit failed: {job.label}")


def _job_cli(
    job: ProbeJob,
    protocol_path: Path,
    probe: Path,
    probe_spec: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "embed_optim.state_operator_factorial_probe",
        "--worker",
        "--kind",
        job.kind,
        "--label",
        job.label,
        "--checkpoint",
        str(job.checkpoint),
        "--export",
        str(job.export),
        "--metrics",
        str(job.metrics),
        "--protocol",
        str(protocol_path),
        "--probe",
        str(probe),
        "--probe-spec",
        str(probe_spec),
        "--batch-size",
        str(args.batch_size),
        "--model-dtype",
        args.model_dtype,
        "--storage-dtype",
        args.storage_dtype,
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


@contextmanager
def _factorial_scheduler_contract(
    jobs: list[ProbeJob],
    protocol_path: Path,
    probe: Path,
    probe_spec: Path,
    args: argparse.Namespace,
) -> Iterator[None]:
    by_label = {job.label: job for job in jobs}
    original_complete = probe_matrix.probe_job_complete
    original_cli = probe_matrix._job_cli

    def complete(job: ProbeJob) -> bool:
        if by_label.get(job.label) != job:
            raise RuntimeError("Probe scheduler received a job outside the factorial contract")
        return padded_probe_job_complete(job, protocol_path, probe, probe_spec, args)

    def command(job: ProbeJob, _args: argparse.Namespace) -> list[str]:
        if by_label.get(job.label) != job:
            raise RuntimeError("Probe scheduler received a job outside the factorial contract")
        return _job_cli(job, protocol_path, probe, probe_spec, args)

    probe_matrix.probe_job_complete = complete
    probe_matrix._job_cli = command
    try:
        yield
    finally:
        probe_matrix.probe_job_complete = original_complete
        probe_matrix._job_cli = original_cli


def _audit_counts(
    jobs: list[ProbeJob],
    protocol_path: Path,
    probe: Path,
    probe_spec: Path,
    args: argparse.Namespace,
) -> dict[str, int]:
    complete = sum(
        padded_probe_job_complete(job, protocol_path, probe, probe_spec, args) for job in jobs
    )
    return {"complete": complete, "expected": len(jobs), "pending": len(jobs) - complete}


def run(args: argparse.Namespace) -> dict[str, Any]:
    require_factorial_implementation()
    protocol_path, protocol, configs = load_all_configs(
        protocol_path=args.protocol,
        matrix_root=args.matrix_root,
        calibration_root=args.calibration_root,
    )
    probe = args.probe.resolve()
    probe_spec = args.probe_spec.resolve()
    probe_identity = _requested_probe_identity(probe, probe_spec)
    if args.dense_reference_checkpoint is not None:
        reference = args.dense_reference_checkpoint.resolve()
        if not reference.is_dir():
            raise FileNotFoundError(reference)
    else:
        reference = _resolve_reference(
            load_matrix(Path("configs/dense_no_packing_retrain.yaml"))[0], None
        )
    jobs = build_probe_jobs(
        configs,
        protocol,
        reference,
        args.output_root,
        probe_identity,
    )
    if args.audit_only:
        counts = _audit_counts(jobs, protocol_path, probe, probe_spec, args)
        if counts["pending"]:
            raise RuntimeError(f"Factorial probe outputs are incomplete: {counts}")
        return {"status": "complete", **counts}

    dataset = audit_branch_data(protocol_path, deep=True)
    training = audit_derived_training_artifacts(configs, dataset, deep=True)
    if training.get("complete") is not True or training.get("errors"):
        details = "; ".join(str(item) for item in training.get("errors", [])[:10])
        raise RuntimeError(f"Factorial probe training preflight failed: {details or 'incomplete'}")
    _atomic_json(
        args.training_audit_receipt,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "scientific_protocol_sha256": _sha256(protocol_path),
            "verified_runs": training["verified_runs"],
            "verified_checkpoints": training["verified_checkpoints"],
            "dataset_manifest_sha256": dataset["manifest_sha256"],
            "errors": [],
        },
    )
    with _factorial_scheduler_contract(jobs, protocol_path, probe, probe_spec, args):
        if args.dry_run:
            failures = probe_matrix.run_probe_matrix(jobs, args)
        else:
            tokens = tuple(sorted(parse_gpu_tokens(args.gpus), key=int))
            with acquire_gpu_lease(
                tokens,
                lock_dir=args.gpu_lock_dir.resolve(),
                timeout_seconds=args.gpu_lock_timeout_seconds,
                purpose="state-operator-factorial-probe",
                ledger_path=args.log_dir.resolve() / f"gpu-lease-{os.getpid()}.json",
            ):
                failures = probe_matrix.run_probe_matrix(jobs, args)
    if failures:
        raise RuntimeError(f"Factorial probe workers failed: {failures}")
    counts = _audit_counts(jobs, protocol_path, probe, probe_spec, args)
    if not args.dry_run and counts["pending"]:
        raise RuntimeError(f"Factorial probe outputs are incomplete: {counts}")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "dry_run" if args.dry_run else "complete",
        "scientific_protocol_sha256": _sha256(protocol_path),
        **counts,
    }
    if not args.dry_run:
        _atomic_json(args.receipt, result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=SCIENTIFIC_PROTOCOL)
    parser.add_argument("--matrix-root", type=Path, default=MATRIX_ROOT)
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=Path("results/dense-no-packing-state-operator/calibration"),
    )
    parser.add_argument("--probe", type=Path, default=PROBE_ROOT)
    parser.add_argument("--probe-spec", type=Path, default=PROBE_SPEC)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--log-dir", type=Path, default=LOG_ROOT)
    parser.add_argument("--dense-reference-checkpoint", type=Path)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--gpu-lock-dir", type=Path, default=GPU_LOCK_ROOT)
    parser.add_argument("--gpu-lock-timeout-seconds", type=float, default=86_400.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--cpu-threads-per-worker", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-flash-attention", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--training-audit-receipt",
        type=Path,
        default=Path("reports/state-operator-factorial/training-audit.json"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("reports/state-operator-factorial/probe-evaluation.json"),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--kind", choices=("reference", "checkpoint"), help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--export", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--metrics", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--reference-export", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (
        args.batch_size <= 0
        or args.max_retries < 0
        or args.cpu_threads_per_worker < 0
        or args.gpu_lock_timeout_seconds <= 0
    ):
        parser.error("Invalid batch/retry/thread/lease setting")
    if args.dry_run and args.audit_only:
        parser.error("--dry-run and --audit-only are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.worker:
        required = (args.kind, args.label, args.checkpoint, args.export, args.metrics)
        if any(value is None for value in required):
            raise ValueError("Factorial probe worker is missing required fields")
        torch.set_num_threads(
            args.cpu_threads_per_worker
            or max(
                1,
                (os.cpu_count() or 1)
                // max(1, len({item for item in args.gpus.split(",") if item.strip()})),
            )
        )
        torch.set_num_interop_threads(1)
        identity = _requested_probe_identity(args.probe.resolve(), args.probe_spec.resolve())
        run_padded_probe_job(
            ProbeJob(
                kind=args.kind,
                family="dense",
                label=args.label,
                checkpoint=args.checkpoint.resolve(),
                export=args.export.resolve(),
                metrics=args.metrics.resolve(),
                reference_export=(
                    None if args.reference_export is None else args.reference_export.resolve()
                ),
                probe_manifest_sha256=identity[0],
                probe_spec_sha256=identity[1],
            ),
            args.protocol.resolve(),
            args.probe.resolve(),
            args.probe_spec.resolve(),
            args,
        )
        return
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
