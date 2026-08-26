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

from huggingface_hub import snapshot_download

from .config import ModelFamily, RunConfig, load_matrix, resolve_matrix_path
from .geometry import SCHEMA_VERSION, _sha256
from .gradient_probe import export_gradient_probe
from .update_geometry import ALGORITHMS, UpdateOperatorConfig, analyze_common_state_updates


@dataclass(frozen=True)
class CommonStateJob:
    family: ModelFamily
    label: str
    checkpoint: Path
    gradient_dir: Path
    update_dir: Path
    gradient_steps: int
    hidden_tensors: int
    hidden_parameters: int


@dataclass
class RunningJob:
    job: CommonStateJob
    process: subprocess.Popen
    log_handle: Any
    attempts: int


def resolve_common_state_spec(path: str | Path, prefix: Path | None = None) -> Path:
    path = Path(path)
    if path.is_file() or path.is_absolute() or path.parent != Path("configs"):
        return path
    prefix = Path(sys.prefix) if prefix is None else prefix
    installed = prefix / "share" / "embedding-optimizer-study" / "configs" / path.name
    return installed if installed.is_file() else path


def _load_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported common-state specification schema: {path}")
    anchor = spec.get("anchor_protocol")
    selection = spec.get("selection")
    gradient = spec.get("gradient_protocol")
    operator = spec.get("operator_protocol")
    analysis = spec.get("analysis_protocol")
    if not all(
        isinstance(value, dict) for value in (anchor, selection, gradient, operator, analysis)
    ):
        raise ValueError(f"Common-state specification is missing a protocol section: {path}")
    required_anchor = {
        "include_pretrained",
        "run_ids",
        "checkpoint_fractions",
        "expected_families",
        "expected_anchors_per_family",
        "expected_total_anchors",
        "expected_hidden_partition",
        "selection_basis",
        "freeze_context",
    }
    if set(anchor) != required_anchor:
        raise ValueError(
            "anchor_protocol fields differ from the frozen schema: "
            f"expected={sorted(required_anchor)}, observed={sorted(anchor)}"
        )
    if anchor["include_pretrained"] is not True:
        raise ValueError("The formal common-state matrix must include the pretrained anchor")
    run_ids = anchor["run_ids"]
    fractions = anchor["checkpoint_fractions"]
    families = anchor["expected_families"]
    if (
        not isinstance(run_ids, list)
        or not run_ids
        or len(run_ids) != len(set(run_ids))
        or not all(isinstance(value, str) and value for value in run_ids)
    ):
        raise ValueError("anchor_protocol.run_ids must be unique non-empty strings")
    if (
        not isinstance(fractions, list)
        or not fractions
        or fractions != sorted(set(fractions))
        or not all(isinstance(value, (int, float)) and 0 < value <= 1 for value in fractions)
    ):
        raise ValueError("anchor checkpoint fractions must be unique, increasing, and in (0, 1]")
    if families != ["dense", "late"]:
        raise ValueError("The frozen common-state matrix must declare dense and late families")
    expected_per_family = 1 + len(run_ids) * len(fractions)
    if anchor["expected_anchors_per_family"] != expected_per_family:
        raise ValueError("expected_anchors_per_family disagrees with the anchor grid")
    if anchor["expected_total_anchors"] != expected_per_family * len(families):
        raise ValueError("expected_total_anchors disagrees with the anchor grid")
    partition = anchor["expected_hidden_partition"]
    if (
        not isinstance(partition, dict)
        or set(partition) != {"tensors", "parameters"}
        or not all(isinstance(value, int) and value > 0 for value in partition.values())
    ):
        raise ValueError("expected_hidden_partition must contain positive tensor/parameter counts")
    freeze_context = anchor["freeze_context"]
    valid_units = (
        freeze_context.get("strict_beir_valid_units") if isinstance(freeze_context, dict) else None
    )
    expected_units = (
        freeze_context.get("strict_beir_expected_units")
        if isinstance(freeze_context, dict)
        else None
    )
    if (
        not isinstance(freeze_context, dict)
        or set(freeze_context)
        != {
            "frozen_at_utc",
            "strict_beir_valid_units",
            "strict_beir_expected_units",
            "partial_results_already_observed",
        }
        or not isinstance(freeze_context["frozen_at_utc"], str)
        or isinstance(valid_units, bool)
        or not isinstance(valid_units, int)
        or isinstance(expected_units, bool)
        or not isinstance(expected_units, int)
        or not 0 <= valid_units < expected_units
        or freeze_context["partial_results_already_observed"] is not True
    ):
        raise ValueError("freeze_context must disclose the partial-result state at protocol freeze")
    gradient_steps = selection.get("gradient_steps")
    if (
        isinstance(gradient_steps, bool)
        or not isinstance(gradient_steps, int)
        or gradient_steps <= 0
    ):
        raise ValueError("selection.gradient_steps must be a positive integer")
    return spec, anchor


def _checkpoint_for_fraction(config: RunConfig, fraction: float) -> Path:
    schedule_path = config.output_dir / "checkpoint_schedule.json"
    try:
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        fractions = [float(value) for value in schedule["fractions"]]
        steps = [int(value) for value in schedule["steps"]]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid checkpoint schedule {schedule_path}: {error}") from error
    if len(fractions) != len(steps) or len(steps) != 5 or steps != sorted(set(steps)):
        raise ValueError(f"Invalid five-stage checkpoint schedule {schedule_path}")
    matches = [
        step for declared, step in zip(fractions, steps, strict=True) if declared == fraction
    ]
    if len(matches) != 1:
        raise ValueError(f"Fraction {fraction} is missing or ambiguous in {schedule_path}")
    checkpoint = (config.output_dir / f"checkpoint-{matches[0]}").resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def build_common_state_jobs(
    configs: list[RunConfig],
    references: dict[ModelFamily, Path],
    spec: dict[str, Any],
    output_root: Path,
) -> list[CommonStateJob]:
    anchor = spec["anchor_protocol"]
    families = sorted({config.model_family for config in configs})
    run_ids = list(anchor["run_ids"])
    fractions = [float(value) for value in anchor["checkpoint_fractions"]]
    partition = anchor["expected_hidden_partition"]
    selected: dict[tuple[ModelFamily, str], RunConfig] = {}
    for config in configs:
        if config.run_id in run_ids:
            key = (config.model_family, config.run_id)
            if key in selected:
                raise ValueError(f"Duplicate matrix configuration for {key}")
            selected[key] = config
    missing_references = [family for family in families if family not in references]
    if missing_references:
        raise ValueError(f"Missing pretrained references for families: {missing_references}")
    missing_runs = [
        f"{family}/{run_id}"
        for family in families
        for run_id in run_ids
        if (family, run_id) not in selected
    ]
    if missing_runs:
        raise ValueError(f"Common-state anchor runs are absent from the matrix: {missing_runs}")

    def job(family: ModelFamily, label: str, checkpoint: Path) -> CommonStateJob:
        root = output_root.resolve() / label
        return CommonStateJob(
            family=family,
            label=label,
            checkpoint=checkpoint.resolve(),
            gradient_dir=root / "gradients",
            update_dir=root / "updates",
            gradient_steps=int(spec["selection"]["gradient_steps"]),
            hidden_tensors=int(partition["tensors"]),
            hidden_parameters=int(partition["parameters"]),
        )

    jobs: list[CommonStateJob] = []
    for family in families:
        jobs.append(job(family, f"{family}/pretrained", references[family]))
        for run_id in run_ids:
            config = selected[(family, run_id)]
            for fraction in fractions:
                checkpoint = _checkpoint_for_fraction(config, fraction)
                jobs.append(
                    job(
                        family,
                        f"{family}/{run_id}/{checkpoint.name}",
                        checkpoint,
                    )
                )
    expected = len(families) * int(anchor["expected_anchors_per_family"])
    if len(jobs) != expected:
        raise AssertionError(f"Built {len(jobs)} common-state jobs, expected {expected}")
    return jobs


def _declared_file_valid(root: Path, item: Any, *, verify_hashes: bool) -> bool:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        return False
    path = root / item["path"]
    if not path.is_file() or path.stat().st_size != item.get("bytes"):
        return False
    return not verify_hashes or item.get("sha256") == _sha256(path)


def common_state_job_complete(
    job: CommonStateJob,
    common_state_spec: Path,
    *,
    verify_hashes: bool = False,
) -> bool:
    gradient_manifest_path = job.gradient_dir / "manifest.json"
    update_manifest_path = job.update_dir / "manifest.json"
    if not gradient_manifest_path.is_file() or not update_manifest_path.is_file():
        return False
    try:
        gradient = json.loads(gradient_manifest_path.read_text(encoding="utf-8"))
        update = json.loads(update_manifest_path.read_text(encoding="utf-8"))
        spec_sha256 = _sha256(common_state_spec)
        shards = gradient["gradient_shards"]
        hidden = gradient["partition_summary"]["hidden"]
        outputs = update["outputs"]
        if (
            gradient.get("schema_version") != SCHEMA_VERSION
            or gradient.get("status") != "complete"
            or Path(gradient["checkpoint"]["path"]).resolve() != job.checkpoint
            or gradient["common_state_spec"]["sha256"] != spec_sha256
            or len(shards) != job.gradient_steps
            or hidden != {"tensors": job.hidden_tensors, "parameters": job.hidden_parameters}
            or update.get("schema_version") != SCHEMA_VERSION
            or Path(update["checkpoint"]["path"]).resolve() != job.checkpoint
            or update["common_state_spec"]["sha256"] != spec_sha256
            or update["gradient_steps"] != job.gradient_steps
            or update["tensors"] != job.hidden_tensors
            or update["parameters"] != job.hidden_parameters
            or set(outputs) != {"metrics", *(f"{name}_matched" for name in ALGORITHMS)}
        ):
            return False
        if not all(
            _declared_file_valid(job.gradient_dir, item, verify_hashes=verify_hashes)
            for item in shards
        ):
            return False
        if not all(
            _declared_file_valid(job.update_dir, item, verify_hashes=verify_hashes)
            for item in outputs.values()
        ):
            return False
        gradient_identity = update["gradient_manifest"]
        return (
            Path(gradient_identity["path"]).resolve() == gradient_manifest_path.resolve()
            and gradient_identity["bytes"] == gradient_manifest_path.stat().st_size
            and gradient_identity["sha256"] == _sha256(gradient_manifest_path)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_common_state_job(
    job: CommonStateJob,
    *,
    probe: Path,
    probe_spec: Path,
    common_state_spec: Path,
    device: str,
) -> None:
    spec, _ = _load_protocol(common_state_spec)
    selection = spec["selection"]
    gradient = spec["gradient_protocol"]
    analysis = spec["analysis_protocol"]
    operator = UpdateOperatorConfig(**spec["operator_protocol"])
    export_gradient_probe(
        job.checkpoint,
        probe,
        job.gradient_dir,
        family=job.family,
        probe_spec=probe_spec,
        common_state_spec=common_state_spec,
        gradient_steps=int(selection["gradient_steps"]),
        examples_per_gradient=int(selection["examples_per_gradient"]),
        micro_batch_size=int(gradient["micro_batch_size"]),
        seed=int(selection["seed"]),
        max_grad_norm=float(gradient["max_grad_norm"]),
        model_dtype=gradient["model_dtype"],
        forward_dtype=gradient["forward_dtype"],
        storage_dtype=gradient["storage_dtype"],
        device=device,
        flash_attention=True,
        train_mode=gradient["model_mode"] == "train",
        gradient_checkpointing=bool(gradient["gradient_checkpointing"]),
    )
    update_manifest_path = job.update_dir / "manifest.json"
    partial_update = not update_manifest_path.is_file() and any(
        (job.update_dir / name).is_file()
        for name in ["metrics.jsonl", *(f"{value}-matched.safetensors" for value in ALGORITHMS)]
    )
    analyze_common_state_updates(
        job.checkpoint,
        job.gradient_dir / "manifest.json",
        job.update_dir,
        operator_config=operator,
        common_state_spec=common_state_spec,
        operator_device=device,
        sketch_rank=int(analysis["sketch_rank"]),
        oversample=int(analysis["oversample"]),
        power_iterations=int(analysis["power_iterations"]),
        seed=int(analysis["seed"]),
        storage_dtype=analysis["matched_update_storage_dtype"],
        overwrite=partial_update,
    )
    if not common_state_job_complete(job, common_state_spec):
        raise RuntimeError(f"Common-state job did not pass its completion audit: {job.label}")


def _job_cli(job: CommonStateJob, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "embed_optim.common_state_matrix",
        "--worker",
        "--family",
        job.family,
        "--label",
        job.label,
        "--checkpoint",
        str(job.checkpoint),
        "--gradient-dir",
        str(job.gradient_dir),
        "--update-dir",
        str(job.update_dir),
        "--hidden-tensors",
        str(job.hidden_tensors),
        "--hidden-parameters",
        str(job.hidden_parameters),
        "--probe",
        str(args.probe.resolve()),
        "--probe-spec",
        str(args.probe_spec.resolve()),
        "--common-state-spec",
        str(args.common_state_spec.resolve()),
        "--device",
        "cuda",
    ]


def _launch(job: CommonStateJob, gpu: str, args: argparse.Namespace, attempts: int) -> RunningJob:
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


def run_common_state_matrix(jobs: list[CommonStateJob], args: argparse.Namespace) -> int:
    pending = [
        job
        for job in jobs
        if not common_state_job_complete(
            job, args.common_state_spec, verify_hashes=args.verify_hashes
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
            if return_code == 0 and common_state_job_complete(
                running_job.job, args.common_state_spec
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
                print(f"failed {running_job.job.label} after {running_job.attempts} attempts")
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


def _resolve_reference(config: RunConfig, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    return Path(
        snapshot_download(repo_id=config.model_name, revision=config.model_revision)
    ).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered common-state gradient and update-geometry matrix"
    )
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--families", nargs="+", choices=("dense", "late"), default=["dense", "late"]
    )
    parser.add_argument("--dense-reference-checkpoint", type=Path)
    parser.add_argument("--late-reference-checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results/common-state"))
    parser.add_argument("--probe", type=Path, default=Path("data/probes/training-1024-seed1729"))
    parser.add_argument(
        "--probe-spec", type=Path, default=Path("configs/representation_probe.json")
    )
    parser.add_argument(
        "--common-state-spec", type=Path, default=Path("configs/common_state_probe.json")
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/common-state"))
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-hashes", action="store_true")

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--family", choices=("dense", "late"), help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--gradient-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--update-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--hidden-tensors", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--hidden-parameters", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative")
    args.common_state_spec = resolve_common_state_spec(args.common_state_spec).resolve()
    spec, anchor = _load_protocol(args.common_state_spec)
    if args.worker:
        required = (
            args.family,
            args.label,
            args.checkpoint,
            args.gradient_dir,
            args.update_dir,
            args.hidden_tensors,
            args.hidden_parameters,
        )
        if any(value is None for value in required):
            raise ValueError("Worker invocation is missing required job fields")
        run_common_state_job(
            CommonStateJob(
                family=args.family,
                label=args.label,
                checkpoint=args.checkpoint.resolve(),
                gradient_dir=args.gradient_dir.resolve(),
                update_dir=args.update_dir.resolve(),
                gradient_steps=int(spec["selection"]["gradient_steps"]),
                hidden_tensors=args.hidden_tensors,
                hidden_parameters=args.hidden_parameters,
            ),
            probe=args.probe.resolve(),
            probe_spec=args.probe_spec.resolve(),
            common_state_spec=args.common_state_spec,
            device=args.device,
        )
        return

    matrix_path = resolve_matrix_path(args.matrix).resolve()
    all_configs = load_matrix(matrix_path)
    configs = [config for config in all_configs if config.model_family in args.families]
    if not configs:
        raise ValueError("No matrix configurations matched the requested families")
    expected_families = set(anchor["expected_families"])
    if not set(args.families).issubset(expected_families):
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
    jobs = build_common_state_jobs(configs, references, spec, args.output_root)
    failures = run_common_state_matrix(jobs, args)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
