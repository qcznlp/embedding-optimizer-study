from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from .config import ModelFamily, RunConfig, load_matrix, resolve_matrix_path
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .gpu_lease import acquire_gpu_lease, parse_gpu_tokens
from .probe_matrix import (
    ProbeJob,
    _declared_checkpoint_steps,
    _requested_probe_identity,
    _resolve_reference,
    probe_job_complete,
    run_probe_matrix,
)
from .scope import ALL_FAMILIES, normalize_families, resolve_scope
from .short_branch import (
    audit_short_branch_matrices,
    audit_short_branch_subset,
    load_short_branch_protocol,
)
from .supplemental_training_audit import audit_derived_training_artifacts
from .validation_data import audit_validation_data, load_validation_spec
from .validation_matrix import run_matrix as run_validation_matrix
from .validation_matrix import validation_job_complete


@dataclasses.dataclass(frozen=True)
class ShortBranchValidationJob:
    config: RunConfig
    seed: int
    step: int
    checkpoint: Path
    output_dir: Path

    @property
    def label(self) -> str:
        return (
            f"{self.config.model_family}/seed{self.seed}/{self.config.run_id}/"
            f"checkpoint-{self.step}"
        )


def _load_branch_configs(
    protocol_path: str | Path,
    *,
    experiment_matrix: str | Path,
    matrix_dir: str | Path | None,
    audit_matrices: bool,
    families: tuple[str, ...] = ALL_FAMILIES,
) -> tuple[Path, dict[str, Any], dict[int, list[RunConfig]], Path]:
    families = normalize_families(families)
    resolved_protocol, protocol = load_short_branch_protocol(protocol_path)
    generated = Path(matrix_dir or protocol["training"]["matrix_output_dir"]).resolve()
    source_matrix = resolve_matrix_path(experiment_matrix).resolve()
    if audit_matrices:
        audit_short_branch_matrices(
            resolved_protocol,
            experiment_matrix=source_matrix,
            output_dir=generated,
        )
    configs: dict[int, list[RunConfig]] = {}
    for seed in protocol["training"]["order_seeds"]:
        matrix_path = generated / f"seed{seed}.yaml"
        runs = [run for run in load_matrix(matrix_path) if run.model_family in families]
        expected_per_seed = len(families) * len(protocol["training"]["optimizer_operators"])
        if (
            len(runs) != expected_per_seed
            or {run.seed for run in runs} != {seed}
            or {run.model_family for run in runs} != set(families)
        ):
            raise ValueError(f"Seed {seed}: short-branch evaluation matrix coverage differs")
        configs[int(seed)] = runs
    expected_runs = (
        len(protocol["training"]["order_seeds"])
        * len(families)
        * len(protocol["training"]["optimizer_operators"])
    )
    if sum(map(len, configs.values())) != expected_runs:
        raise ValueError(f"Short-branch evaluation did not load exactly {expected_runs} runs")
    return resolved_protocol, protocol, configs, generated


def audit_short_branch_training(
    protocol_path: str | Path,
    configs: dict[int, list[RunConfig]],
    families: tuple[str, ...] = ALL_FAMILIES,
) -> dict[str, Any]:
    """Deep-audit the shared 50K subset and all 18 short-branch runs."""

    resolved_protocol, protocol = load_short_branch_protocol(protocol_path)
    families = normalize_families(families)
    expected_seeds = {int(seed) for seed in protocol["training"]["order_seeds"]}
    expected_runs = int(protocol["training"]["expected_runs"]) * len(families) // len(ALL_FAMILIES)
    observed_families = {
        getattr(run, "model_family", None) for runs in configs.values() for run in runs
    }
    family_mismatch = None not in observed_families and observed_families != set(families)
    if (
        set(configs) != expected_seeds
        or sum(map(len, configs.values())) != expected_runs
        or family_mismatch
    ):
        raise ValueError(
            f"Short-branch training audit requires the active {expected_runs}-run matrix"
        )
    dataset = audit_short_branch_subset(resolved_protocol)
    per_seed: dict[str, Any] = {}
    errors: list[str] = []
    verified_runs = 0
    verified_checkpoints = 0
    for seed, runs in sorted(configs.items()):
        audit = audit_derived_training_artifacts(runs, dataset, deep=True)
        per_seed[str(seed)] = audit
        verified_runs += int(audit.get("verified_runs", 0))
        verified_checkpoints += int(audit.get("verified_checkpoints", 0))
        errors.extend(f"seed {seed}: {error}" for error in audit.get("errors", []))
    expected_checkpoints = expected_runs * 5
    complete = (
        not errors
        and verified_runs == expected_runs
        and verified_checkpoints == expected_checkpoints
        and all(audit.get("complete") is True for audit in per_seed.values())
    )
    if not complete and not errors:
        errors.append(
            "short-branch training coverage is "
            f"{verified_runs}/{expected_runs} runs and "
            f"{verified_checkpoints}/{expected_checkpoints} checkpoints"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "complete": complete,
        "protocol_sha256": _sha256(resolved_protocol),
        "dataset": dataset,
        "verified_runs": verified_runs,
        "expected_runs": expected_runs,
        "verified_checkpoints": verified_checkpoints,
        "expected_checkpoints": expected_checkpoints,
        "per_seed": per_seed,
        "errors": errors,
    }


def build_short_branch_validation_jobs(
    configs: dict[int, list[RunConfig]], output_root: str | Path
) -> list[ShortBranchValidationJob]:
    root = Path(output_root).resolve()
    jobs: list[ShortBranchValidationJob] = []
    for seed, runs in sorted(configs.items()):
        for config in sorted(runs, key=lambda item: (item.model_family, item.run_id)):
            for step in _declared_checkpoint_steps(config):
                jobs.append(
                    ShortBranchValidationJob(
                        config=config,
                        seed=seed,
                        step=step,
                        checkpoint=(config.output_dir / f"checkpoint-{step}").resolve(),
                        output_dir=(
                            root
                            / f"seed{seed}"
                            / config.model_family
                            / config.run_id
                            / f"checkpoint-{step}"
                        ),
                    )
                )
    labels = [job.label for job in jobs]
    expected = sum(len(runs) for runs in configs.values()) * 5
    if len(jobs) != expected or len(labels) != len(set(labels)):
        raise ValueError(f"Short-branch validation requires {expected} unique checkpoint jobs")
    return jobs


def build_short_branch_probe_jobs(
    configs: dict[int, list[RunConfig]],
    references: dict[ModelFamily, Path],
    output_root: str | Path,
    probe_identity: tuple[str, str] | None = None,
) -> list[ProbeJob]:
    root = Path(output_root).resolve()
    families = sorted({run.model_family for runs in configs.values() for run in runs})
    if not families or set(references) != set(families):
        raise ValueError("Short-branch probe evaluation requires every scoped pretrained reference")
    probe_manifest_sha256, probe_spec_sha256 = probe_identity or (None, None)
    jobs: list[ProbeJob] = []
    for family in families:
        export = root / "exports" / family / "pretrained.npz"
        jobs.append(
            ProbeJob(
                kind="reference",
                family=family,
                label=f"{family}/pretrained",
                checkpoint=references[family].resolve(),
                export=export,
                metrics=root / "metrics" / family / "pretrained.json",
                reference_export=None,
                probe_manifest_sha256=probe_manifest_sha256,
                probe_spec_sha256=probe_spec_sha256,
            )
        )
    for seed, runs in sorted(configs.items()):
        for config in sorted(runs, key=lambda item: (item.model_family, item.run_id)):
            reference_export = root / "exports" / config.model_family / "pretrained.npz"
            for step in _declared_checkpoint_steps(config):
                relative = (
                    Path(config.model_family) / f"seed{seed}" / config.run_id / f"checkpoint-{step}"
                )
                jobs.append(
                    ProbeJob(
                        kind="checkpoint",
                        family=config.model_family,
                        label=str(relative),
                        checkpoint=(config.output_dir / f"checkpoint-{step}").resolve(),
                        export=root / "exports" / relative.with_suffix(".npz"),
                        metrics=root / "metrics" / relative.with_suffix(".json"),
                        reference_export=reference_export,
                        probe_manifest_sha256=probe_manifest_sha256,
                        probe_spec_sha256=probe_spec_sha256,
                    )
                )
    labels = [job.label for job in jobs]
    expected = sum(len(runs) for runs in configs.values()) * 5 + len(families)
    if len(jobs) != expected or len(labels) != len(set(labels)):
        raise ValueError(
            "Short-branch unseen probe requires every checkpoint plus one reference per family"
        )
    return jobs


def _reference_checkpoints(
    experiment_matrix: Path,
    *,
    families: tuple[str, ...],
    dense: Path | None,
    late: Path | None,
    dry_run: bool,
) -> dict[ModelFamily, Path]:
    source = load_matrix(experiment_matrix)
    by_family = {
        family: next(run for run in source if run.model_family == family) for family in families
    }
    result: dict[ModelFamily, Path] = {}
    explicit_by_family = {"dense": dense, "late": late}
    for family in families:
        explicit = explicit_by_family[family]
        config = by_family[family]
        if explicit is not None:
            path = explicit.resolve()
            if not dry_run and not path.is_dir():
                raise FileNotFoundError(path)
            result[family] = path
        elif dry_run:
            result[family] = Path(config.model_name).resolve()
        else:
            result[family] = _resolve_reference(config, None)
    return result


def _audit_counts(
    validation_jobs: list[ShortBranchValidationJob],
    probe_jobs: list[ProbeJob],
    validation_spec: Path,
) -> dict[str, int]:
    complete_validation = sum(
        validation_job_complete(job, validation_spec, verify_hashes=True) for job in validation_jobs
    )
    complete_probe = sum(probe_job_complete(job) for job in probe_jobs)
    return {
        "validation_complete": complete_validation,
        "validation_expected": len(validation_jobs),
        "unseen_probe_complete": complete_probe,
        "unseen_probe_expected": len(probe_jobs),
    }


def _run_gpu_tiers(
    args: argparse.Namespace,
    validation_jobs: list[ShortBranchValidationJob],
    probe_jobs: list[ProbeJob],
) -> int:
    failures = 0
    if args.tier in {"validation", "both"}:
        validation_args = argparse.Namespace(
            **{
                **vars(args),
                "probe": args.validation_probe,
                "log_dir": Path("logs/short-branch/query-disjoint"),
            }
        )
        failures += run_validation_matrix(validation_jobs, validation_args)
    if args.tier in {"unseen", "both"}:
        probe_args = argparse.Namespace(
            **{
                **vars(args),
                "probe": args.unseen_probe,
                "probe_spec": args.unseen_probe_spec,
                "log_dir": Path("logs/short-branch/unseen-representation"),
            }
        )
        failures += run_probe_matrix(probe_jobs, probe_args)
    return failures


def _run_gpu_tiers_with_lease(
    args: argparse.Namespace,
    families: tuple[str, ...],
    validation_jobs: list[ShortBranchValidationJob],
    probe_jobs: list[ProbeJob],
) -> int:
    gpu_tokens = tuple(sorted(parse_gpu_tokens(args.gpus), key=int))
    lease_ledger = Path("logs/short-branch") / f"gpu-lease-{os.getpid()}.json"
    with acquire_gpu_lease(
        gpu_tokens,
        lock_dir=args.gpu_lock_dir,
        timeout_seconds=args.gpu_lock_timeout_seconds,
        purpose=f"short-branch:{','.join(families)}:{args.tier}",
        ledger_path=lease_ledger,
    ):
        return _run_gpu_tiers(args, validation_jobs, probe_jobs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all scale-matched short-branch checkpoints on both frozen probes"
    )
    parser.add_argument("--protocol", type=Path, default=Path("configs/short_branch_protocol.json"))
    parser.add_argument("--experiment-matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--matrix-dir", type=Path)
    parser.add_argument("--families", nargs="+", choices=("dense", "late"), default=["dense"])
    parser.add_argument("--scope-amendment", type=Path)
    parser.add_argument("--tier", choices=("validation", "unseen", "both"), default="both")
    parser.add_argument("--validation-probe", type=Path)
    parser.add_argument(
        "--validation-spec", type=Path, default=Path("configs/validation_probe.json")
    )
    parser.add_argument("--unseen-probe", type=Path)
    parser.add_argument(
        "--unseen-probe-spec",
        type=Path,
        default=Path("configs/beir_representation_probe.json"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("results/short-branch"))
    parser.add_argument("--dense-reference-checkpoint", type=Path)
    parser.add_argument("--late-reference-checkpoint", type=Path)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--gpu-lock-dir",
        type=Path,
        default=Path("logs/dense-only-runtime/gpu-leases"),
    )
    parser.add_argument("--gpu-lock-timeout-seconds", type=float, default=86_400.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--cpu-threads-per-worker",
        type=int,
        default=0,
        help=(
            "CPU threads used by each unseen-probe worker; 0 divides available CPUs "
            "across the requested GPU workers"
        ),
    )
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--training-audit-only", action="store_true")
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--no-flash-attention", action="store_true")
    parser.add_argument(
        "--receipt", type=Path, default=Path("reports/short-branch/evaluation-receipt.json")
    )
    parser.add_argument(
        "--training-audit-receipt",
        type=Path,
        default=Path("reports/short-branch/training-audit.json"),
    )
    args = parser.parse_args(argv)
    if (
        args.batch_size <= 0
        or args.max_retries < 0
        or args.cpu_threads_per_worker < 0
        or args.gpu_lock_timeout_seconds <= 0
    ):
        parser.error(
            "--batch-size/--gpu-lock-timeout-seconds must be positive and "
            "--max-retries/--cpu-threads-per-worker must be non-negative"
        )
    if sum((args.dry_run, args.audit_only, args.training_audit_only)) > 1:
        parser.error("--dry-run, --audit-only, and --training-audit-only are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    families, scope = resolve_scope(args.families, args.scope_amendment)
    protocol_path, protocol, configs, matrix_dir = _load_branch_configs(
        args.protocol,
        experiment_matrix=args.experiment_matrix,
        matrix_dir=args.matrix_dir,
        audit_matrices=True,
        families=families,
    )
    if args.training_audit_only or (not args.audit_only and not args.dry_run):
        training_audit = audit_short_branch_training(protocol_path, configs, families)
        if not training_audit["complete"]:
            details = "; ".join(training_audit["errors"][:10])
            raise RuntimeError(f"Short-branch training preflight failed: {details}")
        _atomic_json(args.training_audit_receipt, training_audit)
        if args.training_audit_only:
            print(json.dumps(training_audit, indent=2, sort_keys=True))
            return
    args.experiment_matrix = resolve_matrix_path(args.experiment_matrix).resolve()
    args.validation_spec, validation_spec = load_validation_spec(args.validation_spec)
    args.validation_probe = Path(
        args.validation_probe or protocol["evaluation"]["query_disjoint_probe"]
    ).resolve()
    args.unseen_probe = Path(
        args.unseen_probe or protocol["evaluation"]["unseen_retrieval_probe"]
    ).resolve()
    args.unseen_probe_spec = resolve_matrix_path(args.unseen_probe_spec).resolve()
    args.output_root = args.output_root.resolve()

    if not args.dry_run:
        audit_validation_data(
            args.validation_probe,
            validation_spec["source"]["training_data"],
            spec_path=args.validation_spec,
        )
    validation_jobs = build_short_branch_validation_jobs(
        configs, args.output_root / "query-disjoint"
    )
    references = _reference_checkpoints(
        args.experiment_matrix,
        families=families,
        dense=args.dense_reference_checkpoint,
        late=args.late_reference_checkpoint,
        dry_run=args.dry_run,
    )
    probe_identity = _requested_probe_identity(args.unseen_probe, args.unseen_probe_spec)
    probe_jobs = build_short_branch_probe_jobs(
        configs,
        references,
        args.output_root / "unseen-representation",
        probe_identity,
    )

    if args.audit_only:
        counts = _audit_counts(validation_jobs, probe_jobs, args.validation_spec)
        print(json.dumps(counts, indent=2, sort_keys=True))
        if (
            counts["validation_complete"] != counts["validation_expected"]
            or counts["unseen_probe_complete"] != counts["unseen_probe_expected"]
        ):
            raise SystemExit(1)
    elif args.dry_run:
        failures = _run_gpu_tiers(args, validation_jobs, probe_jobs)
        if failures:
            raise SystemExit(1)
    else:
        failures = _run_gpu_tiers_with_lease(args, families, validation_jobs, probe_jobs)
        if failures:
            raise SystemExit(1)

    counts = _audit_counts(validation_jobs, probe_jobs, args.validation_spec)
    if (
        counts["validation_complete"] == counts["validation_expected"]
        and counts["unseen_probe_complete"] == counts["unseen_probe_expected"]
    ):
        manifest_path = matrix_dir / "manifest.json"
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "families": list(families),
            "scope_amendment": scope,
            "protocol_sha256": _sha256(protocol_path),
            "matrix_manifest_sha256": _sha256(manifest_path),
            "validation_spec_sha256": _sha256(args.validation_spec),
            "unseen_probe_spec_sha256": _sha256(args.unseen_probe_spec),
            **counts,
        }
        _atomic_json(args.receipt, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
