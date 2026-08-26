from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from .config import ModelFamily, RunConfig, load_matrix, resolve_matrix_path
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .probe_matrix import (
    ProbeJob,
    _declared_checkpoint_steps,
    _requested_probe_identity,
    _resolve_reference,
    probe_job_complete,
    run_probe_matrix,
)
from .short_branch import audit_short_branch_matrices, load_short_branch_protocol
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
) -> tuple[Path, dict[str, Any], dict[int, list[RunConfig]], Path]:
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
        runs = load_matrix(matrix_path)
        if len(runs) != protocol["training"]["runs_per_seed"] or {run.seed for run in runs} != {
            seed
        }:
            raise ValueError(f"Seed {seed}: short-branch evaluation matrix coverage differs")
        configs[int(seed)] = runs
    if sum(map(len, configs.values())) != protocol["training"]["expected_runs"]:
        raise ValueError("Short-branch evaluation did not load exactly 18 runs")
    return resolved_protocol, protocol, configs, generated


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
    if len(jobs) != 90 or len(labels) != len(set(labels)):
        raise ValueError("Short-branch validation requires 90 unique checkpoint jobs")
    return jobs


def build_short_branch_probe_jobs(
    configs: dict[int, list[RunConfig]],
    references: dict[ModelFamily, Path],
    output_root: str | Path,
    probe_identity: tuple[str, str] | None = None,
) -> list[ProbeJob]:
    root = Path(output_root).resolve()
    families = sorted({run.model_family for runs in configs.values() for run in runs})
    if set(families) != {"dense", "late"} or set(references) != set(families):
        raise ValueError("Short-branch probe evaluation requires both pretrained references")
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
    if len(jobs) != 92 or len(labels) != len(set(labels)):
        raise ValueError("Short-branch unseen probe requires 90 checkpoints plus two references")
    return jobs


def _reference_checkpoints(
    experiment_matrix: Path,
    *,
    dense: Path | None,
    late: Path | None,
    dry_run: bool,
) -> dict[ModelFamily, Path]:
    source = load_matrix(experiment_matrix)
    by_family = {
        family: next(run for run in source if run.model_family == family)
        for family in ("dense", "late")
    }
    result: dict[ModelFamily, Path] = {}
    for family, explicit in (("dense", dense), ("late", late)):
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all scale-matched short-branch checkpoints on both frozen probes"
    )
    parser.add_argument("--protocol", type=Path, default=Path("configs/short_branch_protocol.json"))
    parser.add_argument("--experiment-matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--matrix-dir", type=Path)
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--no-flash-attention", action="store_true")
    parser.add_argument(
        "--receipt", type=Path, default=Path("reports/short-branch/evaluation-receipt.json")
    )
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.max_retries < 0:
        parser.error("--batch-size must be positive and --max-retries must be non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    protocol_path, protocol, configs, matrix_dir = _load_branch_configs(
        args.protocol,
        experiment_matrix=args.experiment_matrix,
        matrix_dir=args.matrix_dir,
        audit_matrices=True,
    )
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
    else:
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
        if failures:
            raise SystemExit(1)

    counts = _audit_counts(validation_jobs, probe_jobs, args.validation_spec)
    if counts == {
        "validation_complete": 90,
        "validation_expected": 90,
        "unseen_probe_complete": 92,
        "unseen_probe_expected": 92,
    }:
        manifest_path = matrix_dir / "manifest.json"
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
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
