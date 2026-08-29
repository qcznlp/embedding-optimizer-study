from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .aggregate import (
    audit_training_artifacts,
    collect_system_metrics,
    collect_training_history,
)
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .probe_matrix import ProbeJob, _requested_probe_identity
from .probes import resolve_probe_spec_path
from .representation_summary import (
    ExpectedMetric,
    summarize_probe_metrics,
)
from .short_branch import audit_short_branch_subset
from .short_branch_evaluation import (
    ShortBranchValidationJob,
    _audit_counts,
    _load_branch_configs,
    build_short_branch_probe_jobs,
    build_short_branch_validation_jobs,
)
from .validation_data import load_validation_spec
from .validation_evaluation import METRICS

FAMILIES = ("dense", "late")
OPERATORS = ("adamw", "muon", "normuon")
CONTRASTS = (("muon", "adamw"), ("normuon", "adamw"), ("normuon", "muon"))
PRIMARY_VALIDATION_METRICS = (
    "contrastive_loss",
    "positive_margin",
    "reciprocal_rank",
    "top1_accuracy",
)


def _canonical_operator(name: str) -> str:
    return "adamw" if name == "hybrid_adamw" else name


def expected_short_branch_probe_metrics(
    configs: dict[int, list[Any]], probe_jobs: list[ProbeJob]
) -> list[ExpectedMetric]:
    by_label = {job.label: job for job in probe_jobs}
    expected = []
    for family in FAMILIES:
        label = f"{family}/pretrained"
        expected.append(ExpectedMetric(by_label[label], "", "", "pretrained", 0, 0.0, 0))
    for seed, runs in sorted(configs.items()):
        for config in sorted(runs, key=lambda item: (item.model_family, item.run_id)):
            schedule = json.loads(
                (config.output_dir / "checkpoint_schedule.json").read_text(encoding="utf-8")
            )
            steps = [int(step) for step in schedule["steps"]]
            if len(steps) != 5 or steps != sorted(set(steps)):
                raise ValueError(f"Seed {seed}: short-branch checkpoint schedule differs")
            for stage, (step, fraction) in enumerate(
                zip(steps, config.checkpoint_fractions, strict=True), start=1
            ):
                label = f"{config.model_family}/seed{seed}/{config.run_id}/checkpoint-{step}"
                expected.append(
                    ExpectedMetric(
                        by_label[label],
                        _canonical_operator(config.optimizer.name),
                        config.optimizer.lr,
                        config.run_id,
                        stage,
                        float(fraction),
                        step,
                        seed,
                    )
                )
    if len(expected) != 92 or len({item.job.label for item in expected}) != 92:
        raise ValueError("Short-branch representation summary requires 92 unique jobs")
    return expected


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"Empty short-branch metric table: {path}")
    return rows


def load_short_branch_validation_rows(
    jobs: list[ShortBranchValidationJob], validation_spec: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _, spec = load_validation_spec(validation_spec)
    expected_groups = {"__all__", *spec["expected"]["quotas"]}
    checkpoint_rows = []
    group_rows = []
    sources = []
    for job in jobs:
        manifest_path = job.output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        group_item = manifest["outputs"]["group_metrics"]
        group_path = job.output_dir / group_item["path"]
        if (
            group_path.stat().st_size != group_item["bytes"]
            or _sha256(group_path) != group_item["sha256"]
        ):
            raise ValueError(f"Short-branch validation output differs: {group_path}")
        rows = _jsonl(group_path)
        indexed = {str(row["group"]): row for row in rows}
        if len(rows) != 8 or set(indexed) != expected_groups:
            raise ValueError(f"Short-branch validation groups differ: {job.label}")
        steps = [
            int(step)
            for step in json.loads(
                (job.config.output_dir / "checkpoint_schedule.json").read_text(encoding="utf-8")
            )["steps"]
        ]
        stage = steps.index(job.step) + 1
        identity = {
            "family": job.config.model_family,
            "seed": job.seed,
            "operator": _canonical_operator(job.config.optimizer.name),
            "run_id": job.config.run_id,
            "stage": stage,
            "fraction": float(job.config.checkpoint_fractions[stage - 1]),
            "step": job.step,
            "label": job.label,
        }
        for group, row in sorted(indexed.items()):
            metrics = {metric: float(row[metric]) for metric in METRICS}
            if not all(math.isfinite(value) for value in metrics.values()):
                raise ValueError(f"Non-finite short-branch validation metric: {job.label}/{group}")
            if group == "__all__":
                checkpoint_rows.append({**identity, **metrics})
            else:
                group_rows.append({**identity, "group": group, **metrics})
        sources.append(
            {
                "label": job.label,
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": _sha256(manifest_path),
                "group_metrics_path": str(group_path.resolve()),
                "group_metrics_sha256": _sha256(group_path),
            }
        )
    if len(checkpoint_rows) != 90 or len(group_rows) != 630 or len(sources) != 90:
        raise ValueError("Short-branch query-disjoint summary coverage differs")
    return checkpoint_rows, group_rows, sources


def summarize_short_branch_contrasts(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = {
        (int(row["seed"]), str(row["family"]), int(row["stage"]), str(row["operator"])): row
        for row in rows
    }
    if len(indexed) != 90:
        raise ValueError("Short-branch contrast input requires 90 unique checkpoint rows")
    seeds = sorted({identity[0] for identity in indexed})
    expected = {
        (seed, family, stage, operator)
        for seed in seeds
        for family in FAMILIES
        for stage in range(1, 6)
        for operator in OPERATORS
    }
    if len(seeds) != 3 or set(indexed) != expected:
        raise ValueError("Short-branch contrast input coverage differs")
    contrasts = []
    for seed in seeds:
        for family in FAMILIES:
            for stage in range(1, 6):
                for treatment, baseline in CONTRASTS:
                    left = indexed[(seed, family, stage, treatment)]
                    right = indexed[(seed, family, stage, baseline)]
                    contrasts.append(
                        {
                            "family": family,
                            "seed": seed,
                            "stage": stage,
                            "fraction": left["fraction"],
                            "treatment": treatment,
                            "baseline": baseline,
                            **{
                                f"delta_{metric}": float(left[metric]) - float(right[metric])
                                for metric in PRIMARY_VALIDATION_METRICS
                            },
                        }
                    )
    summaries = []
    grouped: dict[tuple[str, int, str, str, str], list[float]] = defaultdict(list)
    for row in contrasts:
        for metric in PRIMARY_VALIDATION_METRICS:
            grouped[
                (
                    str(row["family"]),
                    int(row["stage"]),
                    str(row["treatment"]),
                    str(row["baseline"]),
                    metric,
                )
            ].append(float(row[f"delta_{metric}"]))
    for (family, stage, treatment, baseline, metric), values in sorted(grouped.items()):
        lower_is_better = metric == "contrastive_loss"
        wins = sum(value < 0 if lower_is_better else value > 0 for value in values)
        ties = sum(value == 0 for value in values)
        summaries.append(
            {
                "family": family,
                "stage": stage,
                "fraction": stage / 5,
                "treatment": treatment,
                "baseline": baseline,
                "metric": metric,
                "seeds": len(values),
                "mean_delta": statistics.mean(values),
                "seed_delta_standard_deviation": statistics.stdev(values),
                "treatment_seed_wins": wins,
                "seed_ties": ties,
                "treatment_seed_losses": len(values) - wins - ties,
                "beneficial_direction": "negative" if lower_is_better else "positive",
            }
        )
    if len(contrasts) != 90 or len(summaries) != 120:
        raise AssertionError("Short-branch contrast cardinality invariant failed")
    return contrasts, summaries


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bridge_rows(
    validation: list[dict[str, Any]], representation_dir: Path
) -> list[dict[str, Any]]:
    checkpoint = [
        row
        for row in _read_csv(representation_dir / "checkpoint_metrics.csv")
        if row["kind"] == "checkpoint"
    ]
    geometries = [
        row
        for row in _read_csv(representation_dir / "representation_metrics.csv")
        if row["kind"] == "checkpoint"
        and (
            (row["family"] == "dense" and row["representation_role"] == "queries")
            or (row["family"] == "late" and row["representation_role"] == "query_tokens")
        )
    ]

    def key(row: dict[str, Any]) -> tuple[int, str, str, int]:
        return int(row["seed"]), str(row["family"]), str(row["optimizer"]), int(row["stage"])

    validation_index = {key(row): row for row in validation}
    checkpoint_index = {key(row): row for row in checkpoint}
    geometry_index = {key(row): row for row in geometries}
    if not (
        len(validation_index) == len(checkpoint_index) == len(geometry_index) == 90
        and set(validation_index) == set(checkpoint_index) == set(geometry_index)
    ):
        raise ValueError("Short-branch validation and representation checkpoints do not align")
    output = []
    for identity in sorted(validation_index):
        validation_row = validation_index[identity]
        score = checkpoint_index[identity]
        geometry = geometry_index[identity]
        output.append(
            {
                "seed": identity[0],
                "family": identity[1],
                "operator": identity[2],
                "stage": identity[3],
                "fraction": validation_row["fraction"],
                **{
                    f"validation_{metric}": validation_row[metric]
                    for metric in PRIMARY_VALIDATION_METRICS
                },
                "unseen_margin_mean": score["margin_mean"],
                "unseen_top1_accuracy": score["top1_accuracy"],
                "unseen_mean_reciprocal_rank": score["mean_reciprocal_rank"],
                "pretrained_top_k_overlap": score["reference_mean_top_k_overlap"],
                "pretrained_top1_agreement": score["reference_top1_agreement"],
                "pretrained_score_drift_rms": score["reference_score_drift_rms"],
                "query_geometry_role": geometry["representation_role"],
                "query_entropy_effective_rank": geometry["entropy_effective_rank"],
                "query_normalized_effective_rank": geometry["normalized_effective_rank"],
                "query_mean_pairwise_cosine": geometry["mean_pairwise_cosine"],
                "late_token_evidence_entropy": score["token_evidence_entropy_mean"],
                "late_document_token_coverage": score["document_token_coverage_mean"],
                "late_repeated_token_dominance": score["repeated_token_dominance_mean"],
            }
        )
    return output


def build_short_branch_report(
    protocol_path: str | Path = "configs/short_branch_protocol.json",
    *,
    experiment_matrix: str | Path = "configs/experiment.yaml",
    matrix_dir: str | Path | None = None,
    results_root: str | Path = "results/short-branch",
    output_dir: str | Path = "reports/short-branch",
    validation_spec: str | Path = "configs/validation_probe.json",
    unseen_probe: str | Path | None = None,
    unseen_probe_spec: str | Path = "configs/beir_representation_probe.json",
) -> dict[str, Any]:
    protocol_path, protocol, configs, generated = _load_branch_configs(
        protocol_path,
        experiment_matrix=experiment_matrix,
        matrix_dir=matrix_dir,
        audit_matrices=True,
    )
    validation_spec_path, _ = load_validation_spec(validation_spec)
    root = Path(results_root).resolve()
    validation_jobs = build_short_branch_validation_jobs(configs, root / "query-disjoint")
    probe_path = Path(unseen_probe or protocol["evaluation"]["unseen_retrieval_probe"]).resolve()
    probe_spec_path = resolve_probe_spec_path(unseen_probe_spec).resolve()
    probe_identity = _requested_probe_identity(probe_path, probe_spec_path)
    probe_jobs = build_short_branch_probe_jobs(
        configs,
        {"dense": Path("."), "late": Path(".")},
        root / "unseen-representation",
        probe_identity,
    )
    counts = _audit_counts(validation_jobs, probe_jobs, validation_spec_path)
    if counts != {
        "validation_complete": 90,
        "validation_expected": 90,
        "unseen_probe_complete": 92,
        "unseen_probe_expected": 92,
    }:
        raise ValueError(f"Short-branch evaluation matrix is incomplete: {counts}")
    training_audits = {}
    system_rows = []
    history_rows = []
    dataset = audit_short_branch_subset(protocol_path)
    for seed, runs in sorted(configs.items()):
        training = audit_training_artifacts(
            runs,
            deep=True,
            expected_dataset_fingerprint=dataset["training_view_fingerprint"],
            expected_dataset_rows=dataset["rows"],
        )
        if not training.get("complete"):
            raise ValueError(f"Seed {seed}: short-branch training artifacts failed strict audit")
        training_audits[str(seed)] = {"dataset": dataset, "training": training}
        for row in collect_system_metrics(runs):
            system_rows.append(
                {
                    "seed": seed,
                    **row,
                    "optimizer": _canonical_operator(str(row["optimizer"])),
                }
            )
        for row in collect_training_history(runs):
            history_rows.append(
                {
                    "seed": seed,
                    **row,
                    "optimizer": _canonical_operator(str(row["optimizer"])),
                }
            )
    if len(system_rows) != 18 or not history_rows:
        raise ValueError("Short-branch training dynamics or system metrics are incomplete")

    validation_rows, group_rows, validation_sources = load_short_branch_validation_rows(
        validation_jobs, validation_spec_path
    )
    contrast_rows, contrast_summaries = summarize_short_branch_contrasts(validation_rows)
    output = Path(output_dir).resolve()
    representation_manifest = summarize_probe_metrics(
        expected_short_branch_probe_metrics(configs, probe_jobs),
        output / "unseen-representation",
        probe_manifest_path=probe_path / "manifest.json",
        probe_spec_path=probe_spec_path,
    )
    bridge_rows = _bridge_rows(validation_rows, output / "unseen-representation")
    from .confirmatory_summary import _atomic_csv

    tables = {
        "validation_checkpoints": _atomic_csv(
            output / "validation_checkpoint_metrics.csv", validation_rows
        ),
        "validation_groups": _atomic_csv(output / "validation_group_metrics.csv", group_rows),
        "paired_contrasts": _atomic_csv(output / "paired_checkpoint_contrasts.csv", contrast_rows),
        "paired_summary": _atomic_csv(output / "paired_dynamics_summary.csv", contrast_summaries),
        "mechanism_bridge": _atomic_csv(output / "mechanism_bridge.csv", bridge_rows),
        "system_metrics": _atomic_csv(output / "system_metrics.csv", system_rows),
        "training_history": _atomic_csv(output / "training_history.csv", history_rows),
    }
    matrix_manifest = generated / "manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "matrix_manifest": {
            "path": str(matrix_manifest),
            "sha256": _sha256(matrix_manifest),
        },
        "evaluation_counts": counts,
        "training_audits": training_audits,
        "validation_sources": validation_sources,
        "representation_summary": {
            "path": str((output / "unseen-representation" / "summary_manifest.json")),
            "sha256": _sha256(output / "unseen-representation" / "summary_manifest.json"),
            "valid_jobs": representation_manifest["valid_jobs"],
        },
        "coverage": {
            "runs": 18,
            "checkpoints": len(validation_rows),
            "validation_group_rows": len(group_rows),
            "paired_checkpoint_contrasts": len(contrast_rows),
            "paired_dynamics_summaries": len(contrast_summaries),
            "bridge_rows": len(bridge_rows),
            "system_rows": len(system_rows),
            "training_history_rows": len(history_rows),
        },
        "outputs": tables,
        "interpretation": (
            "These shared-start, scale-calibrated branches test accumulated optimizer-direction "
            "effects on frozen functional probes; they are not a second full-corpus BEIR study."
        ),
    }
    _atomic_json(output / "summary_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly summarize all shared-checkpoint scale-matched short branches"
    )
    parser.add_argument("--protocol", type=Path, default=Path("configs/short_branch_protocol.json"))
    parser.add_argument("--experiment-matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--matrix-dir", type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results/short-branch"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/short-branch"))
    parser.add_argument(
        "--validation-spec", type=Path, default=Path("configs/validation_probe.json")
    )
    parser.add_argument("--unseen-probe", type=Path)
    parser.add_argument(
        "--unseen-probe-spec",
        type=Path,
        default=Path("configs/beir_representation_probe.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_short_branch_report(
        args.protocol,
        experiment_matrix=args.experiment_matrix,
        matrix_dir=args.matrix_dir,
        results_root=args.results_root,
        output_dir=args.output_dir,
        validation_spec=args.validation_spec,
        unseen_probe=args.unseen_probe,
        unseen_probe_spec=args.unseen_probe_spec,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
