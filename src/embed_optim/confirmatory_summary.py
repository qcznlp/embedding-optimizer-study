from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .aggregate import (
    collect_evaluations,
    collect_system_metrics,
)
from .config import load_matrix
from .confirmatory_data import audit_confirmatory_view, load_confirmatory_protocol
from .confirmatory_evaluation import audit_confirmatory_evaluations
from .confirmatory_matrix import audit_confirmatory_matrices
from .decontamination import DECONTAMINATED_TASK_NAMES
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .scope import ALL_FAMILIES, normalize_families, resolve_scope
from .supplemental_training_audit import audit_derived_training_artifacts

FAMILIES = ("dense", "late")
OPTIMIZERS = ("adamw", "muon", "normuon")
CONTRASTS = (("muon", "adamw"), ("normuon", "adamw"), ("normuon", "muon"))
FAMILYWISE_CONTRASTS = len(FAMILIES) * len(CONTRASTS)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"Refusing to write an empty confirmatory table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "path": str(path.resolve()),
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.blake2b("/".join((str(seed), *parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def hierarchical_seed_task_bootstrap(
    effects: np.ndarray,
    *,
    samples: int = 20_000,
    seed: int = 20260826,
    familywise_contrasts: int = FAMILYWISE_CONTRASTS,
) -> dict[str, float | int]:
    values = np.asarray(effects, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] < 2
        or not np.isfinite(values).all()
        or samples < 1
        or familywise_contrasts < 1
    ):
        raise ValueError("Hierarchical bootstrap requires a finite seed-by-task matrix")
    generator = np.random.default_rng(seed)
    seed_indices = generator.integers(0, values.shape[0], size=(samples, values.shape[0]))
    task_indices = generator.integers(0, values.shape[1], size=(samples, values.shape[1]))
    draws = values[seed_indices[:, :, None], task_indices[:, None, :]].mean(axis=(1, 2))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    familywise_alpha = 0.05 / familywise_contrasts
    familywise_lower, familywise_upper = np.quantile(
        draws,
        [familywise_alpha / 2, 1 - familywise_alpha / 2],
    )
    return {
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "bootstrap_ci_95_lower": float(lower),
        "bootstrap_ci_95_upper": float(upper),
        "familywise_method": "bonferroni",
        "familywise_contrasts": familywise_contrasts,
        "familywise_ci_95_lower": float(familywise_lower),
        "familywise_ci_95_upper": float(familywise_upper),
        "bootstrap_probability_positive": float(np.mean(draws > 0)),
        "bootstrap_probability_negative": float(np.mean(draws < 0)),
    }


def summarize_confirmatory_scores(
    rows: list[dict[str, Any]],
    seeds: list[int],
    *,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 20260826,
    families: tuple[str, ...] = ALL_FAMILIES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    families = normalize_families(families)
    # The prospective claim protocol froze one six-comparison family before the
    # post-hoc Dense-only scope amendment. Filtering which rows are displayed must
    # not make the interval less conservative.
    familywise_contrasts = FAMILYWISE_CONTRASTS
    tasks = list(DECONTAMINATED_TASK_NAMES)
    expected = {
        (seed, family, optimizer, task)
        for seed in seeds
        for family in families
        for optimizer in OPTIMIZERS
        for task in tasks
    }
    indexed: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for row in rows:
        optimizer = str(row["optimizer"])
        if optimizer not in OPTIMIZERS or int(row["stage"]) != 5:
            raise ValueError(f"Unexpected confirmatory score row: {row}")
        identity = (
            int(row["seed"]),
            str(row["model_family"]),
            optimizer,
            str(row["task"]),
        )
        score = float(row["ndcg_at_10"])
        if identity in indexed or not math.isfinite(score):
            raise ValueError(f"Duplicate or non-finite confirmatory score: {identity}")
        indexed[identity] = {**row, "ndcg_at_10": score}
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        unexpected = sorted(set(indexed) - expected)
        raise ValueError(
            f"Confirmatory score coverage differs: missing={missing[:3]}, unexpected={unexpected[:3]}"
        )

    seed_scores = []
    for seed in seeds:
        for family in families:
            for optimizer in OPTIMIZERS:
                task_scores = [
                    indexed[(seed, family, optimizer, task)]["ndcg_at_10"] for task in tasks
                ]
                seed_scores.append(
                    {
                        "seed": seed,
                        "model_family": family,
                        "optimizer": optimizer,
                        "tasks": len(tasks),
                        "mean_ndcg_at_10": statistics.mean(task_scores),
                        "median_ndcg_at_10": statistics.median(task_scores),
                    }
                )

    contrast_rows = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for family in families:
        for treatment, baseline in CONTRASTS:
            for seed in seeds:
                for task in tasks:
                    treatment_score = indexed[(seed, family, treatment, task)]["ndcg_at_10"]
                    baseline_score = indexed[(seed, family, baseline, task)]["ndcg_at_10"]
                    record = {
                        "model_family": family,
                        "treatment": treatment,
                        "baseline": baseline,
                        "seed": seed,
                        "task": task,
                        "treatment_ndcg_at_10": treatment_score,
                        "baseline_ndcg_at_10": baseline_score,
                        "delta_ndcg_at_10": treatment_score - baseline_score,
                    }
                    contrast_rows.append(record)
                    grouped[(family, treatment, baseline)].append(record)

    summaries = []
    for (family, treatment, baseline), values in sorted(grouped.items()):
        by_seed_task = {
            (int(row["seed"]), str(row["task"])): float(row["delta_ndcg_at_10"]) for row in values
        }
        effect_matrix = np.asarray(
            [[by_seed_task[(seed, task)] for task in tasks] for seed in seeds],
            dtype=np.float64,
        )
        seed_means = effect_matrix.mean(axis=1)
        task_means = effect_matrix.mean(axis=0)
        interval = hierarchical_seed_task_bootstrap(
            effect_matrix,
            samples=bootstrap_samples,
            seed=_stable_seed(bootstrap_seed, family, treatment, baseline),
            familywise_contrasts=familywise_contrasts,
        )
        summaries.append(
            {
                "model_family": family,
                "treatment": treatment,
                "baseline": baseline,
                "seeds": len(seeds),
                "tasks": len(tasks),
                "mean_delta_ndcg_at_10": float(effect_matrix.mean()),
                "median_seed_task_delta": float(np.median(effect_matrix)),
                "seed_mean_standard_deviation": float(np.std(seed_means, ddof=1)),
                "seed_wins": int(np.sum(seed_means > 0)),
                "seed_ties": int(np.sum(seed_means == 0)),
                "seed_losses": int(np.sum(seed_means < 0)),
                "task_wins_after_seed_average": int(np.sum(task_means > 0)),
                "task_ties_after_seed_average": int(np.sum(task_means == 0)),
                "task_losses_after_seed_average": int(np.sum(task_means < 0)),
                **interval,
            }
        )
    expected_seed_scores = len(seeds) * len(families) * len(OPTIMIZERS)
    expected_contrasts = len(seeds) * len(families) * len(CONTRASTS) * len(tasks)
    expected_summaries = len(families) * len(CONTRASTS)
    if (
        len(seed_scores) != expected_seed_scores
        or len(contrast_rows) != expected_contrasts
        or len(summaries) != expected_summaries
    ):
        raise AssertionError("Confirmatory summary cardinality invariant failed")
    return seed_scores, contrast_rows, summaries


def build_confirmatory_report(
    protocol_path: str | Path = "configs/confirmatory_protocol.json",
    *,
    experiment_matrix: str | Path = "configs/experiment.yaml",
    validation_spec: str | Path = "configs/validation_probe.json",
    matrix_dir: str | Path | None = None,
    results_root: str | Path = "results/confirmatory-beir",
    output_dir: str | Path = "reports/confirmatory",
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 20260826,
    families: tuple[str, ...] = ALL_FAMILIES,
    scope_amendment: str | Path | None = None,
) -> dict[str, Any]:
    families, scope = resolve_scope(families, scope_amendment)
    resolved_protocol, protocol = load_confirmatory_protocol(protocol_path)
    generated = Path(matrix_dir or protocol["training"]["matrix_output_dir"]).resolve()
    matrix_audit = audit_confirmatory_matrices(
        resolved_protocol,
        experiment_matrix=experiment_matrix,
        validation_spec=validation_spec,
        output_dir=generated,
    )
    evaluation_audit = audit_confirmatory_evaluations(
        resolved_protocol,
        experiment_matrix=experiment_matrix,
        validation_spec=validation_spec,
        matrix_dir=generated,
        results_root=results_root,
        families=families,
        scope_amendment=scope_amendment,
    )
    seeds = [int(seed) for seed in protocol["confirmatory_data"]["seeds"]]
    rows = []
    system_rows = []
    training_audits: dict[str, Any] = {}
    for seed in seeds:
        matrix_path = generated / f"seed{seed}.yaml"
        configs = [config for config in load_matrix(matrix_path) if config.model_family in families]
        dataset = audit_confirmatory_view(resolved_protocol, seed)
        training = audit_derived_training_artifacts(configs, dataset, deep=True)
        if not training.get("complete"):
            raise ValueError(f"Seed {seed}: confirmatory training artifacts failed strict audit")
        training_audits[str(seed)] = {"dataset": dataset, "training": training}
        for row in collect_evaluations(Path(results_root) / f"seed{seed}", configs):
            rows.append({"seed": seed, **row})
        for row in collect_system_metrics(configs):
            system_rows.append({"seed": seed, **row})
    seed_scores, contrasts, summaries = summarize_confirmatory_scores(
        rows,
        seeds,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        families=families,
    )
    output = Path(output_dir).resolve()
    tables = {
        "task_seed_scores": _atomic_csv(output / "task_seed_scores.csv", rows),
        "seed_scores": _atomic_csv(output / "seed_scores.csv", seed_scores),
        "paired_task_seed_contrasts": _atomic_csv(
            output / "paired_task_seed_contrasts.csv", contrasts
        ),
        "paired_summary": _atomic_csv(output / "paired_summary.csv", summaries),
        "system_metrics": _atomic_csv(output / "system_metrics.csv", system_rows),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "families": list(families),
        "scope_amendment": scope,
        "protocol": {"path": str(resolved_protocol), "sha256": _sha256(resolved_protocol)},
        "matrix_audit": matrix_audit,
        "evaluation_audit": evaluation_audit,
        "training_audits": training_audits,
        "coverage": {
            "seeds": len(seeds),
            "runs": len(seed_scores),
            "tasks": len(DECONTAMINATED_TASK_NAMES),
            "evaluation_units": len(rows),
            "paired_contrast_units": len(contrasts),
        },
        "inference": {
            "levels": ["seed", "task"],
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "nominal_interval": "hierarchical seed-by-task bootstrap 95% interval",
            "familywise_method": "bonferroni",
            "familywise_contrasts": FAMILYWISE_CONTRASTS,
            "familywise_interval": (
                "simultaneous familywise 95% interval over all six contrasts frozen before the "
                "post-hoc Dense-only scope amendment"
            ),
            "headline_interval": "familywise_ci_95",
            "query_level_inference": False,
            "note": (
                "Intervals resample seeds and tasks independently. Aggregate MTEB JSON does not "
                "contain per-query rankings, so this report makes no query-level significance claim. "
                "Both nominal and Bonferroni familywise intervals are reported; only the familywise "
                "interval can determine positive, negative, or inconclusive headline language."
            ),
        },
        "outputs": tables,
    }
    _atomic_json(output / "summary_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly audit and summarize the three-seed confirmatory BEIR study"
    )
    parser.add_argument("--protocol", type=Path, default=Path("configs/confirmatory_protocol.json"))
    parser.add_argument("--experiment-matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--validation-spec", type=Path, default=Path("configs/validation_probe.json")
    )
    parser.add_argument("--matrix-dir", type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results/confirmatory-beir"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/confirmatory"))
    parser.add_argument(
        "--families", nargs="+", choices=("dense", "late"), default=["dense", "late"]
    )
    parser.add_argument("--scope-amendment", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260826)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_confirmatory_report(
        args.protocol,
        experiment_matrix=args.experiment_matrix,
        validation_spec=args.validation_spec,
        matrix_dir=args.matrix_dir,
        results_root=args.results_root,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        families=tuple(args.families),
        scope_amendment=args.scope_amendment,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
