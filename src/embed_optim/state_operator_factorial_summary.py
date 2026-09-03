"""Summarize the prospective 2x2 DenseOn weight-state-by-operator factorial."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .aggregate import (
    CHECKPOINT_PATTERN,
    _base_task_name,
    _result_provenance,
    _run_for_result,
)
from .config import RunConfig, load_matrix
from .corrected_beir_evaluation import audit_requested_results
from .decontamination import DECONTAMINATED_BEIR, DECONTAMINATED_TASK_NAMES
from .evaluate_matrix import audit_evaluation_artifacts
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .state_operator_factorial import MATRIX_ROOT, SCIENTIFIC_PROTOCOL, load_factorial_protocol
from .state_operator_factorial_contract import require_factorial_implementation
from .state_operator_factorial_evaluation import _source_manifest, matrix_path
from .state_operator_factorial_probe import _all_finite, _file_identity

FULL_BEIR_ROOT = Path("results/state-operator-factorial/full-beir")
PROBE_ROOT = Path("results/state-operator-factorial/probe")
OUTPUT_ROOT = Path("reports/state-operator-factorial")
ESTIMANDS = ("weight_state_effect", "operator_effect", "state_operator_interaction")


def _canonical_operator(config: RunConfig) -> str:
    return "adamw" if config.optimizer.name == "hybrid_adamw" else config.optimizer.name


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"Cannot write an empty factorial table: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"Factorial table columns differ: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return _file_identity(path)


def _load_all_configs(
    protocol: dict[str, Any], matrix_root: Path
) -> dict[tuple[str, int], tuple[Path, list[RunConfig]]]:
    output = {}
    for state in protocol["factorial_design"]["factors"]["weight_state"]:
        for seed in protocol["branch_data"]["order_seeds"]:
            path = matrix_path(state, int(seed), matrix_root)
            configs = load_matrix(path)
            if len(configs) != 2:
                raise ValueError(f"Factorial summary requires two runs in {path}")
            output[(state, int(seed))] = (path, configs)
    if sum(len(configs) for _, configs in output.values()) != 12:
        raise ValueError("Factorial summary requires exactly 12 runs")
    return output


def _runtime_versions(results_root: Path, expected_sources: dict[str, Any]) -> dict[str, str]:
    path = results_root / "evaluation_runtime.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        versions = payload["versions"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid factorial evaluation runtime: {path}") from error
    if (
        payload.get("schema_version") != 2
        or payload.get("source_files") != expected_sources
        or not isinstance(versions, dict)
        or not versions
        or any(not isinstance(value, str) or not value for value in versions.values())
    ):
        raise ValueError(f"Factorial evaluation runtime identity differs: {path}")
    return versions


def _collect_beir_cell(
    results_root: Path,
    configs: list[RunConfig],
    protocol_path: Path,
    matrix: Path,
    matrix_manifest: Path,
    state: str,
    seed: int,
    expected_sources: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract_path = results_root / "factorial_execution.json"
    completion_path = results_root / "completion.json"
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Missing factorial full-BEIR contract under {results_root}") from error
    checkpoints = [config.output_dir / "checkpoint-391" for config in configs]
    expected_contract = {
        "scientific_protocol_sha256": _sha256(protocol_path),
        "matrix_manifest_sha256": _sha256(matrix_manifest),
        "matrix_sha256": _sha256(matrix),
        "state": state,
        "seed": seed,
    }
    if (
        contract.get("scientific_protocol", {}).get("sha256")
        != expected_contract["scientific_protocol_sha256"]
        or contract.get("matrix_manifest", {}).get("sha256")
        != expected_contract["matrix_manifest_sha256"]
        or contract.get("matrix", {}).get("sha256") != expected_contract["matrix_sha256"]
        or contract.get("source_state") != state
        or contract.get("order_seed") != seed
        or contract.get("stages") != [5]
        or contract.get("tasks") != list(DECONTAMINATED_TASK_NAMES)
        or contract.get("expected_task_units") != 28
        or contract.get("source_files") != expected_sources
        or completion.get("status") != "complete"
        or completion.get("scientific_protocol_sha256")
        != expected_contract["scientific_protocol_sha256"]
        or completion.get("matrix_sha256") != expected_contract["matrix_sha256"]
        or completion.get("task_units") != 28
    ):
        raise ValueError(f"Factorial full-BEIR contract differs under {results_root}")
    versions = _runtime_versions(results_root, expected_sources)
    coverage = audit_requested_results(results_root, checkpoints)
    if coverage.get("task_units") != 28:
        raise ValueError(f"Factorial full-BEIR coverage differs under {results_root}")

    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for path in results_root.rglob("*Decontaminated.json"):
        config = _run_for_result(path, configs)
        match = CHECKPOINT_PATTERN.search(str(path))
        if config is None or match is None:
            continue
        step = int(match.group(1))
        if step != 391:
            raise ValueError(f"Unexpected non-final factorial BEIR result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        task = _base_task_name(payload["task_name"])
        if task not in DECONTAMINATED_BEIR:
            raise ValueError(f"Unexpected factorial BEIR task: {path}")
        _result_provenance(path, payload, config, step, task, versions)
        score_rows = [item for values in payload["scores"].values() for item in values]
        scores = [float(item["ndcg_at_10"]) for item in score_rows]
        if len(scores) != 1 or not math.isfinite(scores[0]):
            raise ValueError(f"Invalid factorial nDCG@10 result: {path}")
        operator = _canonical_operator(config)
        row = {
            "state": state,
            "operator": operator,
            "seed": seed,
            "run_id": config.run_id,
            "task": task,
            "ndcg_at_10": scores[0],
            "result_path": str(path.resolve()),
        }
        key = (operator, task)
        if key in indexed:
            raise ValueError(f"Duplicate factorial BEIR result: {key}")
        indexed[key] = row
    expected = {
        (operator, task) for operator in ("adamw", "muon") for task in DECONTAMINATED_TASK_NAMES
    }
    if set(indexed) != expected:
        raise ValueError(f"Factorial BEIR result identities differ under {results_root}")
    rows = [indexed[key] for key in sorted(indexed)]
    cache_audit = audit_evaluation_artifacts(results_root, checkpoints, rows)
    return rows, {
        "execution_contract": _file_identity(contract_path),
        "completion": _file_identity(completion_path),
        "evaluation_cache": cache_audit,
    }


def _effect_rows(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (int(row["seed"]), str(row["task"]), str(row["state"]), str(row["operator"])): float(
            row["ndcg_at_10"]
        )
        for row in score_rows
    }
    seeds = sorted({key[0] for key in indexed})
    tasks = sorted({key[1] for key in indexed})
    expected = {
        (seed, task, state, operator)
        for seed in seeds
        for task in tasks
        for state in ("adamw_state", "muon_state")
        for operator in ("adamw", "muon")
    }
    if len(seeds) != 3 or tasks != sorted(DECONTAMINATED_TASK_NAMES) or set(indexed) != expected:
        raise ValueError("Factorial effect table lacks exact 3x14x2x2 coverage")
    rows = []
    for seed in seeds:
        for task in tasks:
            aa = indexed[(seed, task, "adamw_state", "adamw")]
            am = indexed[(seed, task, "adamw_state", "muon")]
            ma = indexed[(seed, task, "muon_state", "adamw")]
            mm = indexed[(seed, task, "muon_state", "muon")]
            values = {
                "weight_state_effect": 0.5 * ((ma - aa) + (mm - am)),
                "operator_effect": 0.5 * ((am - aa) + (mm - ma)),
                "state_operator_interaction": (mm - ma) - (am - aa),
            }
            rows.extend(
                {
                    "estimand": estimand,
                    "seed": seed,
                    "task": task,
                    "contrast_ndcg_at_10": value,
                }
                for estimand, value in values.items()
            )
    if len(rows) != len(ESTIMANDS) * 3 * 14:
        raise AssertionError("Factorial estimand-cell cardinality changed")
    return rows


def two_way_cluster_bootstrap(
    values: np.ndarray,
    *,
    samples: int = 100_000,
    seed: int = 20_260_904,
) -> dict[str, Any]:
    if values.shape != (3, 14) or not np.isfinite(values).all():
        raise ValueError("Two-way factorial bootstrap requires a finite [3, 14] matrix")
    if samples != 100_000 or seed != 20_260_904:
        raise ValueError("Factorial bootstrap settings differ from the prospective lock")
    generator = np.random.default_rng(seed)
    seed_indices = generator.integers(0, values.shape[0], size=(samples, values.shape[0]))
    task_indices = generator.integers(0, values.shape[1], size=(samples, values.shape[1]))
    draws = values[seed_indices[:, :, None], task_indices[:, None, :]].mean(axis=(1, 2))
    lower, upper = np.quantile(draws, [0.025, 0.975], method="linear")
    point = float(values.mean())
    if lower > 0:
        decision = "supported_positive"
    elif upper < 0:
        decision = "supported_negative"
    else:
        decision = "inconclusive"
    return {
        "point_estimate": point,
        "bootstrap_ci_95_lower": float(lower),
        "bootstrap_ci_95_upper": float(upper),
        "decision": decision,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "seed_clusters": values.shape[0],
        "task_clusters": values.shape[1],
    }


def _estimand_summaries(effect_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seeds = sorted({int(row["seed"]) for row in effect_rows})
    tasks = sorted({str(row["task"]) for row in effect_rows})
    indexed = {
        (str(row["estimand"]), int(row["seed"]), str(row["task"])): float(
            row["contrast_ndcg_at_10"]
        )
        for row in effect_rows
    }
    summaries = []
    for estimand in ESTIMANDS:
        values = np.asarray(
            [[indexed[(estimand, seed, task)] for task in tasks] for seed in seeds],
            dtype=np.float64,
        )
        summaries.append({"estimand": estimand, **two_way_cluster_bootstrap(values)})
    return summaries


def _cell_summaries(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in score_rows:
        grouped[(str(row["state"]), str(row["operator"]))].append(float(row["ndcg_at_10"]))
    output = []
    for (state, operator), values in sorted(grouped.items()):
        if len(values) != 42:
            raise ValueError(f"Factorial cell {state}/{operator} has {len(values)}/42 values")
        output.append(
            {
                "state": state,
                "operator": operator,
                "seed_task_cells": len(values),
                "mean_ndcg_at_10": statistics.mean(values),
                "population_std_ndcg_at_10": statistics.pstdev(values),
            }
        )
    if len(output) != 4:
        raise ValueError("Factorial summary requires four cells")
    return output


def _probe_rows(
    root: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
    matrix_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    expected_labels = set()
    for state in protocol["factorial_design"]["factors"]["weight_state"]:
        for seed in protocol["branch_data"]["order_seeds"]:
            for config in load_matrix(matrix_root / f"{state}-seed{seed}.yaml"):
                for stage, step in enumerate(
                    protocol["factorial_design"]["training"]["expected_checkpoint_steps"],
                    start=1,
                ):
                    expected_labels.add(
                        (
                            str(
                                Path("dense")
                                / state
                                / f"seed{seed}"
                                / config.run_id
                                / f"checkpoint-{step}"
                            ),
                            state,
                            _canonical_operator(config),
                            int(seed),
                            stage,
                            int(step),
                        )
                    )
    by_label = {item[0]: item[1:] for item in expected_labels}
    overall_rows = []
    task_rows = []
    sources = []
    for label, (state, operator, seed, stage, step) in sorted(by_label.items()):
        relative = Path(label).relative_to("dense")
        path = root / "metrics/dense" / relative
        metric_path = path.with_name(f"{path.name}.factorial.json")
        try:
            payload = json.loads(metric_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Missing factorial probe metric: {metric_path}") from error
        receipt_path = root / "exports/dense" / relative.with_suffix(".npz.padded-execution.json")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Missing factorial probe receipt: {receipt_path}") from error
        if (
            payload.get("status") != "complete"
            or payload.get("label") != label
            or payload.get("scientific_protocol", {}).get("sha256") != _sha256(protocol_path)
            or set(payload.get("by_task", {})) != set(DECONTAMINATED_TASK_NAMES)
            or not _all_finite(payload)
            or receipt.get("status") != "complete"
            or receipt.get("observed_input_execution")
            != {
                "mode": "independently_padded",
                "sentence_transformers_can_flatten_inputs": False,
            }
            or receipt.get("factorial_metrics") != _file_identity(metric_path)
        ):
            raise ValueError(f"Factorial probe identity differs: {label}")
        identity = {
            "state": state,
            "operator": operator,
            "seed": seed,
            "stage": stage,
            "fraction": stage / 5,
            "step": step,
            "label": label,
        }
        overall_rows.append({**identity, **payload["overall"]})
        for task, values in sorted(payload["by_task"].items()):
            task_rows.append({**identity, "task": task, **values})
        sources.append(
            {
                "label": label,
                "metric": _file_identity(metric_path),
                "padded_execution_receipt": _file_identity(receipt_path),
            }
        )
    if len(overall_rows) != 60 or len(task_rows) != 840 or len(sources) != 60:
        raise ValueError("Factorial probe summary requires 60 checkpoints and 840 task rows")
    return overall_rows, task_rows, sources


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    implementation_path, implementation = require_factorial_implementation()
    repository = Path(__file__).resolve().parents[2]
    protocol_path, protocol = load_factorial_protocol(args.protocol)
    matrix_root = args.matrix_root.resolve()
    matrix_manifest = matrix_root / "manifest.json"
    if not matrix_manifest.is_file():
        raise FileNotFoundError(matrix_manifest)
    cells = _load_all_configs(protocol, matrix_root)
    expected_sources = _source_manifest(repository)
    score_rows = []
    beir_sources = []
    for (state, seed), (matrix, configs) in sorted(cells.items()):
        rows, source = _collect_beir_cell(
            args.full_beir_root.resolve() / state / f"seed{seed}",
            configs,
            protocol_path,
            matrix,
            matrix_manifest,
            state,
            seed,
            expected_sources,
        )
        score_rows.extend(rows)
        beir_sources.append({"state": state, "seed": seed, **source})
    if len(score_rows) != 168:
        raise ValueError(f"Factorial BEIR summary requires 168 scores, found {len(score_rows)}")
    effect_rows = _effect_rows(score_rows)
    estimand_summaries = _estimand_summaries(effect_rows)
    cell_summaries = _cell_summaries(score_rows)
    probe_rows, probe_task_rows, probe_sources = _probe_rows(
        args.probe_root.resolve(), protocol_path, protocol, matrix_root
    )
    output = args.output_root.resolve()
    outputs = {
        "beir_seed_task_scores": _atomic_csv(output / "beir_seed_task_scores.csv", score_rows),
        "factorial_cell_summary": _atomic_csv(
            output / "factorial_cell_summary.csv", cell_summaries
        ),
        "estimand_seed_task_contrasts": _atomic_csv(
            output / "estimand_seed_task_contrasts.csv", effect_rows
        ),
        "estimand_summary": _atomic_csv(output / "estimand_summary.csv", estimand_summaries),
        "probe_checkpoint_metrics": _atomic_csv(
            output / "probe_checkpoint_metrics.csv", probe_rows
        ),
        "probe_task_metrics": _atomic_csv(output / "probe_task_metrics.csv", probe_task_rows),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scientific_protocol": _file_identity(protocol_path),
        "implementation_protocol": _file_identity(implementation_path),
        "matrix_manifest": _file_identity(matrix_manifest),
        "coverage": {
            "training_runs": 12,
            "beir_seed_task_scores": len(score_rows),
            "estimand_seed_task_contrasts": len(effect_rows),
            "estimands": len(estimand_summaries),
            "probe_checkpoints": len(probe_rows),
            "probe_task_rows": len(probe_task_rows),
        },
        "inference": {
            "method": "fixed-seed two-way seed/task cluster percentile bootstrap",
            "samples": 100_000,
            "seed": 20_260_904,
            "decision_rule": protocol["estimands"]["decision_rule"],
        },
        "implementation_commit": implementation["implementation_commit"],
        "outputs": outputs,
        "beir_sources": beir_sources,
        "probe_sources": probe_sources,
    }
    _atomic_json(output / "summary_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=SCIENTIFIC_PROTOCOL)
    parser.add_argument("--matrix-root", type=Path, default=MATRIX_ROOT)
    parser.add_argument("--full-beir-root", type=Path, default=FULL_BEIR_ROOT)
    parser.add_argument("--probe-root", type=Path, default=PROBE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(summarize(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
