"""Descriptive historical-versus-corrected Dense execution sensitivity."""

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

from .config import RunConfig, load_matrix, resolve_matrix_path
from .geometry import _atomic_json, _sha256

SCHEMA_VERSION = 1
OPTIMIZERS = ("adamw", "muon", "normuon")


def _finite(value: Any, *, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value for {context}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value for {context}: {result}")
    return result


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"Refusing to write empty execution-sensitivity table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _validate_matrix(configs: list[RunConfig]) -> None:
    grouped: dict[str, list[RunConfig]] = defaultdict(list)
    for config in configs:
        grouped[config.optimizer.name].append(config)
    if (
        len(configs) != 12
        or set(grouped) != set(OPTIMIZERS)
        or any(len(grouped[name]) != 4 for name in OPTIMIZERS)
        or any(config.model_family != "dense" for config in configs)
        or any(config.dense_can_flatten_inputs for config in configs)
        or any(len(config.checkpoint_fractions) != 5 for config in configs)
    ):
        raise ValueError("Execution sensitivity requires the frozen 12-run corrected Dense matrix")


def _index_historical(
    rows: list[dict[str, Any]], configs: list[RunConfig]
) -> dict[tuple[str, float, int], dict[str, Any]]:
    expected = {
        (config.optimizer.name, config.optimizer.lr, stage)
        for config in configs
        for stage in range(1, 6)
    }
    indexed = {}
    for row in rows:
        optimizer = str(row.get("optimizer"))
        learning_rate = _finite(row.get("learning_rate"), context="historical learning rate")
        stage = int(row.get("stage", -1))
        key = (optimizer, learning_rate, stage)
        score = _finite(row.get("mean_ndcg_at_10"), context=f"historical {key} score")
        fraction = _finite(row.get("fraction"), context=f"historical {key} fraction")
        if (
            key in indexed
            or key not in expected
            or row.get("model_family") != "dense"
            or int(row.get("tasks_completed", -1)) != 14
            or not 0 <= score <= 1
            or not math.isclose(fraction, stage / 5, abs_tol=1e-12)
        ):
            raise ValueError(f"Invalid historical checkpoint row: {key}")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("Historical Dense coverage is not four rates by five stages per optimizer")
    return indexed


def _index_corrected(
    rows: list[dict[str, Any]], configs: list[RunConfig]
) -> dict[tuple[str, float, int], dict[str, Any]]:
    config_by_id = {config.run_id: config for config in configs}
    expected = {
        (config.optimizer.name, config.optimizer.lr, stage)
        for config in configs
        for stage in range(1, 6)
    }
    indexed = {}
    for row in rows:
        run_id = str(row.get("run_id"))
        config = config_by_id.get(run_id)
        optimizer = str(row.get("optimizer"))
        learning_rate = _finite(row.get("learning_rate"), context="corrected learning rate")
        stage = int(row.get("stage", -1))
        key = (optimizer, learning_rate, stage)
        score = _finite(row.get("mean_ndcg_at_10"), context=f"corrected {key} score")
        fraction = _finite(row.get("progress_fraction"), context=f"corrected {key} fraction")
        if (
            config is None
            or config.optimizer.name != optimizer
            or config.optimizer.lr != learning_rate
            or key in indexed
            or key not in expected
            or int(row.get("tasks", -1)) != 14
            or not 0 <= score <= 1
            or not math.isclose(fraction, stage / 5, abs_tol=1e-12)
        ):
            raise ValueError(f"Invalid corrected checkpoint row: {key}")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("Corrected Dense coverage is not four rates by five stages per optimizer")
    return indexed


def _descending_average_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda optimizer: (-values[optimizer], optimizer))
    ranks = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average = (start + end - 1) / 2 + 1
        for optimizer in ordered[start:end]:
            ranks[optimizer] = average
        start = end
    return ranks


def _direction(value: float) -> str:
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def assemble_sensitivity_tables(
    historical_rows: list[dict[str, Any]],
    corrected_rows: list[dict[str, Any]],
    configs: list[RunConfig],
) -> dict[str, list[dict[str, Any]]]:
    """Match normalized stages and fixed rates without pooling the two executions."""

    _validate_matrix(configs)
    historical = _index_historical(historical_rows, configs)
    corrected = _index_corrected(corrected_rows, configs)
    dose_indices = {}
    for optimizer in OPTIMIZERS:
        members = sorted(
            (config for config in configs if config.optimizer.name == optimizer),
            key=lambda config: config.optimizer.lr,
        )
        for dose_index, config in enumerate(members, start=1):
            dose_indices[(optimizer, config.optimizer.lr)] = dose_index

    matched = []
    for config in configs:
        optimizer = config.optimizer.name
        learning_rate = config.optimizer.lr
        for stage in range(1, 6):
            key = (optimizer, learning_rate, stage)
            historical_score = _finite(
                historical[key]["mean_ndcg_at_10"], context=f"historical {key} score"
            )
            corrected_score = _finite(
                corrected[key]["mean_ndcg_at_10"], context=f"corrected {key} score"
            )
            matched.append(
                {
                    "optimizer": optimizer,
                    "learning_rate": learning_rate,
                    "dose_index": dose_indices[(optimizer, learning_rate)],
                    "stage": stage,
                    "progress_fraction": stage / 5,
                    "historical_run_id": historical[key]["run_id"],
                    "corrected_run_id": corrected[key]["run_id"],
                    "historical_packed_training_mean_ndcg_at_10": historical_score,
                    "corrected_padded_training_mean_ndcg_at_10": corrected_score,
                    "corrected_minus_historical_ndcg_at_10": (corrected_score - historical_score),
                }
            )

    optimizer_stage = []
    stage_rankings = []
    optimizer_stage_index = {}
    for stage in range(1, 6):
        historical_means = {}
        corrected_means = {}
        for optimizer in OPTIMIZERS:
            members = [
                row for row in matched if row["stage"] == stage and row["optimizer"] == optimizer
            ]
            if len(members) != 4:
                raise ValueError(f"Expected four matched rates for {optimizer} stage {stage}")
            historical_means[optimizer] = statistics.fmean(
                float(row["historical_packed_training_mean_ndcg_at_10"]) for row in members
            )
            corrected_means[optimizer] = statistics.fmean(
                float(row["corrected_padded_training_mean_ndcg_at_10"]) for row in members
            )
        historical_ranks = _descending_average_ranks(historical_means)
        corrected_ranks = _descending_average_ranks(corrected_means)
        historical_order = ">".join(
            sorted(OPTIMIZERS, key=lambda name: (historical_ranks[name], name))
        )
        corrected_order = ">".join(
            sorted(OPTIMIZERS, key=lambda name: (corrected_ranks[name], name))
        )
        stage_rankings.append(
            {
                "stage": stage,
                "progress_fraction": stage / 5,
                "historical_optimizer_order": historical_order,
                "corrected_optimizer_order": corrected_order,
                "ranking_changed": historical_ranks != corrected_ranks,
            }
        )
        for optimizer in OPTIMIZERS:
            row = {
                "optimizer": optimizer,
                "stage": stage,
                "progress_fraction": stage / 5,
                "learning_rates": 4,
                "historical_packed_training_mean_across_rates": historical_means[optimizer],
                "corrected_padded_training_mean_across_rates": corrected_means[optimizer],
                "corrected_minus_historical_ndcg_at_10": (
                    corrected_means[optimizer] - historical_means[optimizer]
                ),
                "historical_rank": historical_ranks[optimizer],
                "corrected_rank": corrected_ranks[optimizer],
                "rank_improvement": historical_ranks[optimizer] - corrected_ranks[optimizer],
            }
            optimizer_stage.append(row)
            optimizer_stage_index[(optimizer, stage)] = row

    contrasts = []
    for stage in range(1, 6):
        historical_adamw = float(
            optimizer_stage_index[("adamw", stage)]["historical_packed_training_mean_across_rates"]
        )
        corrected_adamw = float(
            optimizer_stage_index[("adamw", stage)]["corrected_padded_training_mean_across_rates"]
        )
        for optimizer in ("muon", "normuon"):
            historical_delta = (
                float(
                    optimizer_stage_index[(optimizer, stage)][
                        "historical_packed_training_mean_across_rates"
                    ]
                )
                - historical_adamw
            )
            corrected_delta = (
                float(
                    optimizer_stage_index[(optimizer, stage)][
                        "corrected_padded_training_mean_across_rates"
                    ]
                )
                - corrected_adamw
            )
            contrasts.append(
                {
                    "optimizer": optimizer,
                    "baseline": "adamw",
                    "stage": stage,
                    "progress_fraction": stage / 5,
                    "historical_optimizer_minus_adamw": historical_delta,
                    "corrected_optimizer_minus_adamw": corrected_delta,
                    "corrected_minus_historical_contrast_shift": (
                        corrected_delta - historical_delta
                    ),
                    "historical_direction": _direction(historical_delta),
                    "corrected_direction": _direction(corrected_delta),
                    "direction_changed": _direction(historical_delta)
                    != _direction(corrected_delta),
                }
            )
    output = {
        "matched_run_stage_sensitivity": matched,
        "optimizer_stage_sensitivity": optimizer_stage,
        "optimizer_minus_adamw_sensitivity": contrasts,
        "stage_optimizer_rankings": stage_rankings,
    }
    expected = {
        "matched_run_stage_sensitivity": 60,
        "optimizer_stage_sensitivity": 15,
        "optimizer_minus_adamw_sensitivity": 10,
        "stage_optimizer_rankings": 5,
    }
    if {name: len(rows) for name, rows in output.items()} != expected:
        raise ValueError("Execution-sensitivity output coverage is incomplete")
    return output


def _load_protocol(path: Path, repository: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "corrected_execution_sensitivity_implementation_lock":
        raise ValueError(f"Unexpected execution-sensitivity protocol status: {path}")
    for group in ("parent_bindings", "source_bindings", "historical_bindings"):
        for identity in payload.get(group, {}).values():
            source = repository / identity["path"]
            if (
                not source.is_file()
                or _sha256(source) != identity["sha256"]
                or ("bytes" in identity and source.stat().st_size != int(identity["bytes"]))
            ):
                raise ValueError(f"Execution-sensitivity {group} mismatch: {source}")
    return payload


def _load_historical(
    coverage_path: Path, repository: Path
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    coverage_path = coverage_path.resolve()
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if not all(
        coverage.get(key)
        for key in ("complete", "contract_complete", "training_complete", "evaluation_complete")
    ):
        raise ValueError(f"Historical Dense coverage is incomplete: {coverage_path}")
    identity = coverage.get("outputs", {}).get("checkpoint_summary", {})
    table_path = repository / identity.get("path", "")
    if (
        int(identity.get("rows", -1)) != 60
        or not table_path.is_file()
        or table_path.stat().st_size != int(identity.get("bytes", -1))
        or _sha256(table_path) != identity.get("sha256")
    ):
        raise ValueError("Historical checkpoint-summary provenance mismatch")
    with table_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 60:
        raise ValueError(f"Expected 60 historical checkpoint rows, found {len(rows)}")
    return rows, {
        "coverage_path": str(coverage_path),
        "coverage_bytes": coverage_path.stat().st_size,
        "coverage_sha256": _sha256(coverage_path),
        "table_path": str(table_path),
        "table_bytes": table_path.stat().st_size,
        "table_sha256": _sha256(table_path),
        "rows": len(rows),
    }


def _load_corrected(
    outcomes_dir: Path, expected_protocol_sha: str
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    outcomes_dir = outcomes_dir.resolve()
    manifest_path = outcomes_dir / "summary_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest.get("outputs", {}).get("run_stage_scores", {})
    table_path = outcomes_dir / "run_stage_scores.csv"
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("coverage", {}).get("task_units") != 840
        or manifest.get("protocol", {}).get("sha256") != expected_protocol_sha
        or int(identity.get("rows", -1)) != 60
        or not table_path.is_file()
        or table_path.stat().st_size != int(identity.get("bytes", -1))
        or _sha256(table_path) != identity.get("sha256")
    ):
        raise ValueError("Corrected run-stage summary provenance mismatch")
    with table_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 60:
        raise ValueError(f"Expected 60 corrected checkpoint rows, found {len(rows)}")
    return rows, {
        "manifest_path": str(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "manifest_sha256": _sha256(manifest_path),
        "table_path": str(table_path),
        "table_bytes": table_path.stat().st_size,
        "table_sha256": _sha256(table_path),
        "rows": len(rows),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path, repository)
    matrix_path = resolve_matrix_path(args.matrix).resolve()
    if _sha256(matrix_path) != protocol["parent_bindings"]["matrix"]["sha256"]:
        raise ValueError("Execution-sensitivity matrix differs from protocol")
    configs = load_matrix(matrix_path)
    _validate_matrix(configs)
    expected_historical_coverage = (
        repository / protocol["historical_bindings"]["coverage"]["path"]
    ).resolve()
    if args.historical_coverage.resolve() != expected_historical_coverage:
        raise ValueError("Historical coverage path differs from the source-bound protocol")
    historical_rows, historical_source = _load_historical(args.historical_coverage, repository)
    corrected_rows, corrected_source = _load_corrected(
        args.outcomes_dir,
        protocol["parent_bindings"]["outcome_protocol"]["sha256"],
    )
    tables = assemble_sensitivity_tables(historical_rows, corrected_rows, configs)
    output_dir = args.output_dir.resolve()
    outputs = {
        f"{name}.csv": _atomic_csv(output_dir / f"{name}.csv", rows)
        for name, rows in tables.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": "historical_packed_training_vs_corrected_padded_training_dense",
        "coverage": {
            "historical_run_stage_rows": len(historical_rows),
            "corrected_run_stage_rows": len(corrected_rows),
            "matched_rows": len(tables["matched_run_stage_sensitivity"]),
            "normalized_stages": 5,
            "optimizers": 3,
            "rates_per_optimizer": 4,
        },
        "estimand": (
            "Descriptive change in optimizer rankings and optimizer-minus-AdamW retrieval "
            "contrasts between the historical packed-training execution and corrected "
            "independently padded training execution at matched rates and normalized stages."
        ),
        "no_pooling": True,
        "claim_boundary": protocol["claim_boundary"],
        "protocol": {
            "path": str(protocol_path),
            "bytes": protocol_path.stat().st_size,
            "sha256": _sha256(protocol_path),
        },
        "sources": {
            "historical": historical_source,
            "corrected": corrected_source,
        },
        "outputs": outputs,
    }
    _atomic_json(output_dir / "summary_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/dense_no_packing_sensitivity_implementation_protocol.json"),
    )
    parser.add_argument(
        "--matrix", type=Path, default=Path("configs/dense_no_packing_retrain.yaml")
    )
    parser.add_argument(
        "--historical-coverage", type=Path, default=Path("reports/dense-discovery/coverage.json")
    )
    parser.add_argument(
        "--outcomes-dir", type=Path, default=Path("reports/dense-no-packing-outcomes")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/dense-no-packing-sensitivity")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(build_report(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
