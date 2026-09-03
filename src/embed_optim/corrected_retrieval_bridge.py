"""Source-bound geometry-to-retrieval bridge for corrected Dense runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import RunConfig, load_matrix, resolve_matrix_path
from .geometry import _atomic_json, _sha256

SCHEMA_VERSION = 1
OPTIMIZERS = ("adamw", "muon", "normuon")
DISPLACEMENT_KINDS = ("saved_segment", "cumulative")
FEATURES = (
    "log_saved_segment_to_weight_ratio",
    "saved_segment_stable_rank_fraction",
    "saved_segment_sketch_effective_rank_fraction",
    "saved_segment_row_norm_cv",
    "saved_segment_top_1pct_row_energy",
    "cumulative_displacement_to_weight_ratio",
    "cumulative_stable_rank_fraction",
    "mean_saved_segment_subspace_overlap_to_adamw",
    "mean_cumulative_subspace_overlap_to_adamw",
)


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
        raise ValueError(f"Refusing to write empty corrected bridge table: {path}")
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
        raise ValueError("Corrected bridge requires the frozen 12-run padded Dense matrix")
    for optimizer in OPTIMIZERS:
        rates = [config.optimizer.lr for config in grouped[optimizer]]
        if len(set(rates)) != 4 or any(rate <= 0 for rate in rates):
            raise ValueError(f"Corrected bridge requires four positive rates for {optimizer}")


def _read_manifest_table(
    directory: Path,
    filename: str,
    *,
    expected_rows: int,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    directory = directory.resolve()
    manifest_path = directory / "summary_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source summary manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise ValueError(f"Incomplete source summary manifest: {manifest_path}")
    identity = manifest.get("outputs", {}).get(filename)
    if not isinstance(identity, dict):
        raise ValueError(f"Source manifest does not bind {filename}: {manifest_path}")
    table_path = directory / filename
    if (
        not table_path.is_file()
        or table_path.stat().st_size != int(identity.get("bytes", -1))
        or _sha256(table_path) != identity.get("sha256")
        or int(identity.get("rows", -1)) != expected_rows
    ):
        raise ValueError(f"Source table provenance mismatch: {table_path}")
    with table_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in {table_path}, found {len(rows)}")
    source = {
        "manifest_path": str(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "manifest_sha256": _sha256(manifest_path),
        "table_path": str(table_path),
        "table_bytes": table_path.stat().st_size,
        "table_sha256": _sha256(table_path),
        "rows": len(rows),
    }
    return rows, manifest, source


def _index_checkpoint_geometry(
    rows: list[dict[str, Any]], configs: list[RunConfig]
) -> dict[tuple[str, int], dict[str, Any]]:
    config_by_id = {config.run_id: config for config in configs}
    expected = {(config.run_id, stage) for config in configs for stage in range(1, 6)}
    indexed = {}
    for row in rows:
        key = (str(row.get("run_id")), int(row.get("stage", -1)))
        config = config_by_id.get(key[0])
        if (
            key in indexed
            or key not in expected
            or config is None
            or row.get("optimizer") != config.optimizer.name
            or _finite(row.get("learning_rate"), context=f"{key} learning rate")
            != config.optimizer.lr
        ):
            raise ValueError(f"Invalid corrected checkpoint-geometry row: {key}")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("Corrected checkpoint-geometry coverage is not 12 runs by five stages")
    return indexed


def _index_scores(
    rows: list[dict[str, Any]], configs: list[RunConfig]
) -> dict[tuple[str, int], dict[str, Any]]:
    config_by_id = {config.run_id: config for config in configs}
    expected = {(config.run_id, stage) for config in configs for stage in range(1, 6)}
    indexed = {}
    for row in rows:
        key = (str(row.get("run_id")), int(row.get("stage", -1)))
        config = config_by_id.get(key[0])
        score = _finite(row.get("mean_ndcg_at_10"), context=f"{key} nDCG@10")
        if (
            key in indexed
            or key not in expected
            or config is None
            or row.get("optimizer") != config.optimizer.name
            or _finite(row.get("learning_rate"), context=f"{key} learning rate")
            != config.optimizer.lr
            or not 0 <= score <= 1
        ):
            raise ValueError(f"Invalid corrected run-stage score row: {key}")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("Corrected run-stage outcome coverage is not 12 runs by five stages")
    return indexed


def _adamw_overlap_index(
    pair_rows: list[dict[str, Any]], configs: list[RunConfig]
) -> dict[tuple[str, int, str], float]:
    config_by_id = {config.run_id: config for config in configs}
    expected_pairs = math.comb(len(configs), 2) * 5 * len(DISPLACEMENT_KINDS)
    if len(pair_rows) != expected_pairs:
        raise ValueError(
            f"Expected {expected_pairs} corrected subspace rows, found {len(pair_rows)}"
        )
    pair_keys = set()
    candidates: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in pair_rows:
        first = str(row.get("first_run_id"))
        second = str(row.get("second_run_id"))
        stage = int(row.get("stage", -1))
        kind = str(row.get("displacement_kind"))
        first_config = config_by_id.get(first)
        second_config = config_by_id.get(second)
        unordered = tuple(sorted((first, second)))
        pair_key = (unordered, stage, kind)
        if (
            first == second
            or first_config is None
            or second_config is None
            or stage not in range(1, 6)
            or kind not in DISPLACEMENT_KINDS
            or pair_key in pair_keys
            or row.get("first_optimizer") != first_config.optimizer.name
            or row.get("second_optimizer") != second_config.optimizer.name
        ):
            raise ValueError(f"Invalid corrected run-pair subspace row: {pair_key}")
        pair_keys.add(pair_key)
        overlap = _finite(row.get("mean_subspace_overlap"), context=f"{pair_key} overlap")
        if not 0 <= overlap <= 1 + 1e-5:
            raise ValueError(f"Subspace overlap outside tolerance for {pair_key}: {overlap}")
        for focal, other in ((first_config, second_config), (second_config, first_config)):
            if other.optimizer.name != "adamw" or (
                focal.optimizer.name == "adamw" and focal.run_id == other.run_id
            ):
                continue
            candidates[(focal.run_id, stage, kind)].append(overlap)
    expected_keys = {
        (config.run_id, stage, kind)
        for config in configs
        for stage in range(1, 6)
        for kind in DISPLACEMENT_KINDS
    }
    if set(candidates) != expected_keys:
        raise ValueError("AdamW overlap coverage is incomplete")
    result = {}
    for key, values in candidates.items():
        expected_count = 3 if config_by_id[key[0]].optimizer.name == "adamw" else 4
        if len(values) != expected_count:
            raise ValueError(
                f"Expected {expected_count} AdamW overlaps for {key}, found {len(values)}"
            )
        result[key] = float(np.mean(values))
    return result


def assemble_bridge_rows(
    checkpoint_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    configs: list[RunConfig],
) -> list[dict[str, Any]]:
    """Join the locked 60-row geometry and retrieval panels without selection."""

    _validate_matrix(configs)
    geometry = _index_checkpoint_geometry(checkpoint_rows, configs)
    scores = _index_scores(score_rows, configs)
    overlaps = _adamw_overlap_index(pair_rows, configs)
    dose_indices = {}
    centered_log_rates = {}
    for optimizer in OPTIMIZERS:
        members = sorted(
            (config for config in configs if config.optimizer.name == optimizer),
            key=lambda config: config.optimizer.lr,
        )
        mean_log_rate = float(np.mean([math.log10(config.optimizer.lr) for config in members]))
        for dose_index, config in enumerate(members, start=1):
            dose_indices[config.run_id] = dose_index
            centered_log_rates[config.run_id] = math.log10(config.optimizer.lr) - mean_log_rate
    output = []
    for config in configs:
        for stage in range(1, 6):
            key = (config.run_id, stage)
            geometry_row = geometry[key]
            segment_ratio = _finite(
                geometry_row.get("saved_segment_to_weight_ratio"),
                context=f"{key} saved segment ratio",
            )
            if segment_ratio <= 0:
                raise ValueError(f"Saved-segment ratio must be positive for log feature: {key}")
            output.append(
                {
                    "run_id": config.run_id,
                    "optimizer": config.optimizer.name,
                    "learning_rate": config.optimizer.lr,
                    "dose_index": dose_indices[config.run_id],
                    "centered_log10_learning_rate": centered_log_rates[config.run_id],
                    "stage": stage,
                    "progress_fraction": stage / 5,
                    "mean_ndcg_at_10": _finite(
                        scores[key].get("mean_ndcg_at_10"), context=f"{key} nDCG@10"
                    ),
                    "log_saved_segment_to_weight_ratio": math.log(segment_ratio),
                    "saved_segment_stable_rank_fraction": _finite(
                        geometry_row.get("saved_segment_stable_rank_fraction_parameter_weighted"),
                        context=f"{key} saved stable-rank fraction",
                    ),
                    "saved_segment_sketch_effective_rank_fraction": _finite(
                        geometry_row.get(
                            "saved_segment_sketch_effective_rank_fraction_parameter_weighted"
                        ),
                        context=f"{key} saved effective-rank fraction",
                    ),
                    "saved_segment_row_norm_cv": _finite(
                        geometry_row.get("saved_segment_row_cv_parameter_weighted"),
                        context=f"{key} saved row CV",
                    ),
                    "saved_segment_top_1pct_row_energy": _finite(
                        geometry_row.get("saved_segment_top_1pct_row_energy_parameter_weighted"),
                        context=f"{key} saved top-row energy",
                    ),
                    "cumulative_displacement_to_weight_ratio": _finite(
                        geometry_row.get("cumulative_displacement_to_weight_ratio"),
                        context=f"{key} cumulative displacement ratio",
                    ),
                    "cumulative_stable_rank_fraction": _finite(
                        geometry_row.get("cumulative_stable_rank_fraction_parameter_weighted"),
                        context=f"{key} cumulative stable-rank fraction",
                    ),
                    "mean_saved_segment_subspace_overlap_to_adamw": overlaps[
                        (config.run_id, stage, "saved_segment")
                    ],
                    "mean_cumulative_subspace_overlap_to_adamw": overlaps[
                        (config.run_id, stage, "cumulative")
                    ],
                }
            )
    if len(output) != 60:
        raise ValueError(f"Expected 60 corrected bridge rows, found {len(output)}")
    return output


def _baseline_design(rows: list[dict[str, Any]]) -> np.ndarray:
    matrix = []
    for row in rows:
        optimizer = str(row["optimizer"])
        stage = int(row["stage"])
        matrix.append(
            [
                1.0,
                float(optimizer == "muon"),
                float(optimizer == "normuon"),
                *(float(stage == candidate) for candidate in range(2, 6)),
                _finite(
                    row["centered_log10_learning_rate"],
                    context="centered log10 learning rate",
                ),
            ]
        )
    result = np.asarray(matrix, dtype=np.float64)
    if result.shape != (60, 8) or np.linalg.matrix_rank(result) != result.shape[1]:
        raise ValueError(f"Corrected bridge baseline design is invalid: {result.shape}")
    return result


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(observed - predicted))))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Correlation vectors must have equal non-trivial length")
    if float(np.std(left)) == 0 or float(np.std(right)) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def evaluate_bridge_features(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the locked four-fold prediction and residual-association tests."""

    if len(rows) != 60 or {int(row["dose_index"]) for row in rows} != {1, 2, 3, 4}:
        raise ValueError("Corrected bridge inference requires the complete 60-row dose panel")
    baseline = _baseline_design(rows)
    outcome = np.asarray(
        [_finite(row["mean_ndcg_at_10"], context="bridge outcome") for row in rows],
        dtype=np.float64,
    )
    dose_indices = np.asarray([int(row["dose_index"]) for row in rows], dtype=np.int64)
    baseline_residual = outcome - baseline @ np.linalg.lstsq(baseline, outcome, rcond=None)[0]
    fold_rows = []
    summary_rows = []
    association_rows = []
    for feature in FEATURES:
        values = np.asarray(
            [_finite(row[feature], context=feature) for row in rows], dtype=np.float64
        )
        pooled_observed = []
        pooled_baseline = []
        pooled_feature = []
        folds_improved = 0
        for fold in range(1, 5):
            test = dose_indices == fold
            train = ~test
            if int(train.sum()) != 45 or int(test.sum()) != 15:
                raise ValueError(f"Dose fold {fold} is not 45 train / 15 test")
            baseline_beta = np.linalg.lstsq(baseline[train], outcome[train], rcond=None)[0]
            mean = float(np.mean(values[train]))
            scale = float(np.std(values[train]))
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"Feature {feature} is constant in dose fold {fold}")
            standardized = (values - mean) / scale
            augmented = np.column_stack((baseline, standardized))
            feature_beta = np.linalg.lstsq(augmented[train], outcome[train], rcond=None)[0]
            baseline_prediction = baseline[test] @ baseline_beta
            feature_prediction = augmented[test] @ feature_beta
            baseline_rmse = _rmse(outcome[test], baseline_prediction)
            feature_rmse = _rmse(outcome[test], feature_prediction)
            improvement = baseline_rmse - feature_rmse
            folds_improved += int(improvement > 0)
            fold_rows.append(
                {
                    "feature": feature,
                    "held_out_dose_index": fold,
                    "train_rows": int(train.sum()),
                    "test_rows": int(test.sum()),
                    "baseline_rmse": baseline_rmse,
                    "feature_rmse": feature_rmse,
                    "rmse_reduction": improvement,
                    "feature_improves": improvement > 0,
                    "training_feature_mean": mean,
                    "training_feature_scale": scale,
                }
            )
            pooled_observed.extend(outcome[test])
            pooled_baseline.extend(baseline_prediction)
            pooled_feature.extend(feature_prediction)
        pooled_observed_array = np.asarray(pooled_observed)
        pooled_baseline_rmse = _rmse(
            pooled_observed_array, np.asarray(pooled_baseline, dtype=np.float64)
        )
        pooled_feature_rmse = _rmse(
            pooled_observed_array, np.asarray(pooled_feature, dtype=np.float64)
        )
        pooled_reduction = pooled_baseline_rmse - pooled_feature_rmse
        supported = pooled_reduction > 0 and folds_improved >= 3
        summary_rows.append(
            {
                "feature": feature,
                "pooled_rows": len(pooled_observed),
                "pooled_baseline_rmse": pooled_baseline_rmse,
                "pooled_feature_rmse": pooled_feature_rmse,
                "pooled_rmse_reduction": pooled_reduction,
                "folds_improved": folds_improved,
                "folds_total": 4,
                "predictively_useful": supported,
            }
        )
        feature_residual = values - baseline @ np.linalg.lstsq(baseline, values, rcond=None)[0]
        association_rows.append(
            {
                "feature": feature,
                "rows": len(rows),
                "pearson_residual_association": _correlation(feature_residual, baseline_residual),
                "spearman_residual_association": _correlation(
                    _average_ranks(feature_residual), _average_ranks(baseline_residual)
                ),
            }
        )
    return fold_rows, summary_rows, association_rows


def _load_implementation_protocol(path: Path, repository: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "corrected_retrieval_bridge_implementation_lock":
        raise ValueError(f"Unexpected corrected bridge protocol status: {path}")
    for group in ("parent_bindings", "source_bindings"):
        for identity in payload.get(group, {}).values():
            source = repository / identity["path"]
            if (
                not source.is_file()
                or _sha256(source) != identity["sha256"]
                or ("bytes" in identity and source.stat().st_size != int(identity["bytes"]))
            ):
                raise ValueError(f"Corrected bridge {group} mismatch: {source}")
    return payload


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    implementation_protocol = _load_implementation_protocol(args.protocol.resolve(), repository)
    matrix_path = resolve_matrix_path(args.matrix).resolve()
    configs = load_matrix(matrix_path)
    _validate_matrix(configs)
    if _sha256(matrix_path) != implementation_protocol["parent_bindings"]["matrix"]["sha256"]:
        raise ValueError("Corrected bridge matrix differs from implementation protocol")
    checkpoint_rows, geometry_manifest, checkpoint_source = _read_manifest_table(
        args.geometry_dir,
        "checkpoint_geometry.csv",
        expected_rows=60,
    )
    pair_rows, pair_manifest, pair_source = _read_manifest_table(
        args.geometry_dir,
        "run_pair_subspace_overlap.csv",
        expected_rows=660,
    )
    score_rows, outcome_manifest, score_source = _read_manifest_table(
        args.outcomes_dir,
        "run_stage_scores.csv",
        expected_rows=60,
    )
    analysis_sha = implementation_protocol["parent_bindings"]["analysis_protocol"]["sha256"]
    outcome_sha = implementation_protocol["parent_bindings"]["outcome_protocol"]["sha256"]
    if (
        geometry_manifest.get("protocol", {}).get("sha256") != analysis_sha
        or pair_manifest.get("protocol", {}).get("sha256") != analysis_sha
        or outcome_manifest.get("protocol", {}).get("sha256") != outcome_sha
    ):
        raise ValueError("Corrected bridge input summaries use an unexpected frozen protocol")
    bridge_rows = assemble_bridge_rows(checkpoint_rows, pair_rows, score_rows, configs)
    fold_rows, summary_rows, association_rows = evaluate_bridge_features(bridge_rows)
    output_dir = args.output_dir.resolve()
    outputs = {
        "bridge_rows.csv": _atomic_csv(output_dir / "bridge_rows.csv", bridge_rows),
        "leave_dose_fold_metrics.csv": _atomic_csv(
            output_dir / "leave_dose_fold_metrics.csv", fold_rows
        ),
        "feature_prediction_summary.csv": _atomic_csv(
            output_dir / "feature_prediction_summary.csv", summary_rows
        ),
        "residual_associations.csv": _atomic_csv(
            output_dir / "residual_associations.csv", association_rows
        ),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": "corrected_dense_no_packing_retrieval_bridge",
        "coverage": {
            "runs": 12,
            "stages": 5,
            "bridge_rows": len(bridge_rows),
            "features": len(FEATURES),
            "leave_dose_fold_rows": len(fold_rows),
        },
        "analysis": {
            "outcome": "mean decontaminated BEIR nDCG@10 across 14 tasks",
            "baseline": (
                "intercept, Muon/NorMuon indicators, stage-2 through stage-5 indicators, "
                "and within-optimizer centered log10 learning rate"
            ),
            "folds": (
                "four leave-dose-index-out folds; each ordered rate index is held out for "
                "all optimizers and stages"
            ),
            "feature_scaling": (
                "each added feature is z-standardized using only the training partition; "
                "unregularized OLS predictions are invariant to this affine scaling"
            ),
            "support_rule": (
                "pooled held-out RMSE decreases and held-out RMSE decreases in at least "
                "three of four folds"
            ),
            "feature_log": "natural logarithm for saved-segment-to-weight ratio",
            "association": (
                "Pearson and average-rank Spearman correlations after separately "
                "residualizing outcome and feature on the full locked baseline"
            ),
        },
        "claim_boundary": implementation_protocol["claim_boundary"],
        "protocol": {
            "path": str(args.protocol.resolve()),
            "bytes": args.protocol.resolve().stat().st_size,
            "sha256": _sha256(args.protocol.resolve()),
        },
        "sources": {
            "checkpoint_geometry": checkpoint_source,
            "run_pair_subspace_overlap": pair_source,
            "run_stage_scores": score_source,
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
        default=Path("configs/dense_no_packing_bridge_implementation_protocol.json"),
    )
    parser.add_argument(
        "--matrix", type=Path, default=Path("configs/dense_no_packing_retrain.yaml")
    )
    parser.add_argument(
        "--geometry-dir", type=Path, default=Path("reports/dense-no-packing-weight-space")
    )
    parser.add_argument(
        "--outcomes-dir", type=Path, default=Path("reports/dense-no-packing-outcomes")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/dense-no-packing-retrieval-bridge")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(build_report(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
