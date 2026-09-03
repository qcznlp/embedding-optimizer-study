import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import pytest

from embed_optim.config import load_matrix
from embed_optim.corrected_retrieval_bridge import (
    DISPLACEMENT_KINDS,
    FEATURES,
    _load_implementation_protocol,
    assemble_bridge_rows,
    build_report,
    evaluate_bridge_features,
)


def _synthetic_inputs():
    configs = load_matrix("configs/dense_no_packing_retrain.yaml")
    run_index = {config.run_id: index for index, config in enumerate(configs)}
    checkpoint_rows = []
    score_rows = []
    for config in configs:
        index = run_index[config.run_id]
        for stage in range(1, 6):
            base = 1 + index + stage
            checkpoint_rows.append(
                {
                    "run_id": config.run_id,
                    "optimizer": config.optimizer.name,
                    "learning_rate": config.optimizer.lr,
                    "stage": stage,
                    "saved_segment_to_weight_ratio": 0.001 * base,
                    "saved_segment_stable_rank_fraction_parameter_weighted": 0.01 * base,
                    "saved_segment_sketch_effective_rank_fraction_parameter_weighted": (
                        0.008 * base
                    ),
                    "saved_segment_row_cv_parameter_weighted": 0.2 + 0.001 * base,
                    "saved_segment_top_1pct_row_energy_parameter_weighted": (0.1 + 0.001 * base),
                    "cumulative_displacement_to_weight_ratio": 0.002 * base,
                    "cumulative_stable_rank_fraction_parameter_weighted": 0.009 * base,
                }
            )
            score_rows.append(
                {
                    "run_id": config.run_id,
                    "optimizer": config.optimizer.name,
                    "learning_rate": config.optimizer.lr,
                    "stage": stage,
                    "mean_ndcg_at_10": 0.5 + 0.001 * base,
                }
            )
    pair_rows = []
    for first, second in itertools.combinations(configs, 2):
        first_index = run_index[first.run_id]
        second_index = run_index[second.run_id]
        for stage in range(1, 6):
            for kind_index, kind in enumerate(DISPLACEMENT_KINDS):
                pair_rows.append(
                    {
                        "stage": stage,
                        "displacement_kind": kind,
                        "first_run_id": first.run_id,
                        "first_optimizer": first.optimizer.name,
                        "second_run_id": second.run_id,
                        "second_optimizer": second.optimizer.name,
                        "mean_subspace_overlap": (
                            0.1
                            + 0.001 * first_index
                            + 0.002 * second_index
                            + 0.003 * stage
                            + 0.004 * kind_index
                        ),
                    }
                )
    return configs, checkpoint_rows, pair_rows, score_rows


def test_assemble_bridge_rows_uses_all_adamw_comparators() -> None:
    configs, checkpoint_rows, pair_rows, score_rows = _synthetic_inputs()
    rows = assemble_bridge_rows(checkpoint_rows, pair_rows, score_rows, configs)
    assert len(rows) == 60
    assert {row["dose_index"] for row in rows} == {1, 2, 3, 4}
    assert all(set(FEATURES) <= set(row) for row in rows)

    target = next(row for row in rows if row["run_id"] == "padded-muon-1e-4" and row["stage"] == 1)
    source = next(
        row for row in checkpoint_rows if row["run_id"] == target["run_id"] and row["stage"] == 1
    )
    assert target["log_saved_segment_to_weight_ratio"] == pytest.approx(
        math.log(source["saved_segment_to_weight_ratio"])
    )
    expected = []
    for pair in pair_rows:
        if pair["stage"] != 1 or pair["displacement_kind"] != "saved_segment":
            continue
        if pair["first_run_id"] == target["run_id"] and pair["second_optimizer"] == "adamw":
            expected.append(pair["mean_subspace_overlap"])
        if pair["second_run_id"] == target["run_id"] and pair["first_optimizer"] == "adamw":
            expected.append(pair["mean_subspace_overlap"])
    assert len(expected) == 4
    assert target["mean_saved_segment_subspace_overlap_to_adamw"] == pytest.approx(
        sum(expected) / 4
    )

    adamw = next(row for row in rows if row["run_id"] == "padded-adamw-1e-6" and row["stage"] == 1)
    adamw_expected = []
    for pair in pair_rows:
        if pair["stage"] != 1 or pair["displacement_kind"] != "saved_segment":
            continue
        if pair["first_run_id"] == adamw["run_id"] and pair["second_optimizer"] == "adamw":
            adamw_expected.append(pair["mean_subspace_overlap"])
        if pair["second_run_id"] == adamw["run_id"] and pair["first_optimizer"] == "adamw":
            adamw_expected.append(pair["mean_subspace_overlap"])
    assert len(adamw_expected) == 3
    assert adamw["mean_saved_segment_subspace_overlap_to_adamw"] == pytest.approx(
        sum(adamw_expected) / 3
    )


def test_assemble_bridge_rows_rejects_incomplete_subspace_grid() -> None:
    configs, checkpoint_rows, pair_rows, score_rows = _synthetic_inputs()
    with pytest.raises(ValueError, match="Expected 660"):
        assemble_bridge_rows(checkpoint_rows, pair_rows[:-1], score_rows, configs)


def test_prediction_support_uses_pooled_and_three_of_four_fold_rule() -> None:
    rows = []
    row_index = 0
    for optimizer_index, optimizer in enumerate(("adamw", "muon", "normuon")):
        for dose_index in range(1, 5):
            centered_rate = dose_index - 2.5
            for stage in range(1, 6):
                signal = centered_rate * (stage - 3)
                row = {
                    "run_id": f"{optimizer}-{dose_index}",
                    "optimizer": optimizer,
                    "learning_rate": 10 ** (-6 + dose_index),
                    "dose_index": dose_index,
                    "centered_log10_learning_rate": centered_rate,
                    "stage": stage,
                    "progress_fraction": stage / 5,
                    "mean_ndcg_at_10": (
                        0.45
                        + 0.01 * optimizer_index
                        + 0.004 * stage
                        + 0.002 * centered_rate
                        + 0.02 * signal
                    ),
                }
                for feature_index, feature in enumerate(FEATURES):
                    row[feature] = (
                        signal
                        if feature_index == 0
                        else math.sin((row_index + 1) * (feature_index + 1) * 0.137) + 0.01 * signal
                    )
                rows.append(row)
                row_index += 1

    folds, summaries, associations = evaluate_bridge_features(rows)
    assert len(folds) == 36
    assert len(summaries) == len(FEATURES)
    assert len(associations) == len(FEATURES)
    assert all(row["train_rows"] == 45 and row["test_rows"] == 15 for row in folds)
    signal_summary = next(row for row in summaries if row["feature"] == FEATURES[0])
    assert signal_summary["pooled_rmse_reduction"] > 0
    assert signal_summary["folds_improved"] == 4
    assert signal_summary["predictively_useful"] is True
    signal_association = next(row for row in associations if row["feature"] == FEATURES[0])
    assert signal_association["pearson_residual_association"] == pytest.approx(1.0)


def test_prediction_rejects_incomplete_panel() -> None:
    with pytest.raises(ValueError, match="complete 60-row"):
        evaluate_bridge_features([])


def test_checked_in_bridge_protocol_binds_current_sources() -> None:
    repository = Path(__file__).resolve().parents[1]
    payload = _load_implementation_protocol(
        repository / "configs/dense_no_packing_bridge_implementation_protocol.json",
        repository,
    )
    assert payload["expected_outputs"] == {
        "bridge_rows": 60,
        "features": 9,
        "leave_dose_fold_rows": 36,
        "feature_prediction_summary_rows": 9,
        "residual_association_rows": 9,
    }


def _write_bound_table(directory: Path, filename: str, rows: list[dict]) -> dict:
    path = directory / filename
    directory.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "path": str(path),
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_build_report_audits_and_writes_complete_bridge(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    configs, checkpoint_rows, pair_rows, score_rows = _synthetic_inputs()
    geometry_dir = tmp_path / "geometry"
    outcomes_dir = tmp_path / "outcomes"
    geometry_outputs = {
        "checkpoint_geometry.csv": _write_bound_table(
            geometry_dir, "checkpoint_geometry.csv", checkpoint_rows
        ),
        "run_pair_subspace_overlap.csv": _write_bound_table(
            geometry_dir, "run_pair_subspace_overlap.csv", pair_rows
        ),
    }
    analysis_protocol = repository / "configs/dense_no_packing_analysis_protocol.json"
    (geometry_dir / "summary_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "protocol": {"sha256": hashlib.sha256(analysis_protocol.read_bytes()).hexdigest()},
                "outputs": geometry_outputs,
            }
        ),
        encoding="utf-8",
    )
    outcome_outputs = {
        "run_stage_scores.csv": _write_bound_table(outcomes_dir, "run_stage_scores.csv", score_rows)
    }
    outcome_protocol = repository / "configs/dense_no_packing_outcome_protocol.json"
    (outcomes_dir / "summary_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "protocol": {"sha256": hashlib.sha256(outcome_protocol.read_bytes()).hexdigest()},
                "outputs": outcome_outputs,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "bridge"
    report = build_report(
        argparse.Namespace(
            protocol=repository / "configs/dense_no_packing_bridge_implementation_protocol.json",
            matrix=repository / "configs/dense_no_packing_retrain.yaml",
            geometry_dir=geometry_dir,
            outcomes_dir=outcomes_dir,
            output_dir=output_dir,
        )
    )
    assert report["status"] == "complete"
    assert report["coverage"] == {
        "runs": 12,
        "stages": 5,
        "bridge_rows": 60,
        "features": 9,
        "leave_dose_fold_rows": 36,
    }
    assert (output_dir / "summary_manifest.json").is_file()
    assert report["outputs"]["bridge_rows.csv"]["rows"] == 60
    assert report["outputs"]["feature_prediction_summary.csv"]["rows"] == 9
