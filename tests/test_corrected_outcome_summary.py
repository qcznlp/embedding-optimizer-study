from __future__ import annotations

from pathlib import Path

import pytest

from embed_optim.config import load_matrix
from embed_optim.corrected_outcome_summary import (
    CONTRASTS,
    _load_outcome_protocol,
    paired_max_t_intervals,
    summarize_score_rows,
)
from embed_optim.decontamination import DECONTAMINATED_TASK_NAMES

REPOSITORY = Path(__file__).resolve().parents[1]


def _configs(root: Path):
    rates = {
        "adamw": (1e-6, 3e-6, 1e-5, 3e-5),
        "muon": (1e-4, 3e-4, 1e-3, 3e-3),
        "normuon": (1e-4, 3e-4, 1e-3, 3e-3),
    }
    entries = []
    for optimizer, values in rates.items():
        for index, value in enumerate(values):
            entries.append(
                f"  - id: padded-{optimizer}-{index}\n    name: {optimizer}\n    lr: {value:.1e}\n"
            )
    matrix = root / "matrix.yaml"
    matrix.write_text(
        (
            "common:\n"
            f"  output_root: {root / 'outputs'}\n"
            f"  dataset_path: {root / 'data'}\n"
            "  dense_can_flatten_inputs: false\n"
            "  checkpoint_fractions: [0.2, 0.4, 0.6, 0.8, 1.0]\n"
            "models:\n"
            "  dense:\n"
            "    model_name: example/base\n"
            "    model_revision: revision\n"
            "optimizers:\n" + "".join(entries)
        ),
        encoding="utf-8",
    )
    return load_matrix(matrix)


def _score_rows(configs):
    optimizer_effect = {"adamw": 0.0, "muon": 0.02, "normuon": 0.01}
    task_slope = {"adamw": 0.0, "muon": 0.0002, "normuon": -0.0001}
    rate_scale = {"adamw": 0.0003, "muon": 0.0007, "normuon": 0.0005}
    optimizer_index = {name: 0 for name in optimizer_effect}
    rate_index = {}
    for config in configs:
        index = optimizer_index[config.optimizer.name]
        rate_index[config.run_id] = index
        optimizer_index[config.optimizer.name] += 1
    rows = []
    for config in configs:
        for stage in range(1, 6):
            for task_index, task in enumerate(DECONTAMINATED_TASK_NAMES):
                score = (
                    0.3
                    + stage * 0.002
                    + task_index * 0.001
                    + optimizer_effect[config.optimizer.name]
                    + task_slope[config.optimizer.name] * task_index
                    + rate_scale[config.optimizer.name] * rate_index[config.run_id]
                )
                rows.append(
                    {
                        "model_family": "dense",
                        "optimizer": config.optimizer.name,
                        "learning_rate": config.optimizer.lr,
                        "run_id": config.run_id,
                        "stage": stage,
                        "task": task,
                        "ndcg_at_10": score,
                    }
                )
    return rows


def test_paired_max_t_is_common_resample_deterministic_and_simultaneous():
    effects = {
        contrast: [
            0.01 * (index + 1) + task_index * 0.0001 * (index + 1)
            for task_index in range(len(DECONTAMINATED_TASK_NAMES))
        ]
        for index, contrast in enumerate(CONTRASTS)
    }
    first = paired_max_t_intervals(effects, samples=2_000, seed=17)
    second = paired_max_t_intervals(effects, samples=2_000, seed=17)

    assert first == second
    assert {row["simultaneous_critical_value"] for row in first.values()}.__len__() == 1
    assert all(row["support"] == "positive" for row in first.values())
    assert all(
        row["simultaneous_ci_95_lower"]
        <= row["mean_delta_ndcg_at_10"]
        <= row["simultaneous_ci_95_upper"]
        for row in first.values()
    )


def test_corrected_score_summary_uses_full_lr_family_and_validation_selection(tmp_path: Path):
    configs = _configs(tmp_path)
    rows = _score_rows(configs)
    selections = {
        optimizer: next(config.run_id for config in configs if config.optimizer.name == optimizer)
        for optimizer in ("adamw", "muon", "normuon")
    }

    summary = summarize_score_rows(
        rows,
        configs,
        selections,
        bootstrap_samples=2_000,
        bootstrap_seed=23,
    )

    assert len(summary["primary_task_effects"]) == 14
    assert len(summary["primary_summary"]) == 3
    assert len(summary["secondary_task_effects"]) == 14
    assert len(summary["secondary_summary"]) == 3
    assert len(summary["run_stage_scores"]) == 60
    assert len(summary["optimizer_stage_scores"]) == 15
    assert len(summary["run_observed_auc"]) == 12
    primary = {(row["treatment"], row["baseline"]): row for row in summary["primary_summary"]}
    assert primary[("muon", "adamw")]["support"] == "positive"
    assert primary[("normuon", "adamw")]["support"] == "positive"
    assert primary[("normuon", "muon")]["support"] == "negative"
    secondary = {(row["treatment"], row["baseline"]): row for row in summary["secondary_summary"]}
    assert secondary[("muon", "adamw")]["treatment_run_id"] == selections["muon"]
    assert all(float(row["observed_mean_20_to_100"]) > 0 for row in summary["run_observed_auc"])


def test_corrected_score_summary_rejects_incomplete_840_unit_grid(tmp_path: Path):
    configs = _configs(tmp_path)
    rows = _score_rows(configs)
    selections = {
        optimizer: next(config.run_id for config in configs if config.optimizer.name == optimizer)
        for optimizer in ("adamw", "muon", "normuon")
    }
    with pytest.raises(ValueError, match="coverage differs"):
        summarize_score_rows(rows[:-1], configs, selections, bootstrap_samples=10)


def test_checked_in_corrected_outcome_protocol_matches_current_sources():
    protocol = _load_outcome_protocol(
        REPOSITORY / "configs/dense_no_packing_outcome_protocol.json",
        REPOSITORY,
    )
    assert protocol["visibility_at_freeze"]["corrected_beir_outputs_visible"] is False
    assert protocol["inference"]["bootstrap_samples"] == 50_000
    assert protocol["expected_outputs"]["run_stage_score_rows"] == 60
