from __future__ import annotations

import json
from pathlib import Path

import pytest

from embed_optim.config import OptimizerConfig, RunConfig
from embed_optim.training_dynamics import optimizer_system_summary, stage_training_dynamics


def _configs(tmp_path: Path):
    configs = []
    for family in ("dense", "late"):
        for optimizer in ("adamw", "muon", "normuon"):
            for index in range(4):
                config = RunConfig(
                    run_id=f"{optimizer}-{index}",
                    model_family=family,
                    optimizer=OptimizerConfig(name=optimizer, lr=1e-5 * (index + 1)),
                    model_name="unused",
                    dataset_path="unused",
                    output_root=str(tmp_path / "outputs"),
                )
                config.output_dir.mkdir(parents=True)
                (config.output_dir / "checkpoint_schedule.json").write_text(
                    json.dumps({"steps": [782, 1563, 2345, 3126, 3907]}),
                    encoding="utf-8",
                )
                configs.append(config)
    return configs


def _histories(configs):
    rows = []
    steps = [1, *range(10, 3901, 10)]
    assert len(steps) == 391
    for config in configs:
        for step in steps:
            rows.append(
                {
                    "model_family": config.model_family,
                    "optimizer": config.optimizer.name,
                    "learning_rate_config": config.optimizer.lr,
                    "run_id": config.run_id,
                    "step": step,
                    "loss": 1 / step,
                    "grad_norm": 2 / step,
                    "learning_rate": config.optimizer.lr,
                    "epoch": step / 3907,
                }
            )
    return rows


def test_stage_training_dynamics_covers_five_stages_for_24_runs(tmp_path: Path):
    configs = _configs(tmp_path)
    rows = stage_training_dynamics(configs, _histories(configs))

    assert len(rows) == 120
    assert {row["stage"] for row in rows} == {1, 2, 3, 4, 5}
    assert all(row["window_observations"] == 10 for row in rows)
    assert {row["observed_step"] for row in rows if row["stage"] == 5} == {3900}


def _systems():
    rows = []
    for family in ("dense", "late"):
        for optimizer_index, optimizer in enumerate(("adamw", "muon", "normuon"), start=1):
            for index in range(4):
                rows.append(
                    {
                        "model_family": family,
                        "optimizer": optimizer,
                        "run_id": f"{optimizer}-{index}",
                        "wall_time_hours": 10 + optimizer_index,
                        "samples_per_second": 100 / optimizer_index,
                        "steps_per_second": 1 / optimizer_index,
                        "peak_allocated_gib": 20 + optimizer_index,
                        "peak_reserved_gib": 21 + optimizer_index,
                        "checkpoint_gib": 2,
                        "optimizer_state_gib": 3 / optimizer_index,
                    }
                )
    return rows


def test_optimizer_system_summary_uses_adamw_as_within_family_baseline():
    rows = optimizer_system_summary(_systems())

    assert len(rows) == 6
    dense_muon = next(
        row for row in rows if row["model_family"] == "dense" and row["optimizer"] == "muon"
    )
    assert dense_muon["throughput_to_adamw_ratio"] == pytest.approx(0.5)
    assert dense_muon["optimizer_state_to_adamw_ratio"] == pytest.approx(0.5)


def test_training_dynamics_rejects_partial_history(tmp_path: Path):
    configs = _configs(tmp_path)
    histories = _histories(configs)
    histories.pop()

    with pytest.raises(ValueError, match="391 unique loss records"):
        stage_training_dynamics(configs, histories)
