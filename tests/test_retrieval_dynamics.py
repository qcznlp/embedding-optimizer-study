from __future__ import annotations

from pathlib import Path

import pytest

from embed_optim.decontamination import DECONTAMINATED_TASK_NAMES
from embed_optim.retrieval_dynamics import (
    _quality_figure,
    load_retrieval_dynamics_protocol,
    render_task_stability_blog,
    summarize_retrieval_dynamics,
    summarize_task_delta_stability,
    task_delta_dynamics,
)


def _sources():
    runs = []
    checkpoints = []
    for family in ("dense", "late"):
        for optimizer in ("adamw", "muon", "normuon"):
            for index in range(4):
                run_id = f"{optimizer}-{index}"
                learning_rate = (index + 1) * (1e-6 if optimizer == "adamw" else 1e-4)
                runs.append(
                    {
                        "model_family": family,
                        "optimizer": optimizer,
                        "learning_rate": learning_rate,
                        "run_id": run_id,
                        "wall_time_hours": 10 + index,
                        "world_size": 4,
                    }
                )
                if optimizer == "adamw":
                    final_score = 0.40 + 0.02 * index
                elif optimizer == "muon":
                    final_score = 0.50 + 0.01 * index
                else:
                    final_score = 0.34 + 0.01 * index
                for stage, step in enumerate((782, 1563, 2345, 3126, 3907), start=1):
                    checkpoints.append(
                        {
                            "model_family": family,
                            "optimizer": optimizer,
                            "learning_rate": learning_rate,
                            "run_id": run_id,
                            "stage": stage,
                            "fraction": stage / 5,
                            "checkpoint_step": step,
                            "mean_ndcg_at_10": final_score * stage / 5,
                            "tasks_completed": 14,
                        }
                    )
    return checkpoints, runs


def test_retrieval_dynamics_keeps_all_sweep_points_and_censoring():
    checkpoints, runs = _sources()

    dynamics, first_passage, groups = summarize_retrieval_dynamics(checkpoints, runs)

    assert len(dynamics) == 120
    assert len(first_passage) == 24
    assert len(groups) == 6
    targets = {row["adamw_median_final_target"] for row in groups}
    assert len(targets) == 1
    assert targets.pop() == pytest.approx(0.43)
    dense_muon = next(
        row for row in groups if row["model_family"] == "dense" and row["optimizer"] == "muon"
    )
    dense_normuon = next(
        row for row in groups if row["model_family"] == "dense" and row["optimizer"] == "normuon"
    )
    assert dense_muon["points_reaching_target"] == 4
    assert dense_normuon["points_reaching_target"] == 0
    assert dense_normuon["points_right_censored"] == 4
    assert dense_normuon["median_observed_useful_wall_time_hours"] is None
    assert all(row["wall_time_estimation"] for row in dynamics)


def test_retrieval_dynamics_rejects_partial_checkpoint_matrix():
    checkpoints, runs = _sources()
    checkpoints.pop()

    with pytest.raises(ValueError, match="24 complete five-checkpoint trajectories"):
        summarize_retrieval_dynamics(checkpoints, runs)


def test_retrieval_dynamics_plot_is_renderable(tmp_path: Path):
    checkpoints, runs = _sources()
    dynamics, _, _ = summarize_retrieval_dynamics(checkpoints, runs)

    result = _quality_figure(dynamics, tmp_path / "quality.svg")

    path = tmp_path / "quality.svg"
    assert path.read_text(encoding="utf-8").startswith("<?xml")
    assert result["bytes"] == path.stat().st_size


def test_retrieval_dynamics_protocol_is_frozen_before_complete_beir():
    path, protocol = load_retrieval_dynamics_protocol()

    assert path.name == "retrieval_dynamics_protocol.json"
    assert protocol["freeze_context"]["strict_beir_valid_units"] == 160
    assert protocol["freeze_context"]["strict_beir_expected_units"] == 1_680
    assert protocol["freeze_context"]["complete_retrieval_matrix_visible"] is False
    assert protocol["reference_target"]["uses_muon_or_normuon_outcomes"] is False
    assert "not a preregistration" in protocol["claim_boundary"]


def test_posthoc_task_delta_stability_tracks_adjacent_checkpoint_directions(tmp_path: Path):
    evaluation_rows = []
    optimizer_rows = []
    for family in ("dense", "late"):
        for optimizer in ("adamw", "muon", "normuon"):
            run_id = f"{family}-{optimizer}-best"
            optimizer_rows.append(
                {
                    "model_family": family,
                    "optimizer": optimizer,
                    "best_run_id": run_id,
                }
            )
            for stage in range(1, 6):
                for task_index, task in enumerate(DECONTAMINATED_TASK_NAMES):
                    baseline = 0.4 + task_index / 1_000
                    if optimizer == "adamw":
                        delta = 0.0
                    elif optimizer == "muon":
                        delta = (task_index - 7) / 1_000 * stage
                    else:
                        delta = (7 - task_index) / 1_000 * stage
                    evaluation_rows.append(
                        {
                            "model_family": family,
                            "optimizer": optimizer,
                            "run_id": run_id,
                            "stage": stage,
                            "fraction": stage / 5,
                            "task": task,
                            "ndcg_at_10": baseline + delta,
                        }
                    )

    dynamics = task_delta_dynamics(evaluation_rows, optimizer_rows)
    stability = summarize_task_delta_stability(dynamics)

    assert len(dynamics) == 2 * 2 * 5 * 14
    assert len(stability) == 2 * 2 * 4
    assert all(row["same_direction_tasks"] == 14 for row in stability)
    assert all(row["pearson_correlation"] == pytest.approx(1.0) for row in stability)
    assert all(row["spearman_correlation"] == pytest.approx(1.0) for row in stability)

    blog = tmp_path / "blog.md"
    blog.write_text(
        "before\n<!-- TASK-DELTA-STABILITY:BEGIN -->\nold\n"
        "<!-- TASK-DELTA-STABILITY:END -->\nafter\n",
        encoding="utf-8",
    )
    render_task_stability_blog(blog, stability)
    rendered = blog.read_text(encoding="utf-8")
    assert "Exploratory task-effect stability across checkpoints" in rendered
    assert "20%→40%" in rendered
    assert "14/14" in rendered
    assert "post-hoc" in rendered
    assert "old" not in rendered
