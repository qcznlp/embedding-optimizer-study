from __future__ import annotations

import json
from pathlib import Path

import pytest

from embed_optim.config import OptimizerConfig, RunConfig
from embed_optim.short_branch_evaluation import build_short_branch_probe_jobs
from embed_optim.short_branch_summary import (
    expected_short_branch_probe_metrics,
    summarize_short_branch_contrasts,
)


def _configs(tmp_path: Path) -> dict[int, list[RunConfig]]:
    result = {}
    for seed in (314159, 271828, 161803):
        runs = []
        for family in ("dense", "late"):
            for operator in ("adamw", "muon", "normuon"):
                config = RunConfig(
                    run_id=f"{operator}-scale-matched",
                    model_family=family,
                    optimizer=OptimizerConfig(
                        name="hybrid_adamw" if operator == "adamw" else operator,
                        lr=1e-4,
                    ),
                    model_name="unused",
                    dataset_path="unused",
                    output_root=str(tmp_path / "outputs" / f"seed{seed}"),
                    seed=seed,
                )
                config.output_dir.mkdir(parents=True)
                steps = [10, 20, 30, 40, 50]
                (config.output_dir / "checkpoint_schedule.json").write_text(
                    json.dumps({"steps": steps}), encoding="utf-8"
                )
                for step in steps:
                    (config.output_dir / f"checkpoint-{step}").mkdir()
                runs.append(config)
        result[seed] = runs
    return result


def test_short_branch_expected_probe_metrics_preserve_seed_and_canonical_operator(tmp_path: Path):
    configs = _configs(tmp_path)
    jobs = build_short_branch_probe_jobs(
        configs,
        {"dense": tmp_path / "dense-base", "late": tmp_path / "late-base"},
        tmp_path / "results",
        ("manifest", "spec"),
    )

    expected = expected_short_branch_probe_metrics(configs, jobs)

    assert len(expected) == 92
    checkpoints = [item for item in expected if item.job.kind == "checkpoint"]
    assert {item.seed for item in checkpoints} == {314159, 271828, 161803}
    assert {item.optimizer for item in checkpoints} == {"adamw", "muon", "normuon"}
    assert {item.stage for item in checkpoints} == {1, 2, 3, 4, 5}


def _validation_rows():
    rows = []
    for seed_index, seed in enumerate((314159, 271828, 161803)):
        for family in ("dense", "late"):
            for stage in range(1, 6):
                for operator_index, operator in enumerate(("adamw", "muon", "normuon")):
                    rows.append(
                        {
                            "family": family,
                            "seed": seed,
                            "operator": operator,
                            "stage": stage,
                            "fraction": stage / 5,
                            "contrastive_loss": 1.0 - operator_index * 0.1 + seed_index * 0.001,
                            "positive_margin": operator_index * 0.2 + seed_index * 0.001,
                            "reciprocal_rank": 0.5 + operator_index * 0.1,
                            "top1_accuracy": 0.4 + operator_index * 0.1,
                        }
                    )
    return rows


def test_short_branch_contrasts_cover_all_stages_pairs_and_metrics():
    contrasts, summaries = summarize_short_branch_contrasts(_validation_rows())

    assert len(contrasts) == 90
    assert len(summaries) == 120
    assert {row["stage"] for row in summaries} == {1, 2, 3, 4, 5}
    assert {row["metric"] for row in summaries} == {
        "contrastive_loss",
        "positive_margin",
        "reciprocal_rank",
        "top1_accuracy",
    }
    assert all(row["treatment_seed_wins"] == 3 for row in summaries)


def test_short_branch_contrasts_reject_partial_matrix():
    rows = _validation_rows()
    rows.pop()

    with pytest.raises(ValueError, match="90 unique"):
        summarize_short_branch_contrasts(rows)
