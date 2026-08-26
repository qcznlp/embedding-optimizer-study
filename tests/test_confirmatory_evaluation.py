from __future__ import annotations

import numpy as np
import pytest

from embed_optim.confirmatory_summary import (
    hierarchical_seed_task_bootstrap,
    summarize_confirmatory_scores,
)
from embed_optim.decontamination import DECONTAMINATED_TASK_NAMES


def _rows():
    rows = []
    for seed_index, seed in enumerate((314159, 271828, 161803)):
        for family_index, family in enumerate(("dense", "late")):
            for optimizer_index, optimizer in enumerate(("adamw", "muon", "normuon")):
                for task_index, task in enumerate(DECONTAMINATED_TASK_NAMES):
                    rows.append(
                        {
                            "seed": seed,
                            "model_family": family,
                            "optimizer": optimizer,
                            "stage": 5,
                            "task": task,
                            "ndcg_at_10": (
                                0.2
                                + seed_index * 0.001
                                + family_index * 0.01
                                + optimizer_index * 0.002
                                + task_index * 0.0001
                            ),
                        }
                    )
    return rows


def test_hierarchical_bootstrap_is_deterministic_and_contains_constant_effect():
    effects = np.full((3, 14), 0.002)

    first = hierarchical_seed_task_bootstrap(effects, samples=500, seed=7)
    second = hierarchical_seed_task_bootstrap(effects, samples=500, seed=7)

    assert first == second
    assert first["bootstrap_ci_95_lower"] == pytest.approx(0.002)
    assert first["bootstrap_ci_95_upper"] == pytest.approx(0.002)
    assert first["bootstrap_probability_positive"] == 1.0


def test_confirmatory_summary_covers_three_fixed_contrasts():
    seed_scores, contrasts, summaries = summarize_confirmatory_scores(
        _rows(),
        [314159, 271828, 161803],
        bootstrap_samples=500,
    )

    assert len(seed_scores) == 18
    assert len(contrasts) == 252
    assert len(summaries) == 6
    assert {(row["treatment"], row["baseline"]) for row in summaries} == {
        ("muon", "adamw"),
        ("normuon", "adamw"),
        ("normuon", "muon"),
    }
    by_pair = {
        (row["treatment"], row["baseline"]): row
        for row in summaries
        if row["model_family"] == "dense"
    }
    assert by_pair[("muon", "adamw")]["mean_delta_ndcg_at_10"] == pytest.approx(0.002)
    assert by_pair[("normuon", "adamw")]["mean_delta_ndcg_at_10"] == pytest.approx(0.004)
    assert by_pair[("normuon", "muon")]["mean_delta_ndcg_at_10"] == pytest.approx(0.002)


def test_confirmatory_summary_rejects_partial_coverage():
    rows = _rows()
    rows.pop()

    with pytest.raises(ValueError, match="coverage differs"):
        summarize_confirmatory_scores(rows, [314159, 271828, 161803], bootstrap_samples=10)
