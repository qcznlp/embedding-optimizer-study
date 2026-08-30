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
    assert first["familywise_method"] == "bonferroni"
    assert first["familywise_contrasts"] == 6
    assert first["familywise_ci_95_lower"] == pytest.approx(0.002)
    assert first["familywise_ci_95_upper"] == pytest.approx(0.002)
    assert first["bootstrap_probability_positive"] == 1.0


def test_familywise_bootstrap_interval_contains_the_nominal_interval():
    effects = np.linspace(-0.01, 0.02, 42).reshape(3, 14)

    interval = hierarchical_seed_task_bootstrap(effects, samples=5_000, seed=11)

    assert interval["familywise_ci_95_lower"] < interval["bootstrap_ci_95_lower"]
    assert interval["familywise_ci_95_upper"] > interval["bootstrap_ci_95_upper"]


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
    assert all(row["familywise_contrasts"] == 6 for row in summaries)
    assert all(
        row["familywise_ci_95_lower"] <= row["bootstrap_ci_95_lower"]
        and row["familywise_ci_95_upper"] >= row["bootstrap_ci_95_upper"]
        for row in summaries
    )


def test_confirmatory_summary_rejects_partial_coverage():
    rows = _rows()
    rows.pop()

    with pytest.raises(ValueError, match="coverage differs"):
        summarize_confirmatory_scores(rows, [314159, 271828, 161803], bootstrap_samples=10)


def test_dense_scope_uses_three_familywise_contrasts():
    dense_rows = [row for row in _rows() if row["model_family"] == "dense"]

    seed_scores, contrasts, summaries = summarize_confirmatory_scores(
        dense_rows,
        [314159, 271828, 161803],
        bootstrap_samples=100,
        families=("dense",),
    )

    assert len(seed_scores) == 9
    assert len(contrasts) == 126
    assert len(summaries) == 3
    assert all(row["familywise_contrasts"] == 3 for row in summaries)
