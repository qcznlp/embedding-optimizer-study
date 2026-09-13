import copy

import numpy as np
import pytest

from scripts.audit_dense_probe_coupling import compare_seed_pairings


def fixture():
    seeds = {seed: (np.eye(6)[:, 2 * seed : 2 * seed + 2],) * 2 for seed in range(3)}
    return {("a", "w"): seeds, ("b", "w"): copy.deepcopy(seeds)}, {
        ("a", "w"): seeds[0],
        ("b", "w"): seeds[0],
    }


def test_full_crossed_seeds_separate_shared_probe_similarity():
    spaces, exact = fixture()
    result = compare_seed_pairings(spaces, exact, ["a", "b"], ["w"], rank=2)
    assert len(result["rows"]) == 9
    row = result["summaries"][0]
    assert row["same_seed_pairs"] == 3 and row["different_seed_pairs"] == 6
    assert row["same_seed_mean_overlap"] == row["exact_mean_overlap"] == 1
    assert row["different_seed_mean_overlap"] == 0


def test_pairing_results_ignore_basis_signs():
    spaces, exact = fixture()
    first = compare_seed_pairings(spaces, exact, ["a", "b"], ["w"], rank=2)
    spaces["a", "w"][0] = tuple(-basis for basis in spaces["a", "w"][0])
    assert compare_seed_pairings(spaces, exact, ["a", "b"], ["w"], rank=2) == first


@pytest.mark.parametrize(
    "change", ["missing_seed", "extra_seed", "population", "rank", "nonorthogonal"]
)
def test_incomplete_or_malformed_pairing_inputs_are_refused(change):
    spaces, exact = fixture()
    if change == "missing_seed":
        spaces["a", "w"].pop(2)
    elif change == "extra_seed":
        spaces["a", "w"][3] = spaces["a", "w"][0]
    elif change == "population":
        spaces.pop(("b", "w"))
    elif change == "rank":
        spaces["a", "w"][0] = (np.eye(6)[:, :1],) * 2
    else:
        spaces["a", "w"][0] = (np.zeros((6, 2)),) * 2
    with pytest.raises(ValueError):
        compare_seed_pairings(spaces, exact, ["a", "b"], ["w"], rank=2)
