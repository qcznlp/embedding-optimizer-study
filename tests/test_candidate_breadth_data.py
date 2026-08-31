from __future__ import annotations

import copy

import pytest

from embed_optim.candidate_breadth_data import (
    load_candidate_breadth_protocol,
    nested_candidate_ids,
    select_validation_rows,
)
from embed_optim.data import SPLITS


def _validation_rows(per_source: int = 40) -> list[dict]:
    rows = []
    sample_id = 0
    for source in SPLITS:
        for query_id in range(per_source):
            rows.append(
                {
                    "sample_id": sample_id,
                    "source": source,
                    "query_id": query_id,
                    "positive_id": 100_000 + sample_id,
                    "negative_ids": list(range(7)),
                    "negative_pool_indices": list(range(7)),
                }
            )
            sample_id += 1
    return rows


def test_protocol_is_post_hoc_and_fixes_nested_widths() -> None:
    _, protocol = load_candidate_breadth_protocol("configs/candidate_breadth_probe.json")
    assert protocol["status"] == "post_hoc_shortlist_corpus_mechanism_probe"
    assert protocol["candidate_construction"]["negative_widths"] == [
        7,
        10,
        32,
        128,
        512,
        2048,
    ]
    assert protocol["timing"]["candidate_breadth_data_or_scores_visible"] is False


def test_balanced_selection_is_deterministic_and_source_complete() -> None:
    rows = _validation_rows()
    selected = select_validation_rows(rows, count=224, seed=20260901)
    repeated = select_validation_rows(reversed(rows), count=224, seed=20260901)
    assert selected == repeated
    assert len(selected) == 224
    assert {source: sum(row["source"] == source for row in selected) for source in SPLITS} == {
        source: 32 for source in SPLITS
    }
    assert [row["sample_id"] for row in selected] == sorted(row["sample_id"] for row in selected)


def test_nested_candidates_preserve_canonical_seven_and_extend_in_mined_order() -> None:
    raw_ids = [90_000, *range(1, 2049)]
    raw_scores = [1.0, *([0.9] * 2048)]
    pool_indices = [1, 3, 4, 5, 6, 7, 8]
    canonical = [2, 4, 5, 6, 7, 8, 9]
    row = {
        "positive_id": 90_000,
        "negative_ids": canonical,
        "negative_pool_indices": pool_indices,
    }
    candidates = nested_candidate_ids(
        raw_ids,
        raw_scores,
        row,
        threshold=0.95,
        maximum_width=2048,
    )
    assert candidates[:7] == canonical
    assert candidates[:10] == [*canonical, 1, 3, 10]
    assert len(candidates) == len(set(candidates)) == 2048
    assert 90_000 not in candidates
    for smaller, larger in zip((7, 10, 32, 128, 512), (10, 32, 128, 512, 2048)):
        assert candidates[:smaller] == candidates[:larger][:smaller]


def test_nested_candidates_reject_a_nonreproducible_canonical_seven() -> None:
    raw_ids = [90_000, *range(1, 2049)]
    raw_scores = [1.0, *([0.9] * 2048)]
    row = {
        "positive_id": 90_000,
        "negative_ids": [2, 4, 5, 6, 7, 8, 11],
        "negative_pool_indices": [1, 3, 4, 5, 6, 7, 8],
    }
    with pytest.raises(ValueError, match="canonical seven"):
        nested_candidate_ids(
            raw_ids,
            raw_scores,
            copy.deepcopy(row),
            threshold=0.95,
            maximum_width=2048,
        )
