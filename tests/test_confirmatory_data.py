from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from embed_optim.confirmatory_data import (
    _scan_negative_pools,
    _selected_pool_indices,
    load_confirmatory_protocol,
)


def _score_fixture(tmp_path: Path, *, duplicate_pool_id: bool = False):
    score_root = tmp_path / "scores"
    score_root.mkdir()
    pool_ids = list(range(200, 210))
    if duplicate_pool_id:
        pool_ids[-1] = pool_ids[-2]
    documents = [100, 999, *pool_ids]
    scores = [1.0, 0.99, *[0.94 - index * 0.01 for index in range(10)]]
    table = pa.table(
        {
            "query_id": pa.array([7], type=pa.int64()),
            "document_ids": pa.array([documents], type=pa.list_(pa.int64())),
            "scores": pa.array([scores], type=pa.list_(pa.float64())),
        }
    )
    pq.write_table(table, score_root / "fever-00000-of-00001.parquet")
    indices = _selected_pool_indices(42, "fever", 7)
    base = {
        "sample_id": 0,
        "source": "fever",
        "query_id": 7,
        "positive_id": 100,
        "negative_pool_indices": indices,
        "negative_ids": [pool_ids[index] for index in indices],
    }
    return base, list(range(200, 210))


def _append_alternate_positive(tmp_path: Path):
    path = tmp_path / "scores" / "fever-00000-of-00001.parquet"
    original = pq.read_table(path)
    alternate = pa.table(
        {
            "query_id": pa.array([7], type=pa.int64()),
            "document_ids": pa.array([[101, *range(300, 310)]], type=pa.list_(pa.int64())),
            "scores": pa.array(
                [[1.0, *[0.9 - index * 0.01 for index in range(10)]]],
                type=pa.list_(pa.float64()),
            ),
        }
    )
    pq.write_table(pa.concat_tables([original, alternate]), path)


def test_frozen_confirmatory_protocol_has_three_new_seeds():
    path, protocol = load_confirmatory_protocol("configs/confirmatory_protocol.json")

    assert path.name == "confirmatory_protocol.json"
    assert protocol["confirmatory_data"]["seeds"] == [314159, 271828, 161803]
    assert protocol["source"]["exploratory_seed"] not in protocol["confirmatory_data"]["seeds"]
    assert protocol["training"]["expected_runs"] == 18
    assert protocol["training"]["expected_beir_units"] == 252
    assert protocol["freeze_context"]["strict_beir_valid_units"] == 150
    assert protocol["freeze_context"]["query_disjoint_validation_outputs_visible"] is False
    assert protocol["freeze_context"]["confirmatory_model_outputs_exist"] is False


def test_protocol_rejects_reusing_exploratory_seed(tmp_path: Path):
    _, protocol = load_confirmatory_protocol("configs/confirmatory_protocol.json")
    protocol["confirmatory_data"]["seeds"] = [42, 271828, 161803]
    protocol["confirmatory_data"]["outputs"]["42"] = protocol["confirmatory_data"]["outputs"].pop(
        "314159"
    )
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="exploratory seed"):
        load_confirmatory_protocol(path)


def test_scan_reconstructs_the_exact_seed42_pool(tmp_path: Path):
    base, expected_pool = _score_fixture(tmp_path)

    observed = _scan_negative_pools(
        tmp_path,
        "fever",
        [base],
        exploratory_seed=42,
        threshold=0.95,
        pool_size=10,
        sampled_negatives=7,
        batch_size=8,
    )

    assert observed == [
        {
            "sample_id": 0,
            "source": "fever",
            "query_id": 7,
            "positive_id": 100,
            "negative_pool_ids": expected_pool,
        }
    ]


def test_scan_matches_original_first_eligible_positive_rule(tmp_path: Path):
    base, expected_pool = _score_fixture(tmp_path)
    _append_alternate_positive(tmp_path)

    observed = _scan_negative_pools(
        tmp_path,
        "fever",
        [base],
        exploratory_seed=42,
        threshold=0.95,
        pool_size=10,
        sampled_negatives=7,
        batch_size=8,
    )

    assert observed[0]["positive_id"] == 100
    assert observed[0]["negative_pool_ids"] == expected_pool


def test_scan_rejects_duplicate_pool_documents(tmp_path: Path):
    base, _ = _score_fixture(tmp_path, duplicate_pool_id=True)

    with pytest.raises(ValueError, match="duplicate/positive pool"):
        _scan_negative_pools(
            tmp_path,
            "fever",
            [base],
            exploratory_seed=42,
            threshold=0.95,
            pool_size=10,
            sampled_negatives=7,
            batch_size=8,
        )


def test_confirmatory_selection_is_deterministic_distinct_and_seeded():
    first = _selected_pool_indices(314159, "nq", 123)
    repeated = _selected_pool_indices(314159, "nq", 123)
    alternate = _selected_pool_indices(271828, "nq", 123)

    assert first == repeated
    assert first == sorted(set(first))
    assert len(first) == 7
    assert all(0 <= index < 10 for index in first)
    assert first != alternate
