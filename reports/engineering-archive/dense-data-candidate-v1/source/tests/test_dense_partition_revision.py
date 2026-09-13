import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset

from embed_optim.data import _candidate_ranks, _scan_eligible_candidates
from scripts.materialize_dense_partition_revision import (
    PinnedSource,
    changed_positions,
    fixed_priority,
    group_row,
    mine_record,
    partition_isolation,
    revised_table,
    select_replacements,
)


def mined_record(positive=100):
    return {
        "query_id": 7,
        "document_ids": [positive, *range(200, 214)],
        "scores": [1.0, 0.95, *([0.2] * 13)],
    }


@pytest.mark.parametrize("seed", [42, 20260826])
@pytest.mark.parametrize("split", ["fiqa", "hotpotqa", "nq", "fever"])
def test_mining_matches_original_scanner(tmp_path, seed, split):
    root = tmp_path / "scores"
    root.mkdir()
    records = [
        {"query_id": 7, "document_ids": [100, 1], "scores": [1.0, 0.1]},
        mined_record(101),
        mined_record(102),
    ]
    pq.write_table(pa.Table.from_pylist(records), root / f"{split}-00000.parquet", row_group_size=1)
    original = _scan_eligible_candidates(tmp_path, split, {7: 0}, seed, 0.95, 10, 7, 2)[7]
    actual = mine_record(records[1], seed, split, 7)
    assert mine_record(records[0], seed, split, 7) is None
    assert actual == {
        "query_id": 7,
        "positive_id": original.positive_id,
        "negative_ids": list(original.negative_ids),
        "negative_pool_indices": list(original.negative_pool_indices),
    }
    assert original.positive_id == 101
    assert 200 not in actual["negative_ids"]  # Equality to the threshold is not eligible.


@pytest.mark.parametrize("partition", ["training", "validation"])
def test_priority_keeps_original_population_before_new_exclusions(partition):
    values = np.arange(30)
    old_training = {1, 7, 9}
    population = values if partition == "training" else np.setdiff1d(values, sorted(old_training))
    manifest = {
        "scorable_query_counts": {"fiqa": 30},
        "available_disjoint_query_counts": {"fiqa": 27},
        "quotas": {"fiqa": 2},
        "seed": 42,
        "candidate_margin": 0.5,
    }
    assert fixed_priority(values, old_training, partition, "fiqa", manifest) == _candidate_ranks(
        population, 2, 42, "fiqa", 0.5
    )
    manifest[
        "scorable_query_counts" if partition == "training" else "available_disjoint_query_counts"
    ]["fiqa"] -= 1
    with pytest.raises(ValueError, match="population changed"):
        fixed_priority(values, old_training, partition, "fiqa", manifest)


@pytest.mark.parametrize("invalid", ["", " \n", None])
def test_empty_positive_is_not_imputed(invalid):
    mined = mine_record(mined_record(), 42, "fiqa", 7)
    documents = {i: str(i) for i in [mined["positive_id"], *mined["negative_ids"]]}
    documents[mined["positive_id"]] = invalid
    assert group_row(0, "fiqa", "query", mined, documents) == (None, "empty_or_missing_text")


def test_first_eligible_positive_does_not_change_for_empty_text():
    source = PinnedSource.__new__(PinnedSource)
    source.split, source.snapshot = "fiqa", Path("/fixture")
    path = source.snapshot / "scores/fiqa.parquet"
    source.score_locations = {7: [(path, 0, 0), (path, 0, 1)]}
    records = [mined_record(100), mined_record(101)]
    source.score_record = lambda p, rg, offset: records[offset]
    mined, origin = source.first_mined(7, 42)
    assert mined["positive_id"] == 100 and origin["offset"] == 0


def test_exact_target_value_preservation_and_quota_guard():
    original = Dataset.from_list(
        [{"sample_id": i, "source": "fiqa", "text": str(i)} for i in range(5)]
    )
    replacement = {2: {"sample_id": 2, "source": "fiqa", "text": "replacement"}}
    after = revised_table(original, replacement)
    assert changed_positions(original.data.table, after) == [2]
    assert after.to_pylist()[:2] == original.to_list()[:2]
    replacement[2]["source"] = "nq"
    with pytest.raises(ValueError, match="source quota"):
        revised_table(original, replacement)


@pytest.mark.parametrize("bad", ["text_overlap", "id_overlap", "beir_overlap", "val_duplicate"])
def test_partition_admission_rejects_each_leak(bad):
    train = [{"source": "fiqa", "query_id": 1, "query": "training"}]
    val = [{"source": "nq", "query_id": 2, "query": "validation"}]
    beir = set()
    if bad == "text_overlap":
        val[0]["query"] = " TRAINING "
    elif bad == "id_overlap":
        val[0].update(source="fiqa", query_id=1)
    elif bad == "beir_overlap":
        from scripts.prepare_dense_data_protection import query_digest

        beir.add(query_digest("training"))
    else:
        val.append({"source": "nq", "query_id": 3, "query": "validation"})
    with pytest.raises(ValueError, match="not isolated"):
        partition_isolation(Dataset.from_list(train), Dataset.from_list(val), beir)


def test_selection_reserves_accepted_query_before_later_partition():
    def fake_first(query_id, seed):
        return {
            "query_id": query_id,
            "positive_id": 100,
            "negative_ids": list(range(101, 108)),
            "negative_pool_indices": list(range(7)),
        }, {"fixture": True}

    source = SimpleNamespace(
        scorable=np.arange(10),
        queries={i: f"query-{i}" for i in range(10)},
        first_mined=fake_first,
        documents=lambda ids: {i: str(i) for i in ids},
    )
    parents = {
        p: Dataset.from_list(
            [{"sample_id": 0, "source": "fiqa", "query_id": q, "query": f"query-{q}"}]
        )
        for p, q in (("training", 0), ("validation", 1))
    }
    manifests = {
        p: {
            "scorable_query_counts": {"fiqa": 10},
            "available_disjoint_query_counts": {"fiqa": 9},
            "quotas": {"fiqa": 1},
            "seed": 42,
            "candidate_margin": 0.5,
        }
        for p in parents
    }
    proposal = {"partitions": {p: {"rows": [{"source": "fiqa", "sample_id": 0}]} for p in parents}}
    decisions = io.StringIO()
    selected, _ = select_replacements(
        parents, manifests, proposal, {"fiqa": source}, set(), decisions
    )
    ids = [selected[p][0]["row"]["query_id"] for p in parents]
    assert len(set(ids)) == 2 and not set(ids) & {0, 1}
