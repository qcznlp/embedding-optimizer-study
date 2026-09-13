import json

import numpy as np
import pytest
from datasets import Dataset

from embed_optim.data import _candidate_ranks
from scripts.materialize_dense_partition_revision import write_partition
from scripts.replay_dense_partition_revision import materialize_reference, priority


@pytest.mark.parametrize("seed", [42, 20260826])
@pytest.mark.parametrize("size", [17, 25000])
def test_independent_priority_matches_original(seed, size):
    population = np.arange(size) * 3 + 7
    manifest = {"seed": seed, "quotas": {"nq": 10}, "candidate_margin": 0.5}
    assert priority(population, manifest, "nq") == list(
        _candidate_ranks(population, 10, seed, "nq", 0.5)
    )


def make_row(index, query_id):
    row = {
        "sample_id": index,
        "source": "fiqa",
        "query_id": query_id,
        "positive_id": 100,
        "query": f"query-{query_id}",
        "positive": "positive",
        "length": 20,
    }
    for i in range(7):
        row[f"negative_{i}"] = f"negative-{i}"
        row[f"negative_{i}_id"] = 200 + i
    return row


def ledger(row):
    return {
        **{k: row[k] for k in ("sample_id", "source", "query_id", "positive_id")},
        "negative_ids": [row[f"negative_{i}_id"] for i in range(7)],
        "negative_pool_indices": list(range(7)),
    }


def test_independent_generator_writer_matches_splice_writer(tmp_path):
    original = Dataset.from_list([make_row(i, i) for i in range(20)])
    root = tmp_path / "original"
    original.save_to_disk(str(root / "dataset"))
    with (root / "rows.jsonl").open("w") as stream:
        for row in original:
            stream.write(json.dumps(ledger(row), sort_keys=True) + "\n")
    (root / "manifest.json").write_text(json.dumps({"quotas": {"fiqa": 20}}))
    replacements = {3: {"row": make_row(3, 1000), "ledger": ledger(make_row(3, 1000))}}
    original = Dataset.load_from_disk(str(root / "dataset"))
    actual = write_partition(
        "training", root, original, replacements, tmp_path / "prepared", "fixture", None
    )
    reference = materialize_reference(
        root, replacements, tmp_path / "prepared", tmp_path / "reference"
    )
    assert actual["changed_sample_ids"] == reference["changed_sample_ids"] == [3]
    assert reference["every_prepared_value_matches_independent_materialization"]
    assert reference["every_non_target_value_matches_original"]
