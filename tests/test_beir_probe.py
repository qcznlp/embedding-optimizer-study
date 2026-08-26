import json
from pathlib import Path

import pytest
from datasets import Dataset

from embed_optim.beir_probe import (
    LoadedTask,
    materialize_task_rows,
    prepare_beir_probe,
    select_task_anchors,
)
from embed_optim.geometry import _sha256


def _task_fixture() -> tuple[dict[str, str], dict[str, dict[str, int]], dict[str, str]]:
    queries = {f"q{index}": f"shared topic {index}" for index in range(12)}
    qrels = {f"q{index}": {f"d{index}": 1} for index in range(12)}
    documents = {
        f"d{index}": f"shared topic document {index} with token-{index % 3}" for index in range(12)
    }
    return queries, qrels, documents


def test_beir_anchor_and_negative_selection_is_deterministic_and_nonrelevant():
    queries, qrels, documents = _task_fixture()
    first = select_task_anchors(
        "Fixture", queries, qrels, output_count=3, candidate_pool_count=10, seed=17
    )
    second = select_task_anchors(
        "Fixture", queries, qrels, output_count=3, candidate_pool_count=10, seed=17
    )
    assert first == second
    assert sum(anchor.is_output for anchor in first) == 3

    rows, ledger = materialize_task_rows("Fixture", first, documents, seed=17)
    repeated_rows, repeated_ledger = materialize_task_rows("Fixture", second, documents, seed=17)
    assert rows == repeated_rows
    assert ledger == repeated_ledger
    assert len(rows) == 3
    assert len({row["sample_id"] for row in rows}) == 3
    for row, record in zip(rows, ledger, strict=True):
        negative_ids = [row[f"negative_{index}_id"] for index in range(7)]
        assert len(set(negative_ids)) == 7
        assert set(negative_ids).isdisjoint(record["relevant_ids"])
        assert record["negative_lexical_scores"] == sorted(
            record["negative_lexical_scores"], reverse=True
        )


def test_beir_probe_requires_enough_candidates():
    queries, qrels, _ = _task_fixture()
    with pytest.raises(ValueError, match="candidate pool"):
        select_task_anchors(
            "Fixture", queries, qrels, output_count=3, candidate_pool_count=20, seed=17
        )


def test_prepare_beir_probe_derives_then_enforces_frozen_hashes(tmp_path: Path, monkeypatch):
    queries, qrels, documents = _task_fixture()
    identity = {
        "name": "Fixture",
        "dataset": "fixture/repo",
        "revision": "a" * 40,
        "split": "test",
        "query_count": 3,
        "candidate_pool_count": 10,
        "queries_rows": 12,
        "queries_fingerprint": "queries",
        "qrels_rows": 12,
        "qrels_fingerprint": "qrels",
        "corpus_rows": 12,
        "corpus_fingerprint": "corpus",
    }
    monkeypatch.setattr(
        "embed_optim.beir_probe._load_task",
        lambda task_spec, seed: LoadedTask(queries, qrels, documents, identity),
    )
    spec = {
        "schema_version": 1,
        "output": str(tmp_path / "probe"),
        "seed": 17,
        "tasks": [
            {
                "name": "Fixture",
                "dataset": "fixture/repo",
                "revision": "a" * 40,
                "split": "test",
                "query_count": 3,
                "candidate_pool_count": 10,
            }
        ],
        "expected": {},
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec) + "\n")

    with pytest.raises(ValueError, match="not frozen"):
        prepare_beir_probe(spec_path, output=tmp_path / "rejected")
    output = prepare_beir_probe(spec_path, allow_unfrozen=True)
    manifest = json.loads((output / "manifest.json").read_text())
    serialized = Dataset.load_from_disk(str(output / "dataset"))
    assert len(serialized) == 3
    assert manifest["selection_sha256"] == _sha256(output / "selection.jsonl")

    expected_keys = (
        "task_counts",
        "selected_sample_ids_sha256",
        "selection_sha256",
        "probe_dataset_fingerprint",
        "serialized_probe_dataset_fingerprint",
    )
    spec["expected"] = {key: manifest[key] for key in expected_keys}
    spec["expected"]["manifest_sha256"] = _sha256(output / "manifest.json")
    spec_path.write_text(json.dumps(spec) + "\n")
    frozen = prepare_beir_probe(spec_path, output=tmp_path / "frozen")
    assert _sha256(frozen / "manifest.json") == spec["expected"]["manifest_sha256"]

    spec["expected"]["selection_sha256"] = "0" * 64
    spec_path.write_text(json.dumps(spec) + "\n")
    with pytest.raises(ValueError, match="selection_sha256"):
        prepare_beir_probe(spec_path, output=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
