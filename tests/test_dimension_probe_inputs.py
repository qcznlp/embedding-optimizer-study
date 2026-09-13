import copy
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from embed_optim.primary_contract import file_identity
from scripts import audit_dimension_probe_inputs as audit


def task_fixture(tmp_path):
    task = {
        "name": "test-task",
        "dataset": "test/repo",
        "revision": "a" * 40,
        "split": "test",
        "query_count": 1,
        "candidate_pool_count": 8,
    }
    query_rows = [{"_id": f"q{i}", "text": f"query {i}"} for i in range(8)]
    qrel_rows = [{"query-id": f"q{i}", "corpus-id": f"d{i}", "score": 1} for i in range(8)]
    doc_rows = [{"_id": f"d{i}", "title": f"title {i}", "text": f"text {i}"} for i in range(8)]
    chosen = min(range(8), key=lambda i: audit.selection_rank(4242, task["name"], f"q{i}"))
    ids = [f"d{chosen}", *(f"d{i}" for i in range(8) if i != chosen)]
    row = {
        "source": task["name"],
        "sample_id": audit.selection_rank(4242, "sample", task["name"], f"q{chosen}")
        & ((1 << 63) - 1),
        "query_id": f"q{chosen}",
        "positive_id": ids[0],
        "query": f"query {chosen}",
        "positive": f"title {chosen}\ntext {chosen}",
    }
    for i, doc in enumerate(ids[1:]):
        number = int(doc[1:])
        row[f"negative_{i}_id"] = doc
        row[f"negative_{i}"] = f"title {number}\ntext {number}"
    row["length"] = max(len(row[key]) for key in audit.TEXT_COLUMNS)
    ledger = {key: row[key] for key in ("sample_id", "source", "query_id", "positive_id")}
    ledger.update(
        negative_ids=ids[1:],
        relevant_ids=[ids[0]],
        selection_rank_hex=f"{audit.selection_rank(4242, task['name'], row['query_id']):032x}",
    )
    files = []
    for name, rows in (
        ("queries.parquet", query_rows),
        ("qrels_test.parquet", qrel_rows),
        ("corpus.parquet", doc_rows),
    ):
        path = tmp_path / name
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=3)
        files.append({"repo_path": name, **audit.binding(path)})
    return task, row, ledger, files


def test_original_row_and_all_nine_source_texts_verify(tmp_path):
    task, row, ledger, files = task_fixture(tmp_path)
    identity = audit.row_identity(row, ledger, 0, seed=4242)
    assert identity["candidate_ids_positive_first"] == [row["positive_id"], *ledger["negative_ids"]]
    result = audit.verify_task(task, [row], [ledger], files, 4242)
    assert result["verified_text_fields"] == 9
    assert result["negative_priority_regenerated"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "sample",
        "source",
        "empty",
        "length",
        "positive",
        "negative_order",
        "duplicate",
        "priority",
        "relevance",
    ],
)
def test_row_ledger_identity_refuses_malformed_input(tmp_path, mutation):
    _, row, ledger, _ = task_fixture(tmp_path)
    if mutation == "sample":
        row["sample_id"] += 1
    elif mutation == "source":
        row["source"] = "another-task"
    elif mutation == "empty":
        row["negative_0"] = " "
    elif mutation == "length":
        row["length"] += 1
    elif mutation == "positive":
        row["positive_id"] = "another"
    elif mutation == "negative_order":
        ledger["negative_ids"] = ledger["negative_ids"][::-1]
    elif mutation == "duplicate":
        row["negative_0_id"] = row["positive_id"]
    elif mutation == "priority":
        ledger["selection_rank_hex"] = "0" * 32
    else:
        ledger["relevant_ids"].append(row["negative_0_id"])
    with pytest.raises(ValueError):
        audit.row_identity(row, ledger, 0, seed=4242)


@pytest.mark.parametrize(
    "mutation", ["query_text", "positive_text", "negative_text", "qrels", "pool"]
)
def test_source_reconstruction_refuses_relabelled_content(tmp_path, mutation):
    task, row, ledger, files = task_fixture(tmp_path)
    if mutation.endswith("text"):
        key = {"query_text": "query", "positive_text": "positive", "negative_text": "negative_2"}[
            mutation
        ]
        row[key] += " changed"
    elif mutation == "qrels":
        ledger["relevant_ids"].append(row["negative_0_id"])
    else:
        ledger["negative_ids"][0] = "unknown-pool-document"
    with pytest.raises(ValueError):
        audit.verify_task(task, [row], [ledger], files, 4242)


@pytest.mark.parametrize("mutation", [None, "digest", "size", "missing"])
def test_only_hf_cache_role_resolves_symlink_then_checks_immutable_bytes(
    tmp_path, monkeypatch, mutation
):
    task = {"dataset": "synthetic/repo", "revision": "b" * 40, "split": "test"}
    snapshot = tmp_path / "snapshot"
    blobs = tmp_path / "blobs"
    snapshot.mkdir()
    blobs.mkdir()
    entries = []
    for name in ("queries.parquet", "qrels_test.parquet", "corpus.parquet"):
        path = blobs / name
        path.write_bytes(name.encode())
        (snapshot / name).symlink_to(path)
        bound = file_identity(path)
        entries.append(
            SimpleNamespace(
                path=name, size=bound["bytes"], lfs=SimpleNamespace(sha256=bound["sha256"])
            )
        )
    if mutation == "digest":
        entries[0].lfs.sha256 = "0" * 64
    elif mutation == "size":
        entries[0].size += 1
    elif mutation == "missing":
        entries = entries[:-1]
    monkeypatch.setattr(audit, "snapshot_download", lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(
        audit, "HfApi", lambda: SimpleNamespace(get_paths_info=lambda *args, **kwargs: entries)
    )
    if mutation:
        with pytest.raises(ValueError):
            audit.upstream_files(task)
    else:
        observed = audit.upstream_files(task)
        assert all(not (snapshot / row["repo_path"]).resolve().is_symlink() for row in observed)
        assert all(row["path"].startswith(str(blobs)) for row in observed)
        assert audit.upstream_files(task, copy.deepcopy(observed)) == observed


def test_local_probe_directory_still_refuses_symlinks(tmp_path):
    root = tmp_path / "original"
    root.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="ordinary fixed probe"):
        audit.read_probe(link, {})
