"""Authenticate all fixed probe texts and ordered candidates against pinned BEIR files.

The frozen ledger, not a newly selected query/candidate population, is consumed.
This read-only audit does not encode a model or certify any primary checkpoint.
"""

import argparse
import hashlib
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import HfApi, snapshot_download

from embed_optim.primary_contract import digest, file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_validation_io import read_record_file, write_new
from scripts.audit_dense_natural_data import handoff

PROBE_SPEC = "configs/beir_representation_probe.json"
PROBE_SPEC_SHA = "474aa62282ab200ebb522bc6897db9f2077dd39f627322ea476392b988f71f50"
TEXT_COLUMNS = ("query", "positive", *(f"negative_{i}" for i in range(7)))


def binding(path):
    return {"path": str(Path(path).resolve()), **file_identity(path)}


def selection_rank(seed, *parts):
    value = ":".join(("blake2b-128-query-and-positive-v1", str(seed), *parts))
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=16).digest(), "big")


def row_identity(row, ledger, index, *, seed):
    ids = [row["positive_id"], *(row[f"negative_{i}_id"] for i in range(7))]
    if (
        type(row["sample_id"]) is not int
        or row["sample_id"] < 0
        or any(not isinstance(x, str) or not x for x in [row["source"], row["query_id"], *ids])
        or len(set(ids)) != 8
        or any(not isinstance(row[k], str) or not row[k].strip() for k in TEXT_COLUMNS)
        or row["length"] != max(len(row[k]) for k in TEXT_COLUMNS)
    ):
        raise ValueError("Malformed, repeated or empty original probe row")
    require_same(
        {key: row[key] for key in ("sample_id", "source", "query_id", "positive_id")},
        {key: ledger[key] for key in ("sample_id", "source", "query_id", "positive_id")},
    )
    require_same(ids[1:], ledger["negative_ids"])
    rank = selection_rank(seed, row["source"], row["query_id"])
    sample = selection_rank(seed, "sample", row["source"], row["query_id"]) & ((1 << 63) - 1)
    if sample != row["sample_id"] or f"{rank:032x}" != ledger["selection_rank_hex"]:
        raise ValueError("Probe sample ID or query priority differs from the frozen seed")
    if set(ids[1:]) & set(ledger["relevant_ids"]) or ids[0] not in ledger["relevant_ids"]:
        raise ValueError("Ordered probe candidates violate the frozen relevance record")
    return {
        "position": index,
        "sample_id": sample,
        "source": row["source"],
        "query_id": row["query_id"],
        "candidate_ids_positive_first": ids,
        "row_sha256": digest(row),
    }


def read_probe(root, spec):
    files = sorted(root.rglob("*"))
    if root.is_symlink() or any(p.is_symlink() for p in files):
        raise ValueError("Require ordinary fixed probe inputs")
    names = {p.relative_to(root).as_posix() for p in files if p.is_file()}
    if names != {
        "manifest.json",
        "selection.jsonl",
        "dataset/data-00000-of-00001.arrow",
        "dataset/dataset_info.json",
        "dataset/state.json",
    }:
        raise ValueError("Unexpected fixed probe file inventory")
    inputs = [binding(root / name) for name in sorted(names)]
    for name, key in (
        ("manifest.json", "manifest_sha256"),
        ("selection.jsonl", "selection_sha256"),
    ):
        if file_identity(root / name)["sha256"] != spec["expected"][key]:
            raise ValueError("Probe does not match the immutable manifest/ordered ledger")
    rows = list(Dataset.load_from_disk(str(root / "dataset")))
    ledger = read_record_file(root / "selection.jsonl")
    if (
        len(rows) != 224
        or len(ledger) != 224
        or Counter(row["source"] for row in rows) != Counter(spec["expected"]["task_counts"])
    ):
        raise ValueError("Probe does not have all 224 rows and fourteen equal task groups")
    if (
        len({r["sample_id"] for r in rows}) != 224
        or len({(r["source"], r["query_id"]) for r in rows}) != 224
    ):
        raise ValueError("Duplicate probe sample or task-qualified query")
    identities = [
        row_identity(row, record, i, seed=spec["seed"])
        for i, (row, record) in enumerate(zip(rows, ledger, strict=True))
    ]
    sample_digest = hashlib.sha256(
        "".join(f"{row['sample_id']}\n" for row in identities).encode()
    ).hexdigest()
    if sample_digest != spec["expected"]["selected_sample_ids_sha256"]:
        raise ValueError("Probe row order differs from the frozen sample identity")
    return rows, ledger, identities, inputs


def upstream_files(task, recorded=None):
    names = [
        "queries.parquet",
        f"qrels_{'validation' if task['split'] == 'dev' else task['split']}.parquet",
        "corpus.parquet",
    ]
    if recorded is not None:
        if [row["repo_path"] for row in recorded] != names:
            raise ValueError("Replay upstream file roles differ")
        for row in recorded:
            verify_file(row["path"], row)
        return recorded
    root = Path(
        snapshot_download(
            task["dataset"], revision=task["revision"], repo_type="dataset", local_files_only=True
        )
    )
    remote = HfApi().get_paths_info(
        task["dataset"], paths=names, revision=task["revision"], repo_type="dataset", expand=True
    )
    indexed = {entry.path: entry for entry in remote}
    if set(indexed) != set(names):
        raise ValueError("Pinned upstream files are missing")
    files = []
    for name in names:
        entry = indexed[name]
        # HF snapshots legitimately link to content-addressed cache blobs. Resolve
        # that one upstream cache role, then retain the ordinary-file guard and
        # independently compare its actual bytes with the pinned remote digest.
        actual = binding((root / name).resolve(strict=True))
        if (
            entry.lfs is None
            or actual["sha256"] != entry.lfs.sha256
            or actual["bytes"] != entry.size
        ):
            raise ValueError("Upstream payload disagrees with independent immutable HF metadata")
        files.append({"repo_path": name, **actual, "immutable_hf_lfs_sha256": entry.lfs.sha256})
    return files


def verify_task(task, rows, ledger, files, seed):
    queries = {}
    for row in pq.read_table(files[0]["path"], columns=["_id", "text"]).to_pylist():
        key = str(row["_id"])
        if key in queries:
            raise ValueError("Duplicate upstream query identity")
        queries[key] = str(row["text"])
    qrels = {}
    for row in pq.read_table(files[1]["path"]).to_pylist():
        qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    anchors = []
    for query_id, scores in qrels.items():
        if query_id not in queries or not queries[query_id].strip():
            continue
        positives = {doc: score for doc, score in scores.items() if score > 0}
        if not positives:
            continue
        highest = max(positives.values())
        positive = min(
            (doc for doc, score in positives.items() if score == highest),
            key=lambda doc: (selection_rank(seed, task["name"], query_id, doc), doc),
        )
        anchors.append((selection_rank(seed, task["name"], query_id), query_id, positive))
    anchors.sort()
    pool = anchors[: task["candidate_pool_count"]]
    require_same(
        [(row["query_id"], row["positive_id"]) for row in rows],
        [(q, d) for _, q, d in pool[: task["query_count"]]],
    )
    pool_ids = {doc for _, _, doc in pool}
    wanted = {doc for row in ledger for doc in [row["positive_id"], *row["negative_ids"]]}
    if not wanted.issubset(pool_ids):
        raise ValueError("A recorded candidate is outside the frozen cross-query positive pool")
    corpus = pq.ParquetFile(files[2]["path"])
    documents = {}
    for group in range(corpus.num_row_groups):
        values = corpus.read_row_group(group, columns=["_id"], use_threads=False)["_id"].to_pylist()
        positions = [i for i, value in enumerate(values) if str(value) in wanted]
        if not positions:
            continue
        for row in corpus.read_row_group(group, use_threads=False).take(positions).to_pylist():
            key = str(row["_id"])
            title, text = str(row.get("title") or "").strip(), str(row.get("text") or "").strip()
            if key in documents:
                raise ValueError("Duplicate upstream document identity")
            documents[key] = f"{title}\n{text}" if title and text else title or text
    if set(documents) != wanted:
        raise ValueError("Missing upstream probe document")
    checked = 0
    for row, record in zip(rows, ledger, strict=True):
        relevant = sorted(doc for doc, score in qrels[row["query_id"]].items() if score > 0)
        require_same(record["relevant_ids"], relevant)
        if row["query"] != queries[row["query_id"]]:
            raise ValueError("Probe query text differs from its immutable source")
        for column in TEXT_COLUMNS[1:]:
            if row[column] != documents[row[column + "_id"]]:
                raise ValueError("Probe document text differs from its immutable ordered source")
        checked += 9
    return {
        "task": task["name"],
        "repo": task["dataset"],
        "revision": task["revision"],
        "files": files,
        "rows": len(rows),
        "verified_text_fields": checked,
        "distinct_used_documents": len(wanted),
        "ordered_ledger_candidates_verified": True,
        "negative_priority_regenerated": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "probe-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--replay-sha256")
    args = parser.parse_args()
    if sys.flags.optimize or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Require a new CPU-only input audit result")
    if (args.replay is None) != (args.replay_sha256 is None):
        raise ValueError("Replay requires an exact trusted receipt hash")
    root = args.repository.resolve()
    spec_path = root / PROBE_SPEC
    if file_identity(spec_path)["sha256"] != PROBE_SPEC_SHA:
        raise ValueError("Frozen probe specification differs")
    spec = read_json(spec_path)
    rows, ledger, identities, inputs = read_probe(args.probe_root, spec)
    source = [binding(Path(__file__)), binding(spec_path)]
    original = handoff()
    previous = None
    if args.replay:
        if file_identity(args.replay)["sha256"] != args.replay_sha256:
            raise ValueError("Replay parent differs")
        previous = read_json(args.replay)
        require_same(previous["source"], source)
        require_same(previous["inputs"], inputs)
    tasks = []
    for index, task in enumerate(spec["tasks"]):
        files = upstream_files(
            task, None if previous is None else previous["tasks"][index]["files"]
        )
        selected = [i for i, row in enumerate(rows) if row["source"] == task["name"]]
        checked = verify_task(
            task, [rows[i] for i in selected], [ledger[i] for i in selected], files, spec["seed"]
        )
        tasks.append(checked)
        print(
            {"verified_task": task["name"], "text_fields": checked["verified_text_fields"]},
            flush=True,
        )
    for row in [*inputs, *source, *(record for task in tasks for record in task["files"])]:
        verify_file(row["path"], row)
    require_same(handoff(), original)
    if previous:
        for key, value in (("tasks", tasks), ("row_identities", identities)):
            require_same(previous[key], value)
    result = {
        "scope": "engineering_pinned_beir_dimension_probe_input_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "input_validation_passed": True,
        "source": source,
        "inputs": inputs,
        "tasks": tasks,
        "row_identities": identities,
        "row_identities_sha256": digest(identities),
        "verified_text_fields": sum(task["verified_text_fields"] for task in tasks),
        "negative_priority_regenerated": False,
        "boundary": "Authenticate every frozen ordered candidate and its relevance/pool membership, all original query/document texts, and query/positive seeded priority. Do not redraw or claim regeneration of the frozen lexical-negative ranking. This is input evidence, not vector encoding or a primary model result.",
        "fresh_replay": previous is not None,
        "replay_parent": binding(args.replay) if args.replay else None,
        "scientific_completion": False,
        "model_updates": 0,
        "gpu_workers": 0,
        "post_execution_dispatchers": original,
    }
    write_new(args.output, result)
    print(
        {
            "input_validation_passed": True,
            "texts": result["verified_text_fields"],
            **file_identity(args.output),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
