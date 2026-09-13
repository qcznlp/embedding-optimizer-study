"""Read-only held-out query protection before the proposed six-group data revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import HfApi, snapshot_download

from embed_optim.decontamination import DECONTAMINATED_BEIR
from embed_optim.primary_contract import file_identity, verify_file
from embed_optim.validation_data import audit_validation_data
from scripts.audit_dense_natural_data import input_files, write_new


def query_digest(text):
    if not isinstance(text, str):
        raise ValueError("Protected query text must be a string")
    normalized = " ".join(unicodedata.normalize("NFKD", text.lower()).split())
    if not normalized:
        raise ValueError("Protected query is empty")
    return hashlib.sha256(normalized.encode()).hexdigest()


def beir_queries(item):
    task, (repo, revision) = item
    root = Path(
        snapshot_download(repo, revision=revision, repo_type="dataset", local_files_only=True)
    )
    split = "validation" if task == "MSMARCO" else "test"
    names = ["queries.parquet", f"qrels_{split}.parquet"]
    files = []
    remote = HfApi().get_paths_info(
        repo, paths=names, revision=revision, repo_type="dataset", expand=True
    )
    if sorted(e.path for e in remote) != sorted(names):
        raise ValueError("Pinned BEIR query/qrel files are missing")
    for entry in remote:
        local = root / entry.path
        actual = file_identity(local.resolve())
        if (
            entry.lfs is None
            or actual["sha256"] != entry.lfs.sha256
            or actual["bytes"] != entry.size
        ):
            raise ValueError("Pinned BEIR query/qrel digest differs")
        files.append(
            {
                "path": str(local.resolve()),
                "repo_path": entry.path,
                **actual,
                "independent_hf_lfs_sha256": entry.lfs.sha256,
            }
        )
    ids = {
        str(value)
        for value in pq.read_table(root / names[1], columns=["query-id"])["query-id"].to_pylist()
    }
    table = pq.read_table(root / names[0], columns=["_id", "text"])
    selected = [
        {"task": task, "query_id": str(row["_id"]), "normalized_sha256": query_digest(row["text"])}
        for row in table.to_pylist()
        if str(row["_id"]) in ids
    ]
    if {r["query_id"] for r in selected} != ids or len(selected) != len(ids):
        raise ValueError("BEIR evaluation queries are missing or duplicated")
    return {
        "task": task,
        "repo": repo,
        "revision": revision,
        "evaluation_split": "dev" if task == "MSMARCO" else "test",
        "query_count": len(selected),
        "files": files,
        "queries": selected,
    }


def overlaps(dataset, protected):
    seen = []
    for batch in dataset.select_columns(["sample_id", "source", "query_id", "query"]).iter(
        batch_size=2048
    ):
        for index, query in enumerate(batch["query"]):
            key = query_digest(query)
            if key in protected:
                seen.append(
                    {
                        "sample_id": batch["sample_id"][index],
                        "source": batch["source"][index],
                        "query_id": batch["query_id"][index],
                        "normalized_sha256": key,
                        "held_out_matches": protected[key],
                    }
                )
    return seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Require CPU-only inspection and a fresh output receipt")
    inputs, original_files = input_files(args.repository)
    validation_root = args.experiment_root / "data/validation-4096-seed20260826"
    validation_audit = audit_validation_data(
        validation_root,
        args.experiment_root / "data/denseon-sft-500k-seed42",
        spec_path=args.repository / "configs/validation_probe.json",
    )
    training = Dataset.load_from_disk(
        str(Path(inputs["data_linkage_audit"]["dataset_path"]) / "dataset")
    )
    validation = Dataset.load_from_disk(str(validation_root / "dataset"))
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = list(pool.map(beir_queries, DECONTAMINATED_BEIR.items()))
    protected = {}
    for task in tasks:
        for row in task["queries"]:
            protected.setdefault(row["normalized_sha256"], []).append(
                {"task": row["task"], "query_id": row["query_id"]}
            )
    training_overlap, validation_overlap = (
        overlaps(training, protected),
        overlaps(validation, protected),
    )
    validation_hashes = {}
    for row in validation.select_columns(["source", "query_id", "query"]):
        validation_hashes.setdefault(query_digest(row["query"]), []).append(
            {"source": row["source"], "query_id": row["query_id"]}
        )
    cross_partition_text_overlap = overlaps(training, validation_hashes)
    for row in original_files:
        verify_file(Path(row["path"]), row)
    result = {
        "scope": "prepared_dense_data_protected_queries",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "scientific_completion": False,
        "replacement_rows_selected": 0,
        "data_modified": False,
        "training_executed": False,
        "normalization": "lowercase, Unicode NFKD, collapsed whitespace, SHA-256; exact normalized query equality only; no near-duplicate or document-overlap claim",
        "beir": tasks,
        "frozen_validation_audit": validation_audit,
        "training_beir_query_overlaps": training_overlap,
        "validation_beir_query_overlaps": validation_overlap,
        "training_validation_text_overlaps": cross_partition_text_overlap,
        "original_inputs_unchanged": original_files,
        "source_bindings": [
            {"path": str(p), **file_identity(p)}
            for p in (
                Path(__file__).resolve(),
                args.repository / "src/embed_optim/decontamination.py",
                args.repository / "src/embed_optim/validation_data.py",
                args.repository / "scripts/audit_dense_natural_data.py",
            )
        ],
        "boundary": "Authenticate all fourteen official evaluation query/qrel files at fixed revisions, using MSMARCO dev/validation and the other thirteen test splits. Only query identity/text and split membership are used, never model outcomes or score-based sample selection. Audit both original partitions before proposing replacements; report rather than silently repair any pre-existing overlap. These protected normalized-query digests supplement, not replace, source-qualified query-ID exclusion.",
    }
    write_new(args.output, result)
    print(
        json.dumps(
            {
                "beir_tasks": len(tasks),
                "beir_query_records": sum(t["query_count"] for t in tasks),
                "training_beir_overlaps": len(training_overlap),
                "validation_beir_overlaps": len(validation_overlap),
                "training_validation_normalized_text_overlaps": len(cross_partition_text_overlap),
                "output": str(args.output),
                **file_identity(args.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
