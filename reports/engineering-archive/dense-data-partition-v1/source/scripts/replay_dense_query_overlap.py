"""Independently recheck protected-query hits with strings, not digest joins."""

import argparse
import hashlib
import json
import os
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset

from embed_optim.primary_contract import file_identity, read_json, verify_file
from scripts.audit_dense_natural_data import write_new


def normalize(value):
    return " ".join(unicodedata.normalize("NFKD", value.lower()).split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--producer-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Require CPU-only replay with a new output path")
    if file_identity(args.producer)["sha256"] != args.producer_sha256:
        raise ValueError("Protected-query producer receipt differs")
    producer = read_json(args.producer)
    for row in producer["original_inputs_unchanged"] + producer["source_bindings"]:
        verify_file(Path(row["path"]), row)
    beir = defaultdict(list)
    for task in producer["beir"]:
        files = {r["repo_path"]: r for r in task["files"]}
        for row in files.values():
            verify_file(Path(row["path"]), row)
        qfile = files["queries.parquet"]["path"]
        rfile = next(r["path"] for name, r in files.items() if name.startswith("qrels_"))
        ids = {str(v) for v in pq.read_table(rfile, columns=["query-id"])["query-id"].to_pylist()}
        for row in pq.read_table(qfile, columns=["_id", "text"]).to_pylist():
            if str(row["_id"]) in ids:
                beir[normalize(row["text"])].append(
                    {"task": task["task"], "query_id": str(row["_id"]), "raw_text": row["text"]}
                )
    root = Path("/root/embedding-optimizer-study/data")
    training = Dataset.load_from_disk(str(root / "denseon-sft-500k-seed42/dataset"))
    validation = Dataset.load_from_disk(str(root / "validation-4096-seed20260826/dataset"))
    validation_by_text = defaultdict(list)
    for row in validation.select_columns(["sample_id", "source", "query_id", "query"]):
        validation_by_text[normalize(row["query"])].append(row)
    hits, test_hits = [], []
    for batch in training.select_columns(["sample_id", "source", "query_id", "query"]).iter(
        batch_size=2048
    ):
        for i, text in enumerate(batch["query"]):
            key = normalize(text)
            identity = {k: batch[k][i] for k in ("sample_id", "source", "query_id")}
            for other in validation_by_text.get(key, []):
                hits.append(
                    {
                        "training": identity,
                        "validation": {k: other[k] for k in ("sample_id", "source", "query_id")},
                        "raw_text_equal": text == other["query"],
                        "normalized_sha256": hashlib.sha256(key.encode()).hexdigest(),
                    }
                )
            for other in beir.get(key, []):
                test_hits.append(
                    {
                        "training": identity,
                        "beir": {k: other[k] for k in ("task", "query_id")},
                        "raw_text_equal": text == other["raw_text"],
                        "normalized_sha256": hashlib.sha256(key.encode()).hexdigest(),
                    }
                )
    expected_pairs = {
        (r["sample_id"], m["source"], m["query_id"])
        for r in producer["training_validation_text_overlaps"]
        for m in r["held_out_matches"]
    }
    if expected_pairs != {
        (r["training"]["sample_id"], r["validation"]["source"], r["validation"]["query_id"])
        for r in hits
    }:
        raise ValueError("Independent string join disagrees on train/validation pairs")
    expected_test = {
        (r["sample_id"], m["task"], m["query_id"])
        for r in producer["training_beir_query_overlaps"]
        for m in r["held_out_matches"]
    }
    if expected_test != {
        (r["training"]["sample_id"], r["beir"]["task"], r["beir"]["query_id"]) for r in test_hits
    }:
        raise ValueError("Independent string join disagrees on train/BEIR pairs")
    duplicates = [
        [{k: row[k] for k in ("sample_id", "source", "query_id")} for row in rows]
        for rows in validation_by_text.values()
        if len(rows) > 1
    ]
    result = {
        "scope": "engineering_independent_query_string_overlap_replay",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "replay_passed": True,
        "scientific_completion": False,
        "training_executed": False,
        "data_modified": False,
        "producer": {"path": str(args.producer), **file_identity(args.producer)},
        "training_validation_pairs": hits,
        "training_beir_pairs": test_hits,
        "validation_internal_normalized_duplicates": duplicates,
        "raw_equal_train_validation_pairs": sum(r["raw_text_equal"] for r in hits),
        "raw_equal_train_beir_pairs": sum(r["raw_text_equal"] for r in test_hits),
        "affected_validation_rows": sorted({r["validation"]["sample_id"] for r in hits}),
        "validation_source_counts": dict(
            Counter(
                source
                for source, _ in {
                    (r["validation"]["source"], r["validation"]["sample_id"]) for r in hits
                }
            )
        ),
        "source": {"path": str(Path(__file__).resolve()), **file_identity(Path(__file__))},
        "boundary": "Independent whole-string equality joins over every actual training query after direct lower/NFKD/whitespace normalization; raw equality reported separately. No hash collision assumption, model outcomes, corpus-document exclusion, semantic near-duplicate claim or automatic data revision.",
    }
    write_new(args.output, result)
    print(
        json.dumps(
            {
                "passed": True,
                "pairs": len(hits),
                "affected_validation_rows": len(result["affected_validation_rows"]),
                "raw_equal_train_validation_pairs": result["raw_equal_train_validation_pairs"],
                "beir_pairs": len(test_hits),
                "raw_equal_beir_pairs": result["raw_equal_train_beir_pairs"],
                "validation_duplicate_groups": len(duplicates),
                "output": str(args.output),
                **file_identity(args.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
