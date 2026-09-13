"""Independent original-scanner/raw-string reconstruction of prepared replacement data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset

from embed_optim.data import (
    SPLITS,
    _load_document_texts,
    _read_query_table,
    _scan_eligible_candidates,
    _scorable_query_ids,
)
from embed_optim.primary_contract import file_identity, read_json, verify_file
from scripts.audit_dense_natural_data import write_new
from scripts.replay_dense_query_overlap import normalize


def priority(population, manifest, source):
    """Independent reconstruction of the original fixed-priority prefix."""
    seed_bytes = f"{manifest['seed']}:{source}:query-sample".encode()
    seed = int.from_bytes(hashlib.blake2b(seed_bytes, digest_size=8).digest(), "little")
    quota = manifest["quotas"][source]
    count = min(
        len(population), quota + max(10000, math.ceil(quota * manifest["candidate_margin"]))
    )
    order = np.random.default_rng(seed).permutation(len(population))[:count]
    return [int(population[i]) for i in order]


def original_row(sample_id, source, query, candidate, documents):
    ids = [candidate.positive_id, *candidate.negative_ids]
    if len(set(ids)) != 8:
        return None, "repeated_document_id"
    if any(i not in documents for i in ids):
        return None, "missing_document"
    texts = [query, *(documents[i] for i in ids)]
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        return None, "empty_or_missing_text"
    if len(set(texts[1:])) != 8:
        return None, "repeated_document_text"
    row = {
        "sample_id": sample_id,
        "source": source,
        "query_id": candidate.query_id,
        "positive_id": candidate.positive_id,
        "query": query,
        "positive": texts[1],
        "length": max(len(t) for t in texts),
    }
    for i, negative_id in enumerate(candidate.negative_ids):
        row[f"negative_{i}"] = texts[2 + i]
        row[f"negative_{i}_id"] = negative_id
    return row, None


def replay_selection(snapshot, parents, manifests, proposal, selection, decisions, beir):
    reserved_ids, train_ids, protected = set(), defaultdict(set), set(beir)
    for name, dataset in parents.items():
        for row in dataset.select_columns(["source", "query_id", "query"]):
            reserved_ids.add((row["source"], row["query_id"]))
            protected.add(normalize(row["query"]))
            if name == "training":
                train_ids[row["source"]].add(row["query_id"])
    groups, outputs, summaries = defaultdict(list), {p: {} for p in parents}, []
    for row in decisions:
        groups[(row["partition"], row["source"])].append(row)
    selected = {
        p: {r["row"]["sample_id"]: r for r in rows} for p, rows in selection["selected"].items()
    }
    query_tables = {}
    for partition in ("training", "validation"):
        manifest = manifests[partition]
        for source in SPLITS:
            targets = sorted(
                r["sample_id"]
                for r in proposal["partitions"][partition]["rows"]
                if r["source"] == source
            )
            if not targets:
                if groups.get((partition, source)):
                    raise ValueError("Unexpected source traversal")
                continue
            if source not in query_tables:
                table = _read_query_table(snapshot, source)
                raw_ids = table["query_id"].to_numpy()
                query_tables[source] = (
                    dict(zip(table["query_id"].to_pylist(), table["query"].to_pylist())),
                    _scorable_query_ids(snapshot, source, raw_ids),
                )
            queries, population = query_tables[source]
            if partition == "validation":
                population = np.array([i for i in population if int(i) not in train_ids[source]])
            expected_order = priority(population, manifest, source)
            traversal = groups[(partition, source)]
            if [(r["query_id"], r["rank"]) for r in traversal] != list(
                zip(expected_order[: len(traversal)], range(len(traversal)))
            ):
                raise ValueError("Recorded decisions are not the original priority prefix")
            # Compute score requirements ourselves; do not trust a producer's skipped-score label.
            score_ranks = {
                r["query_id"]: r["rank"]
                for r in traversal
                if (source, r["query_id"]) not in reserved_ids
                and isinstance(queries[r["query_id"]], str)
                and normalize(queries[r["query_id"]])
                and normalize(queries[r["query_id"]]) not in protected
            }
            print(
                json.dumps(
                    {"reference_scan": partition, "source": source, "queries": len(score_ranks)}
                ),
                flush=True,
            )
            eligible = _scan_eligible_candidates(
                snapshot, source, score_ranks, manifest["seed"], 0.95, 10, 7, 2048
            )
            wanted = [i for c in eligible.values() for i in (c.positive_id, *c.negative_ids)]
            documents = _load_document_texts(snapshot, source, wanted) if wanted else {}
            accepted, counts = 0, Counter()
            for decision in traversal:
                query_id, status, new = decision["query_id"], None, None
                text = queries[query_id]
                if (source, query_id) in reserved_ids:
                    status = "reserved_query_id"
                elif not isinstance(text, str) or not normalize(text):
                    status = "empty_or_missing_query"
                elif normalize(text) in protected:
                    status = "reserved_query_text"
                elif query_id not in eligible:
                    status = "no_original_eligible_score_record"
                else:
                    if accepted >= len(targets):
                        raise ValueError("Producer continued beyond the fixed replacement quota")
                    index, candidate = targets[accepted], eligible[query_id]
                    new, status = original_row(index, source, text, candidate, documents)
                    expected_mined = {
                        "query_id": query_id,
                        "positive_id": candidate.positive_id,
                        "negative_ids": list(candidate.negative_ids),
                        "negative_pool_indices": list(candidate.negative_pool_indices),
                    }
                    if decision.get("first_mined") != expected_mined:
                        raise ValueError("Actual original scanner differs from prepared mining")
                    if status is None:
                        declared = selected[partition][index]
                        ledger = {"sample_id": index, "source": source, **expected_mined}
                        if (
                            new != declared["row"]
                            or ledger != declared["ledger"]
                            or decision.get("sample_id") != index
                        ):
                            raise ValueError(
                                "Independent source text/mining reconstruction differs"
                            )
                        outputs[partition][index] = {"row": new, "ledger": ledger}
                        reserved_ids.add((source, query_id))
                        protected.add(normalize(text))
                        accepted += 1
                if decision["status"] != (status or "accepted"):
                    raise ValueError("The actual reason for a candidate decision differs")
                counts[decision["status"]] += 1
            if accepted != len(targets) or traversal[-1]["status"] != "accepted":
                raise ValueError("The independently reconstructed prefix does not end at the quota")
            summaries.append(
                {
                    "partition": partition,
                    "source": source,
                    "decisions": dict(counts),
                    "reference_scored_queries": len(score_ranks),
                }
            )
    if len(groups) != len(summaries):
        raise ValueError("Unconsumed traversal records")
    return outputs, summaries


def reconstruct_rows(parent_path, replacements):
    parent = Dataset.load_from_disk(str(Path(parent_path) / "dataset"))
    for batch in parent.iter(batch_size=1024):
        for offset, sample_id in enumerate(batch["sample_id"]):
            yield replacements.get(
                sample_id, {key: values[offset] for key, values in batch.items()}
            )


def materialize_reference(parent_root, replacements, prepared_root, output):
    """A row-generator writer independent of the producer's Arrow slice/splice writer."""
    parent = Dataset.load_from_disk(str(parent_root / "dataset"))
    output.mkdir()
    rows = {i: r["row"] for i, r in replacements.items()}
    reference = Dataset.from_generator(
        reconstruct_rows,
        features=parent.features,
        cache_dir=str(output / "builder-cache"),
        gen_kwargs={"parent_path": str(parent_root), "replacements": rows},
    )
    reference.save_to_disk(str(output / "dataset"), num_shards=14, num_proc=1)
    reference = Dataset.load_from_disk(str(output / "dataset"))
    prepared = Dataset.load_from_disk(str(prepared_root / "dataset"))
    if (
        len(reference) != len(parent)
        or len(prepared) != len(parent)
        or reference.features != parent.features
    ):
        raise ValueError("Independent serialization changed count/schema")
    actual_changes = []
    content_sha = hashlib.sha256()
    for start in range(0, len(parent), 1024):
        end = min(len(parent), start + 1024)
        old, new, other = parent[start:end], prepared[start:end], reference[start:end]
        if new != other:
            raise ValueError("Independent complete dataset values differ")
        for offset in range(end - start):
            row = {key: values[offset] for key, values in new.items()}
            before = {key: values[offset] for key, values in old.items()}
            if row != before:
                actual_changes.append(start + offset)
            content_sha.update(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                + b"\n"
            )
    if actual_changes != sorted(replacements):
        raise ValueError("Independent whole-value audit found an undeclared change")
    ledger_sha = hashlib.sha256()
    with (parent_root / "rows.jsonl").open() as source, (output / "rows.jsonl").open("xb") as dest:
        for index, line in enumerate(source):
            row = replacements[index]["ledger"] if index in replacements else json.loads(line)
            encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            ledger_sha.update(encoded)
            dest.write(encoded)
    if ledger_sha.hexdigest() != read_json(prepared_root / "manifest.json")["row_manifest_sha256"]:
        raise ValueError("Independent canonical row-ledger hash differs")
    return {
        "rows": len(reference),
        "changed_sample_ids": actual_changes,
        "every_prepared_value_matches_independent_materialization": True,
        "every_non_target_value_matches_original": True,
        "canonical_all_fields_sha256": content_sha.hexdigest(),
        "canonical_row_ledger_sha256": ledger_sha.hexdigest(),
        "files": [
            {"path": str(p), **file_identity(p)} for p in sorted(output.rglob("*")) if p.is_file()
        ],
        "boundary": "All row values agree. Different writers/cache paths need not have identical Arrow container bytes or fingerprints; only the prepared parent-bound version is a prospective primary input.",
    }


def full_string_isolation(prepared, beir):
    texts, ids = {}, {}
    for partition in ("training", "validation"):
        dataset = Dataset.load_from_disk(str(prepared / partition / "dataset"))
        rows = list(dataset.select_columns(["source", "query_id", "query"]))
        texts[partition] = [normalize(r["query"]) for r in rows]
        ids[partition] = {(r["source"], r["query_id"]) for r in rows}
        if len(ids[partition]) != len(dataset):
            raise ValueError("Independent check found repeated IDs")
    result = {
        "train_validation_strings": len(set(texts["training"]) & set(texts["validation"])),
        "train_beir_strings": len(set(texts["training"]) & beir),
        "validation_beir_strings": len(set(texts["validation"]) & beir),
        "train_validation_ids": len(ids["training"] & ids["validation"]),
        "validation_internal_duplicate_strings": len(texts["validation"])
        - len(set(texts["validation"])),
    }
    if any(result.values()):
        raise ValueError("Independent full-string partition check failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--producer-sha256", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or not args.output.is_dir()
        or any(args.output.iterdir())
        or args.output.is_symlink()
    ):
        raise ValueError("Require CPU-only reconstruction in a new empty output directory")
    producer_path = args.prepared / "result.json"
    if file_identity(producer_path)["sha256"] != args.producer_sha256:
        raise ValueError("The prepared materialization receipt changed")
    producer = read_json(producer_path)
    verify_file(args.prepared / "selection.json", producer["selection"])
    selection = read_json(args.prepared / "selection.json")
    verify_file(args.prepared / "decisions.jsonl", selection["decisions"])
    with (args.prepared / "decisions.jsonl").open() as handle:
        decisions = [json.loads(line) for line in handle]
    archive = args.repository / "reports/engineering-archive/dense-data-partition-v1"
    for name, expected in (
        ("revision-proposal.json", producer["proposal_sha256"]),
        ("validation.json", producer["partition_audit_sha256"]),
    ):
        if file_identity(archive / name)["sha256"] != expected:
            raise ValueError("The deciding pre-selection audit/proposal changed")
    proposal, audit = (
        read_json(archive / "revision-proposal.json"),
        read_json(archive / "validation.json"),
    )
    original_files = audit["external_payload_checks"]
    prepared_files = [row for p in producer["partitions"].values() for row in p["files"]]
    checked = [*original_files, *prepared_files, *selection["source_files"], producer["source"]]
    for row in checked:
        verify_file(Path(row["path"]), row)
    for row in selection["source_files"]:
        if (args.snapshot / row["repo_path"]).resolve() != Path(row["path"]):
            raise ValueError("The reference scanner would read an unauthenticated snapshot path")
    protection_path = archive / "protected-queries.json"
    if (
        file_identity(protection_path)["sha256"]
        != "f20353ce500807a7bfe4ddf30169560bfe6ef415b082d2291060102946e37f66"
    ):
        raise ValueError("The evaluation query declaration changed")
    beir = set()
    for task in read_json(protection_path)["beir"]:
        files = {row["repo_path"]: row for row in task["files"]}
        for row in files.values():
            verify_file(Path(row["path"]), row)
        qrels = next(row["path"] for name, row in files.items() if name.startswith("qrels_"))
        ids = {str(i) for i in pq.read_table(qrels, columns=["query-id"])["query-id"].to_pylist()}
        for row in pq.read_table(
            files["queries.parquet"]["path"], columns=["_id", "text"]
        ).to_pylist():
            if str(row["_id"]) in ids:
                beir.add(normalize(row["text"]))
    data = Path("/root/embedding-optimizer-study/data")
    roots = {
        "training": data / "denseon-sft-500k-seed42",
        "validation": data / "validation-4096-seed20260826",
    }
    parents = {p: Dataset.load_from_disk(str(root / "dataset")) for p, root in roots.items()}
    manifests = {p: read_json(root / "manifest.json") for p, root in roots.items()}
    replacements, traversals = replay_selection(
        args.snapshot, parents, manifests, proposal, selection, decisions, beir
    )
    partitions = {}
    for partition in ("training", "validation"):
        print(json.dumps({"independent_materialization": partition}), flush=True)
        partitions[partition] = materialize_reference(
            roots[partition],
            replacements[partition],
            args.prepared / partition,
            args.output / partition,
        )
    isolation = full_string_isolation(args.prepared, beir)
    for row in checked:
        verify_file(Path(row["path"]), row)
    receipt = {
        "scope": "independent_dense_partition_reconstruction_v2",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "replay_passed": True,
        "scientific_completion": False,
        "execution_authorized": False,
        "gpu_workers_started": 0,
        "producer": {"path": str(producer_path), **file_identity(producer_path)},
        "source": {"path": str(Path(__file__).resolve()), **file_identity(Path(__file__))},
        "original_scanner": {
            "path": str(args.repository / "src/embed_optim/data.py"),
            **file_identity(args.repository / "src/embed_optim/data.py"),
        },
        "traversals": traversals,
        "partitions": partitions,
        "full_string_isolation": isolation,
        "upstream_and_prepared_payloads_unchanged": checked,
        "boundary": "Independent original hard-negative scanner and full document loader, independently reconstructed priority prefixes, raw normalized strings instead of hash joins, and a row-generator serializer. No semantic near-duplicate claim, formal training authorization, historical result adoption or change to original data.",
    }
    write_new(args.output / "result.json", receipt)
    print(
        json.dumps(
            {
                "replay_passed": True,
                "replacements": {p: len(v) for p, v in replacements.items()},
                "full_string_isolation": isolation,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
