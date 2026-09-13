"""Prepare, never deploy, the authenticated 6/52-group Dense data revision."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import HfApi

from embed_optim.data import SOURCE_REPO, SOURCE_REVISION, SPLITS, _candidate_ranks, _seed_for
from embed_optim.primary_contract import digest, file_identity, read_json, verify_file
from scripts.audit_dense_natural_data import write_new
from scripts.audit_dense_text_integrity import content_audit, manifest_linkage
from scripts.prepare_dense_data_protection import query_digest

PARTITION_AUDIT_SHA = "ef667f2d8c3e5b860c55dbe7b08cdc2f6977214d37abd5dd40fa60855d85c730"
PROPOSAL_SHA = "f05c32c367f2adc531eb90aa89b4a96a6b3b6af0ee0766729e38036eb8181d76"


def mine_record(record, seed, split, query_id):
    """The original first-eligible-score rule, before any text eligibility test."""
    ids, scores = record["document_ids"], record["scores"]
    if not ids or not scores or len(ids) != len(scores):
        return None
    pool = [i for i, score in enumerate(scores[1:], 1) if score < 0.95 * scores[0]][:10]
    if len(pool) < 10:
        return None
    rng = np.random.default_rng(_seed_for(seed, split, query_id, "negatives"))
    choices = np.sort(rng.choice(10, size=7, replace=False)).tolist()
    return {
        "query_id": query_id,
        "positive_id": int(ids[0]),
        "negative_ids": [int(ids[pool[i]]) for i in choices],
        "negative_pool_indices": choices,
    }


def group_row(sample_id, split, query, mined, documents):
    ids = [mined["positive_id"], *mined["negative_ids"]]
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
        "source": split,
        "query_id": mined["query_id"],
        "positive_id": mined["positive_id"],
        "query": query,
        "positive": documents[mined["positive_id"]],
        "length": max(map(len, texts)),
    }
    for n, doc_id in enumerate(mined["negative_ids"]):
        row[f"negative_{n}"] = documents[doc_id]
        row[f"negative_{n}_id"] = doc_id
    return row, None


def fixed_priority(scorable, original_training_ids, partition, split, manifest):
    """Rebuild the old population; NEVER remove newly protected IDs before shuffling."""
    population = scorable
    if partition == "validation":
        population = scorable[~np.isin(scorable, sorted(original_training_ids))]
    count_field = (
        "scorable_query_counts" if partition == "training" else "available_disjoint_query_counts"
    )
    if len(population) != manifest[count_field][split]:
        raise ValueError("The original candidate population changed")
    return _candidate_ranks(
        population,
        manifest["quotas"][split],
        manifest["seed"],
        split,
        manifest["candidate_margin"],
    )


class PinnedSource:
    """CPU-only authenticated parquet access; preserve physical score-record order."""

    def __init__(self, snapshot, split):
        self.split, self.snapshot = split, snapshot
        self.paths = {
            kind: sorted((snapshot / kind).glob(f"{split}-*.parquet"))
            for kind in ("queries", "scores", "documents")
        }
        if not all(self.paths.values()):
            raise ValueError("The pinned source snapshot is incomplete")
        paths = [p for values in self.paths.values() for p in values]
        names = [str(p.relative_to(snapshot)) for p in paths]
        entries = HfApi().get_paths_info(
            SOURCE_REPO, paths=names, revision=SOURCE_REVISION, repo_type="dataset", expand=True
        )
        if sorted(e.path for e in entries) != sorted(names):
            raise ValueError("Independent source file listing differs")

        def authenticate(entry):
            path = snapshot / entry.path
            actual = file_identity(path.resolve())
            if entry.lfs is None or actual != {"bytes": entry.size, "sha256": entry.lfs.sha256}:
                raise ValueError("An upstream source file differs from its immutable HF digest")
            return {
                "path": str(path.resolve()),
                "snapshot_path": str(path),
                "repo_path": entry.path,
                **actual,
                "independent_hf_lfs_sha256": entry.lfs.sha256,
            }

        with ThreadPoolExecutor(max_workers=3) as pool:
            self.files = list(pool.map(authenticate, entries))
        query_table = pq.read_table(self.paths["queries"], columns=["query_id", "query"])
        self.queries = dict(
            zip(query_table["query_id"].to_pylist(), query_table["query"].to_pylist())
        )
        if len(self.queries) != len(query_table):
            raise ValueError("Source query IDs are not unique")
        self.parquets = {p: pq.ParquetFile(p, memory_map=True) for p in paths}
        self.score_locations = defaultdict(list)
        for path in self.paths["scores"]:
            parquet = self.parquets[path]
            for rg in range(parquet.num_row_groups):
                ids = parquet.read_row_group(rg, columns=["query_id"])["query_id"].to_pylist()
                for offset, query_id in enumerate(ids):
                    self.score_locations[query_id].append((path, rg, offset))
        self.scorable = np.intersect1d(list(self.queries), list(self.score_locations))
        self.cache = OrderedDict()

    def score_record(self, path, rg, offset):
        key = (path, rg)
        if key not in self.cache:
            self.cache[key] = self.parquets[path].read_row_group(
                rg, columns=["query_id", "document_ids", "scores"]
            )
        self.cache.move_to_end(key)
        # Preparation-only bounded CPU cache. The largest declared source has 60
        # score row groups; retain them during the original randomized traversal.
        # Eviction changes IO only, never candidate priority or eligibility.
        while len(self.cache) > 64:
            self.cache.popitem(last=False)
        table = self.cache[key]
        return {name: table[name][offset].as_py() for name in table.column_names}

    def first_mined(self, query_id, seed):
        for path, rg, offset in self.score_locations[query_id]:
            record = self.score_record(path, rg, offset)
            if record["query_id"] != query_id:
                raise ValueError("Indexed score record differs from its query")
            mined = mine_record(record, seed, self.split, query_id)
            if mined is not None:
                return mined, {
                    "repo_path": str(path.relative_to(self.snapshot)),
                    "row_group": rg,
                    "offset": offset,
                    "record_sha256": digest(record),
                }
        return None, None

    def documents(self, wanted):
        wanted, result = sorted(set(wanted)), {}
        for path in self.paths["documents"]:
            parquet = self.parquets[path]
            column = parquet.schema_arrow.get_field_index("document_id")
            for rg in range(parquet.num_row_groups):
                stats = parquet.metadata.row_group(rg).column(column).statistics
                if stats is not None and stats.has_min_max:
                    if not any(stats.min <= i <= stats.max for i in wanted):
                        continue
                table = parquet.read_row_group(rg, columns=["document_id", "document"])
                selected = table.filter(pc.is_in(table["document_id"], value_set=pa.array(wanted)))
                for row in selected.to_pylist():
                    if row["document_id"] in result:
                        raise ValueError("Duplicate document ID in pinned source")
                    result[row["document_id"]] = row["document"]
        return result


def revised_table(original, replacements):
    """Retain every non-target Arrow value; no tokenization/cleaning of old rows."""
    if original._indices is not None:
        raise ValueError("Original parent must be the directly materialized row order")
    table, pieces, cursor = original.data.table, [], 0
    for index, row in sorted(replacements.items()):
        if not 0 <= index < len(table) or row["sample_id"] != index:
            raise ValueError("A replacement is outside its declared position")
        if set(row) != set(table.column_names):
            raise ValueError("A replacement does not retain the full parent schema")
        if original[index]["source"] != row["source"]:
            raise ValueError("A replacement changes its source quota")
        pieces.extend(
            [table.slice(cursor, index - cursor), pa.Table.from_pylist([row], schema=table.schema)]
        )
        cursor = index + 1
    pieces.append(table.slice(cursor))
    return pa.concat_tables(pieces)


def select_replacements(parents, manifests, proposal, sources, protected_beir, decisions):
    reserved_ids, old_train_ids, reserved_texts = set(), defaultdict(set), set(protected_beir)
    for partition, dataset in parents.items():
        for row in dataset.select_columns(["source", "query_id", "query"]):
            reserved_ids.add((row["source"], row["query_id"]))
            reserved_texts.add(query_digest(row["query"]))
            if partition == "training":
                old_train_ids[row["source"]].add(row["query_id"])
    selected, traversals = {p: {} for p in parents}, []
    for partition in ("training", "validation"):
        manifest = manifests[partition]
        for split in SPLITS:
            targets = sorted(
                r["sample_id"]
                for r in proposal["partitions"][partition]["rows"]
                if r["source"] == split
            )
            if not targets:
                continue
            source, accepted, counts = sources[split], 0, Counter()
            ranks = fixed_priority(
                source.scorable, old_train_ids[split], partition, split, manifest
            )
            for query_id, rank in ranks.items():
                decision = {
                    "partition": partition,
                    "source": split,
                    "query_id": query_id,
                    "rank": rank,
                }
                reason = None
                if (split, query_id) in reserved_ids:
                    reason = "reserved_query_id"
                else:
                    try:
                        key = query_digest(source.queries[query_id])
                    except ValueError:
                        reason = "empty_or_missing_query"
                    if reason is None and key in reserved_texts:
                        reason = "reserved_query_text"
                if reason is None:
                    mined, origin = source.first_mined(query_id, manifest["seed"])
                    decision["first_mined"] = mined
                    decision["score_origin"] = origin
                    if mined is None:
                        reason = "no_original_eligible_score_record"
                    else:
                        documents = source.documents([mined["positive_id"], *mined["negative_ids"]])
                        row, reason = group_row(
                            targets[accepted], split, source.queries[query_id], mined, documents
                        )
                        if reason is None:
                            index = targets[accepted]
                            selected[partition][index] = {
                                "row": row,
                                "ledger": {"sample_id": index, "source": split, **mined},
                                "original_row_sha256": digest(parents[partition][index]),
                                "replacement_row_sha256": digest(row),
                                "rank": rank,
                                "score_origin": origin,
                            }
                            decision["sample_id"] = index
                            reserved_ids.add((split, query_id))
                            reserved_texts.add(key)
                            accepted += 1
                decision["status"] = reason or "accepted"
                counts[decision["status"]] += 1
                decisions.write(json.dumps(decision, sort_keys=True) + "\n")
                if accepted == len(targets):
                    break
            decisions.flush()
            if accepted != len(targets):
                raise ValueError(
                    "The original candidate priority window cannot fill the declared quota"
                )
            traversals.append(
                {
                    "partition": partition,
                    "source": split,
                    "decisions": dict(counts),
                    "last_rank": rank,
                }
            )
            print(json.dumps(traversals[-1]), flush=True)
    return selected, traversals


def changed_positions(before, after):
    if before.schema != after.schema or len(before) != len(after):
        raise ValueError("A revision changed the parent schema or total count")
    changed = pa.chunked_array([pa.array([False] * len(before))])
    for column in before.column_names:
        left, right = before[column], after[column]
        unequal = pc.or_(
            pc.fill_null(pc.invert(pc.equal(left, right)), False),
            pc.not_equal(pc.is_null(left), pc.is_null(right)),
        )
        changed = pc.or_(changed, unequal)
    return pc.indices_nonzero(changed).to_pylist()


def write_partition(partition, parent_root, original, selected, output, amendment, training_output):
    output.mkdir()
    replacements = {i: r["row"] for i, r in selected.items()}
    table = revised_table(original, replacements)
    if changed_positions(original.data.table, table) != sorted(selected):
        raise ValueError("The actual changed-row set differs from the explicit amendment")
    logical = digest(
        {"parent": file_identity(parent_root / "manifest.json"), "replacements": replacements}
    )
    dataset = Dataset(table, info=copy.deepcopy(original.info), fingerprint=logical[:16])
    dataset.save_to_disk(str(output / "dataset"), num_shards=14, num_proc=1)
    reloaded = Dataset.load_from_disk(str(output / "dataset"))
    if changed_positions(original.data.table, reloaded.data.table) != sorted(selected):
        raise ValueError("Serialized data no longer retains the declared changed-row set")
    row_sha = hashlib.sha256()
    count = 0
    with (
        (parent_root / "rows.jsonl").open("rb") as source,
        (output / "rows.jsonl").open("xb") as dest,
    ):
        for index, line in enumerate(source):
            record = json.loads(line)
            if record["sample_id"] != index:
                raise ValueError("The original row ledger no longer has its declared order")
            if index in selected:
                record = selected[index]["ledger"]
                line = (json.dumps(record, sort_keys=True) + "\n").encode()
            dest.write(line)
            row_sha.update(
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            )
            count += 1
    if count != len(original):
        raise ValueError("Row ledger count differs from the materialized parent")
    manifest = read_json(parent_root / "manifest.json")
    manifest.update(
        {
            "scope": "prepared_dense_partition_revision_v2",
            "execution_authorized": False,
            "parent_manifest": {
                "path": str(parent_root / "manifest.json"),
                **file_identity(parent_root / "manifest.json"),
            },
            "amendment_sha256": amendment,
            "row_manifest_sha256": row_sha.hexdigest(),
            "materialized_dataset_fingerprint": dataset._fingerprint,
            "dataset_fingerprint": reloaded._fingerprint,
            "dataset_files": [
                {"path": str(p.relative_to(output)), **file_identity(p)}
                for p in sorted((output / "dataset").iterdir())
                if p.is_file()
            ],
            "replaced_sample_ids": sorted(selected),
        }
    )
    if partition == "validation":
        with (
            (output / "rows.jsonl").open("rb") as source,
            (output / "selection.jsonl").open("xb") as dest,
        ):
            for chunk in iter(lambda: source.read(1 << 20), b""):
                dest.write(chunk)
        training_manifest = read_json(training_output / "manifest.json")
        ids_sha, samples_sha = hashlib.sha256(), hashlib.sha256()
        for row in reloaded.select_columns(["source", "query_id", "sample_id"]):
            ids_sha.update(f"{row['source']}:{row['query_id']}\n".encode())
            samples_sha.update(f"{row['sample_id']}\n".encode())
        manifest.update(
            {
                "training_manifest_sha256": file_identity(training_output / "manifest.json")[
                    "sha256"
                ],
                "training_row_ledger_sha256": file_identity(training_output / "rows.jsonl")[
                    "sha256"
                ],
                "training_row_manifest_sha256": training_manifest["row_manifest_sha256"],
                "selected_source_query_ids_sha256": ids_sha.hexdigest(),
                "selected_sample_ids_sha256": samples_sha.hexdigest(),
                "selection_sha256": file_identity(output / "selection.jsonl")["sha256"],
                "serialized_probe_dataset_fingerprint": reloaded._fingerprint,
            }
        )
    content, linkage = (
        content_audit(reloaded.data.table),
        manifest_linkage(reloaded, output / "rows.jsonl"),
    )
    if content["affected_rows_union"] or not linkage["every_materialized_id_matches_manifest"]:
        raise ValueError("The revised data failed full content/ID admission")
    if linkage["source_counts"] != manifest["quotas"]:
        raise ValueError("The revision changed source quotas")
    write_new(output / "manifest.json", manifest)
    return {
        "rows": len(reloaded),
        "changed_sample_ids": sorted(selected),
        "every_non_target_value_preserved": True,
        "content": content,
        "linkage": linkage,
        "files": [
            {"path": str(p), **file_identity(p)} for p in sorted(output.rglob("*")) if p.is_file()
        ],
    }


def partition_isolation(training, validation, beir):
    ids, texts = {}, {}
    for name, dataset in (("training", training), ("validation", validation)):
        records = list(dataset.select_columns(["source", "query_id", "query"]))
        ids[name] = {(r["source"], r["query_id"]) for r in records}
        texts[name] = [query_digest(r["query"]) for r in records]
        if len(ids[name]) != len(dataset):
            raise ValueError("A revised partition repeats a source-qualified query ID")
    result = {
        "training_validation_query_id_overlap": len(ids["training"] & ids["validation"]),
        "training_validation_query_text_overlap": len(
            set(texts["training"]) & set(texts["validation"])
        ),
        "training_beir_query_text_overlap": len(set(texts["training"]) & beir),
        "validation_beir_query_text_overlap": len(set(texts["validation"]) & beir),
        "validation_internal_duplicate_query_texts": len(texts["validation"])
        - len(set(texts["validation"])),
    }
    if any(result.values()):
        raise ValueError(f"Revised query partitions are not isolated: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or not output.is_dir() or any(output.iterdir()):
        raise ValueError("Require CPU-only execution and a new empty explicit output directory")
    if args.output.is_symlink() or args.snapshot.name != SOURCE_REVISION:
        raise ValueError("Require an ordinary output root and the pinned source snapshot")
    own_stat = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    write_new(
        output / "worker.json",
        {
            "pid": os.getpid(),
            "start_time_ticks": int(own_stat[19]),
            "source": {"path": str(Path(__file__).resolve()), **file_identity(Path(__file__))},
            "output": str(output),
            "scope": "owned_cpu_data_preparation_not_training",
        },
    )
    archive = args.repository / "reports/engineering-archive/dense-data-partition-v1"
    if file_identity(archive / "validation.json")["sha256"] != PARTITION_AUDIT_SHA:
        raise ValueError("The deciding partition audit changed")
    if file_identity(archive / "revision-proposal.json")["sha256"] != PROPOSAL_SHA:
        raise ValueError("The prospective data revision changed")
    proposal, audit = (
        read_json(archive / "revision-proposal.json"),
        read_json(archive / "validation.json"),
    )
    for row in audit["external_payload_checks"]:
        verify_file(Path(row["path"]), row)
    protection = read_json(archive / "protected-queries.json")
    if (
        file_identity(archive / "protected-queries.json")["sha256"]
        != "f20353ce500807a7bfe4ddf30169560bfe6ef415b082d2291060102946e37f66"
    ):
        raise ValueError("The protected query set changed")
    beir = {q["normalized_sha256"] for t in protection["beir"] for q in t["queries"]}
    roots = {
        "training": args.experiment_root / "data/denseon-sft-500k-seed42",
        "validation": args.experiment_root / "data/validation-4096-seed20260826",
    }
    authenticated_paths = {r["path"] for r in audit["external_payload_checks"]}
    for root in roots.values():
        required = [root / "manifest.json", root / "rows.jsonl"] + [
            p for p in (root / "dataset").rglob("*") if p.is_file()
        ]
        if any(str(p.resolve()) not in authenticated_paths for p in required):
            raise ValueError("An actual parent input is outside the authenticated audit")
    parents = {p: Dataset.load_from_disk(str(r / "dataset")) for p, r in roots.items()}
    manifests = {p: read_json(r / "manifest.json") for p, r in roots.items()}
    for partition, parent in parents.items():
        for row in proposal["partitions"][partition]["rows"]:
            if digest(parent[row["sample_id"]]) != row["original_row_sha256"]:
                raise ValueError("The actual target differs from the prospective proposal")
    sources = {}
    for split in SPLITS:
        if not any(
            r["source"] == split for p in proposal["partitions"].values() for r in p["rows"]
        ):
            continue
        print(json.dumps({"authenticating_source": split}), flush=True)
        sources[split] = PinnedSource(args.snapshot, split)
        print(
            json.dumps({"authenticated_source": split, "files": len(sources[split].files)}),
            flush=True,
        )
    with (output / "decisions.jsonl").open("x") as stream:
        selected, traversals = select_replacements(
            parents, manifests, proposal, sources, beir, stream
        )
    selection = {
        "scope": "prepared_dense_partition_replacements_v2",
        "execution_authorized": False,
        "proposal_sha256": PROPOSAL_SHA,
        "selected": {p: [rows[i] for i in sorted(rows)] for p, rows in selected.items()},
        "traversals": traversals,
        "source_files": [row for source in sources.values() for row in source.files],
        "decisions": {
            "path": str(output / "decisions.jsonl"),
            **file_identity(output / "decisions.jsonl"),
        },
    }
    write_new(output / "selection.json", selection)
    result = {}
    for partition in ("training", "validation"):
        print(
            json.dumps({"materializing": partition, "replacement_count": len(selected[partition])}),
            flush=True,
        )
        result[partition] = write_partition(
            partition,
            roots[partition],
            parents[partition],
            selected[partition],
            output / partition,
            file_identity(output / "selection.json")["sha256"],
            output / "training",
        )
    isolation = partition_isolation(
        Dataset.load_from_disk(str(output / "training/dataset")),
        Dataset.load_from_disk(str(output / "validation/dataset")),
        beir,
    )
    for row in audit["external_payload_checks"] + selection["source_files"]:
        verify_file(Path(row["path"]), row)
        if "snapshot_path" in row and Path(row["snapshot_path"]).resolve() != Path(row["path"]):
            raise ValueError("The snapshot file target changed while preparing data")
    receipt = {
        "scope": "prepared_dense_partition_materialization_v2",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "scientific_completion": False,
        "execution_authorized": False,
        "original_data_modified": False,
        "training_executed": False,
        "gpu_workers_started": 0,
        "proposal_sha256": PROPOSAL_SHA,
        "partition_audit_sha256": PARTITION_AUDIT_SHA,
        "source": {"path": str(Path(__file__).resolve()), **file_identity(Path(__file__))},
        "selection": {
            "path": str(output / "selection.json"),
            **file_identity(output / "selection.json"),
        },
        "partitions": result,
        "isolation": isolation,
        "independent_replay_pending": True,
    }
    write_new(output / "result.json", receipt)
    print(
        json.dumps(
            {
                "prepared": True,
                "replacements": {p: len(v) for p, v in selected.items()},
                "isolation": isolation,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
