from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import Dataset, concatenate_datasets
from huggingface_hub import snapshot_download

SOURCE_REPO = "lightonai/embeddings-fine-tuning"
SOURCE_REVISION = "1ca463331ed637d25c1058567e932e0d3bad2983"
SPLITS = ("fiqa", "hotpotqa", "msmarco", "nq", "fever", "squadv2", "trivia")


@dataclass(frozen=True)
class Candidate:
    query_id: int
    positive_id: int
    negative_ids: tuple[int, ...]
    negative_pool_indices: tuple[int, ...]


def allocate_quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    """Proportionally allocate ``total`` with a deterministic largest remainder."""
    available = sum(counts.values())
    if total <= 0 or total > available:
        raise ValueError(f"total must be in [1, {available}], got {total}")
    exact = {name: total * count / available for name, count in counts.items()}
    quotas = {name: math.floor(value) for name, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(counts, key=lambda name: (-(exact[name] - quotas[name]), name))
    for name in order[:remaining]:
        quotas[name] += 1
    return quotas


def _seed_for(seed: int, *parts: object) -> int:
    payload = ":".join([str(seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _files(snapshot: Path, config: str, split: str) -> list[Path]:
    files = sorted((snapshot / config).glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {config}/{split}-*.parquet files under {snapshot}")
    return files


def _read_query_table(snapshot: Path, split: str) -> pa.Table:
    return pq.read_table(_files(snapshot, "queries", split), memory_map=True)


def _scorable_query_ids(snapshot: Path, split: str, query_ids: np.ndarray) -> np.ndarray:
    """Sorted query IDs present in both the query and mined-score tables."""

    score_table = pq.read_table(
        _files(snapshot, "scores", split),
        columns=["query_id"],
        memory_map=True,
    )
    score_ids = pc.unique(score_table["query_id"].combine_chunks()).to_numpy(zero_copy_only=False)
    return np.intersect1d(query_ids, score_ids, assume_unique=False)


def _candidate_ranks(
    query_ids: np.ndarray,
    quota: int,
    seed: int,
    split: str,
    margin: float,
) -> dict[int, int]:
    extra = max(10_000, math.ceil(quota * margin))
    candidate_count = min(len(query_ids), quota + extra)
    order = np.random.default_rng(_seed_for(seed, split, "query-sample")).permutation(
        len(query_ids)
    )[:candidate_count]
    return {int(query_ids[index]): rank for rank, index in enumerate(order)}


def _scan_eligible_candidates(
    snapshot: Path,
    split: str,
    ranks: dict[int, int],
    seed: int,
    threshold: float,
    pool_size: int,
    sampled_negatives: int,
    batch_size: int,
) -> dict[int, Candidate]:
    eligible: dict[int, Candidate] = {}
    for path in _files(snapshot, "scores", split):
        parquet = pq.ParquetFile(path, memory_map=True)
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=["query_id", "document_ids", "scores"],
        ):
            query_ids = batch.column(0).to_numpy(zero_copy_only=False)
            for row_index, raw_query_id in enumerate(query_ids):
                query_id = int(raw_query_id)
                if query_id not in ranks or query_id in eligible:
                    continue
                document_ids = batch.column(1)[row_index].as_py()
                scores = batch.column(2)[row_index].as_py()
                if not document_ids or not scores or len(document_ids) != len(scores):
                    continue
                positive_score = scores[0]
                valid_indices: list[int] = []
                for index, score in enumerate(scores[1:], start=1):
                    if score < threshold * positive_score:
                        valid_indices.append(index)
                        if len(valid_indices) == pool_size:
                            break
                if len(valid_indices) < pool_size:
                    continue
                rng = np.random.default_rng(_seed_for(seed, split, query_id, "negatives"))
                pool_choices = np.sort(rng.choice(pool_size, size=sampled_negatives, replace=False))
                chosen_score_indices = [valid_indices[int(index)] for index in pool_choices]
                eligible[query_id] = Candidate(
                    query_id=query_id,
                    positive_id=int(document_ids[0]),
                    negative_ids=tuple(int(document_ids[index]) for index in chosen_score_indices),
                    negative_pool_indices=tuple(int(index) for index in pool_choices),
                )
    return eligible


def _load_document_texts(snapshot: Path, split: str, required_ids: Iterable[int]) -> dict[int, str]:
    required = sorted(set(required_ids))
    table = pq.read_table(
        _files(snapshot, "documents", split),
        columns=["document_id", "document"],
        memory_map=True,
    )
    mask = pc.is_in(table["document_id"], value_set=pa.array(required, type=pa.int64()))
    selected = table.filter(mask)
    ids = selected["document_id"].to_pylist()
    texts = selected["document"].to_pylist()
    result = {int(document_id): text for document_id, text in zip(ids, texts)}
    missing = set(required).difference(result)
    if missing:
        preview = sorted(missing)[:10]
        raise RuntimeError(f"{split}: missing {len(missing)} documents; first ids={preview}")
    return result


def _build_split(
    snapshot: Path,
    split: str,
    scorable_query_ids: np.ndarray,
    quota: int,
    seed: int,
    threshold: float,
    pool_size: int,
    sampled_negatives: int,
    candidate_margin: float,
    score_batch_size: int,
    sample_id_start: int,
) -> tuple[Dataset, list[dict], int]:
    query_table = _read_query_table(snapshot, split)
    query_ids = query_table["query_id"].combine_chunks().to_numpy(zero_copy_only=False)
    query_texts = query_table["query"].to_pylist()
    if len(np.unique(query_ids)) != len(query_ids):
        raise RuntimeError(f"{split}: query_id values are not unique")
    ranks = _candidate_ranks(scorable_query_ids, quota, seed, split, candidate_margin)
    eligible = _scan_eligible_candidates(
        snapshot=snapshot,
        split=split,
        ranks=ranks,
        seed=seed,
        threshold=threshold,
        pool_size=pool_size,
        sampled_negatives=sampled_negatives,
        batch_size=score_batch_size,
    )
    if len(eligible) < quota:
        raise RuntimeError(
            f"{split}: only {len(eligible):,} eligible candidates for quota {quota:,}; "
            "increase --candidate-margin"
        )
    selected = sorted(eligible.values(), key=lambda item: ranks[item.query_id])[:quota]
    query_lookup = {
        int(query_id): text
        for query_id, text in zip(query_ids, query_texts)
        if int(query_id) in eligible
    }
    required_documents = [
        document_id for item in selected for document_id in (item.positive_id, *item.negative_ids)
    ]
    document_lookup = _load_document_texts(snapshot, split, required_documents)

    columns: dict[str, list] = {
        "sample_id": [],
        "source": [],
        "query_id": [],
        "positive_id": [],
        "query": [],
        "positive": [],
        "length": [],
    }
    for index in range(sampled_negatives):
        columns[f"negative_{index}"] = []
        columns[f"negative_{index}_id"] = []
    manifest_rows: list[dict] = []
    for offset, item in enumerate(selected):
        sample_id = sample_id_start + offset
        query = query_lookup[item.query_id]
        positive = document_lookup[item.positive_id]
        negatives = [document_lookup[doc_id] for doc_id in item.negative_ids]
        columns["sample_id"].append(sample_id)
        columns["source"].append(split)
        columns["query_id"].append(item.query_id)
        columns["positive_id"].append(item.positive_id)
        columns["query"].append(query)
        columns["positive"].append(positive)
        columns["length"].append(max(map(len, (query, positive, *negatives))))
        for index, (negative_id, negative) in enumerate(zip(item.negative_ids, negatives)):
            columns[f"negative_{index}"].append(negative)
            columns[f"negative_{index}_id"].append(negative_id)
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "source": split,
                "query_id": item.query_id,
                "positive_id": item.positive_id,
                "negative_ids": list(item.negative_ids),
                "negative_pool_indices": list(item.negative_pool_indices),
            }
        )
    return Dataset.from_dict(columns), manifest_rows, len(query_ids)


def prepare_dataset(
    output: str | Path,
    total_queries: int = 500_000,
    seed: int = 42,
    threshold: float = 0.95,
    pool_size: int = 10,
    sampled_negatives: int = 7,
    candidate_margin: float = 0.5,
    score_batch_size: int = 2_048,
    source_revision: str = SOURCE_REVISION,
    overwrite: bool = False,
) -> Path:
    output = Path(output).resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    snapshot = Path(
        snapshot_download(
            SOURCE_REPO,
            repo_type="dataset",
            revision=source_revision,
        )
    )
    query_tables = {split: _read_query_table(snapshot, split) for split in SPLITS}
    raw_counts = {split: len(table) for split, table in query_tables.items()}
    scorable_ids = {
        split: _scorable_query_ids(
            snapshot,
            split,
            table["query_id"].combine_chunks().to_numpy(zero_copy_only=False),
        )
        for split, table in query_tables.items()
    }
    scorable_counts = {split: len(ids) for split, ids in scorable_ids.items()}
    quotas = allocate_quotas(scorable_counts, total_queries)
    print(f"Source revision: {source_revision}")
    print(f"Raw query counts: {raw_counts}")
    print(f"Scorable query counts: {scorable_counts}")
    print(f"Sampling quotas: {quotas}")

    datasets: list[Dataset] = []
    all_manifest_rows: list[dict] = []
    sample_id = 0
    observed_counts: dict[str, int] = {}
    for split in SPLITS:
        print(f"Preparing {split}: quota={quotas[split]:,}", flush=True)
        dataset, rows, observed = _build_split(
            snapshot=snapshot,
            split=split,
            scorable_query_ids=scorable_ids[split],
            quota=quotas[split],
            seed=seed,
            threshold=threshold,
            pool_size=pool_size,
            sampled_negatives=sampled_negatives,
            candidate_margin=candidate_margin,
            score_batch_size=score_batch_size,
            sample_id_start=sample_id,
        )
        datasets.append(dataset)
        all_manifest_rows.extend(rows)
        observed_counts[split] = observed
        sample_id += len(dataset)

    combined = concatenate_datasets(datasets)
    if len(combined) != total_queries:
        raise AssertionError(f"Expected {total_queries:,} rows, got {len(combined):,}")
    output.mkdir(parents=True)
    dataset_dir = output / "dataset"
    combined.save_to_disk(dataset_dir, num_proc=min(32, len(SPLITS) * 2))
    serialized_fingerprint = Dataset.load_from_disk(str(dataset_dir))._fingerprint

    checksum = hashlib.sha256()
    for row in all_manifest_rows:
        checksum.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        checksum.update(b"\n")
    manifest = {
        "source_repo": SOURCE_REPO,
        "source_revision": source_revision,
        "seed": seed,
        "total_queries": total_queries,
        "raw_query_counts": observed_counts,
        "scorable_query_counts": scorable_counts,
        "quotas": quotas,
        "nv_threshold": threshold,
        "negative_pool_size": pool_size,
        "sampled_negatives": sampled_negatives,
        "candidate_margin": candidate_margin,
        "row_manifest_sha256": checksum.hexdigest(),
        "materialized_dataset_fingerprint": combined._fingerprint,
        "dataset_fingerprint": serialized_fingerprint,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with (output / "rows.jsonl").open("w") as handle:
        for row in all_manifest_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Prepared {len(combined):,} rows at {output}")
    print(f"Manifest SHA256: {manifest['row_manifest_sha256']}")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the shared 500k-query dataset")
    parser.add_argument("--output", default="data/denseon-sft-500k-seed42")
    parser.add_argument("--total-queries", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--sampled-negatives", type=int, default=7)
    parser.add_argument("--candidate-margin", type=float, default=0.5)
    parser.add_argument("--score-batch-size", type=int, default=2_048)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    prepare_dataset(
        output=args.output,
        total_queries=args.total_queries,
        seed=args.seed,
        threshold=args.threshold,
        pool_size=args.pool_size,
        sampled_negatives=args.sampled_negatives,
        candidate_margin=args.candidate_margin,
        score_batch_size=args.score_batch_size,
        source_revision=args.source_revision,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
