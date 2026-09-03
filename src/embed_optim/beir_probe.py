from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, get_dataset_split_names, load_dataset

from .geometry import _atomic_json, _sha256
from .probes import _replace_directory, _validate_expected, resolve_probe_spec_path

SCHEMA_VERSION = 1
SELECTION_ALGORITHM = "blake2b-128-query-and-positive-v1"
NEGATIVE_SELECTION_ALGORITHM = "cross-query-positive-idf-overlap-v1"
TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class TaskAnchor:
    query_id: str
    query: str
    positive_id: str
    relevant_ids: tuple[str, ...]
    selection_rank: int
    is_output: bool


@dataclass(frozen=True)
class LoadedTask:
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    documents: dict[str, str]
    identity: dict[str, Any]


def _rank(seed: int, *parts: str) -> int:
    payload = ":".join((SELECTION_ALGORITHM, str(seed), *parts)).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "big")


def _sample_id(seed: int, task: str, query_id: str) -> int:
    # Hugging Face Dataset stores this as a signed int64. Keep the top bit clear.
    return _rank(seed, "sample", task, query_id) & ((1 << 63) - 1)


def _positive_document(
    task: str,
    query_id: str,
    relevance: dict[str, int],
    seed: int,
) -> str | None:
    positives = [
        (str(document_id), int(score)) for document_id, score in relevance.items() if score > 0
    ]
    if not positives:
        return None
    best_score = max(score for _, score in positives)
    candidates = [document_id for document_id, score in positives if score == best_score]
    return min(
        candidates, key=lambda document_id: (_rank(seed, task, query_id, document_id), document_id)
    )


def select_task_anchors(
    task: str,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    *,
    output_count: int,
    candidate_pool_count: int,
    seed: int,
) -> list[TaskAnchor]:
    if output_count <= 0:
        raise ValueError(f"output_count must be positive, got {output_count}")
    if candidate_pool_count < max(output_count, 8):
        raise ValueError("candidate_pool_count must cover output rows and at least eight documents")

    candidates: list[tuple[int, str, str]] = []
    for query_id, relevance in qrels.items():
        query_id = str(query_id)
        query = queries.get(query_id)
        if query is None or not str(query).strip():
            continue
        positive_id = _positive_document(task, query_id, relevance, seed)
        if positive_id is None:
            continue
        candidates.append((_rank(seed, task, query_id), query_id, positive_id))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    if len(candidates) < candidate_pool_count:
        raise ValueError(
            f"{task} has {len(candidates)} eligible qrels queries, fewer than the requested "
            f"candidate pool of {candidate_pool_count}"
        )

    anchors = []
    for index, (rank, query_id, positive_id) in enumerate(candidates[:candidate_pool_count]):
        relevant_ids = tuple(
            sorted(
                str(document_id) for document_id, score in qrels[query_id].items() if int(score) > 0
            )
        )
        anchors.append(
            TaskAnchor(
                query_id=query_id,
                query=str(queries[query_id]),
                positive_id=positive_id,
                relevant_ids=relevant_ids,
                selection_rank=rank,
                is_output=index < output_count,
            )
        )
    return anchors


def _tokens(text: str) -> frozenset[str]:
    return frozenset(token.casefold() for token in TOKEN_PATTERN.findall(text))


def _document_text(row: dict[str, Any]) -> str:
    text = str(row.get("text") or "").strip()
    title = str(row.get("title") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def materialize_task_rows(
    task: str,
    anchors: list[TaskAnchor],
    documents: dict[str, str],
    *,
    seed: int,
    negative_count: int = 7,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if negative_count != 7:
        raise ValueError("The representation analyzer contract requires exactly seven negatives")
    positive_ids = []
    for anchor in anchors:
        if anchor.positive_id not in documents:
            raise ValueError(f"{task} corpus is missing selected document {anchor.positive_id!r}")
        if anchor.positive_id not in positive_ids:
            positive_ids.append(anchor.positive_id)

    document_tokens = {document_id: _tokens(documents[document_id]) for document_id in positive_ids}
    document_frequency: Counter[str] = Counter()
    for tokens in document_tokens.values():
        document_frequency.update(tokens)
    pool_size = len(document_tokens)
    idf = {
        token: math.log((pool_size + 1) / (frequency + 1)) + 1.0
        for token, frequency in document_frequency.items()
    }

    rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for anchor in (item for item in anchors if item.is_output):
        relevant = set(anchor.relevant_ids)
        query_tokens = _tokens(anchor.query)
        ranked_negatives = []
        for document_id in positive_ids:
            if document_id in relevant:
                continue
            overlap_score = sum(idf[token] for token in query_tokens & document_tokens[document_id])
            tie_rank = _rank(seed, "negative", task, anchor.query_id, document_id)
            ranked_negatives.append((-overlap_score, tie_rank, document_id, overlap_score))
        ranked_negatives.sort(key=lambda item: (item[0], item[1], item[2]))
        if len(ranked_negatives) < negative_count:
            raise ValueError(
                f"{task}/{anchor.query_id} has only {len(ranked_negatives)} non-relevant "
                f"cross-query positives; expected {negative_count}"
            )
        chosen = ranked_negatives[:negative_count]
        sample_id = _sample_id(seed, task, anchor.query_id)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "source": task,
            "query_id": anchor.query_id,
            "positive_id": anchor.positive_id,
            "query": anchor.query,
            "positive": documents[anchor.positive_id],
        }
        for index, (_, _, document_id, _) in enumerate(chosen):
            row[f"negative_{index}_id"] = document_id
            row[f"negative_{index}"] = documents[document_id]
        row["length"] = max(
            len(row["query"]),
            len(row["positive"]),
            *(len(row[f"negative_{index}"]) for index in range(negative_count)),
        )
        rows.append(row)
        ledger.append(
            {
                "candidate_pool_count": len(anchors),
                "negative_ids": [document_id for _, _, document_id, _ in chosen],
                "negative_lexical_scores": [f"{score:.12f}" for _, _, _, score in chosen],
                "positive_id": anchor.positive_id,
                "query_id": anchor.query_id,
                "relevant_ids": list(anchor.relevant_ids),
                "sample_id": sample_id,
                "selection_rank_hex": f"{anchor.selection_rank:032x}",
                "source": task,
            }
        )
    return rows, ledger


def _load_config(
    dataset_path: str,
    revision: str,
    config: str,
    requested_split: str,
) -> Dataset:
    splits = get_dataset_split_names(dataset_path, revision=revision, config_name=config)
    if requested_split in splits:
        split = requested_split
    elif len(splits) == 1:
        split = str(splits[0])
    else:
        raise ValueError(
            f"{dataset_path}/{config} does not contain {requested_split!r}; available={splits}"
        )
    return load_dataset(dataset_path, config, revision=revision, split=split)


def _load_task(task_spec: dict[str, Any], seed: int) -> LoadedTask:
    task = str(task_spec["name"])
    dataset_path = str(task_spec["dataset"])
    revision = str(task_spec["revision"])
    split = str(task_spec["split"])
    qrels_config = f"qrels-{'validation' if split == 'dev' else split}"

    qrels_dataset = _load_config(dataset_path, revision, qrels_config, split)
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_dataset:
        query_id = str(row["query-id"])
        qrels.setdefault(query_id, {})[str(row["corpus-id"])] = int(row["score"])

    queries_dataset = _load_config(dataset_path, revision, "queries", split)
    wanted_queries = set(qrels)
    queries = {
        str(row.get("id", row.get("_id"))): str(row["text"])
        for row in queries_dataset
        if str(row.get("id", row.get("_id"))) in wanted_queries
    }
    anchors = select_task_anchors(
        task,
        queries,
        qrels,
        output_count=int(task_spec["query_count"]),
        candidate_pool_count=int(task_spec["candidate_pool_count"]),
        seed=seed,
    )
    wanted_documents = {anchor.positive_id for anchor in anchors}

    corpus_dataset = _load_config(dataset_path, revision, "corpus", split)
    id_column = "id" if "id" in corpus_dataset.column_names else "_id"
    document_indices = {
        str(document_id): index
        for index, document_id in enumerate(corpus_dataset[id_column])
        if str(document_id) in wanted_documents
    }
    missing = wanted_documents.difference(document_indices)
    if missing:
        preview = sorted(missing)[:5]
        raise ValueError(f"{task} corpus is missing {len(missing)} selected documents: {preview}")
    documents = {
        document_id: _document_text(corpus_dataset[index])
        for document_id, index in document_indices.items()
    }
    if any(not text for text in documents.values()):
        raise ValueError(f"{task} contains an empty selected document")
    return LoadedTask(
        queries=queries,
        qrels=qrels,
        documents=documents,
        identity={
            "name": task,
            "dataset": dataset_path,
            "revision": revision,
            "split": split,
            "query_count": int(task_spec["query_count"]),
            "candidate_pool_count": int(task_spec["candidate_pool_count"]),
            "queries_rows": len(queries_dataset),
            "queries_fingerprint": queries_dataset._fingerprint,
            "qrels_rows": len(qrels_dataset),
            "qrels_fingerprint": qrels_dataset._fingerprint,
            "corpus_rows": len(corpus_dataset),
            "corpus_fingerprint": corpus_dataset._fingerprint,
        },
    )


def prepare_beir_probe(
    spec_path: str | Path,
    *,
    output: str | Path | None = None,
    overwrite: bool = False,
    allow_unfrozen: bool = False,
) -> Path:
    spec_path = resolve_probe_spec_path(spec_path).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported BEIR probe specification schema in {spec_path}")
    required = {"output", "seed", "tasks", "expected"}
    missing = required.difference(spec)
    if missing:
        raise ValueError(f"BEIR probe specification is missing fields: {sorted(missing)}")
    expected = spec["expected"]
    if not isinstance(expected, dict):
        raise ValueError("BEIR probe expected values must be an object")
    if not expected and not allow_unfrozen:
        raise ValueError("BEIR probe is not frozen; pass --allow-unfrozen only to derive hashes")
    tasks = spec["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("BEIR probe tasks must be a non-empty list")

    destination = Path(spec["output"] if output is None else output).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} exists; pass --overwrite to replace it")
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        seed = int(spec["seed"])
        rows: list[dict[str, Any]] = []
        ledger: list[dict[str, Any]] = []
        identities = []
        task_counts = {}
        for task_spec in tasks:
            loaded = _load_task(task_spec, seed)
            anchors = select_task_anchors(
                str(task_spec["name"]),
                loaded.queries,
                loaded.qrels,
                output_count=int(task_spec["query_count"]),
                candidate_pool_count=int(task_spec["candidate_pool_count"]),
                seed=seed,
            )
            task_rows, task_ledger = materialize_task_rows(
                str(task_spec["name"]), anchors, loaded.documents, seed=seed
            )
            rows.extend(task_rows)
            ledger.extend(task_ledger)
            identities.append(loaded.identity)
            task_counts[str(task_spec["name"])] = len(task_rows)

        sample_ids = [int(row["sample_id"]) for row in rows]
        if len(sample_ids) != len(set(sample_ids)):
            raise RuntimeError("Stable BEIR sample IDs collided")
        selection_path = temporary / "selection.jsonl"
        with selection_path.open("wb") as handle:
            for record in ledger:
                encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
                handle.write(encoded + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

        dataset = Dataset.from_list(rows)
        dataset.save_to_disk(str(temporary / "dataset"))
        serialized = Dataset.load_from_disk(str(temporary / "dataset"))
        sample_digest = hashlib.sha256()
        for sample_id in serialized["sample_id"]:
            sample_digest.update(f"{int(sample_id)}\n".encode())
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "probe_kind": "unseen-decontaminated-beir",
            "selection_algorithm": SELECTION_ALGORITHM,
            "negative_selection_algorithm": NEGATIVE_SELECTION_ALGORITHM,
            "seed": seed,
            "count": len(serialized),
            "task_counts": task_counts,
            "tasks": identities,
            "selected_sample_ids_sha256": sample_digest.hexdigest(),
            "selection_sha256": _sha256(selection_path),
            "probe_dataset_fingerprint": dataset._fingerprint,
            "serialized_probe_dataset_fingerprint": serialized._fingerprint,
            "columns": list(serialized.column_names),
            "positive_candidate_index": 0,
            "negative_candidates": 7,
        }
        if expected:
            _validate_expected(manifest, expected)
        _atomic_json(temporary / "manifest.json", manifest)
        if expected.get("manifest_sha256"):
            actual = _sha256(temporary / "manifest.json")
            if actual != expected["manifest_sha256"]:
                raise ValueError(
                    "BEIR probe manifest hash mismatch: "
                    f"expected {expected['manifest_sha256']}, got {actual}"
                )
        destination.parent.mkdir(parents=True, exist_ok=True)
        _replace_directory(temporary, destination, overwrite)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"Prepared {len(rows):,} unseen BEIR probe rows at {destination}")
    print(f"Probe manifest SHA256: {_sha256(destination / 'manifest.json')}")
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic cross-task decontaminated-BEIR representation probe"
    )
    parser.add_argument("--spec", type=Path, default=Path("configs/beir_representation_probe.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-unfrozen",
        action="store_true",
        help="Derive the initial manifest hashes; frozen production runs reject an empty expected block",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    prepare_beir_probe(
        args.spec,
        output=args.output,
        overwrite=args.overwrite,
        allow_unfrozen=args.allow_unfrozen,
    )


if __name__ == "__main__":
    main()
