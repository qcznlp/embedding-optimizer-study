from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import Dataset, concatenate_datasets
from huggingface_hub import snapshot_download

from .config import resolve_matrix_path
from .data import SOURCE_REPO, SOURCE_REVISION, SPLITS, _files, _seed_for
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256

REQUIRED_BASE_COLUMNS = (
    "sample_id",
    "source",
    "query_id",
    "positive_id",
    "query",
    "positive",
    "length",
)


def _canonical(row: dict[str, Any]) -> bytes:
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _identity(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root) if root is not None else path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield row


def load_confirmatory_protocol(
    path: str | Path = "configs/confirmatory_protocol.json",
) -> tuple[Path, dict[str, Any]]:
    protocol_path = resolve_matrix_path(path).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source = protocol.get("source", {})
    pool = protocol.get("negative_pool", {})
    data = protocol.get("confirmatory_data", {})
    training = protocol.get("training", {})
    seeds = data.get("seeds")
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported confirmatory protocol schema in {protocol_path}")
    if protocol.get("status") != "prospective_completion_lock":
        raise ValueError("Confirmatory protocol is not prospectively locked")
    if source.get("repo") != SOURCE_REPO or source.get("revision") != SOURCE_REVISION:
        raise ValueError(
            "Confirmatory source repository or revision differs from the pinned source"
        )
    if source.get("queries") != 500_000 or source.get("sources") != len(SPLITS):
        raise ValueError("Confirmatory source size differs from the frozen 500K/seven-source view")
    if not isinstance(seeds, list) or len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Confirmatory protocol requires exactly three distinct seeds")
    if source.get("exploratory_seed") in seeds:
        raise ValueError("The exploratory seed cannot be reused as a confirmatory seed")
    if set(map(str, seeds)) != set(data.get("outputs", {})):
        raise ValueError("Every confirmatory seed must have exactly one output path")
    if pool.get("size") != 10 or data.get("sampled_negatives") != 7:
        raise ValueError("Confirmatory data requires seven-of-ten negative resampling")
    if not all(
        data.get(key) is True
        for key in (
            "fixed_query_ids",
            "fixed_positive_ids_and_text",
            "fixed_source_quotas",
            "negative_sampling_without_replacement",
            "trainer_seed_equals_view_seed",
        )
    ):
        raise ValueError("Confirmatory invariants must all be enabled")
    if training.get("expected_runs") != len(seeds) * training.get("runs_per_seed", -1):
        raise ValueError("Confirmatory run count disagrees with the frozen seed matrix")
    return protocol_path, protocol


def _load_base_rows(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[dict]]]:
    source = protocol["source"]
    root = Path(source["training_data"]).resolve()
    manifest_path = root / "manifest.json"
    ledger_path = root / "rows.jsonl"
    if _sha256(manifest_path) != source["manifest_sha256"]:
        raise ValueError("The 500K source manifest differs from the frozen digest")
    if _sha256(ledger_path) != source["row_ledger_sha256"]:
        raise ValueError("The 500K source row ledger differs from the frozen digest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("source_repo") != source["repo"]
        or manifest.get("source_revision") != source["revision"]
        or manifest.get("seed") != source["exploratory_seed"]
        or manifest.get("total_queries") != source["queries"]
        or manifest.get("negative_pool_size") != protocol["negative_pool"]["size"]
        or manifest.get("sampled_negatives") != protocol["confirmatory_data"]["sampled_negatives"]
    ):
        raise ValueError("The 500K source manifest violates the confirmatory protocol")

    rows_by_split: dict[str, list[dict]] = {split: [] for split in SPLITS}
    digest = hashlib.sha256()
    seen: set[tuple[str, int]] = set()
    for expected_sample_id, row in enumerate(_iter_jsonl(ledger_path)):
        digest.update(_canonical(row))
        source_name = row.get("source")
        query_id = row.get("query_id")
        positive_id = row.get("positive_id")
        negative_ids = row.get("negative_ids")
        pool_indices = row.get("negative_pool_indices")
        key = (source_name, query_id)
        if (
            row.get("sample_id") != expected_sample_id
            or source_name not in rows_by_split
            or not isinstance(query_id, int)
            or not isinstance(positive_id, int)
            or key in seen
            or not isinstance(negative_ids, list)
            or len(negative_ids) != 7
            or len(set(negative_ids)) != 7
            or positive_id in negative_ids
            or not isinstance(pool_indices, list)
            or pool_indices != sorted(set(pool_indices))
            or len(pool_indices) != 7
            or any(not isinstance(index, int) or index < 0 or index >= 10 for index in pool_indices)
        ):
            raise ValueError(f"Invalid frozen source row at sample {expected_sample_id}")
        seen.add(key)
        rows_by_split[source_name].append(row)
    if len(seen) != source["queries"]:
        raise ValueError(f"Expected {source['queries']:,} source rows, found {len(seen):,}")
    if digest.hexdigest() != manifest.get("row_manifest_sha256"):
        raise ValueError("The recomputed source-row digest differs from its manifest")
    observed_quotas = {split: len(rows_by_split[split]) for split in SPLITS}
    if observed_quotas != manifest.get("quotas"):
        raise ValueError("The source ledger does not match its declared quotas")
    return manifest, rows_by_split


def _scan_negative_pools(
    snapshot: Path,
    split: str,
    base_rows: list[dict],
    *,
    exploratory_seed: int,
    threshold: float,
    pool_size: int,
    sampled_negatives: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    targets = {int(row["query_id"]): row for row in base_rows}
    if len(targets) != len(base_rows):
        raise ValueError(f"{split}: duplicate source query IDs")
    found: dict[int, dict[str, Any]] = {}
    for path in _files(snapshot, "scores", split):
        parquet = pq.ParquetFile(path, memory_map=True)
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=["query_id", "document_ids", "scores"],
        ):
            query_ids = batch.column(0).to_numpy(zero_copy_only=False)
            for row_index, raw_query_id in enumerate(query_ids):
                query_id = int(raw_query_id)
                base = targets.get(query_id)
                if base is None:
                    continue
                if query_id in found:
                    # The mined-score source may contain one row per positive for
                    # a query.  The frozen 500K builder accepted the first
                    # eligible row in file order, then skipped later positives.
                    # Reconstruct that exact selection rule.
                    continue
                document_ids = batch.column(1)[row_index].as_py()
                scores = batch.column(2)[row_index].as_py()
                if not document_ids or not scores or len(document_ids) != len(scores):
                    continue
                positive_score = float(scores[0])
                if not math.isfinite(positive_score):
                    continue
                eligible_indices = [
                    index
                    for index, score in enumerate(scores[1:], start=1)
                    if math.isfinite(float(score)) and float(score) < threshold * positive_score
                ][:pool_size]
                if len(eligible_indices) != pool_size:
                    continue
                if int(document_ids[0]) != int(base["positive_id"]):
                    raise ValueError(
                        f"{split}: first eligible positive ID drift for query {query_id}"
                    )
                pool_ids = [int(document_ids[index]) for index in eligible_indices]
                if len(set(pool_ids)) != pool_size or int(base["positive_id"]) in pool_ids:
                    raise ValueError(
                        f"{split}: query {query_id} has duplicate/positive pool entries"
                    )

                expected_indices = sorted(
                    int(index)
                    for index in np.random.default_rng(
                        _seed_for(exploratory_seed, split, query_id, "negatives")
                    ).choice(pool_size, size=sampled_negatives, replace=False)
                )
                expected_ids = [pool_ids[index] for index in expected_indices]
                if (
                    expected_indices != base["negative_pool_indices"]
                    or expected_ids != base["negative_ids"]
                ):
                    raise ValueError(
                        f"{split}: reconstructed seed-{exploratory_seed} negatives differ for "
                        f"query {query_id}"
                    )
                found[query_id] = {
                    "sample_id": int(base["sample_id"]),
                    "source": split,
                    "query_id": query_id,
                    "positive_id": int(base["positive_id"]),
                    "negative_pool_ids": pool_ids,
                }
    missing = set(targets).difference(found)
    if missing:
        raise ValueError(
            f"{split}: missing {len(missing):,} score rows; first={sorted(missing)[:5]}"
        )
    return [found[int(row["query_id"])] for row in base_rows]


def prepare_negative_pool(
    protocol_path: str | Path = "configs/confirmatory_protocol.json",
    *,
    score_batch_size: int = 2_048,
) -> Path:
    resolved_protocol, protocol = load_confirmatory_protocol(protocol_path)
    output = Path(protocol["negative_pool"]["cache"]).resolve()
    if output.exists():
        audit_negative_pool(resolved_protocol)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    base_manifest, rows_by_split = _load_base_rows(protocol)
    snapshot = Path(
        snapshot_download(
            protocol["source"]["repo"],
            repo_type="dataset",
            revision=protocol["source"]["revision"],
        )
    )
    pool = protocol["negative_pool"]
    data = protocol["confirmatory_data"]
    with tempfile.TemporaryDirectory(prefix=".confirmatory-pools-", dir=output.parent) as temp:
        artifact = Path(temp) / "artifact"
        artifact.mkdir()
        ledger_path = artifact / "negative_pools.jsonl"
        digest = hashlib.sha256()
        count = 0
        with ledger_path.open("w", encoding="utf-8") as handle:
            for split in SPLITS:
                print(f"Reconstructing fixed negative pools for {split}", flush=True)
                rows = _scan_negative_pools(
                    snapshot,
                    split,
                    rows_by_split[split],
                    exploratory_seed=protocol["source"]["exploratory_seed"],
                    threshold=float(pool["threshold"]),
                    pool_size=int(pool["size"]),
                    sampled_negatives=int(data["sampled_negatives"]),
                    batch_size=score_batch_size,
                )
                for row in rows:
                    encoded = _canonical(row)
                    digest.update(encoded)
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    count += 1
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "protocol": _identity(resolved_protocol),
            "source": {
                "repo": protocol["source"]["repo"],
                "revision": protocol["source"]["revision"],
                "training_manifest_sha256": protocol["source"]["manifest_sha256"],
                "training_row_ledger_sha256": protocol["source"]["row_ledger_sha256"],
                "training_row_manifest_sha256": base_manifest["row_manifest_sha256"],
            },
            "rows": count,
            "sources": {split: len(rows_by_split[split]) for split in SPLITS},
            "threshold": pool["threshold"],
            "pool_size": pool["size"],
            "negative_pool_ledger": {
                "path": ledger_path.name,
                "bytes": ledger_path.stat().st_size,
                "sha256": _sha256(ledger_path),
                "canonical_rows_sha256": digest.hexdigest(),
            },
        }
        if count != protocol["source"]["queries"]:
            raise ValueError(f"Expected 500,000 pool rows, found {count:,}")
        _atomic_json(artifact / "manifest.json", manifest)
        os.replace(artifact, output)
    audit_negative_pool(resolved_protocol)
    return output


def audit_negative_pool(
    protocol_path: str | Path = "configs/confirmatory_protocol.json",
    *,
    verify_source: bool = False,
    score_batch_size: int = 2_048,
) -> dict[str, Any]:
    resolved_protocol, protocol = load_confirmatory_protocol(protocol_path)
    base_manifest, rows_by_split = _load_base_rows(protocol)
    root = Path(protocol["negative_pool"]["cache"]).resolve()
    manifest_path = root / "manifest.json"
    ledger_path = root / "negative_pools.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sources = {split: len(rows_by_split[split]) for split in SPLITS}
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("protocol", {}).get("sha256") != _sha256(resolved_protocol)
        or manifest.get("rows") != protocol["source"]["queries"]
        or manifest.get("sources") != expected_sources
        or manifest.get("threshold") != protocol["negative_pool"]["threshold"]
        or manifest.get("pool_size") != protocol["negative_pool"]["size"]
        or manifest.get("source", {}).get("training_row_manifest_sha256")
        != base_manifest["row_manifest_sha256"]
        or manifest.get("negative_pool_ledger", {}).get("sha256") != _sha256(ledger_path)
    ):
        raise ValueError("The cached confirmatory negative-pool manifest is inconsistent")

    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    base_rows = itertools.chain.from_iterable(rows_by_split[split] for split in SPLITS)
    observed_rows = _iter_jsonl(ledger_path)
    count = 0
    for count, (base, observed) in enumerate(zip(base_rows, observed_rows, strict=True), start=1):
        digest.update(_canonical(observed))
        pool_ids = observed.get("negative_pool_ids")
        if (
            observed.get("sample_id") != base["sample_id"]
            or observed.get("source") != base["source"]
            or observed.get("query_id") != base["query_id"]
            or observed.get("positive_id") != base["positive_id"]
            or not isinstance(pool_ids, list)
            or len(pool_ids) != protocol["negative_pool"]["size"]
            or len(set(pool_ids)) != len(pool_ids)
            or base["positive_id"] in pool_ids
        ):
            raise ValueError(f"Invalid cached negative pool at sample {base['sample_id']}")
        expected_base = [pool_ids[index] for index in base["negative_pool_indices"]]
        if expected_base != base["negative_ids"]:
            raise ValueError(f"Cached pool no longer reconstructs sample {base['sample_id']}")
        counts[base["source"]] += 1
    if count != protocol["source"]["queries"] or dict(counts) != expected_sources:
        raise ValueError("Cached negative-pool coverage is incomplete")
    if digest.hexdigest() != manifest["negative_pool_ledger"]["canonical_rows_sha256"]:
        raise ValueError("Cached negative-pool canonical digest differs")

    if verify_source:
        snapshot = Path(
            snapshot_download(
                protocol["source"]["repo"],
                repo_type="dataset",
                revision=protocol["source"]["revision"],
            )
        )
        cached_by_split = {split: [] for split in SPLITS}
        for row in _iter_jsonl(ledger_path):
            cached_by_split[row["source"]].append(row)
        for split in SPLITS:
            reconstructed = _scan_negative_pools(
                snapshot,
                split,
                rows_by_split[split],
                exploratory_seed=protocol["source"]["exploratory_seed"],
                threshold=float(protocol["negative_pool"]["threshold"]),
                pool_size=int(protocol["negative_pool"]["size"]),
                sampled_negatives=int(protocol["confirmatory_data"]["sampled_negatives"]),
                batch_size=score_batch_size,
            )
            if reconstructed != cached_by_split[split]:
                raise ValueError(f"{split}: cached pools differ from the pinned source")
    return {
        "path": str(root),
        "manifest_sha256": _sha256(manifest_path),
        "ledger_sha256": _sha256(ledger_path),
        "rows": count,
        "sources": dict(counts),
        "source_rescanned": verify_source,
    }


def _load_document_texts_streaming(
    snapshot: Path,
    split: str,
    required_ids: Iterable[int],
    *,
    batch_size: int = 65_536,
) -> dict[int, str]:
    required = sorted(set(int(value) for value in required_ids))
    value_set = pa.array(required, type=pa.int64())
    result: dict[int, str] = {}
    for path in _files(snapshot, "documents", split):
        parquet = pq.ParquetFile(path, memory_map=True)
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=["document_id", "document"],
        ):
            mask = pc.is_in(batch.column(0), value_set=value_set)
            selected = pa.Table.from_batches([batch]).filter(mask)
            ids = selected["document_id"].to_pylist()
            texts = selected["document"].to_pylist()
            result.update((int(document_id), text) for document_id, text in zip(ids, texts))
    missing = set(required).difference(result)
    if missing:
        raise ValueError(
            f"{split}: missing {len(missing):,} documents; first={sorted(missing)[:5]}"
        )
    return result


def _selected_pool_indices(seed: int, split: str, query_id: int) -> list[int]:
    return sorted(
        int(index)
        for index in np.random.default_rng(_seed_for(seed, split, query_id, "negatives")).choice(
            10, size=7, replace=False
        )
    )


def _dataset_file_identities(dataset_root: Path) -> list[dict[str, Any]]:
    return [
        _identity(path, root=dataset_root.parent)
        for path in sorted(path for path in dataset_root.rglob("*") if path.is_file())
    ]


def prepare_confirmatory_views(
    protocol_path: str | Path = "configs/confirmatory_protocol.json",
) -> list[Path]:
    resolved_protocol, protocol = load_confirmatory_protocol(protocol_path)
    pool_root = prepare_negative_pool(resolved_protocol)
    pool_audit = audit_negative_pool(resolved_protocol)
    base_manifest, rows_by_split = _load_base_rows(protocol)
    data_spec = protocol["confirmatory_data"]
    seeds = [int(seed) for seed in data_spec["seeds"]]
    outputs = {seed: Path(data_spec["outputs"][str(seed)]).resolve() for seed in seeds}
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        if len(existing) != len(outputs):
            raise FileExistsError(
                "Only a subset of confirmatory views exists; refusing to create a mixed receipt: "
                + ", ".join(map(str, existing))
            )
        for seed in seeds:
            audit_confirmatory_view(resolved_protocol, seed)
        return [outputs[seed] for seed in seeds]

    base_root = Path(protocol["source"]["training_data"]).resolve()
    base_dataset = Dataset.load_from_disk(str(base_root / "dataset"))
    required_columns = {
        *REQUIRED_BASE_COLUMNS,
        *(f"negative_{index}" for index in range(7)),
        *(f"negative_{index}_id" for index in range(7)),
    }
    if len(base_dataset) != protocol["source"]["queries"] or not required_columns.issubset(
        base_dataset.column_names
    ):
        raise ValueError("The materialized 500K source dataset is incomplete")

    snapshot = Path(
        snapshot_download(
            protocol["source"]["repo"],
            repo_type="dataset",
            revision=protocol["source"]["revision"],
        )
    )
    pools_by_split = {split: [] for split in SPLITS}
    for row in _iter_jsonl(pool_root / "negative_pools.jsonl"):
        pools_by_split[row["source"]].append(row)

    parent = next(iter(outputs.values())).parent
    if any(path.parent != parent for path in outputs.values()):
        raise ValueError("Confirmatory view outputs must share a parent for atomic preparation")
    parent.mkdir(parents=True, exist_ok=True)
    changed: Counter[tuple[int, int]] = Counter()
    comparisons = [
        *[(protocol["source"]["exploratory_seed"], seed) for seed in seeds],
        *list(itertools.combinations(seeds, 2)),
    ]
    row_digests = {seed: hashlib.sha256() for seed in seeds}
    query_positive_digests = {seed: hashlib.sha256() for seed in seeds}
    with tempfile.TemporaryDirectory(prefix=".confirmatory-views-", dir=parent) as temp:
        temp_root = Path(temp)
        artifacts = {seed: temp_root / f"seed-{seed}" for seed in seeds}
        row_handles = {}
        for seed, artifact in artifacts.items():
            artifact.mkdir()
            (artifact / "parts").mkdir()
            row_handles[seed] = (artifact / "rows.jsonl").open("w", encoding="utf-8")

        try:
            offset = 0
            for split in SPLITS:
                base_rows = rows_by_split[split]
                pool_rows = pools_by_split[split]
                count = len(base_rows)
                if len(pool_rows) != count:
                    raise ValueError(f"{split}: pool/source quota mismatch")
                source_slice = base_dataset.select(range(offset, offset + count))
                offset += count
                base_columns = {name: list(source_slice[name]) for name in REQUIRED_BASE_COLUMNS}
                if (
                    base_columns["sample_id"] != [row["sample_id"] for row in base_rows]
                    or base_columns["source"] != [split] * count
                    or base_columns["query_id"] != [row["query_id"] for row in base_rows]
                    or base_columns["positive_id"] != [row["positive_id"] for row in base_rows]
                ):
                    raise ValueError(f"{split}: source dataset identity differs from its ledger")

                selected: dict[int, list[tuple[list[int], list[int]]]] = {
                    seed: [] for seed in seeds
                }
                required_document_ids: set[int] = set()
                for base, pool_row in zip(base_rows, pool_rows):
                    choices = {
                        protocol["source"]["exploratory_seed"]: (
                            base["negative_pool_indices"],
                            base["negative_ids"],
                        )
                    }
                    pool_ids = pool_row["negative_pool_ids"]
                    for seed in seeds:
                        indices = _selected_pool_indices(seed, split, int(base["query_id"]))
                        negative_ids = [pool_ids[index] for index in indices]
                        selected[seed].append((indices, negative_ids))
                        required_document_ids.update(negative_ids)
                        choices[seed] = (indices, negative_ids)
                    for left, right in comparisons:
                        if choices[left][1] != choices[right][1]:
                            changed[(left, right)] += 1

                print(
                    f"Materializing {split}: {count:,} rows, "
                    f"{len(required_document_ids):,} required documents",
                    flush=True,
                )
                document_texts = _load_document_texts_streaming(
                    snapshot, split, required_document_ids
                )
                for seed in seeds:
                    columns = {name: list(values) for name, values in base_columns.items()}
                    for index in range(7):
                        columns[f"negative_{index}"] = []
                        columns[f"negative_{index}_id"] = []
                    for row_index, (base, (indices, negative_ids)) in enumerate(
                        zip(base_rows, selected[seed])
                    ):
                        for negative_index, negative_id in enumerate(negative_ids):
                            columns[f"negative_{negative_index}_id"].append(negative_id)
                            columns[f"negative_{negative_index}"].append(
                                document_texts[negative_id]
                            )
                        ledger_row = {
                            "sample_id": int(base["sample_id"]),
                            "source": split,
                            "query_id": int(base["query_id"]),
                            "positive_id": int(base["positive_id"]),
                            "negative_ids": negative_ids,
                            "negative_pool_indices": indices,
                        }
                        encoded = _canonical(ledger_row)
                        row_digests[seed].update(encoded)
                        row_handles[seed].write(json.dumps(ledger_row, sort_keys=True) + "\n")
                        identity_row = {
                            "sample_id": int(base["sample_id"]),
                            "source": split,
                            "query_id": int(base["query_id"]),
                            "positive_id": int(base["positive_id"]),
                            "query": base_columns["query"][row_index],
                            "positive": base_columns["positive"][row_index],
                        }
                        query_positive_digests[seed].update(_canonical(identity_row))
                    part = Dataset.from_dict(columns)
                    part.save_to_disk(str(artifacts[seed] / "parts" / split))
                del document_texts, selected, source_slice, base_columns
            if offset != len(base_dataset):
                raise ValueError("Confirmatory materialization did not consume the complete source")
        finally:
            for handle in row_handles.values():
                handle.close()

        total = protocol["source"]["queries"]
        minimum_changed = float(data_spec["minimum_pairwise_changed_negative_group_fraction"])
        changed_fractions = {
            f"{left}_vs_{right}": changed[(left, right)] / total for left, right in comparisons
        }
        if any(value < minimum_changed for value in changed_fractions.values()):
            raise ValueError(f"Confirmatory negative views changed too little: {changed_fractions}")

        for seed in seeds:
            artifact = artifacts[seed]
            parts = [Dataset.load_from_disk(str(artifact / "parts" / split)) for split in SPLITS]
            combined = concatenate_datasets(parts)
            if len(combined) != total:
                raise ValueError(f"Seed {seed}: expected {total:,} rows, found {len(combined):,}")
            dataset_root = artifact / "dataset"
            combined.save_to_disk(str(dataset_root), num_proc=min(32, len(SPLITS) * 2))
            serialized = Dataset.load_from_disk(str(dataset_root))
            ledger_path = artifact / "rows.jsonl"
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "protocol": _identity(resolved_protocol),
                "seed": seed,
                "rows": total,
                "source_repo": protocol["source"]["repo"],
                "source_revision": protocol["source"]["revision"],
                "source_training_manifest_sha256": protocol["source"]["manifest_sha256"],
                "source_training_row_ledger_sha256": protocol["source"]["row_ledger_sha256"],
                "source_training_row_manifest_sha256": base_manifest["row_manifest_sha256"],
                "negative_pool_manifest_sha256": pool_audit["manifest_sha256"],
                "negative_pool_ledger_sha256": pool_audit["ledger_sha256"],
                "negative_pool_size": protocol["negative_pool"]["size"],
                "sampled_negatives": data_spec["sampled_negatives"],
                "quotas": base_manifest["quotas"],
                "row_manifest_sha256": row_digests[seed].hexdigest(),
                "row_ledger_sha256": _sha256(ledger_path),
                "query_positive_identity_sha256": query_positive_digests[seed].hexdigest(),
                "changed_negative_group_fractions": changed_fractions,
                "materialized_dataset_fingerprint": combined._fingerprint,
                "dataset_fingerprint": serialized._fingerprint,
                "dataset_files": _dataset_file_identities(dataset_root),
            }
            _atomic_json(artifact / "manifest.json", manifest)
            # Intermediate split artifacts are inside the temporary parent and are not moved.
            final_artifact = temp_root / f"final-{seed}"
            final_artifact.mkdir()
            os.replace(dataset_root, final_artifact / "dataset")
            os.replace(ledger_path, final_artifact / "rows.jsonl")
            os.replace(artifact / "manifest.json", final_artifact / "manifest.json")
            os.replace(final_artifact, outputs[seed])

    for seed in seeds:
        audit_confirmatory_view(resolved_protocol, seed)
    return [outputs[seed] for seed in seeds]


def _verify_dataset_files(root: Path, expected: list[dict[str, Any]]) -> None:
    observed = _dataset_file_identities(root / "dataset")
    if observed != expected:
        raise ValueError(f"Dataset file identities differ for {root}")


def audit_confirmatory_view(
    protocol_path: str | Path,
    seed: int,
) -> dict[str, Any]:
    resolved_protocol, protocol = load_confirmatory_protocol(protocol_path)
    if seed not in protocol["confirmatory_data"]["seeds"]:
        raise ValueError(f"Seed {seed} is not in the confirmatory protocol")
    base_manifest, rows_by_split = _load_base_rows(protocol)
    pool_audit = audit_negative_pool(resolved_protocol)
    root = Path(protocol["confirmatory_data"]["outputs"][str(seed)]).resolve()
    manifest_path = root / "manifest.json"
    ledger_path = root / "rows.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("protocol", {}).get("sha256") != _sha256(resolved_protocol)
        or manifest.get("seed") != seed
        or manifest.get("rows") != protocol["source"]["queries"]
        or manifest.get("source_training_manifest_sha256") != protocol["source"]["manifest_sha256"]
        or manifest.get("source_training_row_ledger_sha256")
        != protocol["source"]["row_ledger_sha256"]
        or manifest.get("source_training_row_manifest_sha256")
        != base_manifest["row_manifest_sha256"]
        or manifest.get("negative_pool_manifest_sha256") != pool_audit["manifest_sha256"]
        or manifest.get("negative_pool_ledger_sha256") != pool_audit["ledger_sha256"]
        or manifest.get("row_ledger_sha256") != _sha256(ledger_path)
        or manifest.get("quotas") != base_manifest["quotas"]
    ):
        raise ValueError(f"Seed {seed}: confirmatory manifest is inconsistent")
    _verify_dataset_files(root, manifest["dataset_files"])
    dataset = Dataset.load_from_disk(str(root / "dataset"))
    if (
        len(dataset) != protocol["source"]["queries"]
        or dataset._fingerprint != manifest["dataset_fingerprint"]
    ):
        raise ValueError(f"Seed {seed}: dataset row count or fingerprint differs")

    base_root = Path(protocol["source"]["training_data"]).resolve()
    base_dataset = Dataset.load_from_disk(str(base_root / "dataset"))
    pool_rows = _iter_jsonl(Path(protocol["negative_pool"]["cache"]) / "negative_pools.jsonl")
    ledger_rows = _iter_jsonl(ledger_path)
    digest = hashlib.sha256()
    identity_digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    sample_id = 0
    for base_batch, view_batch in zip(
        base_dataset.iter(batch_size=2_048), dataset.iter(batch_size=2_048), strict=True
    ):
        batch_size = len(view_batch["sample_id"])
        for index in range(batch_size):
            try:
                pool_row = next(pool_rows)
                ledger = next(ledger_rows)
            except StopIteration as error:
                raise ValueError(f"Seed {seed}: ledgers ended before the dataset") from error
            source = str(view_batch["source"][index])
            query_id = int(view_batch["query_id"][index])
            positive_id = int(view_batch["positive_id"][index])
            expected_indices = _selected_pool_indices(seed, source, query_id)
            expected_ids = [pool_row["negative_pool_ids"][item] for item in expected_indices]
            observed_ids = [int(view_batch[f"negative_{item}_id"][index]) for item in range(7)]
            if (
                int(view_batch["sample_id"][index]) != sample_id
                or int(base_batch["sample_id"][index]) != sample_id
                or source != str(base_batch["source"][index])
                or query_id != int(base_batch["query_id"][index])
                or positive_id != int(base_batch["positive_id"][index])
                or view_batch["query"][index] != base_batch["query"][index]
                or view_batch["positive"][index] != base_batch["positive"][index]
                or pool_row["sample_id"] != sample_id
                or ledger.get("sample_id") != sample_id
                or ledger.get("negative_pool_indices") != expected_indices
                or ledger.get("negative_ids") != expected_ids
                or observed_ids != expected_ids
                or len(set(observed_ids)) != 7
                or positive_id in observed_ids
            ):
                raise ValueError(f"Seed {seed}: dataset/ledger drift at sample {sample_id}")
            digest.update(_canonical(ledger))
            identity_digest.update(
                _canonical(
                    {
                        "sample_id": sample_id,
                        "source": source,
                        "query_id": query_id,
                        "positive_id": positive_id,
                        "query": view_batch["query"][index],
                        "positive": view_batch["positive"][index],
                    }
                )
            )
            counts[source] += 1
            sample_id += 1
    if sample_id != protocol["source"]["queries"]:
        raise ValueError(f"Seed {seed}: audited only {sample_id:,} rows")
    try:
        next(pool_rows)
        raise ValueError(f"Seed {seed}: pool ledger has extra rows")
    except StopIteration:
        pass
    try:
        next(ledger_rows)
        raise ValueError(f"Seed {seed}: view ledger has extra rows")
    except StopIteration:
        pass
    if (
        digest.hexdigest() != manifest["row_manifest_sha256"]
        or identity_digest.hexdigest() != manifest["query_positive_identity_sha256"]
        or dict(counts) != base_manifest["quotas"]
    ):
        raise ValueError(f"Seed {seed}: recomputed view identities differ")
    minimum = protocol["confirmatory_data"]["minimum_pairwise_changed_negative_group_fraction"]
    if any(value < minimum for value in manifest["changed_negative_group_fractions"].values()):
        raise ValueError(f"Seed {seed}: changed-negative fraction is below the frozen minimum")
    return {
        "seed": seed,
        "path": str(root),
        "manifest_sha256": _sha256(manifest_path),
        "row_ledger_sha256": _sha256(ledger_path),
        "dataset_fingerprint": dataset._fingerprint,
        "rows": sample_id,
        "query_positive_identity_sha256": identity_digest.hexdigest(),
    }


def audit_confirmatory_data(
    protocol_path: str | Path = "configs/confirmatory_protocol.json",
    *,
    verify_source: bool = False,
) -> dict[str, Any]:
    resolved_protocol, protocol = load_confirmatory_protocol(protocol_path)
    pool = audit_negative_pool(resolved_protocol, verify_source=verify_source)
    views = [
        audit_confirmatory_view(resolved_protocol, int(seed))
        for seed in protocol["confirmatory_data"]["seeds"]
    ]
    query_positive = {view["query_positive_identity_sha256"] for view in views}
    if len(query_positive) != 1:
        raise ValueError("Confirmatory views do not share exact query/positive identities")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "protocol": _identity(resolved_protocol),
        "negative_pool": pool,
        "views": views,
        "query_positive_identity_sha256": next(iter(query_positive)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize and audit fixed-query confirmatory negative views"
    )
    parser.add_argument("--protocol", type=Path, default=Path("configs/confirmatory_protocol.json"))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-source", action="store_true")
    parser.add_argument("--score-batch-size", type=int, default=2_048)
    parser.add_argument(
        "--receipt", type=Path, default=Path("reports/confirmatory-data/receipt.json")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.audit_only:
        prepare_negative_pool(args.protocol, score_batch_size=args.score_batch_size)
        prepare_confirmatory_views(args.protocol)
    receipt = audit_confirmatory_data(args.protocol, verify_source=args.verify_source)
    if not args.audit_only:
        _atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
