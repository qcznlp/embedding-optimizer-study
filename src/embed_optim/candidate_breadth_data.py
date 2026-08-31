from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import snapshot_download

from .data import SOURCE_REPO, SOURCE_REVISION, SPLITS, _files
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256

SELECTION_ALGORITHM = "blake2b-128-per-source-smallest-v1"
SOURCE_RECONSTRUCTION_ALGORITHM = "pinned-score-order-and-document-text-v1"

LEDGER_FIELDS = {
    "query_index",
    "sample_id",
    "source",
    "query_id",
    "positive_id",
    "negative_ids",
    "source_score_file",
    "source_score_row_group",
    "source_score_row_offset",
}
QUERY_FIELDS = {"query_index", "sample_id", "source", "query_id", "query"}
CANDIDATE_FIELDS = {
    "query_index",
    "sample_id",
    "source",
    "query_id",
    "candidate_index",
    "document_id",
    "document",
}


class InsufficientEligibleCandidates(ValueError):
    """A mined-score row cannot reproduce the frozen ten-candidate pool."""


def _canonical_row(record: dict[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _resolve_protocol(path: str | Path, prefix: Path | None = None) -> Path:
    path = Path(path)
    if path.is_file() or path.is_absolute() or path.parent != Path("configs"):
        return path
    prefix = Path(sys.prefix) if prefix is None else prefix
    installed = prefix / "share" / "embedding-optimizer-study" / "configs" / path.name
    return installed if installed.is_file() else path


def load_candidate_breadth_protocol(path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = _resolve_protocol(path).resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "status",
        "frozen_at_utc",
        "purpose",
        "timing",
        "source",
        "query_selection",
        "candidate_construction",
        "evaluation",
        "analysis",
        "claim_boundary",
    }
    if protocol.get("schema_version") != SCHEMA_VERSION or set(protocol) != required:
        raise ValueError(f"Unsupported candidate-breadth protocol: {path}")
    source = protocol["source"]
    selection = protocol["query_selection"]
    construction = protocol["candidate_construction"]
    widths = construction.get("negative_widths")
    if (
        source.get("repo") != SOURCE_REPO
        or source.get("revision") != SOURCE_REVISION
        or source.get("raw_documents_per_query") != 2049
        or source.get("negative_threshold") != 0.95
    ):
        raise ValueError("Candidate-breadth source contract changed")
    if (
        selection.get("count") != 224
        or selection.get("allocation") != "balanced_32_per_source"
        or selection.get("algorithm") != SELECTION_ALGORITHM
    ):
        raise ValueError("Candidate-breadth query-selection contract changed")
    if widths != [7, 10, 32, 128, 512, 2048]:
        raise ValueError("Candidate-breadth widths changed")
    if construction.get("requirements", {}).get("available_unique_negatives_per_query") != 2048:
        raise ValueError("Candidate-breadth uniqueness requirement changed")
    return path, protocol


def _selection_rank(seed: int, source: str, sample_id: int) -> int:
    payload = f"{SELECTION_ALGORITHM}:{seed}:{source}:{sample_id}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "big")


def select_validation_rows(
    rows: Iterable[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in rows]
    if count <= 0 or count % len(SPLITS):
        raise ValueError("Balanced candidate-breadth count must be positive and divisible by 7")
    per_source = count // len(SPLITS)
    grouped: dict[str, list[dict[str, Any]]] = {source: [] for source in SPLITS}
    seen: set[tuple[str, int]] = set()
    for row in rows:
        source = str(row.get("source"))
        key = (source, int(row.get("sample_id", -1)))
        if source not in grouped or key in seen:
            raise ValueError("Validation rows contain an unknown source or duplicate sample")
        seen.add(key)
        grouped[source].append(row)
    selected: list[dict[str, Any]] = []
    for source in SPLITS:
        if len(grouped[source]) < per_source:
            raise ValueError(f"Not enough validation rows for {source}")
        ranked = sorted(
            grouped[source],
            key=lambda row: (
                _selection_rank(seed, source, int(row["sample_id"])),
                int(row["sample_id"]),
            ),
        )[:per_source]
        selected.extend(ranked)
    return sorted(selected, key=lambda row: int(row["sample_id"]))


def nested_candidate_ids(
    raw_document_ids: list[int],
    raw_scores: list[float],
    validation_row: dict[str, Any],
    *,
    threshold: float,
    maximum_width: int,
) -> list[int]:
    """Reconstruct the canonical seven, then extend it in pinned mined order."""

    if (
        len(raw_document_ids) != len(raw_scores)
        or len(raw_document_ids) != maximum_width + 1
        or not raw_scores
    ):
        raise ValueError("Mined-score row has the wrong document or score cardinality")
    positive_id = int(validation_row["positive_id"])
    if int(raw_document_ids[0]) != positive_id:
        raise ValueError("Mined-score positive does not match the validation ledger")
    positive_score = float(raw_scores[0])
    eligible_indices = [
        index
        for index, score in enumerate(raw_scores[1:], start=1)
        if float(score) < threshold * positive_score
    ]
    if len(eligible_indices) < 10:
        raise InsufficientEligibleCandidates(
            "Mined-score row has fewer than ten eligible negatives"
        )
    pool_ids = [int(raw_document_ids[index]) for index in eligible_indices[:10]]
    pool_indices = [int(value) for value in validation_row["negative_pool_indices"]]
    canonical = [int(value) for value in validation_row["negative_ids"]]
    if (
        len(pool_indices) != 7
        or len(set(pool_indices)) != 7
        or any(index < 0 or index >= 10 for index in pool_indices)
        or [pool_ids[index] for index in pool_indices] != canonical
    ):
        raise ValueError("Mined-score row does not reconstruct the canonical seven negatives")

    candidates: list[int] = []
    seen = {positive_id}

    def append_unique(values: Iterable[int]) -> None:
        for raw_value in values:
            value = int(raw_value)
            if value not in seen:
                candidates.append(value)
                seen.add(value)

    append_unique(canonical)
    if len(candidates) != 7:
        raise ValueError("Canonical validation negatives are not seven unique non-positives")
    append_unique(pool_ids)
    if len(candidates) != 10:
        raise ValueError("Reconstructed ten-candidate pool is not unique")
    append_unique(raw_document_ids[1:])
    if len(candidates) != maximum_width:
        raise ValueError(
            f"Expected {maximum_width} unique negatives, reconstructed {len(candidates)}"
        )
    return candidates


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def _source_inputs(
    protocol_path: Path, protocol: dict[str, Any]
) -> tuple[Path, Path, Path, dict[str, Any]]:
    root = protocol_path.parent.parent
    source_spec = protocol["source"]
    validation_root = (root / source_spec["validation_root"]).resolve()
    validation_spec = (root / source_spec["validation_spec"]).resolve()
    validation_manifest = validation_root / "manifest.json"
    validation_rows = validation_root / "rows.jsonl"
    for path, expected in (
        (validation_spec, source_spec["validation_spec_sha256"]),
        (validation_manifest, source_spec["validation_manifest_sha256"]),
        (validation_rows, source_spec["validation_rows_sha256"]),
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Candidate-breadth source binding changed: {path}")
    source_manifest = json.loads(validation_manifest.read_text(encoding="utf-8"))
    declared_files = source_manifest.get("dataset_files")
    source_dataset = validation_root / "dataset"
    observed_files = sorted(path for path in source_dataset.rglob("*") if path.is_file())
    if not isinstance(declared_files, list) or {
        str((validation_root / item.get("path", "")).resolve())
        for item in declared_files
        if isinstance(item, dict)
    } != {str(path.resolve()) for path in observed_files}:
        raise ValueError("Frozen validation Dataset file coverage changed")
    for item in declared_files:
        path = validation_root / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or _sha256(path) != item.get("sha256")
        ):
            raise ValueError(f"Frozen validation Dataset file changed: {path}")
    return validation_root, validation_spec, validation_rows, source_manifest


def _selected_validation_rows(
    protocol_path: Path, protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], Path, Path, dict[str, Any]]:
    validation_root, validation_spec, validation_rows, source_manifest = _source_inputs(
        protocol_path, protocol
    )
    selection = protocol["query_selection"]
    selected = select_validation_rows(
        _read_jsonl(validation_rows),
        count=int(selection["count"]),
        seed=int(selection["seed"]),
    )
    return selected, validation_root, validation_spec, source_manifest


def _indexed_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"query_index": index, **record} for index, record in enumerate(records)]


def _validate_candidate_records(
    records: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    maximum_width: int,
    source_records: list[dict[str, Any]] | None = None,
) -> None:
    selected_by_sample = {int(row["sample_id"]): row for row in selected}
    if len(selected_by_sample) != len(selected):
        raise ValueError("Frozen candidate-breadth selection contains duplicate sample IDs")
    if len(records) != len(selected):
        raise ValueError("Candidate-breadth row ledger changed")
    expected_sources = Counter(str(row["source"]) for row in selected)
    observed_sources: Counter[str] = Counter()
    for index, record in enumerate(records):
        try:
            if set(record) != LEDGER_FIELDS:
                raise ValueError("ledger schema changed")
            sample_id = int(record["sample_id"])
            source_row = selected_by_sample[sample_id]
            source = str(record["source"])
            negatives = [int(value) for value in record["negative_ids"]]
            if (
                int(record["query_index"]) != index
                or source != str(source_row["source"])
                or int(record["query_id"]) != int(source_row["query_id"])
                or int(record["positive_id"]) != int(source_row["positive_id"])
                or negatives[:7] != [int(value) for value in source_row["negative_ids"]]
                or len(negatives) != maximum_width
                or len(set(negatives)) != maximum_width
                or int(record["positive_id"]) in negatives
                or not isinstance(record["source_score_file"], str)
                or Path(record["source_score_file"]).name != record["source_score_file"]
                or int(record["source_score_row_group"]) < 0
                or int(record["source_score_row_offset"]) < 0
            ):
                raise ValueError("ledger identity, provenance, or candidates changed")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid candidate-breadth ledger row {index + 1}") from error
        observed_sources[source] += 1
    if observed_sources != expected_sources:
        raise ValueError("Candidate-breadth source allocation changed")
    if source_records is not None and records != _indexed_records(source_records):
        raise ValueError("Candidate-breadth ledger differs from pinned mined-score reconstruction")


def _candidate_records_for_source(
    snapshot: Path,
    source: str,
    selected: list[dict[str, Any]],
    *,
    threshold: float,
    maximum_width: int,
) -> list[dict[str, Any]]:
    remaining = {int(row["query_id"]): row for row in selected}
    if len(remaining) != len(selected):
        raise ValueError(f"Selected {source} validation queries are not unique")
    records: list[dict[str, Any]] = []
    for path in _files(snapshot, "scores", source):
        parquet = pq.ParquetFile(path, memory_map=True)
        for row_group in range(parquet.metadata.num_row_groups):
            if not remaining:
                break
            statistics = parquet.metadata.row_group(row_group).column(0).statistics
            if (
                statistics
                and statistics.has_min_max
                and not any(
                    int(statistics.min) <= query_id <= int(statistics.max) for query_id in remaining
                )
            ):
                continue
            query_ids = parquet.read_row_group(row_group, columns=["query_id"])[
                "query_id"
            ].to_pylist()
            positions = [
                offset for offset, query_id in enumerate(query_ids) if int(query_id) in remaining
            ]
            if not positions:
                continue
            table = parquet.read_row_group(row_group, columns=["document_ids", "scores"])
            for offset in positions:
                query_id = int(query_ids[offset])
                if query_id not in remaining:
                    continue
                row = remaining[query_id]
                raw_ids = table["document_ids"][offset].as_py()
                raw_scores = table["scores"][offset].as_py()
                try:
                    candidates = nested_candidate_ids(
                        raw_ids,
                        raw_scores,
                        row,
                        threshold=threshold,
                        maximum_width=maximum_width,
                    )
                except InsufficientEligibleCandidates:
                    continue
                records.append(
                    {
                        "sample_id": int(row["sample_id"]),
                        "source": source,
                        "query_id": query_id,
                        "positive_id": int(row["positive_id"]),
                        "negative_ids": candidates,
                        "source_score_file": path.name,
                        "source_score_row_group": row_group,
                        "source_score_row_offset": offset,
                    }
                )
                del remaining[query_id]
        if not remaining:
            break
    if remaining:
        raise RuntimeError(
            f"Could not reconstruct {len(remaining)} selected {source} queries; "
            f"first IDs={sorted(remaining)[:10]}"
        )
    return sorted(records, key=lambda row: int(row["sample_id"]))


def _reconstruct_candidate_records(
    protocol: dict[str, Any],
    selected: list[dict[str, Any]],
    *,
    snapshot: Path | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    source_spec = protocol["source"]
    if snapshot is None:
        snapshot = Path(
            snapshot_download(
                source_spec["repo"],
                repo_type="dataset",
                revision=source_spec["revision"],
            )
        )
    maximum_width = max(
        int(value) for value in protocol["candidate_construction"]["negative_widths"]
    )
    reconstructed: list[dict[str, Any]] = []
    for source in SPLITS:
        source_rows = [row for row in selected if row["source"] == source]
        print(f"Reconstructing {source}: {len(source_rows)} queries", flush=True)
        reconstructed.extend(
            _candidate_records_for_source(
                snapshot,
                source,
                source_rows,
                threshold=float(source_spec["negative_threshold"]),
                maximum_width=maximum_width,
            )
        )
    reconstructed.sort(key=lambda row: int(row["sample_id"]))
    if len(reconstructed) != len(selected):
        raise AssertionError("Candidate reconstruction lost selected queries")
    return snapshot, reconstructed


def _document_texts(snapshot: Path, source: str, required_ids: set[int]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    value_set = pa.array(sorted(required_ids), type=pa.int64())
    for path in _files(snapshot, "documents", source):
        parquet = pq.ParquetFile(path, memory_map=True)
        for batch in parquet.iter_batches(batch_size=65_536, columns=["document_id", "document"]):
            mask = pc.is_in(batch.column(0), value_set=value_set)
            selected = pa.Table.from_batches([batch]).filter(mask)
            for document_id, text in zip(
                selected["document_id"].to_pylist(),
                selected["document"].to_pylist(),
                strict=True,
            ):
                document_id = int(document_id)
                if document_id in lookup and lookup[document_id] != text:
                    raise ValueError(f"Conflicting text for {source} document {document_id}")
                lookup[document_id] = str(text)
    missing = required_ids.difference(lookup)
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} required {source} documents; first={sorted(missing)[:10]}"
        )
    return lookup


def _replace_directory(temporary: Path, output: Path, overwrite: bool) -> None:
    if not output.exists():
        os.replace(temporary, output)
        return
    if not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    backup = output.with_name(f".{output.name}.backup.{os.getpid()}")
    os.replace(output, backup)
    try:
        os.replace(temporary, output)
    except BaseException:
        os.replace(backup, output)
        raise
    shutil.rmtree(backup)


def _file_identities(root: Path, directory: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def _candidate_data_manifest(
    *,
    protocol_path: Path,
    protocol: dict[str, Any],
    validation_spec: Path,
    records: list[dict[str, Any]],
    ledger_path: Path,
    row_manifest_sha256: str,
    query_dataset: Dataset,
    query_files: list[dict[str, Any]],
    candidate_rows: int,
    candidate_fingerprints: dict[str, str],
    candidate_files: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    source_spec = protocol["source"]
    selection = protocol["query_selection"]
    widths = protocol["candidate_construction"]["negative_widths"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "protocol": {
            "path": str(protocol_path),
            "bytes": protocol_path.stat().st_size,
            "sha256": _sha256(protocol_path),
        },
        "source_repo": source_spec["repo"],
        "source_revision": source_spec["revision"],
        "source_reconstruction_algorithm": SOURCE_RECONSTRUCTION_ALGORITHM,
        "validation_spec_sha256": _sha256(validation_spec),
        "validation_manifest_sha256": source_spec["validation_manifest_sha256"],
        "validation_rows_sha256": source_spec["validation_rows_sha256"],
        "selection_algorithm": selection["algorithm"],
        "selection_seed": selection["seed"],
        "queries": len(records),
        "source_counts": dict(sorted(Counter(str(record["source"]) for record in records).items())),
        "negative_widths": widths,
        "maximum_negative_width": max(widths),
        "candidate_rows": candidate_rows,
        "row_manifest_sha256": row_manifest_sha256,
        "rows": {
            "path": "rows.jsonl",
            "bytes": ledger_path.stat().st_size,
            "sha256": _sha256(ledger_path),
        },
        "query_dataset_fingerprint": query_dataset._fingerprint,
        "query_files": query_files,
        "candidate_dataset_fingerprints": candidate_fingerprints,
        "candidate_files": candidate_files,
    }


def prepare_candidate_breadth_data(
    protocol_path: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    protocol_path, protocol = load_candidate_breadth_protocol(protocol_path)
    selected, validation_root, validation_spec, _ = _selected_validation_rows(
        protocol_path, protocol
    )
    widths = protocol["candidate_construction"]["negative_widths"]
    maximum_width = max(int(value) for value in widths)
    snapshot, reconstructed = _reconstruct_candidate_records(protocol, selected)
    selected_by_sample = {int(row["sample_id"]): row for row in selected}
    if len(reconstructed) != len(selected_by_sample):
        raise AssertionError("Candidate reconstruction lost selected queries")

    output = Path(output).resolve()
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        query_rows = []
        ledger_digest = hashlib.sha256()
        ledger_path = temporary / "rows.jsonl"
        with ledger_path.open("wb") as handle:
            for query_index, record in enumerate(reconstructed):
                source_row = selected_by_sample[int(record["sample_id"])]
                full_record = {"query_index": query_index, **record}
                encoded = _canonical_row(full_record)
                handle.write(encoded)
                ledger_digest.update(encoded)
                query_rows.append(
                    {
                        "query_index": query_index,
                        "sample_id": int(record["sample_id"]),
                        "source": str(record["source"]),
                        "query_id": int(record["query_id"]),
                        "query": str(source_row.get("query", "")),
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())

        # Query text is materialized in the frozen Dataset, not its row ledger.
        validation_dataset = Dataset.load_from_disk(str(validation_root / "dataset"))
        text_by_sample = {int(row["sample_id"]): str(row["query"]) for row in validation_dataset}
        for row in query_rows:
            row["query"] = text_by_sample[int(row["sample_id"])]
        query_dir = temporary / "queries"
        Dataset.from_list(query_rows).save_to_disk(str(query_dir))

        candidate_root = temporary / "candidates"
        candidate_root.mkdir()
        candidate_files: dict[str, list[dict[str, Any]]] = {}
        candidate_fingerprints: dict[str, str] = {}
        for source in SPLITS:
            source_records = [row for row in reconstructed if row["source"] == source]
            required = {
                int(document_id)
                for row in source_records
                for document_id in (row["positive_id"], *row["negative_ids"])
            }
            print(f"Loading {source}: {len(required):,} unique documents", flush=True)
            document_lookup = _document_texts(snapshot, source, required)
            materialized = []
            index_by_sample = {
                int(row["sample_id"]): index for index, row in enumerate(reconstructed)
            }
            for row in source_records:
                document_ids = [int(row["positive_id"]), *map(int, row["negative_ids"])]
                for candidate_index, document_id in enumerate(document_ids):
                    materialized.append(
                        {
                            "query_index": index_by_sample[int(row["sample_id"])],
                            "sample_id": int(row["sample_id"]),
                            "source": source,
                            "query_id": int(row["query_id"]),
                            "candidate_index": candidate_index,
                            "document_id": document_id,
                            "document": document_lookup[document_id],
                        }
                    )
            source_dir = candidate_root / source
            Dataset.from_list(materialized).save_to_disk(str(source_dir))
            candidate_files[source] = _file_identities(temporary, source_dir)
            candidate_fingerprints[source] = Dataset.load_from_disk(str(source_dir))._fingerprint

        query_dataset = Dataset.load_from_disk(str(query_dir))
        manifest = _candidate_data_manifest(
            protocol_path=protocol_path,
            protocol=protocol,
            validation_spec=validation_spec,
            records=_indexed_records(reconstructed),
            ledger_path=ledger_path,
            row_manifest_sha256=ledger_digest.hexdigest(),
            query_dataset=query_dataset,
            query_files=_file_identities(temporary, query_dir),
            candidate_rows=len(reconstructed) * (maximum_width + 1),
            candidate_fingerprints=candidate_fingerprints,
            candidate_files=candidate_files,
        )
        _atomic_json(temporary / "manifest.json", manifest)
        _replace_directory(temporary, output, overwrite)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def _verified_file_identities(
    root: Path,
    directory: Path,
    declared: Any,
    *,
    label: str,
) -> list[dict[str, Any]]:
    observed = _file_identities(root, directory)
    if declared != observed:
        raise ValueError(f"Candidate-breadth {label} file coverage or content changed")
    return observed


def audit_candidate_breadth_data(
    protocol_path: str | Path,
    output: str | Path,
    *,
    verify_source: bool = True,
) -> dict[str, Any]:
    protocol_path, protocol = load_candidate_breadth_protocol(protocol_path)
    selected, validation_root, validation_spec, _ = _selected_validation_rows(
        protocol_path, protocol
    )
    output = Path(output).resolve()
    manifest_path = output / "manifest.json"
    ledger_path = output / "rows.jsonl"
    if not manifest_path.is_file() or not ledger_path.is_file():
        raise FileNotFoundError(f"Candidate-breadth data is incomplete under {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_queries = int(protocol["query_selection"]["count"])
    widths = protocol["candidate_construction"]["negative_widths"]
    maximum_width = max(widths)
    digest = hashlib.sha256()
    records = _read_jsonl(ledger_path)
    for record in records:
        digest.update(_canonical_row(record))
    _validate_candidate_records(records, selected, maximum_width=maximum_width)
    source_records = None
    snapshot = None
    if verify_source:
        snapshot, source_records = _reconstruct_candidate_records(protocol, selected)
        _validate_candidate_records(
            records,
            selected,
            maximum_width=maximum_width,
            source_records=source_records,
        )

    query_dir = output / "queries"
    query_files = _verified_file_identities(
        output, query_dir, manifest.get("query_files"), label="query Dataset"
    )
    query_dataset = Dataset.load_from_disk(str(query_dir))
    if set(query_dataset.column_names) != QUERY_FIELDS:
        raise ValueError("Candidate-breadth query Dataset schema changed")
    frozen_dataset = Dataset.load_from_disk(str(validation_root / "dataset"))
    frozen_by_sample = {int(row["sample_id"]): row for row in frozen_dataset}
    if len(frozen_by_sample) != len(frozen_dataset):
        raise ValueError("Frozen validation Dataset contains duplicate sample IDs")
    if len(query_dataset) != expected_queries:
        raise ValueError("Candidate-breadth query Dataset row count changed")
    for index, row in enumerate(query_dataset):
        source_row = selected[index]
        frozen_row = frozen_by_sample[int(source_row["sample_id"])]
        expected = {
            "query_index": index,
            "sample_id": int(source_row["sample_id"]),
            "source": str(source_row["source"]),
            "query_id": int(source_row["query_id"]),
            "query": str(frozen_row["query"]),
        }
        if dict(row) != expected:
            raise ValueError(f"Candidate-breadth query Dataset row {index + 1} changed")

    candidate_rows = 0
    candidate_files: dict[str, list[dict[str, Any]]] = {}
    candidate_fingerprints: dict[str, str] = {}
    for source in SPLITS:
        source_dir = output / "candidates" / source
        declared = manifest.get("candidate_files", {}).get(source)
        candidate_files[source] = _verified_file_identities(
            output, source_dir, declared, label=f"{source} candidate Dataset"
        )
        dataset = Dataset.load_from_disk(str(source_dir))
        if set(dataset.column_names) != CANDIDATE_FIELDS:
            raise ValueError(f"Candidate-breadth {source} candidate Dataset schema changed")
        candidate_fingerprints[source] = dataset._fingerprint
        candidate_rows += len(dataset)
        if len(dataset) != 32 * (maximum_width + 1):
            raise ValueError(f"Candidate-breadth {source} row count changed")
        document_lookup = None
        source_ledger = [record for record in records if record["source"] == source]
        if verify_source:
            assert snapshot is not None
            required = {
                int(document_id)
                for record in source_ledger
                for document_id in (record["positive_id"], *record["negative_ids"])
            }
            document_lookup = _document_texts(snapshot, source, required)
        offset = 0
        for record in source_ledger:
            document_ids = [int(record["positive_id"]), *map(int, record["negative_ids"])]
            stop = offset + len(document_ids)
            block = dataset[offset:stop]
            expected_constant = {
                "query_index": int(record["query_index"]),
                "sample_id": int(record["sample_id"]),
                "source": source,
                "query_id": int(record["query_id"]),
            }
            if (
                [int(value) for value in block["candidate_index"]] != list(range(maximum_width + 1))
                or [int(value) for value in block["document_id"]] != document_ids
                or any(
                    any(value != expected for value in block[field])
                    for field, expected in expected_constant.items()
                )
            ):
                raise ValueError(
                    f"Candidate-breadth {source} materialization changed for "
                    f"sample {record['sample_id']}"
                )
            if document_lookup is not None and [str(value) for value in block["document"]] != [
                document_lookup[document_id] for document_id in document_ids
            ]:
                raise ValueError(
                    f"Candidate-breadth {source} document text differs from pinned source"
                )
            offset = stop
        if offset != len(dataset):
            raise ValueError(f"Candidate-breadth {source} materialization has trailing rows")

    expected_manifest = _candidate_data_manifest(
        protocol_path=protocol_path,
        protocol=protocol,
        validation_spec=validation_spec,
        records=records,
        ledger_path=ledger_path,
        row_manifest_sha256=digest.hexdigest(),
        query_dataset=query_dataset,
        query_files=query_files,
        candidate_rows=candidate_rows,
        candidate_fingerprints=candidate_fingerprints,
        candidate_files=candidate_files,
    )
    if manifest != expected_manifest:
        raise ValueError("Candidate-breadth manifest does not reproduce audited artifacts")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "queries": len(records),
        "candidate_rows": candidate_rows,
        "negative_widths": widths,
        "upstream_reconstruction_verified": verify_source,
        "manifest_sha256": _sha256(manifest_path),
        "protocol_sha256": _sha256(protocol_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or audit the post-hoc nested candidate-breadth probe"
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/candidate_breadth_probe.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/candidate-breadth-224-seed20260901")
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Audit an existing complete output, or prepare it when absent.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.audit_only:
        result = audit_candidate_breadth_data(args.protocol, args.output, verify_source=True)
    elif args.resume and args.output.exists():
        result = audit_candidate_breadth_data(args.protocol, args.output, verify_source=False)
    else:
        output = prepare_candidate_breadth_data(
            args.protocol, args.output, overwrite=args.overwrite
        )
        result = audit_candidate_breadth_data(args.protocol, output, verify_source=False)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
