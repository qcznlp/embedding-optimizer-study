from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset

from .data import allocate_quotas
from .geometry import _atomic_json, _sha256

SCHEMA_VERSION = 1
SELECTION_ALGORITHM = "blake2b-128-per-source-smallest-v1"
Allocation = Literal["balanced", "proportional"]


def resolve_probe_spec_path(path: str | Path, prefix: Path | None = None) -> Path:
    path = Path(path)
    if path.is_file() or path.is_absolute() or path.parent != Path("configs"):
        return path
    prefix = Path(sys.prefix) if prefix is None else prefix
    installed = prefix / "share" / "embedding-optimizer-study" / "configs" / path.name
    return installed if installed.is_file() else path


def allocate_balanced(counts: dict[str, int], total: int) -> dict[str, int]:
    available = sum(counts.values())
    if total <= 0 or total > available:
        raise ValueError(f"total must be in [1, {available}], got {total}")
    quotas = {source: 0 for source in counts}
    remaining = total
    while remaining:
        active = sorted(source for source, count in counts.items() if quotas[source] < count)
        if not active:
            raise AssertionError("balanced allocation exhausted every source too early")
        share = remaining // len(active)
        if share == 0:
            for source in active[:remaining]:
                quotas[source] += 1
            remaining = 0
            break
        assigned = 0
        for source in active:
            addition = min(share, counts[source] - quotas[source])
            quotas[source] += addition
            assigned += addition
        if assigned == 0:
            raise AssertionError("balanced allocation made no progress")
        remaining -= assigned
    return quotas


def _selection_rank(seed: int, source: str, sample_id: int) -> int:
    payload = f"{SELECTION_ALGORITHM}:{seed}:{source}:{sample_id}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "big")


def _resolve_dataset(source: Path) -> tuple[Path, Path]:
    source = source.resolve()
    if (source / "dataset").is_dir():
        dataset_dir = source / "dataset"
        manifest = source / "manifest.json"
    else:
        dataset_dir = source
        manifest = source.parent / "manifest.json"
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)
    if not manifest.is_file():
        raise FileNotFoundError(
            f"A prepared-data manifest is required beside the source dataset: {manifest}"
        )
    return dataset_dir, manifest


def _validate_source(
    dataset: Dataset,
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
) -> None:
    expected_rows = manifest.get("total_queries")
    if expected_rows != len(dataset):
        raise ValueError(
            f"Source row count disagrees with {manifest_path}: expected {expected_rows}, "
            f"got {len(dataset)}"
        )
    expected_fingerprint = manifest.get("dataset_fingerprint")
    if expected_fingerprint != dataset._fingerprint:
        raise ValueError(
            f"Source fingerprint disagrees with {manifest_path}: expected "
            f"{expected_fingerprint!r}, got {dataset._fingerprint!r}"
        )
    required = {
        "sample_id",
        "source",
        "query_id",
        "positive_id",
        "query",
        "positive",
        "length",
        *(f"negative_{index}" for index in range(7)),
        *(f"negative_{index}_id" for index in range(7)),
    }
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Source dataset is missing columns: {sorted(missing)}")
    if manifest.get("sampled_negatives") != 7:
        raise ValueError(
            f"Probe contract requires seven negatives, got {manifest.get('sampled_negatives')!r}"
        )


def _select_indices(
    sample_ids: list[int],
    sources: list[str],
    quotas: dict[str, int],
    seed: int,
) -> list[tuple[int, int]]:
    if len(sample_ids) != len(sources):
        raise ValueError("sample_id and source columns have different lengths")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Source sample_id values must be unique")
    heaps: dict[str, list[tuple[int, int, int, int]]] = {source: [] for source in quotas}
    for index, (sample_id, source) in enumerate(zip(sample_ids, sources)):
        quota = quotas.get(source)
        if quota is None:
            raise ValueError(f"Unexpected source label {source!r}")
        if quota == 0:
            continue
        rank = _selection_rank(seed, source, sample_id)
        # Negative values turn heapq into a bounded max heap. The root is the
        # worst currently accepted (largest rank, then largest sample ID).
        candidate = (-rank, -sample_id, index, rank)
        heap = heaps[source]
        if len(heap) < quota:
            heapq.heappush(heap, candidate)
        elif candidate > heap[0]:
            heapq.heapreplace(heap, candidate)

    selected: list[tuple[int, int]] = []
    for source, quota in quotas.items():
        heap = heaps[source]
        if len(heap) != quota:
            raise RuntimeError(f"Selected {len(heap)} rows for {source}, expected {quota}")
        selected.extend((entry[2], entry[3]) for entry in heap)
    return sorted(selected, key=lambda item: sample_ids[item[0]])


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


def _validate_expected(manifest: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        if key == "manifest_sha256":
            continue
        if key not in manifest:
            raise ValueError(f"Probe specification expects unknown manifest field {key!r}")
        if manifest[key] != value:
            raise ValueError(
                f"Probe specification mismatch for {key}: expected {value!r}, got {manifest[key]!r}"
            )


def prepare_probe(
    source: str | Path,
    output: str | Path,
    *,
    count: int = 1_024,
    seed: int = 1729,
    allocation: Allocation = "balanced",
    overwrite: bool = False,
    expected: dict[str, Any] | None = None,
) -> Path:
    source = Path(source)
    output = Path(output).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    dataset_dir, source_manifest_path = _resolve_dataset(source)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    dataset = Dataset.load_from_disk(str(dataset_dir))
    _validate_source(dataset, source_manifest, manifest_path=source_manifest_path)

    sample_ids = [int(value) for value in dataset["sample_id"]]
    sources = [str(value) for value in dataset["source"]]
    source_counts = dict(sorted(Counter(sources).items()))
    if allocation == "balanced":
        quotas = allocate_balanced(source_counts, count)
    elif allocation == "proportional":
        quotas = allocate_quotas(source_counts, count)
    else:
        raise ValueError(f"Unsupported allocation {allocation!r}")
    selected = _select_indices(sample_ids, sources, quotas, seed)

    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        selection_path = temporary / "selection.jsonl"
        selection_digest = hashlib.sha256()
        selected_indices: list[int] = []
        with selection_path.open("wb") as handle:
            for dataset_index, rank in selected:
                row = dataset[dataset_index]
                record = {
                    "dataset_index": dataset_index,
                    "length": int(row["length"]),
                    "negative_ids": [int(row[f"negative_{index}_id"]) for index in range(7)],
                    "positive_id": int(row["positive_id"]),
                    "query_id": int(row["query_id"]),
                    "sample_id": int(row["sample_id"]),
                    "selection_rank_hex": f"{rank:032x}",
                    "source": str(row["source"]),
                }
                encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
                handle.write(encoded + b"\n")
                selection_digest.update(encoded + b"\n")
                selected_indices.append(dataset_index)
            handle.flush()
            os.fsync(handle.fileno())

        probe = dataset.select(selected_indices)
        probe_dir = temporary / "dataset"
        probe.save_to_disk(str(probe_dir))
        serialized_probe = Dataset.load_from_disk(str(probe_dir))
        if len(serialized_probe) != count:
            raise RuntimeError(
                f"Serialized probe has {len(serialized_probe)} rows, expected {count}"
            )
        selected_id_digest = hashlib.sha256()
        for sample_id in serialized_probe["sample_id"]:
            selected_id_digest.update(f"{int(sample_id)}\n".encode())
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "selection_algorithm": SELECTION_ALGORITHM,
            "allocation": allocation,
            "seed": seed,
            "count": count,
            "source_counts": source_counts,
            "quotas": quotas,
            "selected_sample_ids_sha256": selected_id_digest.hexdigest(),
            "selection_sha256": selection_digest.hexdigest(),
            "probe_dataset_fingerprint": probe._fingerprint,
            "serialized_probe_dataset_fingerprint": serialized_probe._fingerprint,
            "columns": list(serialized_probe.column_names),
            "positive_candidate_index": 0,
            "negative_candidates": 7,
            "source_manifest_sha256": _sha256(source_manifest_path),
            "source_manifest": source_manifest,
        }
        if expected is not None:
            _validate_expected(manifest, expected)
        _atomic_json(temporary / "manifest.json", manifest)
        if expected is not None and "manifest_sha256" in expected:
            actual_manifest_sha256 = _sha256(temporary / "manifest.json")
            if actual_manifest_sha256 != expected["manifest_sha256"]:
                raise ValueError(
                    "Probe specification mismatch for manifest_sha256: expected "
                    f"{expected['manifest_sha256']!r}, got {actual_manifest_sha256!r}"
                )
        output.parent.mkdir(parents=True, exist_ok=True)
        _replace_directory(temporary, output, overwrite)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    print(f"Prepared {count:,} fixed probe rows at {output}")
    print(f"Probe manifest SHA256: {_sha256(output / 'manifest.json')}")
    return output


def prepare_probe_from_spec(
    spec_path: str | Path,
    *,
    output: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    spec_path = resolve_probe_spec_path(spec_path).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported probe specification schema in {spec_path}")
    required = {"source", "output", "count", "seed", "allocation", "expected"}
    missing = required.difference(spec)
    if missing:
        raise ValueError(f"Probe specification is missing fields: {sorted(missing)}")
    if not isinstance(spec["expected"], dict) or not spec["expected"]:
        raise ValueError("Probe specification expected values must be a non-empty object")
    return prepare_probe(
        spec["source"],
        spec["output"] if output is None else output,
        count=int(spec["count"]),
        seed=int(spec["seed"]),
        allocation=spec["allocation"],
        overwrite=overwrite,
        expected=spec["expected"],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a fixed, source-stratified representation probe"
    )
    parser.add_argument("--spec", type=Path, help="Frozen JSON specification with expected hashes")
    parser.add_argument("--source")
    parser.add_argument("--output")
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--allocation", choices=("balanced", "proportional"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.spec is not None:
        overrides = [args.source, args.count, args.seed, args.allocation]
        if any(value is not None for value in overrides):
            raise ValueError(
                "--spec fixes source, count, seed, and allocation; only --output may be overridden"
            )
        prepare_probe_from_spec(args.spec, output=args.output, overwrite=args.overwrite)
        return
    prepare_probe(
        args.source or "data/denseon-sft-500k-seed42",
        args.output or "data/probes/training-1024-seed1729",
        count=1_024 if args.count is None else args.count,
        seed=1729 if args.seed is None else args.seed,
        allocation="balanced" if args.allocation is None else args.allocation,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
