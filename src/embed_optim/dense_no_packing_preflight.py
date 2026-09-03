"""Build or audit the deterministic longest-row Dense padding preflight set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import Dataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_indices(source: Dataset, rows: int) -> list[int]:
    if rows <= 0 or rows > len(source):
        raise ValueError(f"rows must lie in [1, {len(source)}], got {rows}")
    # Materialize each Arrow column once. Repeated scalar indexing on a Dataset
    # Column reconstructs its backing view and makes a 500k-row sort quadratic
    # in practice.
    lengths = list(source["length"])
    sample_ids = list(source["sample_id"])
    ranked = sorted(
        range(len(source)),
        key=lambda index: (-int(lengths[index]), int(sample_ids[index])),
    )
    return ranked[:rows]


def _ids_sha256(values: list[int]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode()
    return hashlib.sha256(payload).hexdigest()


def _expected(source_root: Path, rows: int) -> tuple[Dataset, dict]:
    source_path = source_root / "dataset" if (source_root / "dataset").is_dir() else source_root
    source = Dataset.load_from_disk(str(source_path))
    required = {"sample_id", "length"}
    missing = sorted(required - set(source.column_names))
    if missing:
        raise ValueError(f"Source dataset is missing columns: {missing}")
    indices = _selected_indices(source, rows)
    selected = source.select(indices)
    sample_ids = [int(value) for value in selected["sample_id"]]
    lengths = [int(value) for value in selected["length"]]
    source_manifest = source_root / "manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    manifest = {
        "schema_version": 1,
        "purpose": "engineering-only worst-case memory preflight; never an analysis dataset",
        "source_dataset": str(source_root),
        "source_manifest_sha256": _sha256(source_manifest),
        "selection": "sort by descending declared length, then ascending sample_id",
        "rows": rows,
        "minimum_declared_length": min(lengths),
        "maximum_declared_length": max(lengths),
        "selected_sample_ids_sha256": _ids_sha256(sample_ids),
        "source_dataset_fingerprint": source._fingerprint,
        "selected_dataset_fingerprint": selected._fingerprint,
    }
    return selected, manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/denseon-sft-500k-seed42"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/dense-no-packing-preflight-longest256")
    )
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)

    selected, expected = _expected(args.source, args.rows)
    dataset_path = args.output / "dataset"
    manifest_path = args.output / "manifest.json"
    if args.audit_only:
        if not dataset_path.is_dir() or not manifest_path.is_file():
            raise FileNotFoundError(f"Missing preflight dataset or manifest under {args.output}")
        observed = Dataset.load_from_disk(str(dataset_path))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed_ids = [int(value) for value in observed["sample_id"]]
        materialized_fingerprint = manifest.pop("materialized_dataset_fingerprint", None)
        if manifest != expected:
            raise RuntimeError("Preflight manifest differs from deterministic reconstruction")
        if observed._fingerprint != materialized_fingerprint:
            raise RuntimeError("Preflight dataset fingerprint differs from its manifest")
        if _ids_sha256(observed_ids) != expected["selected_sample_ids_sha256"]:
            raise RuntimeError("Preflight sample IDs differ from deterministic selection")
        print(json.dumps({"status": "complete", **expected}, sort_keys=True))
        return

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing preflight data: {args.output}")
    args.output.mkdir(parents=True)
    selected.save_to_disk(str(dataset_path))
    materialized = Dataset.load_from_disk(str(dataset_path))
    expected["materialized_dataset_fingerprint"] = materialized._fingerprint
    manifest_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", **expected}, sort_keys=True))


if __name__ == "__main__":
    main()
