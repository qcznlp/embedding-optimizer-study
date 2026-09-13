from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, concatenate_datasets
from huggingface_hub import snapshot_download

from .data import (
    SOURCE_REPO,
    SOURCE_REVISION,
    SPLITS,
    _build_split,
    _read_query_table,
    _scorable_query_ids,
)
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .probes import allocate_balanced


def resolve_validation_spec(path: str | Path, prefix: Path | None = None) -> Path:
    path = Path(path)
    if path.is_file() or path.is_absolute() or path.parent != Path("configs"):
        return path
    prefix = Path(sys.prefix) if prefix is None else prefix
    installed = prefix / "share" / "embedding-optimizer-study" / "configs" / path.name
    return installed if installed.is_file() else path


def _canonical_row(record: dict[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def training_query_ids(
    training_root: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_ledger_sha256: str | None = None,
    expected_total: int = 500_000,
) -> tuple[dict[str, set[int]], dict[str, Any]]:
    training_root = Path(training_root).resolve()
    manifest_path = training_root / "manifest.json"
    ledger_path = training_root / "rows.jsonl"
    if not manifest_path.is_file() or not ledger_path.is_file():
        raise FileNotFoundError(f"Training manifest or row ledger is missing under {training_root}")
    if expected_manifest_sha256 and _sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("Training manifest differs from the validation specification")
    if expected_ledger_sha256 and _sha256(ledger_path) != expected_ledger_sha256:
        raise ValueError("Training row ledger differs from the validation specification")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("source_repo") != SOURCE_REPO
        or manifest.get("source_revision") != SOURCE_REVISION
        or manifest.get("total_queries") != expected_total
        or manifest.get("sampled_negatives") != 7
    ):
        raise ValueError("Training manifest differs from the formal 500K data contract")
    query_ids = {split: set() for split in SPLITS}
    digest = hashlib.sha256()
    rows = 0
    with ledger_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                source = str(record["source"])
                query_id = int(record["query_id"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid training ledger row {line_number}") from error
            if source not in query_ids or query_id in query_ids[source]:
                raise ValueError(f"Duplicate or invalid training query at row {line_number}")
            query_ids[source].add(query_id)
            digest.update(_canonical_row(record))
            rows += 1
    if rows != expected_total or digest.hexdigest() != manifest.get("row_manifest_sha256"):
        raise ValueError("Training row ledger does not reproduce its canonical manifest digest")
    if {split: len(values) for split, values in query_ids.items()} != manifest.get("quotas"):
        raise ValueError("Training query counts differ from the source quotas")
    return query_ids, manifest


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


def ensure_selection_ledger(output: str | Path) -> Path:
    """Materialize the probe-compatible selection ledger from the canonical row ledger."""

    output = Path(output).resolve()
    row_path = output / "rows.jsonl"
    selection_path = output / "selection.jsonl"
    if not row_path.is_file():
        raise FileNotFoundError(row_path)
    if selection_path.is_file() and _sha256(selection_path) == _sha256(row_path):
        return selection_path
    temporary = output / f".selection.jsonl.tmp.{os.getpid()}"
    try:
        shutil.copyfile(row_path, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, selection_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return selection_path


def _validate_expected(manifest: dict[str, Any], expected: dict[str, Any] | None) -> None:
    if expected is None:
        return
    for key, value in expected.items():
        if key == "manifest_sha256":
            continue
        if manifest.get(key) != value:
            raise ValueError(
                f"Validation-data expectation differs for {key}: "
                f"expected={value!r}, observed={manifest.get(key)!r}"
            )


def prepare_validation_data(
    training_root: str | Path,
    output: str | Path,
    *,
    count: int = 4_096,
    seed: int = 20_260_826,
    candidate_margin: float = 0.5,
    threshold: float = 0.95,
    pool_size: int = 10,
    sampled_negatives: int = 7,
    score_batch_size: int = 2_048,
    source_revision: str = SOURCE_REVISION,
    expected_training_manifest_sha256: str | None = None,
    expected_training_ledger_sha256: str | None = None,
    expected: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    output = Path(output).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    if source_revision != SOURCE_REVISION:
        raise ValueError("Validation data must use the same pinned source revision as training")
    if sampled_negatives != 7:
        raise ValueError("Validation data requires exactly seven negatives")
    excluded, training_manifest = training_query_ids(
        training_root,
        expected_manifest_sha256=expected_training_manifest_sha256,
        expected_ledger_sha256=expected_training_ledger_sha256,
    )
    snapshot = Path(snapshot_download(SOURCE_REPO, repo_type="dataset", revision=source_revision))
    query_tables = {split: _read_query_table(snapshot, split) for split in SPLITS}
    eligible_ids: dict[str, np.ndarray] = {}
    for split, table in query_tables.items():
        raw_ids = table["query_id"].combine_chunks().to_numpy(zero_copy_only=False)
        scorable = _scorable_query_ids(snapshot, split, raw_ids)
        excluded_values = np.fromiter(excluded[split], dtype=np.int64)
        eligible_ids[split] = scorable[~np.isin(scorable, excluded_values, assume_unique=False)]
    available = {split: len(values) for split, values in eligible_ids.items()}
    quotas = allocate_balanced(available, count)

    datasets = []
    rows: list[dict[str, Any]] = []
    sample_id = 0
    for split in SPLITS:
        print(f"Preparing held-out {split}: quota={quotas[split]:,}", flush=True)
        dataset, split_rows, _ = _build_split(
            snapshot=snapshot,
            split=split,
            scorable_query_ids=eligible_ids[split],
            quota=quotas[split],
            seed=seed,
            threshold=threshold,
            pool_size=pool_size,
            sampled_negatives=sampled_negatives,
            candidate_margin=candidate_margin,
            score_batch_size=score_batch_size,
            sample_id_start=sample_id,
        )
        if any(int(record["query_id"]) in excluded[split] for record in split_rows):
            raise AssertionError(f"Held-out validation leaked a training query in {split}")
        datasets.append(dataset)
        rows.extend(split_rows)
        sample_id += len(dataset)
    combined = concatenate_datasets(datasets)
    if len(combined) != count:
        raise AssertionError(f"Prepared {len(combined)} validation rows, expected {count}")
    observed_counts = dict(sorted(Counter(str(value) for value in combined["source"]).items()))
    if observed_counts != dict(sorted(quotas.items())):
        raise AssertionError("Validation source counts differ from their balanced quotas")

    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        dataset_dir = temporary / "dataset"
        combined.save_to_disk(str(dataset_dir), num_proc=min(14, len(SPLITS) * 2))
        serialized = Dataset.load_from_disk(str(dataset_dir))
        row_path = temporary / "rows.jsonl"
        row_digest = hashlib.sha256()
        with row_path.open("wb") as handle:
            for record in rows:
                encoded = _canonical_row(record)
                handle.write(encoded)
                row_digest.update(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copyfile(row_path, temporary / "selection.jsonl")
        selected_digest = hashlib.sha256()
        sample_id_digest = hashlib.sha256()
        for split, query_id in zip(serialized["source"], serialized["query_id"], strict=True):
            selected_digest.update(f"{split}:{int(query_id)}\n".encode())
        for sample_id_value in serialized["sample_id"]:
            sample_id_digest.update(f"{int(sample_id_value)}\n".encode())
        dataset_files = [
            {
                "path": str(path.relative_to(temporary)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(dataset_dir.rglob("*"))
            if path.is_file()
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "source_repo": SOURCE_REPO,
            "source_revision": source_revision,
            "seed": seed,
            "allocation": "balanced",
            "total_queries": count,
            "count": count,
            "available_disjoint_query_counts": available,
            "quotas": quotas,
            "excluded_training_queries": sum(len(values) for values in excluded.values()),
            "training_manifest_sha256": expected_training_manifest_sha256
            or _sha256(Path(training_root) / "manifest.json"),
            "training_row_ledger_sha256": expected_training_ledger_sha256
            or _sha256(Path(training_root) / "rows.jsonl"),
            "training_row_manifest_sha256": training_manifest["row_manifest_sha256"],
            "query_overlap_with_training": 0,
            "negative_threshold": threshold,
            "negative_pool_size": pool_size,
            "sampled_negatives": sampled_negatives,
            "candidate_margin": candidate_margin,
            "row_manifest_sha256": row_digest.hexdigest(),
            "selection_sha256": row_digest.hexdigest(),
            "selected_source_query_ids_sha256": selected_digest.hexdigest(),
            "selected_sample_ids_sha256": sample_id_digest.hexdigest(),
            "materialized_dataset_fingerprint": combined._fingerprint,
            "dataset_fingerprint": serialized._fingerprint,
            "serialized_probe_dataset_fingerprint": serialized._fingerprint,
            "dataset_files": dataset_files,
            "positive_candidate_index": 0,
            "negative_candidates": 7,
        }
        _validate_expected(manifest, expected)
        _atomic_json(temporary / "manifest.json", manifest)
        if expected and expected.get("manifest_sha256") != _sha256(temporary / "manifest.json"):
            raise ValueError("Validation-data manifest SHA-256 differs from its frozen expectation")
        _replace_directory(temporary, output, overwrite)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def audit_validation_data(
    output: str | Path,
    training_root: str | Path,
    *,
    spec_path: str | Path = "configs/validation_probe.json",
) -> dict[str, Any]:
    output = Path(output).resolve()
    spec_path, spec = load_validation_spec(spec_path)
    manifest_path = output / "manifest.json"
    row_path = output / "rows.jsonl"
    selection_path = output / "selection.jsonl"
    dataset_path = output / "dataset"
    if (
        not manifest_path.is_file()
        or not row_path.is_file()
        or not selection_path.is_file()
        or not dataset_path.is_dir()
    ):
        raise FileNotFoundError(f"Validation output is incomplete under {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = spec.get("expected")
    if not isinstance(expected, dict) or expected.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("Validation output is not bound to the frozen expected manifest")
    _validate_expected(manifest, expected)
    if (
        _sha256(selection_path) != manifest.get("selection_sha256")
        or selection_path.read_bytes() != row_path.read_bytes()
    ):
        raise ValueError("Validation selection and canonical row ledgers differ")
    source = spec["source"]
    excluded, _ = training_query_ids(
        training_root,
        expected_manifest_sha256=source["manifest_sha256"],
        expected_ledger_sha256=source["row_ledger_sha256"],
    )
    declared_files = manifest.get("dataset_files")
    observed_paths = sorted(path for path in dataset_path.rglob("*") if path.is_file())
    if not isinstance(declared_files, list) or len(declared_files) != len(observed_paths):
        raise ValueError("Validation Dataset file coverage differs from its manifest")
    for item in declared_files:
        path = output / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or _sha256(path) != item.get("sha256")
        ):
            raise ValueError(f"Validation Dataset file differs from its manifest: {path}")
    if {str((output / item["path"]).resolve()) for item in declared_files} != {
        str(path.resolve()) for path in observed_paths
    }:
        raise ValueError("Validation Dataset contains undeclared files")

    dataset = Dataset.load_from_disk(str(dataset_path))
    if len(dataset) != manifest.get("total_queries") or dataset._fingerprint != manifest.get(
        "dataset_fingerprint"
    ):
        raise ValueError("Validation Dataset row count or fingerprint changed")
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
    if not required.issubset(dataset.column_names):
        raise ValueError("Validation Dataset lost required positive-first fields")
    digest = hashlib.sha256()
    selected_digest = hashlib.sha256()
    source_counts: Counter[str] = Counter()
    rows = 0
    with row_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            try:
                record = json.loads(line)
                row = dataset[index]
                source_label = str(record["source"])
                query_id = int(record["query_id"])
                positive_id = int(record["positive_id"])
                negative_ids = [int(value) for value in record["negative_ids"]]
            except (json.JSONDecodeError, IndexError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid validation ledger row {index + 1}") from error
            if (
                int(row["sample_id"]) != int(record["sample_id"])
                or str(row["source"]) != source_label
                or int(row["query_id"]) != query_id
                or int(row["positive_id"]) != positive_id
                or [int(row[f"negative_{item}_id"]) for item in range(7)] != negative_ids
            ):
                raise ValueError(f"Validation Dataset disagrees with ledger row {index + 1}")
            if (
                source_label not in excluded
                or query_id in excluded[source_label]
                or len(negative_ids) != 7
                or len(set(negative_ids)) != 7
                or positive_id in negative_ids
            ):
                raise ValueError(
                    f"Validation disjointness or negative contract failed at row {index + 1}"
                )
            digest.update(_canonical_row(record))
            selected_digest.update(f"{source_label}:{query_id}\n".encode())
            source_counts[source_label] += 1
            rows += 1
    if (
        rows != len(dataset)
        or digest.hexdigest() != manifest.get("row_manifest_sha256")
        or selected_digest.hexdigest() != manifest.get("selected_source_query_ids_sha256")
        or dict(sorted(source_counts.items())) != manifest.get("quotas")
    ):
        raise ValueError("Validation ledger digest, selection digest, or quotas changed")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "rows": rows,
        "sources": len(source_counts),
        "query_overlap_with_training": 0,
        "negative_groups_valid": rows,
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        },
        "row_ledger": {
            "path": str(row_path),
            "bytes": row_path.stat().st_size,
            "sha256": _sha256(row_path),
        },
        "spec": {"path": str(spec_path), "sha256": _sha256(spec_path)},
    }


def load_validation_spec(path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = resolve_validation_spec(path).resolve()
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != SCHEMA_VERSION or set(spec) != {
        "schema_version",
        "source",
        "selection",
        "output",
        "evaluation",
        "recipe_selection",
        "freeze_context",
        "expected",
    }:
        raise ValueError(f"Unsupported validation-data specification: {path}")
    source = spec["source"]
    selection = spec["selection"]
    evaluation = spec["evaluation"]
    recipe = spec["recipe_selection"]
    expected = spec["expected"]
    if (
        source.get("repo") != SOURCE_REPO
        or source.get("revision") != SOURCE_REVISION
        or len(source.get("manifest_sha256", "")) != 64
        or len(source.get("row_ledger_sha256", "")) != 64
    ):
        raise ValueError("Validation specification changed its pinned source")
    if selection.get("exclude") != "all 500000 training query IDs within their source split":
        raise ValueError("Validation specification does not enforce training-query disjointness")
    if (
        selection.get("count") != 4096
        or selection.get("allocation") != "balanced"
        or selection.get("sampled_negatives") != 7
    ):
        raise ValueError("Validation specification changed its allocation or negative contract")
    if (
        evaluation.get("expected_jobs") != 24
        or evaluation.get("expected_sample_records_per_job") != 4096
        or evaluation.get("model_dtype") != "float32"
        or evaluation.get("forward_dtype") != "bfloat16"
        or evaluation.get("model_mode") != "eval"
        or evaluation.get("flash_attention") is not True
    ):
        raise ValueError("Validation evaluation protocol changed")
    if (
        recipe.get("checkpoint") != "final checkpoint only"
        or recipe.get("primary_metric") != "mean eight-way contrastive loss"
        or not isinstance(expected, dict)
        or len(expected.get("manifest_sha256", "")) != 64
        or expected.get("query_overlap_with_training") != 0
    ):
        raise ValueError("Validation recipe-selection or expected-output contract changed")
    freeze = spec["freeze_context"]
    if freeze.get(
        "validation_examples_or_model_outputs_visible"
    ) is not False or not 0 <= freeze.get("strict_beir_valid_units", -1) < freeze.get(
        "strict_beir_expected_units", -1
    ):
        raise ValueError("Validation specification has an invalid freeze context")
    return path, spec


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the frozen query-disjoint recipe-selection validation set"
    )
    parser.add_argument("--spec", type=Path, default=Path("configs/validation_probe.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-unfrozen", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--repair-selection-ledger", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _, spec = load_validation_spec(args.spec)
    source = spec["source"]
    selection = spec["selection"]
    if args.repair_selection_ledger:
        output = args.output if args.output is not None else spec["output"]
        print(ensure_selection_ledger(output))
        return
    if args.audit_only:
        print(
            json.dumps(
                audit_validation_data(
                    args.output if args.output is not None else spec["output"],
                    source["training_data"],
                    spec_path=args.spec,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    prepare_validation_data(
        source["training_data"],
        args.output if args.output is not None else spec["output"],
        count=int(selection["count"]),
        seed=int(selection["seed"]),
        candidate_margin=float(selection["candidate_margin"]),
        threshold=float(selection["negative_threshold"]),
        pool_size=int(selection["negative_pool_size"]),
        sampled_negatives=int(selection["sampled_negatives"]),
        source_revision=source["revision"],
        expected_training_manifest_sha256=source["manifest_sha256"],
        expected_training_ledger_sha256=source["row_ledger_sha256"],
        expected=None if args.allow_unfrozen else spec["expected"],
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
