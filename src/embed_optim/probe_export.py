from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from datasets import Dataset

from .geometry import _atomic_json, _sha256
from .probes import resolve_probe_spec_path

SCHEMA_VERSION = 1
EXPECTED_DIMENSIONS = {"dense": 768, "late": 128}
TEXT_COLUMNS = ("positive", *(f"negative_{index}" for index in range(7)))
ModelFamily = Literal["dense", "late"]


def _checkpoint_inputs(checkpoint: Path) -> list[dict[str, Any]]:
    paths = sorted(
        path
        for path in checkpoint.rglob("*")
        if path.is_file() and path.suffix in {".json", ".safetensors"}
    )
    if not any(path.suffix == ".safetensors" for path in paths):
        raise FileNotFoundError(f"No safetensors model weights under {checkpoint}")
    return [
        {
            "path": str(path.relative_to(checkpoint)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def _validate_checkpoint_family(checkpoint: Path, family: ModelFamily) -> dict[str, Any] | None:
    run_config_path = checkpoint.parent / "run_config.json"
    if not run_config_path.is_file():
        return None
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    if run_config.get("model_family") != family:
        raise ValueError(
            f"Checkpoint run_config declares {run_config.get('model_family')!r}, expected {family!r}"
        )
    return {"path": str(run_config_path), "sha256": _sha256(run_config_path)}


def _load_probe(probe_root: Path) -> tuple[Dataset, dict[str, Any], str]:
    probe_root = probe_root.resolve()
    manifest_path = probe_root / "manifest.json"
    selection_path = probe_root / "selection.jsonl"
    dataset_path = probe_root / "dataset"
    if not manifest_path.is_file() or not selection_path.is_file() or not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Expected manifest.json, selection.jsonl, and dataset/ under {probe_root}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported probe manifest schema under {probe_root}")
    if manifest.get("selection_sha256") != _sha256(selection_path):
        raise ValueError("Probe selection ledger digest mismatch")
    dataset = Dataset.load_from_disk(str(dataset_path))
    if manifest.get("count") != len(dataset):
        raise ValueError(
            f"Probe row count mismatch: manifest={manifest.get('count')}, dataset={len(dataset)}"
        )
    if manifest.get("serialized_probe_dataset_fingerprint") != dataset._fingerprint:
        raise ValueError(
            "Probe Dataset fingerprint mismatch: expected "
            f"{manifest.get('serialized_probe_dataset_fingerprint')!r}, got {dataset._fingerprint!r}"
        )
    if manifest.get("positive_candidate_index") != 0 or manifest.get("negative_candidates") != 7:
        raise ValueError("Probe must contain one positive followed by exactly seven negatives")
    required = {"sample_id", "source", "query", *TEXT_COLUMNS}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Probe Dataset is missing columns: {sorted(missing)}")
    sample_ids = [int(value) for value in dataset["sample_id"]]
    selected_digest = hashlib.sha256()
    for sample_id in sample_ids:
        selected_digest.update(f"{sample_id}\n".encode())
    if selected_digest.hexdigest() != manifest.get("selected_sample_ids_sha256"):
        raise ValueError("Probe selected-sample-ID digest mismatch")
    return dataset, manifest, _sha256(manifest_path)


def _validate_probe_spec(
    probe_spec: str | Path | None,
    manifest_sha256: str,
) -> dict[str, Any] | None:
    if probe_spec is None:
        return None
    spec_path = resolve_probe_spec_path(probe_spec).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    expected = spec.get("expected")
    if not isinstance(expected, dict) or "manifest_sha256" not in expected:
        raise ValueError(f"Probe specification lacks expected.manifest_sha256: {spec_path}")
    if expected["manifest_sha256"] != manifest_sha256:
        raise ValueError(
            "Probe manifest does not match the frozen specification: expected "
            f"{expected['manifest_sha256']}, got {manifest_sha256}"
        )
    return {
        "path": str(spec_path),
        "sha256": _sha256(spec_path),
        "expected_manifest_sha256": expected["manifest_sha256"],
    }


def _load_model(
    family: ModelFamily,
    checkpoint: Path,
    *,
    dtype: torch.dtype,
    device: str,
    flash_attention: bool,
):
    attention = "flash_attention_2" if flash_attention else "sdpa"
    model_kwargs = {"dtype": dtype, "attn_implementation": attention}
    if family == "dense":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            str(checkpoint),
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )
        model.max_seq_length = 8192
    else:
        from .pylate_compat import configure_pylate_compatibility

        models = configure_pylate_compatibility()
        model = models.ColBERT(
            str(checkpoint),
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )
        if model.query_length != 8192 or model.document_length != 8192:
            raise ValueError(
                "Late checkpoint context contract changed: "
                f"query={model.query_length}, document={model.document_length}"
            )
        if model.do_query_expansion:
            raise ValueError("Late checkpoint unexpectedly enables query expansion")
        first_module = model._first_module()
        if hasattr(first_module, "can_flatten_inputs"):
            first_module.can_flatten_inputs = False
    model.to(device)
    model.eval()
    return model


def _as_embedding_array(values: Any, expected_dimension: int, label: str) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().numpy()
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != expected_dimension:
        raise ValueError(
            f"{label} embeddings must have shape [items, {expected_dimension}], got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{label} embeddings contain a non-finite value")
    return array


def encode_dense_probe(
    model: Any,
    dataset: Dataset,
    *,
    batch_size: int,
    storage_dtype: np.dtype,
) -> dict[str, np.ndarray]:
    dimension = EXPECTED_DIMENSIONS["dense"]
    queries = _as_embedding_array(
        model.encode(
            list(dataset["query"]),
            prompt="query: ",
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dimension,
        "query",
    )
    if queries.shape[0] != len(dataset):
        raise ValueError(
            f"Dense encoder returned {queries.shape[0]} queries for {len(dataset)} rows"
        )
    documents_text = [str(row[column]) for row in dataset for column in TEXT_COLUMNS]
    documents = _as_embedding_array(
        model.encode(
            documents_text,
            prompt="document: ",
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dimension,
        "document",
    )
    documents = documents.reshape(len(dataset), len(TEXT_COLUMNS), dimension)
    return {
        "query_embeddings": queries.astype(storage_dtype, copy=False),
        "document_embeddings": documents.astype(storage_dtype, copy=False),
    }


def _variable_embeddings(values: Any, expected_dimension: int, label: str) -> list[np.ndarray]:
    if isinstance(values, torch.Tensor):
        values = list(values)
    result = []
    for index, value in enumerate(values):
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        array = np.asarray(value)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != expected_dimension:
            raise ValueError(
                f"{label}[{index}] must have shape [tokens>0, {expected_dimension}], got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{label}[{index}] contains a non-finite value")
        result.append(array)
    return result


def pad_variable_embeddings(
    values: list[np.ndarray],
    *,
    storage_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        raise ValueError("Cannot pad an empty embedding list")
    dimension = values[0].shape[1]
    maximum = max(value.shape[0] for value in values)
    embeddings = np.zeros((len(values), maximum, dimension), dtype=storage_dtype)
    mask = np.zeros((len(values), maximum), dtype=np.bool_)
    for index, value in enumerate(values):
        if value.shape[1] != dimension:
            raise ValueError("Variable token embeddings have inconsistent dimensions")
        length = value.shape[0]
        embeddings[index, :length] = value.astype(storage_dtype, copy=False)
        mask[index, :length] = True
    return embeddings, mask


def pack_variable_embeddings(
    values: list[np.ndarray],
    *,
    storage_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack variable-length token embeddings without global-max padding."""

    if not values:
        raise ValueError("Cannot pack an empty embedding list")
    dimension = values[0].shape[1]
    lengths = np.asarray([value.shape[0] for value in values], dtype=np.int64)
    if np.any(lengths <= 0):
        raise ValueError("Packed token embeddings must contain at least one token per item")
    if any(value.shape[1] != dimension for value in values):
        raise ValueError("Variable token embeddings have inconsistent dimensions")
    offsets = np.empty(len(values) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    embeddings = np.empty((int(offsets[-1]), dimension), dtype=storage_dtype)
    for index, value in enumerate(values):
        embeddings[offsets[index] : offsets[index + 1]] = value.astype(storage_dtype, copy=False)
    return embeddings, offsets


def encode_late_probe(
    model: Any,
    dataset: Dataset,
    *,
    batch_size: int,
    storage_dtype: np.dtype,
) -> dict[str, np.ndarray]:
    dimension = EXPECTED_DIMENSIONS["late"]
    query_values = _variable_embeddings(
        model.encode(
            list(dataset["query"]),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            is_query=True,
        ),
        dimension,
        "query",
    )
    document_texts = [str(row[column]) for row in dataset for column in TEXT_COLUMNS]
    document_values = _variable_embeddings(
        model.encode(
            document_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            is_query=False,
        ),
        dimension,
        "document",
    )
    if len(query_values) != len(dataset):
        raise ValueError(
            f"Late encoder returned {len(query_values)} queries for {len(dataset)} rows"
        )
    expected_documents = len(dataset) * len(TEXT_COLUMNS)
    if len(document_values) != expected_documents:
        raise ValueError(
            f"Late encoder returned {len(document_values)} documents, expected {expected_documents}"
        )
    query_embeddings, query_offsets = pack_variable_embeddings(
        query_values, storage_dtype=storage_dtype
    )
    document_embeddings, document_offsets = pack_variable_embeddings(
        document_values, storage_dtype=storage_dtype
    )
    return {
        "query_embeddings": query_embeddings,
        "document_embeddings": document_embeddings,
        "query_offsets": query_offsets,
        "document_offsets": document_offsets,
    }


def _write_npz(path: Path, arrays: dict[str, np.ndarray], compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        writer = np.savez_compressed if compressed else np.savez
        writer(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def export_probe(
    checkpoint: str | Path,
    probe_root: str | Path,
    output: str | Path,
    *,
    family: ModelFamily,
    batch_size: int = 32,
    model_dtype: Literal["bfloat16", "float32"] = "bfloat16",
    storage_dtype: Literal["float16", "float32"] = "float16",
    device: str = "cuda",
    flash_attention: bool = True,
    compressed: bool = False,
    overwrite: bool = False,
    probe_spec: str | Path | None = None,
) -> tuple[Path, Path]:
    checkpoint = Path(checkpoint).resolve()
    probe_root = Path(probe_root).resolve()
    output = Path(output).resolve()
    manifest_output = output.with_suffix(output.suffix + ".manifest.json")
    if output.suffix != ".npz":
        raise ValueError(f"Probe export must use a .npz path, got {output}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if (output.exists() or manifest_output.exists()) and not overwrite:
        raise FileExistsError(f"{output} or its manifest exists; pass --overwrite to replace")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA output was requested but CUDA is unavailable")
    if not device.startswith("cuda") and flash_attention:
        raise ValueError("FlashAttention export requires a CUDA device")
    torch_dtype = torch.bfloat16 if model_dtype == "bfloat16" else torch.float32
    if device == "cpu" and torch_dtype == torch.bfloat16:
        raise ValueError("Use --model-dtype float32 for CPU export")
    numpy_dtype = np.dtype(storage_dtype)

    dataset, probe_manifest, probe_manifest_sha256 = _load_probe(probe_root)
    probe_spec_identity = _validate_probe_spec(probe_spec, probe_manifest_sha256)
    checkpoint_run_config = _validate_checkpoint_family(checkpoint, family)
    checkpoint_inputs = _checkpoint_inputs(checkpoint)
    model = _load_model(
        family,
        checkpoint,
        dtype=torch_dtype,
        device=device,
        flash_attention=flash_attention,
    )
    try:
        if family == "dense":
            embeddings = encode_dense_probe(
                model,
                dataset,
                batch_size=batch_size,
                storage_dtype=numpy_dtype,
            )
        elif family == "late":
            embeddings = encode_late_probe(
                model,
                dataset,
                batch_size=batch_size,
                storage_dtype=numpy_dtype,
            )
        else:
            raise ValueError(f"Unsupported family {family!r}")
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sample_ids = np.asarray(dataset["sample_id"], dtype=np.int64)
    sample_groups = np.asarray(dataset["source"], dtype=np.str_)
    arrays = {"sample_ids": sample_ids, "sample_groups": sample_groups, **embeddings}
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        _write_npz(temporary, arrays, compressed)
        npz_sha256 = _sha256(temporary)
        array_metadata = {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in sorted(arrays.items())
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "family": family,
            "checkpoint": str(checkpoint),
            "checkpoint_inputs": checkpoint_inputs,
            "checkpoint_run_config": checkpoint_run_config,
            "probe": {
                "path": str(probe_root),
                "manifest_sha256": probe_manifest_sha256,
                "selection_sha256": probe_manifest["selection_sha256"],
                "selected_sample_ids_sha256": probe_manifest["selected_sample_ids_sha256"],
                "dataset_fingerprint": probe_manifest["serialized_probe_dataset_fingerprint"],
                "frozen_spec": probe_spec_identity,
            },
            "encoding": {
                "batch_size": batch_size,
                "model_dtype": model_dtype,
                "storage_dtype": storage_dtype,
                "device": device,
                "flash_attention": flash_attention,
                "compressed": compressed,
                "max_length": 8192,
                "normalized": True,
                "dense_query_prompt": "query: " if family == "dense" else None,
                "dense_document_prompt": "document: " if family == "dense" else None,
                "late_query_expansion": False if family == "late" else None,
                "late_document_skiplist": True if family == "late" else None,
                "late_storage": "ragged_offsets" if family == "late" else None,
                "positive_candidate_index": 0,
            },
            "runtime": {
                "torch": importlib.metadata.version("torch"),
                "sentence_transformers": importlib.metadata.version("sentence-transformers"),
                "pylate": importlib.metadata.version("pylate"),
                "cuda": torch.version.cuda,
                "gpu_name": (
                    torch.cuda.get_device_name(torch.device(device))
                    if device.startswith("cuda")
                    else None
                ),
            },
            "output": {
                "path": str(output),
                "sha256": npz_sha256,
                "bytes": temporary.stat().st_size,
                "arrays": array_metadata,
            },
        }
        os.replace(temporary, output)
        _atomic_json(manifest_output, payload)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output, manifest_output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a fixed representation probe from a checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=Path("data/probes/training-1024-seed1729"))
    parser.add_argument(
        "--probe-spec",
        type=Path,
        default=Path("configs/representation_probe.json"),
        help="Frozen probe specification; pass --allow-unfrozen-probe for a custom probe",
    )
    parser.add_argument(
        "--allow-unfrozen-probe",
        action="store_true",
        help="Analyze a custom probe while retaining its manifest hashes in provenance",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=("dense", "late"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-flash-attention", action="store_true")
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output, manifest = export_probe(
        args.checkpoint,
        args.probe,
        args.output,
        family=args.family,
        batch_size=args.batch_size,
        model_dtype=args.model_dtype,
        storage_dtype=args.storage_dtype,
        device=args.device,
        flash_attention=not args.no_flash_attention,
        compressed=args.compressed,
        overwrite=args.overwrite,
        probe_spec=None if args.allow_unfrozen_probe else args.probe_spec,
    )
    print(f"Exported probe: {output}")
    print(f"Export manifest: {manifest}")


if __name__ == "__main__":
    main()
