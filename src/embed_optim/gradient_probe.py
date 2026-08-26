from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Literal

import torch
from datasets import Dataset

from .collators import TEXT_COLUMNS, DenseGroupCollator, LateGroupCollator
from .geometry import SCHEMA_VERSION, TensorStore, _atomic_json, _atomic_safetensors, _sha256
from .losses import ExplicitDenseInfoNCELoss, ExplicitLateInfoNCELoss
from .optimizers import parameter_partition, parameter_partition_name, partition_summary
from .probe_export import (
    ModelFamily,
    _checkpoint_inputs,
    _load_model,
    _load_probe,
    _validate_checkpoint_family,
    _validate_probe_spec,
)

DEFAULT_TEMPERATURES = {"dense": 0.02, "late": 0.001}


def _selection_key(seed: int, sample_id: int) -> bytes:
    return hashlib.blake2b(f"{seed}:{sample_id}".encode(), digest_size=16).digest()


def balanced_probe_indices(dataset: Dataset, count: int, seed: int) -> list[int]:
    """Choose a stable round-robin sample across probe source groups."""

    if count <= 0 or count > len(dataset):
        raise ValueError(f"count must be in [1, {len(dataset)}], got {count}")
    buckets: dict[str, list[tuple[bytes, int]]] = {}
    for index, (sample_id, source) in enumerate(
        zip(dataset["sample_id"], dataset["source"], strict=True)
    ):
        buckets.setdefault(str(source), []).append((_selection_key(seed, int(sample_id)), index))
    for values in buckets.values():
        values.sort()
    sources = sorted(buckets)
    offsets = {source: 0 for source in sources}
    selected = []
    while len(selected) < count:
        advanced = False
        for source in sources:
            offset = offsets[source]
            if offset >= len(buckets[source]):
                continue
            selected.append(buckets[source][offset][1])
            offsets[source] += 1
            advanced = True
            if len(selected) == count:
                break
        if not advanced:
            raise RuntimeError("Probe selection exhausted before reaching the requested count")
    return selected


def _selection_records(dataset: Dataset, indices: list[int]) -> tuple[list[dict[str, Any]], str]:
    records = [
        {
            "order": order,
            "dataset_index": index,
            "sample_id": int(dataset[index]["sample_id"]),
            "source": str(dataset[index]["source"]),
        }
        for order, index in enumerate(indices)
    ]
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            (
                f"{record['order']}\t{record['dataset_index']}\t"
                f"{record['sample_id']}\t{record['source']}\n"
            ).encode()
        )
    return records, digest.hexdigest()


def _sample_ids_sha256(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(f"{record['sample_id']}\n".encode())
    return digest.hexdigest()


def _validate_common_state_spec(
    spec_path: str | Path | None,
    *,
    probe_manifest_sha256: str,
    config: dict[str, Any],
    selection: list[dict[str, Any]],
    selection_sha256: str,
) -> dict[str, Any] | None:
    if spec_path is None:
        return None
    path = Path(spec_path).resolve()
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported common-state specification schema: {path}")
    if spec.get("probe_manifest_sha256") != probe_manifest_sha256:
        raise ValueError("Common-state specification points to a different frozen probe")
    expected_selection = spec.get("selection")
    expected_gradient = spec.get("gradient_protocol")
    if not isinstance(expected_selection, dict) or not isinstance(expected_gradient, dict):
        raise ValueError("Common-state specification lacks selection or gradient_protocol")
    observed_selection = {
        "seed": config["seed"],
        "gradient_steps": config["gradient_steps"],
        "examples_per_gradient": config["examples_per_gradient"],
        "count": len(selection),
        "expected_selection_sha256": selection_sha256,
        "expected_sample_ids_sha256": _sample_ids_sha256(selection),
        "expected_source_counts": dict(
            sorted(
                {
                    source: sum(record["source"] == source for record in selection)
                    for source in {record["source"] for record in selection}
                }.items()
            )
        ),
    }
    if observed_selection != expected_selection:
        raise ValueError(
            "Common-state selection differs from its frozen specification: "
            f"expected={expected_selection}, observed={observed_selection}"
        )
    observed_gradient = {
        "micro_batch_size": config["micro_batch_size"],
        "max_grad_norm": config["max_grad_norm"],
        "model_dtype": config["model_dtype"],
        "forward_dtype": config["forward_dtype"],
        "storage_dtype": config["storage_dtype"],
        "model_mode": config["model_mode"],
        "weights_advanced": config["weights_advanced"],
    }
    if observed_gradient != expected_gradient:
        raise ValueError(
            "Common-state gradient protocol differs from its frozen specification: "
            f"expected={expected_gradient}, observed={observed_gradient}"
        )
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _probe_components(model: Any, family: ModelFamily, temperature: float):
    if family == "dense":
        return ExplicitDenseInfoNCELoss(model, temperature), DenseGroupCollator(model.preprocess)
    return ExplicitLateInfoNCELoss(model, temperature), LateGroupCollator(model)


def _collect_features(batch: dict[str, Any], device: str) -> list[dict[str, torch.Tensor]]:
    features = []
    for column in TEXT_COLUMNS:
        prefix = f"{column}_"
        feature = {
            key[len(prefix) :]: value.to(device)
            for key, value in batch.items()
            if key.startswith(prefix) and isinstance(value, torch.Tensor)
        }
        if "input_ids" not in feature:
            raise ValueError(f"Collator output has no input_ids for {column!r}")
        features.append(feature)
    return features


def _temperature_from_checkpoint(
    checkpoint: Path, family: ModelFamily, requested: float | None
) -> float:
    run_config_path = checkpoint.parent / "run_config.json"
    declared = None
    if run_config_path.is_file():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if run_config.get("temperature") is not None:
            declared = float(run_config["temperature"])
    temperature = requested if requested is not None else declared
    if temperature is None:
        temperature = DEFAULT_TEMPERATURES[family]
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if requested is not None and declared is not None and not math.isclose(requested, declared):
        raise ValueError(
            f"Requested temperature {requested} differs from checkpoint run config {declared}"
        )
    return temperature


def _verify_shards(root: Path, manifest: dict[str, Any]) -> None:
    for item in manifest.get("gradient_shards", []):
        path = root / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or _sha256(path) != item.get("sha256")
        ):
            raise ValueError(f"Gradient shard does not match its manifest: {path}")


def _hidden_parameter_mapping(
    hidden: list[tuple[str, torch.nn.Parameter]], checkpoint: Path
) -> list[tuple[str, str, torch.nn.Parameter]]:
    """Map runtime module names to canonical SentenceTransformers safetensor names."""

    with TensorStore(checkpoint) as checkpoint_store:
        checkpoint_hidden = {
            name: checkpoint_store.shape(name)
            for name in checkpoint_store.keys()
            if parameter_partition_name(name, len(checkpoint_store.shape(name))) == "hidden"
        }
    mapping = []
    used_checkpoint_names: set[str] = set()
    for model_name, parameter in hidden:
        candidates = [model_name]
        if ".model." in model_name:
            candidates.append(model_name.replace(".model.", ".", 1))
        if model_name.startswith("model."):
            candidates.append(model_name[len("model.") :])
        compatible = [
            name
            for name in candidates
            if name in checkpoint_hidden and checkpoint_hidden[name] == tuple(parameter.shape)
        ]
        if len(compatible) != 1:
            raise ValueError(
                f"Cannot uniquely map runtime hidden tensor {model_name!r} with shape "
                f"{tuple(parameter.shape)} into checkpoint storage; candidates={compatible}"
            )
        checkpoint_name = compatible[0]
        if checkpoint_name in used_checkpoint_names:
            raise ValueError(
                f"Multiple runtime tensors map to checkpoint tensor {checkpoint_name!r}"
            )
        used_checkpoint_names.add(checkpoint_name)
        mapping.append((model_name, checkpoint_name, parameter))
    missing = sorted(set(checkpoint_hidden) - used_checkpoint_names)
    if missing:
        raise ValueError(f"Checkpoint hidden tensors have no runtime parameter mapping: {missing}")
    return mapping


def export_gradient_probe(
    checkpoint: str | Path,
    probe_root: str | Path,
    output_dir: str | Path,
    *,
    family: ModelFamily,
    probe_spec: str | Path | None = None,
    common_state_spec: str | Path | None = None,
    gradient_steps: int = 8,
    examples_per_gradient: int = 4,
    micro_batch_size: int = 1,
    seed: int = 2718,
    temperature: float | None = None,
    max_grad_norm: float = 1.0,
    model_dtype: Literal["bfloat16", "float32"] = "float32",
    forward_dtype: Literal["bfloat16", "float32"] = "bfloat16",
    storage_dtype: Literal["float16", "float32"] = "float32",
    device: str = "cuda",
    flash_attention: bool = True,
    train_mode: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export clipped hidden-matrix gradients without changing checkpoint weights."""

    checkpoint = Path(checkpoint).resolve()
    probe_root = Path(probe_root).resolve()
    output_dir = Path(output_dir).resolve()
    if family not in {"dense", "late"}:
        raise ValueError(f"Unsupported family {family!r}")
    for name, value in (
        ("gradient_steps", gradient_steps),
        ("examples_per_gradient", examples_per_gradient),
        ("micro_batch_size", micro_batch_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if max_grad_norm <= 0:
        raise ValueError(f"max_grad_norm must be positive, got {max_grad_norm}")
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError(f"Unsupported storage dtype {storage_dtype!r}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA output was requested but CUDA is unavailable")
    if not device.startswith("cuda") and flash_attention:
        raise ValueError("FlashAttention gradient export requires CUDA")
    if device == "cpu" and (model_dtype == "bfloat16" or forward_dtype == "bfloat16"):
        raise ValueError("Use float32 model and forward dtypes for CPU gradient export")

    dataset, probe_manifest, probe_manifest_sha256 = _load_probe(probe_root)
    probe_spec_identity = _validate_probe_spec(probe_spec, probe_manifest_sha256)
    sample_count = gradient_steps * examples_per_gradient
    indices = balanced_probe_indices(dataset, sample_count, seed)
    selection, selection_sha256 = _selection_records(dataset, indices)
    effective_temperature = _temperature_from_checkpoint(checkpoint, family, temperature)
    checkpoint_inputs = _checkpoint_inputs(checkpoint)
    checkpoint_run_config = _validate_checkpoint_family(checkpoint, family)
    config = {
        "family": family,
        "gradient_steps": gradient_steps,
        "examples_per_gradient": examples_per_gradient,
        "micro_batch_size": micro_batch_size,
        "seed": seed,
        "temperature": effective_temperature,
        "max_grad_norm": max_grad_norm,
        "model_dtype": model_dtype,
        "forward_dtype": forward_dtype,
        "storage_dtype": storage_dtype,
        "device": device,
        "flash_attention": flash_attention,
        "model_mode": "train" if train_mode else "eval",
        "parameter_partition": "hidden",
        "weights_advanced": False,
    }
    common_state_spec_identity = _validate_common_state_spec(
        common_state_spec,
        probe_manifest_sha256=probe_manifest_sha256,
        config=config,
        selection=selection,
        selection_sha256=selection_sha256,
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": {
            "path": str(checkpoint),
            "inputs": checkpoint_inputs,
            "run_config": checkpoint_run_config,
        },
        "probe": {
            "path": str(probe_root),
            "manifest_sha256": probe_manifest_sha256,
            "selection_sha256": probe_manifest["selection_sha256"],
            "selected_sample_ids_sha256": probe_manifest["selected_sample_ids_sha256"],
            "dataset_fingerprint": probe_manifest["serialized_probe_dataset_fingerprint"],
            "frozen_spec": probe_spec_identity,
        },
        "config": config,
        "selection": selection,
        "selection_sha256": selection_sha256,
        "common_state_spec": common_state_spec_identity,
    }
    manifest_path = output_dir / "manifest.json"
    if overwrite and output_dir.is_dir():
        declared_paths: set[Path] = set(output_dir.glob("gradient-*.safetensors"))
        if manifest_path.is_file():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            declared_paths.update(
                output_dir / item["path"] for item in previous.get("gradient_shards", [])
            )
        for path in sorted(declared_paths):
            if path.is_file():
                path.unlink()
        if manifest_path.is_file():
            manifest_path.unlink()

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if {key: manifest.get(key) for key in identity} != identity:
            raise ValueError("Existing gradient export does not match the requested inputs")
        _verify_shards(output_dir, manifest)
        if manifest.get("status") == "complete":
            if len(manifest["gradient_shards"]) != gradient_steps:
                raise ValueError("Completed gradient manifest has the wrong shard count")
            mapping = manifest.get("parameter_name_mapping")
            hidden_count = (
                (manifest.get("partition_summary") or {}).get("hidden", {}).get("tensors")
            )
            if not isinstance(mapping, list) or len(mapping) != hidden_count:
                raise ValueError("Completed gradient manifest has no complete parameter mapping")
            return manifest
    else:
        existing = sorted(output_dir.glob("gradient-*.safetensors")) if output_dir.is_dir() else []
        if existing:
            raise FileExistsError(f"Unmanifested gradient shards exist under {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            **identity,
            "status": "in_progress",
            "partition_summary": None,
            "gradient_shards": [],
            "runtime": None,
        }
        _atomic_json(manifest_path, manifest)

    completed_steps = len(manifest["gradient_shards"])
    if completed_steps > gradient_steps:
        raise ValueError("Gradient manifest contains more shards than requested")
    torch_dtype = torch.bfloat16 if model_dtype == "bfloat16" else torch.float32
    output_dtype = torch.float16 if storage_dtype == "float16" else torch.float32
    model = _load_model(
        family,
        checkpoint,
        dtype=torch_dtype,
        device=device,
        flash_attention=flash_attention,
    )
    loss, collator = _probe_components(model, family, effective_temperature)
    model.train(mode=train_mode)
    partition = parameter_partition(model)
    hidden = partition["hidden"]
    if not hidden:
        raise ValueError("Loaded checkpoint has no hidden matrices routed to Muon")
    observed_partition = partition_summary(partition)
    if manifest.get("partition_summary") not in (None, observed_partition):
        raise ValueError("Loaded checkpoint partition differs from the resumed gradient export")
    hidden_mapping = _hidden_parameter_mapping(hidden, checkpoint)
    mapping_records = [
        {
            "model_name": model_name,
            "checkpoint_name": checkpoint_name,
            "shape": list(parameter.shape),
        }
        for model_name, checkpoint_name, parameter in hidden_mapping
    ]
    if manifest.get("parameter_name_mapping") not in (None, mapping_records):
        raise ValueError("Runtime/checkpoint parameter mapping differs from resumed export")
    completed_path = checkpoint.parent / "completed.json"
    if completed_path.is_file():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        declared_partition = completed.get("optimizer_partition")
        if declared_partition is not None and declared_partition != observed_partition:
            raise ValueError("Loaded checkpoint partition differs from completed.json")
    manifest["partition_summary"] = observed_partition
    manifest["parameter_name_mapping"] = mapping_records

    try:
        for step_index in range(completed_steps, gradient_steps):
            torch.manual_seed(seed + step_index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + step_index)
            start = step_index * examples_per_gradient
            step_indices = indices[start : start + examples_per_gradient]
            model.zero_grad(set_to_none=True)
            mean_loss = 0.0
            for micro_start in range(0, len(step_indices), micro_batch_size):
                micro_indices = step_indices[micro_start : micro_start + micro_batch_size]
                rows = [dataset[index] for index in micro_indices]
                features = _collect_features(collator(rows), device)
                device_type = torch.device(device).type
                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.bfloat16,
                    enabled=forward_dtype == "bfloat16",
                ):
                    micro_loss = loss(features)
                if not torch.isfinite(micro_loss):
                    raise ValueError(f"Non-finite loss in gradient shard {step_index}")
                weight = len(micro_indices) / len(step_indices)
                (micro_loss * weight).backward()
                mean_loss += float(micro_loss.detach().float().item()) * weight

            pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_grad_norm
            )
            gradients = {}
            for model_name, checkpoint_name, parameter in hidden_mapping:
                if parameter.grad is None:
                    raise ValueError(f"Hidden tensor {model_name!r} has no gradient")
                gradient = parameter.grad.detach().to(device="cpu", dtype=output_dtype)
                if not torch.isfinite(gradient).all():
                    raise ValueError(f"Hidden tensor {model_name!r} has a non-finite gradient")
                gradients[checkpoint_name] = gradient.contiguous()
            path = output_dir / f"gradient-{step_index:04d}.safetensors"
            _atomic_safetensors(
                path,
                gradients,
                metadata={
                    "schema_version": str(SCHEMA_VERSION),
                    "step_index": str(step_index),
                    "storage_dtype": storage_dtype,
                    "weights_advanced": "false",
                },
            )
            sample_ids = [int(dataset[index]["sample_id"]) for index in step_indices]
            manifest["gradient_shards"].append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "step_index": step_index,
                    "sample_ids": sample_ids,
                    "mean_loss": mean_loss,
                    "pre_clip_grad_norm": float(pre_clip_norm.detach().float().item()),
                    "clip_coefficient": min(
                        1.0,
                        max_grad_norm / (float(pre_clip_norm.detach().float().item()) + 1e-6),
                    ),
                }
            )
            _atomic_json(manifest_path, manifest)
    finally:
        del loss, collator, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest["status"] = "complete"
    manifest["runtime"] = {
        "torch": importlib.metadata.version("torch"),
        "sentence_transformers": importlib.metadata.version("sentence-transformers"),
        "pylate": importlib.metadata.version("pylate"),
        "cuda": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None
        ),
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export deterministic common-state gradient sequences from a frozen probe"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=Path("data/probes/training-1024-seed1729"))
    parser.add_argument(
        "--probe-spec", type=Path, default=Path("configs/representation_probe.json")
    )
    parser.add_argument(
        "--common-state-spec", type=Path, default=Path("configs/common_state_probe.json")
    )
    parser.add_argument("--allow-unfrozen-probe", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family", choices=("dense", "late"), required=True)
    parser.add_argument("--gradient-steps", type=int, default=8)
    parser.add_argument("--examples-per-gradient", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--model-dtype", choices=("bfloat16", "float32"), default="float32")
    parser.add_argument("--forward-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-flash-attention", action="store_true")
    parser.add_argument("--train-mode", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest = export_gradient_probe(
        args.checkpoint,
        args.probe,
        args.output_dir,
        family=args.family,
        probe_spec=None if args.allow_unfrozen_probe else args.probe_spec,
        common_state_spec=None if args.allow_unfrozen_probe else args.common_state_spec,
        gradient_steps=args.gradient_steps,
        examples_per_gradient=args.examples_per_gradient,
        micro_batch_size=args.micro_batch_size,
        seed=args.seed,
        temperature=args.temperature,
        max_grad_norm=args.max_grad_norm,
        model_dtype=args.model_dtype,
        forward_dtype=args.forward_dtype,
        storage_dtype=args.storage_dtype,
        device=args.device,
        flash_attention=not args.no_flash_attention,
        train_mode=args.train_mode,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "manifest": str((args.output_dir / "manifest.json").resolve()),
                "gradient_shards": len(manifest["gradient_shards"]),
                "partition_summary": manifest["partition_summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
