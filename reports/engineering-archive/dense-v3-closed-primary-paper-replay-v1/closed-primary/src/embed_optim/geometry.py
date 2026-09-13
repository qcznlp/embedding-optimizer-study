from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .optimizers import parameter_partition_name

SCHEMA_VERSION = 1
PARTITIONS = ("hidden", "aux_decay", "aux_no_decay")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_safetensors(
    path: Path,
    tensors: dict[str, torch.Tensor],
    *,
    metadata: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}.safetensors")
    try:
        save_file(tensors, temporary, metadata=metadata)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


TensorLocation = tuple[Path, str]


def _resolve_plain_directory(source: Path) -> tuple[list[Path], dict[str, TensorLocation] | None]:
    single = source / "model.safetensors"
    if single.is_file():
        return [single.resolve()], None

    index = source / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid safetensors weight_map in {index}")
        mapping = {
            str(name): ((source / str(filename)).resolve(), str(name))
            for name, filename in weight_map.items()
        }
        files = sorted({location[0] for location in mapping.values()})
        missing = [path for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing safetensors shard(s): {missing}")
        return files, mapping

    files = sorted(path.resolve() for path in source.glob("*.safetensors"))
    if not files:
        return [], None
    return files, None


def _resolve_source(source: Path) -> tuple[list[Path], dict[str, TensorLocation] | None]:
    source = source.resolve()
    if source.is_file():
        if source.suffix != ".safetensors":
            raise ValueError(f"Expected a .safetensors file, got {source}")
        return [source], None
    if not source.is_dir():
        raise FileNotFoundError(source)

    modules_path = source / "modules.json"
    if modules_path.is_file():
        modules = json.loads(modules_path.read_text(encoding="utf-8"))
        if not isinstance(modules, list):
            raise ValueError(f"Invalid SentenceTransformers module list in {modules_path}")
        files: set[Path] = set()
        mapping: dict[str, TensorLocation] = {}
        for module in modules:
            if not isinstance(module, dict):
                raise ValueError(f"Invalid module entry in {modules_path}: {module!r}")
            prefix = str(module.get("name", module.get("idx", "")))
            relative_path = str(module.get("path", ""))
            module_dir = (source / relative_path).resolve() if relative_path else source
            module_files, declared = _resolve_plain_directory(module_dir)
            files.update(module_files)
            if declared is None:
                for path in module_files:
                    with safe_open(str(path), framework="pt", device="cpu") as handle:
                        local_names = list(handle.keys())
                    for local_name in local_names:
                        canonical = f"{prefix}.{local_name}" if prefix else local_name
                        if canonical in mapping:
                            raise ValueError(f"Duplicate tensor {canonical!r} under {source}")
                        mapping[canonical] = (path, local_name)
            else:
                for local_name, location in declared.items():
                    canonical = f"{prefix}.{local_name}" if prefix else local_name
                    if canonical in mapping:
                        raise ValueError(f"Duplicate tensor {canonical!r} under {source}")
                    mapping[canonical] = location
        if not files or not mapping:
            raise FileNotFoundError(f"No module safetensors found under {source}")
        return sorted(files), mapping

    files, mapping = _resolve_plain_directory(source)
    if not files:
        raise FileNotFoundError(f"No safetensors files found under {source}")
    return files, mapping


class TensorStore:
    """A streaming reader for single-file or sharded safetensors checkpoints."""

    def __init__(self, source: Path) -> None:
        self.source = source.resolve()
        self.files, self._declared_mapping = _resolve_source(self.source)
        self._stack: ExitStack | None = None
        self._handles: dict[Path, Any] = {}
        self._mapping: dict[str, TensorLocation] = {}

    def __enter__(self) -> TensorStore:
        self._stack = ExitStack()
        for path in self.files:
            self._handles[path] = self._stack.enter_context(
                safe_open(str(path), framework="pt", device="cpu")
            )
        if self._declared_mapping is None:
            for path, handle in self._handles.items():
                for name in handle.keys():
                    if name in self._mapping:
                        raise ValueError(f"Duplicate tensor {name!r} across {self.source}")
                    self._mapping[name] = (path, name)
        else:
            self._mapping = dict(self._declared_mapping)
            for name, (path, local_name) in self._mapping.items():
                if local_name not in self._handles[path].keys():
                    raise ValueError(
                        f"Index maps {name!r} to {path.name}, but the tensor is absent"
                    )
        return self

    def __exit__(self, *args: object) -> None:
        assert self._stack is not None
        self._stack.close()
        self._stack = None
        self._handles.clear()
        self._mapping.clear()

    def keys(self) -> list[str]:
        return sorted(self._mapping)

    def shape(self, name: str) -> tuple[int, ...]:
        path, local_name = self._mapping[name]
        return tuple(self._handles[path].get_slice(local_name).get_shape())

    def tensor(self, name: str) -> torch.Tensor:
        path, local_name = self._mapping[name]
        return self._handles[path].get_tensor(local_name)


def _gini(values: torch.Tensor) -> float:
    values = values.detach().float().flatten()
    if values.numel() == 0:
        return 0.0
    total = values.sum()
    if total <= 0:
        return 0.0
    ordered = values.sort().values
    indices = torch.arange(1, ordered.numel() + 1, dtype=torch.float64)
    numerator = ((2 * indices - ordered.numel() - 1) * ordered.double()).sum()
    return float((numerator / (ordered.numel() * total.double())).item())


def _norm_distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten()
    if values.numel() == 0:
        return {"mean": 0.0, "cv": 0.0, "gini": 0.0, "max_to_median": 0.0}
    mean = values.mean()
    median = values.median()
    cv = values.std(unbiased=False) / mean if mean > 0 else torch.zeros(())
    ratio = values.max() / median if median > 0 else torch.zeros(())
    return {
        "mean": float(mean.item()),
        "cv": float(cv.item()),
        "gini": _gini(values),
        "max_to_median": float(ratio.item()),
    }


def _top_energy_fraction(row_norms: torch.Tensor, fraction: float) -> float:
    energy = row_norms.float().square()
    total = energy.sum()
    if total <= 0:
        return 0.0
    count = max(1, math.ceil(energy.numel() * fraction))
    return float((energy.topk(count).values.sum() / total).item())


def _singular_summary(
    matrix: torch.Tensor,
    *,
    sketch_rank: int,
    oversample: int,
    power_iterations: int,
    seed: int,
) -> dict[str, Any]:
    rows, columns = matrix.shape
    limit = min(rows, columns)
    frobenius_sq = matrix.square().sum()
    if frobenius_sq <= 0 or sketch_rank == 0:
        return {
            "algorithm": "disabled" if sketch_rank == 0 else "zero",
            "rank": 0,
            "spectral_norm": 0.0,
            "approx_stable_rank": 0.0,
            "sketched_nuclear_norm": 0.0,
            "sketched_entropy_effective_rank": 0.0,
            "sketched_condition_number": None,
            "captured_frobenius_energy": 0.0,
        }

    rank = min(sketch_rank, limit)
    if rank == limit:
        singular_values = torch.linalg.svdvals(matrix)
        algorithm = "exact"
    else:
        width = min(limit, rank + oversample)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        omega = torch.randn(columns, width, generator=generator, dtype=matrix.dtype)
        sample = matrix @ omega
        for _ in range(power_iterations):
            basis = torch.linalg.qr(sample, mode="reduced").Q
            sample = matrix @ (matrix.T @ basis)
        basis = torch.linalg.qr(sample, mode="reduced").Q
        projected = basis.T @ matrix
        singular_values = torch.linalg.svdvals(projected)[:rank]
        algorithm = "randomized"

    singular_values = singular_values.float()
    spectral = singular_values[0]
    nuclear = singular_values.sum()
    probabilities = singular_values / nuclear if nuclear > 0 else torch.zeros_like(singular_values)
    nonzero = probabilities > 0
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    threshold = torch.finfo(singular_values.dtype).eps * max(rows, columns) * spectral
    usable = singular_values[singular_values > threshold]
    condition = float((usable[0] / usable[-1]).item()) if usable.numel() else None
    return {
        "algorithm": algorithm,
        "rank": int(singular_values.numel()),
        "spectral_norm": float(spectral.item()),
        "approx_stable_rank": float((frobenius_sq / spectral.square()).item()),
        "sketched_nuclear_norm": float(nuclear.item()),
        "sketched_entropy_effective_rank": float(entropy.exp().item()),
        "sketched_condition_number": condition,
        "captured_frobenius_energy": float(
            (singular_values.square().sum() / frobenius_sq).clamp(max=1).item()
        ),
    }


def matrix_metrics(
    tensor: torch.Tensor,
    *,
    sketch_rank: int = 64,
    oversample: int = 8,
    power_iterations: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    if tensor.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {tuple(tensor.shape)}")
    matrix = tensor.detach().to(device="cpu", dtype=torch.float32)
    row_norms = torch.linalg.vector_norm(matrix, dim=1)
    column_norms = torch.linalg.vector_norm(matrix, dim=0)
    result: dict[str, Any] = {
        "frobenius_norm": float(torch.linalg.vector_norm(matrix).item()),
        "row_norms": _norm_distribution(row_norms),
        "column_norms": _norm_distribution(column_norms),
        "top_1pct_row_energy": _top_energy_fraction(row_norms, 0.01),
        "top_10pct_row_energy": _top_energy_fraction(row_norms, 0.10),
    }
    result.update(
        _singular_summary(
            matrix,
            sketch_rank=sketch_rank,
            oversample=oversample,
            power_iterations=power_iterations,
            seed=seed,
        )
    )
    return result


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.detach().float().flatten()
    right = right.detach().float().flatten()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if denominator <= 0:
        return None
    return float((torch.dot(left, right) / denominator).item())


def _partition_summary(store: TensorStore) -> dict[str, dict[str, int]]:
    summary = {name: {"tensors": 0, "parameters": 0} for name in PARTITIONS}
    for name in store.keys():
        shape = store.shape(name)
        partition = parameter_partition_name(name, len(shape))
        summary[partition]["tensors"] += 1
        summary[partition]["parameters"] += math.prod(shape)
    return summary


def _source_provenance(source: Path) -> dict[str, Any]:
    files, _ = _resolve_source(source)
    source = source.resolve()
    metadata_files = []
    if source.is_dir():
        metadata_files = sorted(
            {
                *source.rglob("modules.json"),
                *source.rglob("model.safetensors.index.json"),
            }
        )
    return {
        "source": str(source),
        "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in files
        ],
        "metadata_files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in metadata_files
        ],
    }


def _record_provenance(record_path: Path, output_dir: Path) -> dict[str, Any]:
    tensor_count = sum(1 for line in record_path.read_text(encoding="utf-8").splitlines() if line)
    return {
        "path": str(record_path.relative_to(output_dir)),
        "tensors": tensor_count,
        "bytes": record_path.stat().st_size,
        "sha256": _sha256(record_path),
    }


def _checkpoint_paths(run_dir: Path, completed: dict[str, Any]) -> list[tuple[int, Path]]:
    checkpoints = completed.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError(f"No checkpoint list in {run_dir / 'completed.json'}")
    result = []
    for raw_step in checkpoints:
        step = int(raw_step)
        path = run_dir / f"checkpoint-{step}"
        _resolve_source(path)
        result.append((step, path))
    return sorted(result)


def _validate_partitions(partitions: tuple[str, ...]) -> None:
    invalid = sorted(set(partitions) - set(PARTITIONS))
    if invalid:
        raise ValueError(f"Unknown partition(s) {invalid}; choose from {PARTITIONS}")
    if not partitions:
        raise ValueError("At least one partition is required")


def analyze_run(
    run_dir: Path,
    output_dir: Path,
    *,
    reference: Path | None = None,
    partitions: tuple[str, ...] = ("hidden",),
    sketch_rank: int = 64,
    oversample: int = 8,
    power_iterations: int = 2,
    seed: int = 42,
    max_checkpoints: int | None = None,
    steps: tuple[int, ...] | None = None,
    tensor_regex: str | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    _validate_partitions(partitions)
    if sketch_rank < 0 or oversample < 0 or power_iterations < 0:
        raise ValueError("Sketch settings must be non-negative")

    completed_path = run_dir / "completed.json"
    run_config_path = run_dir / "run_config.json"
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    checkpoints = _checkpoint_paths(run_dir, completed)
    if max_checkpoints is not None and steps is not None:
        raise ValueError("max_checkpoints and steps are mutually exclusive")
    if max_checkpoints is not None:
        if max_checkpoints <= 0:
            raise ValueError("max_checkpoints must be positive")
        checkpoints = checkpoints[:max_checkpoints]
    if steps is not None:
        if not steps or len(set(steps)) != len(steps):
            raise ValueError("steps must contain distinct checkpoint steps")
        requested_steps = set(steps)
        available_steps = {step for step, _ in checkpoints}
        missing_steps = sorted(requested_steps - available_steps)
        if missing_steps:
            raise ValueError(f"Unknown checkpoint step(s): {missing_steps}")
        checkpoints = [item for item in checkpoints if item[0] in requested_steps]
    pattern = re.compile(tensor_regex) if tensor_regex is not None else None

    input_provenance = [
        {"kind": "checkpoint", "step": step, **_source_provenance(path)}
        for step, path in checkpoints
    ]
    if reference is not None:
        input_provenance.append({"kind": "reference", **_source_provenance(reference)})

    analysis_config = {
        "partitions": list(partitions),
        "sketch_rank": sketch_rank,
        "oversample": oversample,
        "power_iterations": power_iterations,
        "seed": seed,
    }
    if steps is not None:
        analysis_config["steps"] = sorted(steps)
    if tensor_regex is not None:
        analysis_config["tensor_regex"] = tensor_regex
    identity = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "run_dir": str(run_dir),
            "run_id": completed.get("run_id"),
            "model_family": completed.get("model_family"),
            "optimizer": run_config.get("optimizer"),
            "dataset_fingerprint": completed.get("dataset_fingerprint"),
        },
        "analysis_config": analysis_config,
        "inputs": input_provenance,
        "reference": str(reference.resolve()) if reference is not None else None,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed_identity = {key: existing.get(key) for key in identity}
        if observed_identity != identity:
            raise ValueError(
                f"Existing analysis manifest does not match the requested inputs: {manifest_path}"
            )
        manifest = existing
    else:
        manifest = {**identity, "partition_summary": None, "records": {}}
        _atomic_json(manifest_path, manifest)

    previous_path: Path | None = None
    expected_summary = completed.get("optimizer_partition")
    for step, checkpoint_path in checkpoints:
        record_path = output_dir / "records" / f"checkpoint-{step}.jsonl"
        if record_path.exists():
            observed_record = _record_provenance(record_path, output_dir)
            declared_record = manifest["records"].get(str(step))
            if declared_record is not None and declared_record != observed_record:
                raise ValueError(f"Existing record does not match its manifest: {record_path}")
            if declared_record is None:
                manifest["records"][str(step)] = observed_record
                _atomic_json(manifest_path, manifest)
            previous_path = checkpoint_path
            continue

        with ExitStack() as stack:
            current = stack.enter_context(TensorStore(checkpoint_path))
            previous = stack.enter_context(TensorStore(previous_path)) if previous_path else None
            reference_store = stack.enter_context(TensorStore(reference)) if reference else None
            previous_names = set(previous.keys()) if previous is not None else set()
            reference_names = set(reference_store.keys()) if reference_store is not None else set()
            summary = _partition_summary(current)
            if expected_summary is not None and summary != expected_summary:
                raise ValueError(
                    f"Checkpoint {step} partition {summary} does not match completed.json "
                    f"partition {expected_summary}"
                )
            if manifest.get("partition_summary") not in (None, summary):
                raise ValueError(f"Checkpoint {step} partition differs from earlier checkpoints")

            records: list[dict[str, Any]] = []
            for tensor_index, name in enumerate(current.keys()):
                shape = current.shape(name)
                partition = parameter_partition_name(name, len(shape))
                if partition not in partitions:
                    continue
                if pattern is not None and pattern.search(name) is None:
                    continue
                if len(shape) != 2:
                    raise ValueError(f"Selected tensor {name!r} is not two-dimensional")
                weight = current.tensor(name)
                metric_seed = seed + step * 1009 + tensor_index
                record: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "step": step,
                    "tensor": name,
                    "shape": list(shape),
                    "parameters": math.prod(shape),
                    "partition": partition,
                    "weight": matrix_metrics(
                        weight,
                        sketch_rank=sketch_rank,
                        oversample=oversample,
                        power_iterations=power_iterations,
                        seed=metric_seed,
                    ),
                }
                if previous is not None:
                    if name not in previous_names or previous.shape(name) != shape:
                        raise ValueError(
                            f"Tensor {name!r} is incompatible with the previous checkpoint"
                        )
                    delta = weight.float() - previous.tensor(name).float()
                    record["delta_from_previous"] = matrix_metrics(
                        delta,
                        sketch_rank=sketch_rank,
                        oversample=oversample,
                        power_iterations=power_iterations,
                        seed=metric_seed + 1,
                    )
                    record["delta_from_previous"]["cosine_with_weight"] = _cosine(delta, weight)
                if reference_store is not None:
                    if name not in reference_names or reference_store.shape(name) != shape:
                        raise ValueError(
                            f"Tensor {name!r} is incompatible with the reference model"
                        )
                    delta = weight.float() - reference_store.tensor(name).float()
                    record["delta_from_reference"] = matrix_metrics(
                        delta,
                        sketch_rank=sketch_rank,
                        oversample=oversample,
                        power_iterations=power_iterations,
                        seed=metric_seed + 2,
                    )
                    record["delta_from_reference"]["cosine_with_weight"] = _cosine(delta, weight)
                records.append(record)

            if not records:
                raise ValueError(
                    f"No tensors matched partitions={partitions} tensor_regex={tensor_regex!r}"
                )

        _atomic_jsonl(record_path, records)
        manifest["partition_summary"] = summary
        manifest["records"][str(step)] = _record_provenance(record_path, output_dir)
        _atomic_json(manifest_path, manifest)
        previous_path = checkpoint_path

    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream checkpoint tensors and record optimizer-partition-aware geometry"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--partitions", nargs="+", default=["hidden"], choices=PARTITIONS)
    parser.add_argument("--sketch-rank", type=int, default=64)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--power-iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-checkpoints", type=int)
    parser.add_argument("--steps", nargs="+", type=int)
    parser.add_argument("--tensor-regex")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest = analyze_run(
        args.run_dir,
        args.output_dir,
        reference=args.reference,
        partitions=tuple(args.partitions),
        sketch_rank=args.sketch_rank,
        oversample=args.oversample,
        power_iterations=args.power_iterations,
        seed=args.seed,
        max_checkpoints=args.max_checkpoints,
        steps=tuple(args.steps) if args.steps is not None else None,
        tensor_regex=args.tensor_regex,
    )
    print(
        json.dumps(
            {
                "manifest": str((args.output_dir / "manifest.json").resolve()),
                "checkpoints": len(manifest["records"]),
                "partition_summary": manifest["partition_summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
