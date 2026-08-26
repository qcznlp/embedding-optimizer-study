from __future__ import annotations

import argparse
import json
import math
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors import safe_open

from .geometry import (
    SCHEMA_VERSION,
    TensorStore,
    _atomic_json,
    _atomic_jsonl,
    _atomic_safetensors,
    _cosine,
    _sha256,
    matrix_metrics,
)
from .optimizers import _muon_update, _normuon_update, parameter_partition_name
from .probe_export import _checkpoint_inputs

ALGORITHMS = ("adamw", "muon", "normuon")


@dataclass(frozen=True)
class UpdateOperatorConfig:
    """State-transition settings shared with the formal training optimizers."""

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    muon_momentum: float = 0.95
    normuon_beta2: float = 0.95
    ns_steps: int = 5
    adjust_lr_fn: str = "original"

    def validate(self) -> None:
        for name, value in (
            ("adam_beta1", self.adam_beta1),
            ("adam_beta2", self.adam_beta2),
            ("muon_momentum", self.muon_momentum),
            ("normuon_beta2", self.normuon_beta2),
        ):
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1), got {value}")
        if self.adam_eps <= 0:
            raise ValueError(f"adam_eps must be positive, got {self.adam_eps}")
        if self.ns_steps <= 0:
            raise ValueError(f"ns_steps must be positive, got {self.ns_steps}")
        if self.adjust_lr_fn not in {"original", "match_rms_adamw", "none"}:
            raise ValueError(f"Unsupported adjust_lr_fn {self.adjust_lr_fn!r}")


def _adamw_update_direction(
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    step: int,
    beta1: float,
    beta2: float,
    eps: float,
) -> torch.Tensor:
    """Advance Adam's state and return its data-gradient direction before LR and decay."""

    exp_avg.lerp_(gradient, 1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    return (exp_avg / bias_correction1) / ((exp_avg_sq / bias_correction2).sqrt() + eps)


def _muon_aspect_scale(shape: torch.Size, adjust_lr_fn: str) -> float:
    rows, columns = shape
    if adjust_lr_fn == "original":
        return math.sqrt(max(1, rows / columns))
    if adjust_lr_fn == "match_rms_adamw":
        return 0.2 * math.sqrt(max(rows, columns))
    return 1.0


def replay_update_directions(
    gradients: list[torch.Tensor],
    config: UpdateOperatorConfig = UpdateOperatorConfig(),
    *,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Replay one fixed-state gradient sequence through all three optimizer states.

    The parameters are deliberately not advanced. This separates each optimizer's stateful
    transformation from trajectory-dependent changes in the model and its gradients. Returned
    tensors are the data-update directions per unit base learning rate; weight decay is excluded.
    """

    config.validate()
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA update replay was requested but CUDA is unavailable")
    if not gradients:
        raise ValueError("At least one gradient is required")
    shape = gradients[0].shape
    if len(shape) != 2:
        raise ValueError(f"Expected matrix gradients, got {tuple(shape)}")
    normalized = []
    for index, gradient in enumerate(gradients):
        if gradient.shape != shape:
            raise ValueError(
                f"Gradient {index} has shape {tuple(gradient.shape)}, expected {tuple(shape)}"
            )
        value = gradient.detach().to(device=target_device, dtype=torch.float32)
        if not torch.isfinite(value).all():
            raise ValueError(f"Gradient {index} contains a non-finite value")
        normalized.append(value)

    adam_momentum = torch.zeros_like(normalized[0])
    adam_second_moment = torch.zeros_like(normalized[0])
    muon_momentum = torch.zeros_like(normalized[0])
    normuon_momentum = torch.zeros_like(normalized[0])
    normuon_second_moment = torch.zeros_like(normalized[0][..., :1])
    adam_direction = torch.empty_like(normalized[0])
    muon_direction = torch.empty_like(normalized[0])
    normuon_direction = torch.empty_like(normalized[0])

    for step, gradient in enumerate(normalized, start=1):
        adam_direction = _adamw_update_direction(
            gradient,
            adam_momentum,
            adam_second_moment,
            step=step,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            eps=config.adam_eps,
        )
        # Muon's momentum state does not depend on its orthogonalized update, so only the final
        # Newton--Schulz transform is needed. NorMuon's row-wise second moment does depend on every
        # transformed update and is therefore advanced on every replayed gradient.
        muon_momentum.lerp_(gradient, 1 - config.muon_momentum)
        if step == len(normalized):
            nesterov = gradient.lerp(muon_momentum, config.muon_momentum)
            zero_momentum = torch.zeros_like(nesterov)
            muon_direction = _muon_update(
                nesterov,
                zero_momentum,
                momentum=0.0,
                ns_steps=config.ns_steps,
            ).float()
        normuon_direction = _normuon_update(
            gradient,
            normuon_momentum,
            normuon_second_moment,
            momentum=config.muon_momentum,
            beta2=config.normuon_beta2,
            ns_steps=config.ns_steps,
        ).float()

    muon_direction.mul_(_muon_aspect_scale(shape, config.adjust_lr_fn))
    return {
        "adamw": adam_direction,
        "muon": muon_direction,
        "normuon": normuon_direction,
    }


def _resolve_gradient_shards(
    manifest_path: Path,
) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported gradient manifest schema: {manifest.get('schema_version')}")
    if manifest.get("status") != "complete":
        raise ValueError("Gradient manifest is not complete")
    shards = manifest.get("gradient_shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Gradient manifest must declare at least one gradient_shard")
    resolved = []
    for index, item in enumerate(shards):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"Invalid gradient shard {index}: {item!r}")
        path = (manifest_path.parent / item["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if item.get("bytes") != path.stat().st_size or item.get("sha256") != _sha256(path):
            raise ValueError(f"Gradient shard identity mismatch: {path}")
        resolved.append((path, item))
    return manifest, resolved


def _gradient_names(
    stack: ExitStack,
    shards: list[tuple[Path, dict[str, Any]]],
) -> tuple[list[str], list[Any]]:
    handles = [
        stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
        for path, _ in shards
    ]
    expected = sorted(handles[0].keys())
    if not expected:
        raise ValueError("Gradient shard contains no tensors")
    for (path, _), handle in zip(shards[1:], handles[1:], strict=True):
        if sorted(handle.keys()) != expected:
            raise ValueError(f"Gradient tensor set differs in {path}")
    return expected, handles


def _output_provenance(path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(output_dir)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _verify_completed_output(output_dir: Path, manifest: dict[str, Any]) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Existing update analysis manifest has no outputs")
    for label, item in outputs.items():
        path = output_dir / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or _sha256(path) != item.get("sha256")
        ):
            raise ValueError(f"Existing {label} output does not match its manifest: {path}")


def _validate_common_state_spec(
    spec_path: str | Path | None,
    *,
    operator_config: UpdateOperatorConfig,
    operator_device: str,
    gradient_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    if spec_path is None:
        return None
    path = Path(spec_path).resolve()
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported common-state specification schema: {path}")
    if spec.get("operator_protocol") != asdict(operator_config):
        raise ValueError(
            "Update operator settings differ from the frozen common-state specification"
        )
    if spec.get("operator_runtime") != {"device": operator_device}:
        raise ValueError("Update replay device differs from the frozen common-state specification")
    expected_normalization = "per-tensor-frobenius-equals-weight-frobenius"
    if spec.get("matched_update_normalization") != expected_normalization:
        raise ValueError("Frozen common-state specification has an unsupported normalization")
    declared = gradient_manifest.get("common_state_spec")
    if not isinstance(declared, dict) or declared.get("sha256") != _sha256(path):
        raise ValueError("Gradient manifest does not use the requested common-state specification")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def analyze_common_state_updates(
    checkpoint: str | Path,
    gradient_manifest: str | Path,
    output_dir: str | Path,
    *,
    operator_config: UpdateOperatorConfig = UpdateOperatorConfig(),
    common_state_spec: str | Path | None = None,
    operator_device: str = "cpu",
    sketch_rank: int = 64,
    oversample: int = 8,
    power_iterations: int = 2,
    seed: int = 42,
    storage_dtype: Literal["float16", "float32"] = "float32",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Compare optimizer directions at one checkpoint using cached common gradients."""

    checkpoint = Path(checkpoint).resolve()
    gradient_manifest_path = Path(gradient_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    operator_config.validate()
    replay_device = torch.device(operator_device)
    if replay_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA update analysis was requested but CUDA is unavailable")
    if sketch_rank < 0 or oversample < 0 or power_iterations < 0:
        raise ValueError("Sketch settings must be non-negative")
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError(f"Unsupported storage dtype {storage_dtype!r}")
    gradient_manifest_payload, shards = _resolve_gradient_shards(gradient_manifest_path)
    common_state_spec_identity = _validate_common_state_spec(
        common_state_spec,
        operator_config=operator_config,
        operator_device=operator_device,
        gradient_manifest=gradient_manifest_payload,
    )
    checkpoint_inputs = _checkpoint_inputs(checkpoint)
    declared_checkpoint = gradient_manifest_payload.get("checkpoint")
    if not isinstance(declared_checkpoint, dict):
        raise ValueError("Gradient manifest lacks checkpoint provenance")
    if declared_checkpoint.get("inputs") != checkpoint_inputs:
        raise ValueError("Gradient manifest was not produced from the requested checkpoint")

    analysis_config = {
        "operator": asdict(operator_config),
        "sketch_rank": sketch_rank,
        "oversample": oversample,
        "power_iterations": power_iterations,
        "seed": seed,
        "storage_dtype": storage_dtype,
        "operator_device": operator_device,
        "normalization": "per-tensor-frobenius-equals-weight-frobenius",
        "weight_decay_included": False,
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": {"path": str(checkpoint), "inputs": checkpoint_inputs},
        "gradient_manifest": {
            "path": str(gradient_manifest_path),
            "bytes": gradient_manifest_path.stat().st_size,
            "sha256": _sha256(gradient_manifest_path),
        },
        "common_state_spec": common_state_spec_identity,
        "analysis_config": analysis_config,
    }
    manifest_path = output_dir / "manifest.json"
    known_outputs = [
        output_dir / "metrics.jsonl",
        *(output_dir / f"{algorithm}-matched.safetensors" for algorithm in ALGORITHMS),
    ]
    if manifest_path.exists() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if {key: existing.get(key) for key in identity} != identity:
            raise ValueError("Existing update analysis does not match the requested inputs")
        _verify_completed_output(output_dir, existing)
        return existing
    if not overwrite and any(path.exists() for path in known_outputs):
        raise FileExistsError(f"Partial update analysis exists under {output_dir}")
    if overwrite:
        for path in [manifest_path, *known_outputs]:
            if path.is_file():
                path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dtype = torch.float16 if storage_dtype == "float16" else torch.float32
    matched_updates: dict[str, dict[str, torch.Tensor]] = {
        algorithm: {} for algorithm in ALGORITHMS
    }
    records: list[dict[str, Any]] = []
    with ExitStack() as stack:
        weights = stack.enter_context(TensorStore(checkpoint))
        gradient_names, gradient_handles = _gradient_names(stack, shards)
        weight_names = set(weights.keys())
        missing_weights = sorted(set(gradient_names) - weight_names)
        if missing_weights:
            raise ValueError(f"Gradient tensors are absent from checkpoint: {missing_weights}")
        declared_partition = gradient_manifest_payload.get("partition_summary")
        if declared_partition is not None:
            observed_hidden = {
                "tensors": len(gradient_names),
                "parameters": sum(math.prod(weights.shape(name)) for name in gradient_names),
            }
            if declared_partition.get("hidden") != observed_hidden:
                raise ValueError(
                    "Gradient tensor coverage differs from its declared hidden partition: "
                    f"expected={declared_partition.get('hidden')}, observed={observed_hidden}"
                )
        for tensor_index, name in enumerate(gradient_names):
            if name not in weight_names:
                raise ValueError(f"Gradient tensor {name!r} is absent from checkpoint")
            shape = weights.shape(name)
            if len(shape) != 2 or parameter_partition_name(name, len(shape)) != "hidden":
                raise ValueError(f"Gradient tensor {name!r} is not a routed hidden matrix")
            gradients = [handle.get_tensor(name) for handle in gradient_handles]
            if any(tuple(gradient.shape) != shape for gradient in gradients):
                raise ValueError(f"Gradient shape mismatch for {name!r}")
            weight = weights.tensor(name).to(device=replay_device, dtype=torch.float32)
            if not torch.isfinite(weight).all():
                raise ValueError(f"Checkpoint tensor {name!r} contains a non-finite value")
            directions = replay_update_directions(
                gradients, operator_config, device=operator_device
            )
            weight_norm = torch.linalg.vector_norm(weight)
            if weight_norm <= 0:
                raise ValueError(f"Checkpoint tensor {name!r} has zero Frobenius norm")
            final_gradient = gradients[-1].to(device=replay_device, dtype=torch.float32)
            record: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "tensor": name,
                "shape": list(shape),
                "parameters": math.prod(shape),
                "gradient_steps": len(gradients),
                "weight_frobenius_norm": float(weight_norm.item()),
                "final_gradient": matrix_metrics(
                    final_gradient,
                    sketch_rank=sketch_rank,
                    oversample=oversample,
                    power_iterations=power_iterations,
                    seed=seed + tensor_index * 17,
                ),
                "algorithms": {},
                "pairwise_cosine": {},
            }
            for algorithm in ALGORITHMS:
                direction = directions[algorithm]
                direction_norm = torch.linalg.vector_norm(direction)
                if not torch.isfinite(direction).all() or direction_norm <= 0:
                    raise ValueError(f"{algorithm} produced an invalid direction for {name!r}")
                matched = direction * (weight_norm / direction_norm)
                matched_updates[algorithm][name] = matched.to(
                    device="cpu", dtype=output_dtype
                ).contiguous()
                metrics = matrix_metrics(
                    direction,
                    sketch_rank=sketch_rank,
                    oversample=oversample,
                    power_iterations=power_iterations,
                    seed=seed + tensor_index * 17,
                )
                metrics.update(
                    {
                        "cosine_with_final_gradient": _cosine(direction, final_gradient),
                        "cosine_with_weight": _cosine(direction, weight),
                        "per_unit_lr_update_to_weight": float(
                            (direction_norm / weight_norm).item()
                        ),
                        "matched_frobenius_norm": float(weight_norm.item()),
                    }
                )
                record["algorithms"][algorithm] = metrics
            for left_index, left in enumerate(ALGORITHMS):
                for right in ALGORITHMS[left_index + 1 :]:
                    record["pairwise_cosine"][f"{left}__{right}"] = _cosine(
                        directions[left], directions[right]
                    )
            records.append(record)

    metrics_path = output_dir / "metrics.jsonl"
    _atomic_jsonl(metrics_path, records)
    outputs = {"metrics": _output_provenance(metrics_path, output_dir)}
    metadata = {
        "schema_version": str(SCHEMA_VERSION),
        "normalization": analysis_config["normalization"],
        "weight_decay_included": "false",
    }
    for algorithm in ALGORITHMS:
        path = output_dir / f"{algorithm}-matched.safetensors"
        _atomic_safetensors(
            path, matched_updates[algorithm], metadata={**metadata, "algorithm": algorithm}
        )
        outputs[f"{algorithm}_matched"] = _output_provenance(path, output_dir)
    manifest = {
        **identity,
        "gradient_steps": len(shards),
        "tensors": len(records),
        "parameters": sum(record["parameters"] for record in records),
        "outputs": outputs,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "operator_device": operator_device,
            "gpu_name": (
                torch.cuda.get_device_name(replay_device) if replay_device.type == "cuda" else None
            ),
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare AdamW, Muon, and NorMuon on cached common-state gradients"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gradient-manifest", type=Path, required=True)
    parser.add_argument(
        "--common-state-spec", type=Path, default=Path("configs/common_state_probe.json")
    )
    parser.add_argument("--operator-device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sketch-rank", type=int, default=64)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--power-iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--normuon-beta2", type=float, default=0.95)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument(
        "--adjust-lr-fn", choices=("original", "match_rms_adamw", "none"), default="original"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest = analyze_common_state_updates(
        args.checkpoint,
        args.gradient_manifest,
        args.output_dir,
        operator_config=UpdateOperatorConfig(
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_eps=args.adam_eps,
            muon_momentum=args.muon_momentum,
            normuon_beta2=args.normuon_beta2,
            ns_steps=args.ns_steps,
            adjust_lr_fn=args.adjust_lr_fn,
        ),
        common_state_spec=args.common_state_spec,
        operator_device=args.operator_device,
        sketch_rank=args.sketch_rank,
        oversample=args.oversample,
        power_iterations=args.power_iterations,
        seed=args.seed,
        storage_dtype=args.storage_dtype,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "manifest": str((args.output_dir / "manifest.json").resolve()),
                "gradient_steps": manifest["gradient_steps"],
                "tensors": manifest["tensors"],
                "parameters": manifest["parameters"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
