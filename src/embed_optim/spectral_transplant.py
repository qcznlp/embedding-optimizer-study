from __future__ import annotations

import importlib.metadata
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .collators import DenseGroupCollator, LateGroupCollator
from .functional_intervention import (
    InterventionCondition,
    _evaluate_condition,
    _output_identity,
    _verify_declared_file,
    load_intervention_protocol,
)
from .geometry import SCHEMA_VERSION, _atomic_json, _atomic_jsonl, _atomic_safetensors, _sha256
from .gradient_probe import _hidden_parameter_mapping, _temperature_from_checkpoint
from .optimizers import parameter_partition, partition_summary
from .probe_export import (
    ModelFamily,
    _checkpoint_inputs,
    _load_model,
    _load_probe,
    _validate_checkpoint_family,
    _validate_probe_spec,
)

SOURCE_ALGORITHMS = ("adamw", "muon")
NATIVE_CONDITIONS = ("adamw-native", "muon-native")


@dataclass(frozen=True)
class SpectralCondition:
    name: str
    basis_source: str
    spectrum_operation: str
    interpolation_lambda: float | None = None
    band: str | None = None


def resolve_spectral_transplant_spec(path: str | Path, prefix: Path | None = None) -> Path:
    path = Path(path)
    if path.is_file() or path.is_absolute() or path.parent != Path("configs"):
        return path
    prefix = Path(sys.prefix) if prefix is None else prefix
    installed = prefix / "share" / "embedding-optimizer-study" / "configs" / path.name
    return installed if installed.is_file() else path


def spectral_conditions(spec: dict[str, Any]) -> list[SpectralCondition]:
    conditions = []
    for payload in spec["factorization"]["transformed_conditions"]:
        conditions.append(
            SpectralCondition(
                name=payload["name"],
                basis_source=payload["basis_source"],
                spectrum_operation=payload["spectrum_operation"],
                interpolation_lambda=(float(payload["lambda"]) if "lambda" in payload else None),
                band=payload.get("band"),
            )
        )
    names = [condition.name for condition in conditions]
    if len(names) != len(set(names)):
        raise ValueError("Spectral-transplant condition names are not unique")
    return conditions


def load_spectral_transplant_protocol(path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = resolve_spectral_transplant_spec(path).resolve()
    spec = json.loads(path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "analysis_status",
        "frozen_at_utc",
        "objective",
        "source_inputs",
        "anchor_scope",
        "factorization",
        "intervention",
        "evaluation",
        "primary_estimands",
        "freeze_context",
        "claim_boundary",
    }
    if set(spec) != expected_fields or spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported spectral-transplant protocol: {path}")
    if spec["analysis_status"] != "post_hoc_explanatory_intervention":
        raise ValueError("Spectral transplant must disclose its post-hoc explanatory status")

    source = spec["source_inputs"]
    if (
        source.get("input_direction_archives")
        != ["adamw-matched.safetensors", "muon-matched.safetensors"]
        or source.get("weight_decay_included") is not False
    ):
        raise ValueError("Spectral transplant requires the frozen AdamW and Muon data directions")
    root = path.parent.parent
    for path_key, digest_key in (
        ("common_state_spec", "common_state_spec_sha256"),
        ("functional_intervention_spec", "functional_intervention_spec_sha256"),
    ):
        declared = (root / source[path_key]).resolve()
        if not declared.is_file() or _sha256(declared) != source[digest_key]:
            raise ValueError(f"Frozen spectral-transplant source changed: {declared}")

    scope = spec["anchor_scope"]
    if (
        scope.get("selection") != "all-frozen-common-state-anchors"
        or scope.get("families") != ["dense", "late"]
        or scope.get("expected_anchors_per_family") != 10
        or scope.get("expected_total_anchors") != 20
        or scope.get("expected_hidden_tensors_per_anchor") != 88
        or scope.get("expected_hidden_parameters_per_anchor") != 110_297_088
    ):
        raise ValueError("Spectral-transplant anchor scope differs from the frozen grid")

    factorization = spec["factorization"]
    expected_lambdas = [0.25, 0.5, 0.75, 1.0]
    if (
        factorization.get("decomposition") != "torch.linalg.svd-full-economy"
        or factorization.get("compute_dtype") != "float32"
        or factorization.get("stored_direction_dtype") != "float16"
        or factorization.get("spectrum_normalization_before_mixing") != "unit-l2"
        or factorization.get("spectrum_interpolation") != "log-linear-geodesic"
        or factorization.get("final_normalization")
        != "per-tensor-frobenius-equals-source-weight-frobenius"
        or factorization.get("interpolation_lambdas") != expected_lambdas
    ):
        raise ValueError("Unsupported spectral factorization settings")
    floor = factorization.get("singular_value_floor_relative_to_largest")
    if not isinstance(floor, (int, float)) or not 0 < float(floor) < 1:
        raise ValueError("Singular-value floor must be in (0, 1)")
    bands = factorization.get("band_boundaries")
    if bands != {
        "head": [0.0, 0.25],
        "middle": [0.25, 0.75],
        "tail": [0.75, 1.0],
        "rounding": "nearest integer boundaries with complete disjoint coverage",
    }:
        raise ValueError("Spectral bands differ from the frozen quartile partition")
    conditions = spectral_conditions(spec)
    interpolation = [
        condition.interpolation_lambda
        for condition in conditions
        if condition.spectrum_operation == "log_interpolation"
    ]
    band_conditions = [
        condition.band
        for condition in conditions
        if condition.spectrum_operation == "band_transplant"
    ]
    if (
        len(conditions) != 8
        or interpolation != expected_lambdas
        or band_conditions != ["head", "middle", "tail"]
        or sum(condition.spectrum_operation == "adam_spectrum" for condition in conditions) != 1
        or any(condition.basis_source not in SOURCE_ALGORITHMS for condition in conditions)
    ):
        raise ValueError("Transformed conditions differ from the frozen 2x2 and band design")

    intervention = spec["intervention"]
    evaluation = spec["evaluation"]
    if (
        float(intervention.get("relative_scale", math.nan)) != 0.001
        or intervention.get("native_conditions") != list(NATIVE_CONDITIONS)
        or intervention.get("include_unmodified_baseline") is not True
        or intervention.get("expected_conditions_per_anchor") != 3 + len(conditions)
        or intervention.get("expected_sample_records_per_anchor")
        != intervention["expected_conditions_per_anchor"] * evaluation.get("examples", -1)
    ):
        raise ValueError("Spectral-transplant intervention cardinality is inconsistent")
    if (
        evaluation.get("examples") != 224
        or evaluation.get("groups") != 14
        or evaluation.get("model_dtype") != "float32"
        or evaluation.get("forward_dtype") != "bfloat16"
        or evaluation.get("model_mode") != "eval"
        or evaluation.get("normalized_embeddings") is not True
    ):
        raise ValueError("Spectral-transplant evaluation differs from the frozen probe contract")
    freeze = spec["freeze_context"]
    if (
        freeze.get("discovery_beir_valid_units") != 1680
        or freeze.get("discovery_beir_expected_units") != 1680
        or freeze.get("local_global_reversal_analysis_visible") is not True
        or freeze.get("confirmatory_results_available") is not False
        or freeze.get("short_branch_results_available") is not False
    ):
        raise ValueError("Spectral-transplant freeze disclosure is invalid")
    if not isinstance(spec.get("claim_boundary"), str) or not spec["claim_boundary"]:
        raise ValueError("Spectral-transplant claim boundary is missing")
    return path, spec


def _unit_spectrum(values: torch.Tensor, relative_floor: float) -> torch.Tensor:
    if values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("Singular values must be a finite non-empty vector")
    largest = values.max()
    if largest <= 0:
        raise ValueError("Singular values have zero scale")
    floor = largest * relative_floor
    clamped = values.clamp_min(floor)
    return clamped / torch.linalg.vector_norm(clamped)


def _band_slice(rank: int, band: str) -> slice:
    if rank < 1:
        raise ValueError("Matrix rank must be positive")
    head_end = max(1, min(rank, int(math.floor(0.25 * rank + 0.5))))
    tail_start = max(head_end, min(rank, int(math.floor(0.75 * rank + 0.5))))
    intervals = {
        "head": (0, head_end),
        "middle": (head_end, tail_start),
        "tail": (tail_start, rank),
    }
    if band not in intervals:
        raise ValueError(f"Unknown spectral band {band!r}")
    start, stop = intervals[band]
    return slice(start, stop)


def _mixed_spectrum(
    adam_singular: torch.Tensor,
    muon_singular: torch.Tensor,
    condition: SpectralCondition,
    *,
    relative_floor: float,
) -> torch.Tensor:
    adam = _unit_spectrum(adam_singular, relative_floor)
    muon = _unit_spectrum(muon_singular, relative_floor)
    if condition.spectrum_operation == "log_interpolation":
        if condition.interpolation_lambda is None:
            raise ValueError("Log interpolation requires lambda")
        weight = condition.interpolation_lambda
        mixed = torch.exp((1 - weight) * torch.log(adam) + weight * torch.log(muon))
    elif condition.spectrum_operation == "adam_spectrum":
        mixed = adam
    elif condition.spectrum_operation == "band_transplant":
        if condition.band is None:
            raise ValueError("Band transplant requires a named band")
        mixed = adam.clone()
        selected = _band_slice(mixed.numel(), condition.band)
        mixed[selected] = muon[selected]
    else:
        raise ValueError(f"Unknown spectrum operation {condition.spectrum_operation!r}")
    norm = torch.linalg.vector_norm(mixed)
    if not torch.isfinite(mixed).all() or norm <= 0:
        raise ValueError(f"Condition {condition.name} produced an invalid spectrum")
    return mixed / norm


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if denominator <= 0:
        raise ValueError("Cannot compare zero-norm directions")
    return float(torch.sum(left * right).div(denominator).item())


def construct_spectral_transplants(
    adamw: torch.Tensor,
    muon: torch.Tensor,
    conditions: list[SpectralCondition],
    *,
    target_frobenius_norm: float,
    relative_floor: float,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Construct the frozen spectrum/basis factorial from two matched matrix directions."""

    if (
        adamw.ndim != 2
        or muon.ndim != 2
        or adamw.shape != muon.shape
        or adamw.dtype != torch.float32
        or muon.dtype != torch.float32
        or not torch.isfinite(adamw).all()
        or not torch.isfinite(muon).all()
    ):
        raise ValueError("Spectral transplant requires same-shaped finite float32 matrices")
    if not math.isfinite(target_frobenius_norm) or target_frobenius_norm <= 0:
        raise ValueError("Target Frobenius norm must be finite and positive")
    adam_u, adam_s, adam_vh = torch.linalg.svd(adamw, full_matrices=False)
    muon_u, muon_s, muon_vh = torch.linalg.svd(muon, full_matrices=False)
    outputs: dict[str, torch.Tensor] = {}
    diagnostics = []
    for condition in conditions:
        spectrum = _mixed_spectrum(
            adam_s,
            muon_s,
            condition,
            relative_floor=relative_floor,
        )
        if condition.basis_source == "adamw":
            left, right = adam_u, adam_vh
        elif condition.basis_source == "muon":
            left, right = muon_u, muon_vh
        else:
            raise ValueError(f"Unknown singular-vector basis {condition.basis_source!r}")
        direction = (left * spectrum.unsqueeze(0)) @ right
        direction.mul_(target_frobenius_norm / torch.linalg.vector_norm(direction))
        if not torch.isfinite(direction).all():
            raise ValueError(f"Condition {condition.name} produced non-finite values")
        normalized = spectrum / torch.linalg.vector_norm(spectrum)
        probabilities = normalized.square().clamp_min(torch.finfo(torch.float32).tiny)
        entropy_rank = torch.exp(-(probabilities * torch.log(probabilities)).sum())
        diagnostics.append(
            {
                "condition": condition.name,
                "basis_source": condition.basis_source,
                "spectrum_operation": condition.spectrum_operation,
                "interpolation_lambda": condition.interpolation_lambda,
                "band": condition.band,
                "frobenius_norm": float(torch.linalg.vector_norm(direction).item()),
                "cosine_with_adamw": _cosine(direction, adamw),
                "cosine_with_muon": _cosine(direction, muon),
                "stable_rank": float((spectrum.square().sum() / spectrum.max().square()).item()),
                "entropy_rank": float(entropy_rank.item()),
                "rank": spectrum.numel(),
            }
        )
        outputs[condition.name] = direction
    return outputs, diagnostics


def _source_update_inputs(
    update_dir: Path,
    *,
    checkpoint: Path,
    common_state_spec_sha256: str,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, float]]:
    manifest_path = update_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("checkpoint", {}).get("inputs") != _checkpoint_inputs(checkpoint)
        or manifest.get("common_state_spec", {}).get("sha256") != common_state_spec_sha256
        or manifest.get("analysis_config", {}).get("normalization")
        != "per-tensor-frobenius-equals-weight-frobenius"
        or manifest.get("analysis_config", {}).get("weight_decay_included") is not False
        or manifest.get("tensors") != 88
        or manifest.get("parameters") != 110_297_088
    ):
        raise ValueError(f"Common-state update source is inconsistent: {manifest_path}")
    outputs = manifest.get("outputs", {})
    paths = {
        algorithm: _verify_declared_file(update_dir, outputs[f"{algorithm}_matched"])
        for algorithm in SOURCE_ALGORITHMS
    }
    metrics_path = _verify_declared_file(update_dir, outputs["metrics"])
    weights: dict[str, float] = {}
    with metrics_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            name = row.get("tensor")
            value = float(row.get("weight_frobenius_norm", math.nan))
            if (
                not isinstance(name, str)
                or name in weights
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Invalid source metric at {metrics_path}:{line_number}")
            weights[name] = value
    if len(weights) != 88:
        raise ValueError("Source update metrics do not cover all 88 hidden matrices")
    identity = {
        "path": str(manifest_path.resolve()),
        "bytes": manifest_path.stat().st_size,
        "sha256": _sha256(manifest_path),
    }
    return identity, paths, weights


def _verify_direction_manifest(
    direction_dir: Path,
    *,
    spec_path: Path,
    source_manifest: dict[str, Any],
    expected_conditions: list[SpectralCondition],
    verify_hashes: bool,
) -> dict[str, Any]:
    manifest_path = direction_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("spectral_transplant_spec", {}).get("sha256") != _sha256(spec_path)
        or manifest.get("source_update_manifest") != source_manifest
        or manifest.get("tensors") != 88
        or manifest.get("parameters") != 110_297_088
        or manifest.get("condition_records") != 88 * len(expected_conditions)
        or set(manifest.get("outputs", {}))
        != {"direction_metrics", *(condition.name for condition in expected_conditions)}
    ):
        raise ValueError("Spectral direction manifest is inconsistent")
    for item in manifest["outputs"].values():
        path = direction_dir / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or (verify_hashes and _sha256(path) != item.get("sha256"))
        ):
            raise ValueError(f"Spectral direction output differs from its manifest: {path}")
    return manifest


def prepare_spectral_directions(
    checkpoint: str | Path,
    update_dir: str | Path,
    direction_dir: str | Path,
    *,
    spectral_spec: str | Path,
    device: str = "cuda",
    overwrite: bool = False,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    update_dir = Path(update_dir).resolve()
    direction_dir = Path(direction_dir).resolve()
    spec_path, spec = load_spectral_transplant_protocol(spectral_spec)
    conditions = spectral_conditions(spec)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA spectral factorization was requested but CUDA is unavailable")
    source_manifest, paths, weight_norms = _source_update_inputs(
        update_dir,
        checkpoint=checkpoint,
        common_state_spec_sha256=spec["source_inputs"]["common_state_spec_sha256"],
    )
    manifest_path = direction_dir / "manifest.json"
    if manifest_path.is_file() and not overwrite:
        return _verify_direction_manifest(
            direction_dir,
            spec_path=spec_path,
            source_manifest=source_manifest,
            expected_conditions=conditions,
            verify_hashes=True,
        )
    known = [
        direction_dir / "direction_metrics.jsonl",
        *(direction_dir / f"{condition.name}.safetensors" for condition in conditions),
    ]
    if not overwrite and any(path.exists() for path in known):
        raise FileExistsError(f"Partial spectral directions exist under {direction_dir}")
    if overwrite:
        for path in [manifest_path, *known]:
            if path.is_file():
                path.unlink()

    target_device = torch.device(device)
    output_tensors: dict[str, dict[str, torch.Tensor]] = {
        condition.name: {} for condition in conditions
    }
    records = []
    relative_floor = float(spec["factorization"]["singular_value_floor_relative_to_largest"])
    with (
        safe_open(paths["adamw"], framework="pt", device="cpu") as adam_handle,
        safe_open(paths["muon"], framework="pt", device="cpu") as muon_handle,
    ):
        names = sorted(adam_handle.keys())
        if (
            names != sorted(muon_handle.keys())
            or set(names) != set(weight_norms)
            or (adam_handle.metadata() or {}).get("algorithm") != "adamw"
            or (muon_handle.metadata() or {}).get("algorithm") != "muon"
        ):
            raise ValueError("Source direction archives have different tensor identities")
        parameters = 0
        for name in names:
            adamw = adam_handle.get_tensor(name).to(device=target_device, dtype=torch.float32)
            muon = muon_handle.get_tensor(name).to(device=target_device, dtype=torch.float32)
            parameters += adamw.numel()
            transformed, diagnostics = construct_spectral_transplants(
                adamw,
                muon,
                conditions,
                target_frobenius_norm=weight_norms[name],
                relative_floor=relative_floor,
            )
            for condition in conditions:
                output_tensors[condition.name][name] = (
                    transformed[condition.name].to(device="cpu", dtype=torch.float16).contiguous()
                )
            for row in diagnostics:
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "tensor": name,
                        "shape": list(adamw.shape),
                        "parameters": adamw.numel(),
                        "target_frobenius_norm": weight_norms[name],
                        **row,
                    }
                )
            del adamw, muon, transformed
    if len(names) != 88 or parameters != 110_297_088 or len(records) != 88 * len(conditions):
        raise AssertionError("Spectral direction coverage differs from the frozen hidden partition")

    direction_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = direction_dir / "direction_metrics.jsonl"
    _atomic_jsonl(metrics_path, records)
    outputs = {"direction_metrics": _output_identity(metrics_path, direction_dir)}
    metadata = {
        "schema_version": str(SCHEMA_VERSION),
        "spectral_transplant_spec_sha256": _sha256(spec_path),
        "normalization": spec["factorization"]["final_normalization"],
        "weight_decay_included": "false",
    }
    for condition in conditions:
        path = direction_dir / f"{condition.name}.safetensors"
        _atomic_safetensors(
            path,
            output_tensors.pop(condition.name),
            metadata={**metadata, "condition": condition.name},
        )
        outputs[condition.name] = _output_identity(path, direction_dir)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "checkpoint": {"path": str(checkpoint), "inputs": _checkpoint_inputs(checkpoint)},
        "spectral_transplant_spec": {
            "path": str(spec_path),
            "bytes": spec_path.stat().st_size,
            "sha256": _sha256(spec_path),
        },
        "source_update_manifest": source_manifest,
        "conditions": [condition.__dict__ for condition in conditions],
        "tensors": len(names),
        "parameters": parameters,
        "condition_records": len(records),
        "outputs": outputs,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": device,
            "gpu_name": (
                torch.cuda.get_device_name(target_device) if target_device.type == "cuda" else None
            ),
        },
    }
    _atomic_json(manifest_path, manifest)
    if target_device.type == "cuda":
        torch.cuda.empty_cache()
    return manifest


def _load_condition_direction(
    path: Path,
    *,
    expected_metadata_key: str,
    expected_metadata_value: str,
    mapping: list[tuple[str, str, torch.nn.Parameter]],
) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        expected = {checkpoint_name for _, checkpoint_name, _ in mapping}
        if (
            metadata.get(expected_metadata_key) != expected_metadata_value
            or set(handle.keys()) != expected
        ):
            raise ValueError(f"Direction archive has the wrong identity: {path}")
        directions = {}
        for _, checkpoint_name, parameter in mapping:
            value = handle.get_tensor(checkpoint_name)
            if value.shape != parameter.shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid spectral direction {checkpoint_name!r}")
            directions[checkpoint_name] = value.to(device=parameter.device, dtype=parameter.dtype)
    return directions


def run_spectral_transplant_intervention(
    checkpoint: str | Path,
    update_dir: str | Path,
    output_dir: str | Path,
    *,
    family: ModelFamily,
    spectral_spec: str | Path = "configs/spectral_transplant_intervention.json",
    device: str = "cuda",
    overwrite: bool = False,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    update_dir = Path(update_dir).resolve()
    output_dir = Path(output_dir).resolve()
    spec_path, spec = load_spectral_transplant_protocol(spectral_spec)
    if family not in {"dense", "late"}:
        raise ValueError(f"Unsupported family {family!r}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA spectral intervention was requested but CUDA is unavailable")

    repository_root = spec_path.parent.parent
    functional_path = (
        repository_root / spec["source_inputs"]["functional_intervention_spec"]
    ).resolve()
    _, functional = load_intervention_protocol(functional_path)
    probe_root = (repository_root / spec["evaluation"]["probe"]).resolve()
    probe_spec_path = (repository_root / functional["evaluation_probe"]["spec"]).resolve()
    dataset, probe_manifest, probe_manifest_sha256 = _load_probe(probe_root)
    probe_spec_identity = _validate_probe_spec(probe_spec_path, probe_manifest_sha256)
    if (
        probe_manifest_sha256 != spec["evaluation"]["probe_manifest_sha256"]
        or len(dataset) != spec["evaluation"]["examples"]
        or len(set(dataset["source"])) != spec["evaluation"]["groups"]
    ):
        raise ValueError("Spectral intervention probe differs from the frozen identity")

    checkpoint_inputs = _checkpoint_inputs(checkpoint)
    checkpoint_run_config = _validate_checkpoint_family(checkpoint, family)
    source_manifest, native_paths, _ = _source_update_inputs(
        update_dir,
        checkpoint=checkpoint,
        common_state_spec_sha256=spec["source_inputs"]["common_state_spec_sha256"],
    )
    direction_dir = output_dir / "directions"
    direction_overwrite = direction_dir.exists() and not (direction_dir / "manifest.json").is_file()
    direction_manifest = prepare_spectral_directions(
        checkpoint,
        update_dir,
        direction_dir,
        spectral_spec=spec_path,
        device=device,
        overwrite=direction_overwrite,
    )
    transformed_conditions = spectral_conditions(spec)
    direction_manifest = _verify_direction_manifest(
        direction_dir,
        spec_path=spec_path,
        source_manifest=source_manifest,
        expected_conditions=transformed_conditions,
        verify_hashes=True,
    )

    scale = float(spec["intervention"]["relative_scale"])
    conditions = [InterventionCondition("baseline", None, "baseline", 0.0, 0.0)]
    conditions.extend(
        InterventionCondition(name, algorithm, "descent", scale, scale)
        for name, algorithm in zip(NATIVE_CONDITIONS, SOURCE_ALGORITHMS, strict=True)
    )
    conditions.extend(
        InterventionCondition(condition.name, "spectral_transplant", "descent", scale, scale)
        for condition in transformed_conditions
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "checkpoint": {
            "path": str(checkpoint),
            "inputs": checkpoint_inputs,
            "run_config": checkpoint_run_config,
        },
        "source_update_manifest": source_manifest,
        "direction_manifest": {
            "path": str((direction_dir / "manifest.json").resolve()),
            "bytes": (direction_dir / "manifest.json").stat().st_size,
            "sha256": _sha256(direction_dir / "manifest.json"),
        },
        "probe": {
            "path": str(probe_root),
            "manifest_sha256": probe_manifest_sha256,
            "selection_sha256": probe_manifest["selection_sha256"],
            "selected_sample_ids_sha256": probe_manifest["selected_sample_ids_sha256"],
            "dataset_fingerprint": probe_manifest["serialized_probe_dataset_fingerprint"],
            "frozen_spec": probe_spec_identity,
        },
        "spectral_transplant_spec": {
            "path": str(spec_path),
            "bytes": spec_path.stat().st_size,
            "sha256": _sha256(spec_path),
        },
        "conditions": [condition.__dict__ for condition in conditions],
    }
    manifest_path = output_dir / "manifest.json"
    sample_path = output_dir / "sample_metrics.jsonl"
    condition_path = output_dir / "condition_metrics.jsonl"
    if manifest_path.is_file() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if {key: existing.get(key) for key in identity} != identity:
            raise ValueError("Existing spectral intervention has different inputs")
        for item in existing.get("outputs", {}).values():
            _verify_declared_file(output_dir, item)
        if existing.get("status") != "complete":
            raise ValueError("Existing spectral intervention manifest is incomplete")
        return existing
    if not overwrite and any(path.exists() for path in (sample_path, condition_path)):
        raise FileExistsError(f"Partial spectral intervention exists under {output_dir}")
    if overwrite:
        for path in (manifest_path, sample_path, condition_path):
            if path.is_file():
                path.unlink()

    evaluation = spec["evaluation"]
    model = _load_model(
        family,
        checkpoint,
        dtype=torch.float32,
        device=device,
        flash_attention=bool(evaluation["flash_attention"]),
    )
    model.eval()
    partition = parameter_partition(model)
    expected_partition = {
        "tensors": spec["anchor_scope"]["expected_hidden_tensors_per_anchor"],
        "parameters": spec["anchor_scope"]["expected_hidden_parameters_per_anchor"],
    }
    if partition_summary(partition)["hidden"] != expected_partition:
        raise ValueError("Loaded model hidden partition differs from the spectral protocol")
    mapping = _hidden_parameter_mapping(partition["hidden"], checkpoint)
    originals = {
        checkpoint_name: parameter.detach().clone() for _, checkpoint_name, parameter in mapping
    }
    collator = (
        DenseGroupCollator(model.preprocess) if family == "dense" else LateGroupCollator(model)
    )
    batch_size = int(evaluation[f"{family}_batch_size"])
    temperature = _temperature_from_checkpoint(checkpoint, family, None)
    all_samples: list[dict[str, Any]] = []
    all_conditions: list[dict[str, Any]] = []
    try:
        baseline_samples, baseline_summary = _evaluate_condition(
            model,
            dataset,
            family=family,
            collator=collator,
            batch_size=batch_size,
            device=device,
            forward_dtype=evaluation["forward_dtype"],
            temperature=temperature,
            condition=conditions[0],
        )
        all_samples.extend(baseline_samples)
        all_conditions.append(baseline_summary)
        for condition in conditions[1:]:
            if condition.condition == "adamw-native":
                path, metadata_key, metadata_value = native_paths["adamw"], "algorithm", "adamw"
            elif condition.condition == "muon-native":
                path, metadata_key, metadata_value = native_paths["muon"], "algorithm", "muon"
            else:
                output = direction_manifest["outputs"][condition.condition]
                path = direction_dir / output["path"]
                metadata_key, metadata_value = "condition", condition.condition
            directions = _load_condition_direction(
                path,
                expected_metadata_key=metadata_key,
                expected_metadata_value=metadata_value,
                mapping=mapping,
            )
            with torch.no_grad():
                for _, checkpoint_name, parameter in mapping:
                    parameter.copy_(
                        originals[checkpoint_name]
                        - condition.signed_scale * directions[checkpoint_name]
                    )
            samples, summary = _evaluate_condition(
                model,
                dataset,
                family=family,
                collator=collator,
                batch_size=batch_size,
                device=device,
                forward_dtype=evaluation["forward_dtype"],
                temperature=temperature,
                condition=condition,
            )
            all_samples.extend(samples)
            all_conditions.append(summary)
            with torch.no_grad():
                for _, checkpoint_name, parameter in mapping:
                    parameter.copy_(originals[checkpoint_name])
            del directions
    finally:
        with torch.no_grad():
            for _, checkpoint_name, parameter in mapping:
                parameter.copy_(originals[checkpoint_name])
        del originals, collator, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if (
        len(all_samples) != spec["intervention"]["expected_sample_records_per_anchor"]
        or len(all_conditions) != spec["intervention"]["expected_conditions_per_anchor"]
    ):
        raise AssertionError("Spectral intervention output cardinality changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(sample_path, all_samples)
    _atomic_jsonl(condition_path, all_conditions)
    manifest = {
        **identity,
        "status": "complete",
        "temperature": temperature,
        "sample_records": len(all_samples),
        "condition_records": len(all_conditions),
        "outputs": {
            "sample_metrics": _output_identity(sample_path, output_dir),
            "condition_metrics": _output_identity(condition_path, output_dir),
        },
        "runtime": {
            "torch": torch.__version__,
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "pylate": importlib.metadata.version("pylate"),
            "cuda": torch.version.cuda,
            "device": device,
            "gpu_name": (
                torch.cuda.get_device_name(torch.device(device))
                if device.startswith("cuda")
                else None
            ),
        },
        "claim_boundary": spec["claim_boundary"],
    }
    _atomic_json(manifest_path, manifest)
    return manifest
