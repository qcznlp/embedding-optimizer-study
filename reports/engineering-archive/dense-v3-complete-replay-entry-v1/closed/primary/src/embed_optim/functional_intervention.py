from __future__ import annotations

import argparse
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .collators import DenseGroupCollator, LateGroupCollator
from .geometry import SCHEMA_VERSION, _atomic_json, _atomic_jsonl, _sha256
from .gradient_probe import (
    _collect_features,
    _hidden_parameter_mapping,
    _temperature_from_checkpoint,
)
from .optimizers import parameter_partition, partition_summary
from .probe_export import (
    ModelFamily,
    _checkpoint_inputs,
    _load_model,
    _load_probe,
    _validate_checkpoint_family,
    _validate_probe_spec,
)
from .update_geometry import ALGORITHMS


@dataclass(frozen=True)
class InterventionCondition:
    condition: str
    algorithm: str | None
    direction: Literal["baseline", "descent", "sign_reversal"]
    relative_scale: float
    signed_scale: float


def resolve_intervention_spec(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file() or path.is_absolute() or path.parent != Path("configs"):
        return path
    installed = (
        Path(__import__("sys").prefix)
        / "share"
        / "embedding-optimizer-study"
        / "configs"
        / path.name
    )
    return installed if installed.is_file() else path


def intervention_conditions(spec: dict[str, Any]) -> list[InterventionCondition]:
    protocol = spec["intervention"]
    algorithms = protocol["algorithms"]
    scales = protocol["descent_relative_scales"]
    reversal_scale = protocol["sign_reversal_control_scale"]
    conditions = [InterventionCondition("baseline", None, "baseline", 0.0, 0.0)]
    for algorithm in algorithms:
        for scale in scales:
            label = format(float(scale), ".0e")
            conditions.append(
                InterventionCondition(
                    f"{algorithm}-descent-{label}",
                    algorithm,
                    "descent",
                    float(scale),
                    float(scale),
                )
            )
        label = format(float(reversal_scale), ".0e")
        conditions.append(
            InterventionCondition(
                f"{algorithm}-sign-reversal-{label}",
                algorithm,
                "sign_reversal",
                float(reversal_scale),
                -float(reversal_scale),
            )
        )
    names = [condition.condition for condition in conditions]
    if len(names) != len(set(names)):
        raise ValueError("Functional intervention condition names are not unique")
    expected = protocol["expected_conditions_per_anchor"]
    if len(conditions) != expected:
        raise ValueError(f"Built {len(conditions)} conditions, expected {expected}")
    return conditions


def load_intervention_protocol(path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = resolve_intervention_spec(path).resolve()
    spec = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "common_state",
        "evaluation_probe",
        "intervention",
        "evaluation",
        "freeze_context",
        "claim_boundary",
    }
    if set(spec) != required or spec["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported functional intervention specification: {path}")
    common = spec["common_state"]
    probe = spec["evaluation_probe"]
    intervention = spec["intervention"]
    evaluation = spec["evaluation"]
    if common.get("matched_update_normalization") != (
        "per-tensor-frobenius-equals-weight-frobenius"
    ):
        raise ValueError("Intervention requires per-tensor Frobenius-matched directions")
    if common.get("weight_decay_included") is not False:
        raise ValueError("The frozen functional intervention must exclude weight decay")
    if intervention.get("algorithms") != list(ALGORITHMS):
        raise ValueError("Intervention algorithms differ from the common-state analyzer")
    scales = intervention.get("descent_relative_scales")
    if (
        not isinstance(scales, list)
        or scales != sorted(set(scales))
        or not all(isinstance(value, (int, float)) and 0 < value < 1 for value in scales)
    ):
        raise ValueError("descent_relative_scales must be unique, increasing, and in (0, 1)")
    reversal = intervention.get("sign_reversal_control_scale")
    if reversal not in scales:
        raise ValueError("The sign-reversal scale must be one of the frozen descent scales")
    if probe.get("positive_candidate_index") != 0 or probe.get("negative_candidates") != 7:
        raise ValueError("Functional probe must preserve the positive-first eight-way contract")
    if probe.get("count") * intervention.get("expected_conditions_per_anchor") != intervention.get(
        "expected_sample_records_per_anchor"
    ):
        raise ValueError("Expected sample records disagree with probe and condition counts")
    if (
        evaluation.get("model_mode") != "eval"
        or evaluation.get("normalized_embeddings") is not True
    ):
        raise ValueError(
            "Functional intervention must use normalized embeddings in evaluation mode"
        )
    if evaluation.get("model_dtype") != "float32":
        raise ValueError("Functional intervention requires float32 checkpoint parameters")
    freeze = spec["freeze_context"]
    if (
        freeze.get("formal_common_state_outputs_visible") is not False
        or freeze.get("formal_functional_intervention_outputs_visible") is not False
        or not 0
        <= freeze.get("strict_beir_valid_units", -1)
        < freeze.get("strict_beir_expected_units", -1)
    ):
        raise ValueError("Functional intervention freeze context is invalid")
    intervention_conditions(spec)
    return path, spec


def _pad_sequence_dimension(tensor: torch.Tensor, length: int) -> torch.Tensor:
    if tensor.size(1) == length:
        return tensor
    padding = (
        (0, 0, 0, length - tensor.size(1)) if tensor.ndim == 3 else (0, length - tensor.size(1))
    )
    return F.pad(tensor, padding)


def group_scores(
    model: Any,
    features: list[dict[str, torch.Tensor]],
    family: ModelFamily,
) -> torch.Tensor:
    """Return the exact untempered eight-way scores used by formal training."""

    if len(features) != 9:
        raise ValueError(f"Expected query plus eight documents, got {len(features)} columns")
    if family == "dense":
        query = F.normalize(model(features[0])["sentence_embedding"], p=2, dim=-1)
        documents = torch.stack(
            [
                F.normalize(model(feature)["sentence_embedding"], p=2, dim=-1)
                for feature in features[1:]
            ],
            dim=1,
        )
        return torch.einsum("bh,bnh->bn", query, documents)
    if family != "late":
        raise ValueError(f"Unsupported model family {family!r}")

    from pylate.scores import colbert_kd_scores

    embeddings = [
        F.normalize(model(feature)["token_embeddings"], p=2, dim=-1) for feature in features
    ]
    wrapped = model.module if hasattr(model, "module") else model
    query_mask = features[0]["attention_mask"].bool()
    document_masks = [
        torch.logical_and(
            wrapped.skiplist_mask(feature["input_ids"], wrapped.skiplist),
            feature["attention_mask"].bool(),
        )
        for feature in features[1:]
    ]
    max_document_length = max(embedding.size(1) for embedding in embeddings[1:])
    documents = torch.stack(
        [_pad_sequence_dimension(embedding, max_document_length) for embedding in embeddings[1:]],
        dim=1,
    )
    masks = torch.stack(
        [_pad_sequence_dimension(mask, max_document_length) for mask in document_masks], dim=1
    )
    scores = colbert_kd_scores(
        queries_embeddings=embeddings[0],
        documents_embeddings=documents,
        queries_mask=query_mask,
        documents_mask=masks,
        backend="lik",
    )
    return scores / query_mask.sum(dim=-1, keepdim=True).clamp_min(1)


def score_metrics(scores: torch.Tensor, temperature: float) -> dict[str, torch.Tensor]:
    if scores.ndim != 2 or scores.size(1) != 8:
        raise ValueError(f"Expected scores shaped [batch, 8], got {tuple(scores.shape)}")
    if temperature <= 0 or not torch.isfinite(scores).all():
        raise ValueError("Scores must be finite and temperature must be positive")
    targets = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    positive = scores[:, 0]
    hardest_negative = scores[:, 1:].max(dim=1).values
    rank = 1 + (scores[:, 1:] >= positive.unsqueeze(1)).sum(dim=1)
    return {
        "contrastive_loss": F.cross_entropy(scores / temperature, targets, reduction="none"),
        "positive_score": positive,
        "hardest_negative_score": hardest_negative,
        "positive_margin": positive - hardest_negative,
        "reciprocal_rank": rank.float().reciprocal(),
        "top1_accuracy": rank.eq(1).float(),
    }


def _output_identity(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _verify_declared_file(root: Path, item: dict[str, Any]) -> Path:
    path = root / item["path"]
    if (
        not path.is_file()
        or path.stat().st_size != item.get("bytes")
        or _sha256(path) != item.get("sha256")
    ):
        raise ValueError(f"Declared input does not match its manifest: {path}")
    return path


def _update_inputs(
    checkpoint: Path,
    update_dir: Path,
    common_state_spec_sha256: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest_path = update_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported common-state update manifest: {manifest_path}")
    if manifest.get("checkpoint", {}).get("inputs") != _checkpoint_inputs(checkpoint):
        raise ValueError("Common-state update manifest points to different checkpoint inputs")
    if manifest.get("common_state_spec", {}).get("sha256") != common_state_spec_sha256:
        raise ValueError("Common-state update manifest points to a different frozen protocol")
    if manifest.get("analysis_config", {}).get("normalization") != (
        "per-tensor-frobenius-equals-weight-frobenius"
    ):
        raise ValueError("Common-state update directions are not Frobenius matched")
    if manifest.get("analysis_config", {}).get("weight_decay_included") is not False:
        raise ValueError("Common-state update directions unexpectedly include weight decay")
    outputs = manifest.get("outputs", {})
    expected = {f"{algorithm}_matched" for algorithm in ALGORITHMS}
    if not expected.issubset(outputs):
        raise ValueError("Common-state update manifest lacks matched intervention directions")
    paths = {
        algorithm: _verify_declared_file(update_dir, outputs[f"{algorithm}_matched"])
        for algorithm in ALGORITHMS
    }
    return (
        {
            "path": str(manifest_path.resolve()),
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        },
        paths,
    )


def _load_directions(
    path: Path,
    algorithm: str,
    mapping: list[tuple[str, str, torch.nn.Parameter]],
) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = set(handle.keys())
        expected = {checkpoint_name for _, checkpoint_name, _ in mapping}
        if metadata.get("algorithm") != algorithm or keys != expected:
            raise ValueError(f"Matched direction archive has the wrong identity: {path}")
        directions = {}
        for _, checkpoint_name, parameter in mapping:
            direction = handle.get_tensor(checkpoint_name)
            if direction.shape != parameter.shape or not torch.isfinite(direction).all():
                raise ValueError(f"Invalid intervention direction {checkpoint_name!r}")
            directions[checkpoint_name] = direction.to(
                device=parameter.device, dtype=parameter.dtype
            )
    return directions


def _evaluate_condition(
    model: Any,
    dataset: Any,
    *,
    family: ModelFamily,
    collator: Any,
    batch_size: int,
    device: str,
    forward_dtype: str,
    temperature: float,
    condition: InterventionCondition,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    device_type = torch.device(device).type
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            rows = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            features = _collect_features(collator(rows), device)
            with torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=forward_dtype == "bfloat16",
            ):
                scores = group_scores(model, features, family)
            metrics = score_metrics(scores.float(), temperature)
            for offset, row in enumerate(rows):
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "condition": condition.condition,
                    "algorithm": condition.algorithm,
                    "direction": condition.direction,
                    "relative_scale": condition.relative_scale,
                    "signed_scale": condition.signed_scale,
                    "sample_id": int(row["sample_id"]),
                    "group": str(row["source"]),
                }
                record.update(
                    {name: float(values[offset].item()) for name, values in metrics.items()}
                )
                records.append(record)
    metric_names = list(metrics)
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "condition": condition.condition,
        "algorithm": condition.algorithm,
        "direction": condition.direction,
        "relative_scale": condition.relative_scale,
        "signed_scale": condition.signed_scale,
        "samples": len(records),
        **{name: sum(record[name] for record in records) / len(records) for name in metric_names},
    }
    return records, aggregate


def run_functional_intervention(
    checkpoint: str | Path,
    update_dir: str | Path,
    probe_root: str | Path,
    output_dir: str | Path,
    *,
    family: ModelFamily,
    intervention_spec: str | Path = "configs/functional_intervention.json",
    device: str = "cuda",
    overwrite: bool = False,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    update_dir = Path(update_dir).resolve()
    probe_root = Path(probe_root).resolve()
    output_dir = Path(output_dir).resolve()
    spec_path, spec = load_intervention_protocol(intervention_spec)
    if family not in {"dense", "late"}:
        raise ValueError(f"Unsupported family {family!r}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA functional intervention was requested but CUDA is unavailable")
    evaluation = spec["evaluation"]
    if not device.startswith("cuda") and (
        evaluation["flash_attention"] or evaluation["forward_dtype"] == "bfloat16"
    ):
        raise ValueError("CPU intervention requires float32 forward and no FlashAttention")

    common_spec_path = (spec_path.parent.parent / spec["common_state"]["spec"]).resolve()
    probe_spec_path = (spec_path.parent.parent / spec["evaluation_probe"]["spec"]).resolve()
    if _sha256(common_spec_path) != spec["common_state"]["spec_sha256"]:
        raise ValueError("Frozen common-state specification digest changed")
    if _sha256(probe_spec_path) != spec["evaluation_probe"]["spec_sha256"]:
        raise ValueError("Frozen evaluation-probe specification digest changed")
    dataset, probe_manifest, probe_manifest_sha256 = _load_probe(probe_root)
    probe_spec_identity = _validate_probe_spec(probe_spec_path, probe_manifest_sha256)
    if (
        probe_manifest_sha256 != spec["evaluation_probe"]["manifest_sha256"]
        or len(dataset) != spec["evaluation_probe"]["count"]
        or len(set(dataset["source"])) != spec["evaluation_probe"]["groups"]
    ):
        raise ValueError("Evaluation probe differs from the prospectively frozen probe")
    checkpoint_inputs = _checkpoint_inputs(checkpoint)
    checkpoint_run_config = _validate_checkpoint_family(checkpoint, family)
    update_manifest_identity, direction_paths = _update_inputs(
        checkpoint, update_dir, spec["common_state"]["spec_sha256"]
    )
    conditions = intervention_conditions(spec)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "checkpoint": {
            "path": str(checkpoint),
            "inputs": checkpoint_inputs,
            "run_config": checkpoint_run_config,
        },
        "common_state_updates": update_manifest_identity,
        "probe": {
            "path": str(probe_root),
            "manifest_sha256": probe_manifest_sha256,
            "selection_sha256": probe_manifest["selection_sha256"],
            "selected_sample_ids_sha256": probe_manifest["selected_sample_ids_sha256"],
            "dataset_fingerprint": probe_manifest["serialized_probe_dataset_fingerprint"],
            "frozen_spec": probe_spec_identity,
        },
        "intervention_spec": {
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
            raise ValueError("Existing intervention output has different inputs")
        for item in existing.get("outputs", {}).values():
            _verify_declared_file(output_dir, item)
        if existing.get("status") != "complete":
            raise ValueError("Existing intervention manifest is incomplete")
        return existing
    if not overwrite and any(path.exists() for path in (sample_path, condition_path)):
        raise FileExistsError(f"Partial functional intervention exists under {output_dir}")
    if overwrite:
        for path in (manifest_path, sample_path, condition_path):
            if path.is_file():
                path.unlink()

    dtype = torch.float32
    model = _load_model(
        family,
        checkpoint,
        dtype=dtype,
        device=device,
        flash_attention=bool(evaluation["flash_attention"]),
    )
    model.eval()
    partition = parameter_partition(model)
    observed_partition = partition_summary(partition)
    if observed_partition["hidden"] != spec["common_state"]["expected_hidden_partition"]:
        raise ValueError("Loaded model hidden partition differs from the frozen intervention")
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
        for algorithm in ALGORITHMS:
            directions = _load_directions(direction_paths[algorithm], algorithm, mapping)
            for condition in [item for item in conditions if item.algorithm == algorithm]:
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

    expected_records = spec["intervention"]["expected_sample_records_per_anchor"]
    if len(all_samples) != expected_records or len(all_conditions) != len(conditions):
        raise AssertionError("Functional intervention output cardinality changed")
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
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure scale-matched one-step optimizer interventions on a frozen probe"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--update-dir", type=Path, required=True)
    parser.add_argument(
        "--probe", type=Path, default=Path("data/probes/decontaminated-beir-224-seed4242")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family", choices=("dense", "late"), required=True)
    parser.add_argument(
        "--intervention-spec", type=Path, default=Path("configs/functional_intervention.json")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = run_functional_intervention(
        args.checkpoint,
        args.update_dir,
        args.probe,
        args.output_dir,
        family=args.family,
        intervention_spec=args.intervention_spec,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
