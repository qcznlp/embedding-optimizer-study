"""Post-failure audit of DenseOn packed FlashAttention batch invariance.

This is an unplanned implementation diagnostic.  It does not modify any
training artifact or the prospectively frozen candidate-breadth decision.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset

from .collators import DenseGroupCollator
from .functional_intervention import group_scores
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .gradient_probe import _collect_features
from .probe_export import _checkpoint_inputs, _load_model, _validate_checkpoint_family

CONTROL_INDICES = (0, 1)


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def invariance_contrasts(scores: dict[str, np.ndarray]) -> dict[str, float]:
    """Return exact maximum score changes for the three execution contrasts."""

    expected = {
        "packed_batch",
        "packed_singletons",
        "padded_batch",
        "padded_singletons",
    }
    if set(scores) != expected:
        raise ValueError("Packing-invariance score modes changed")
    arrays = {name: np.asarray(value, dtype=np.float32) for name, value in scores.items()}
    if any(value.shape != (2, 8) or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Packing-invariance scores must be finite 2x8 arrays")
    return {
        "packed_batch_vs_singleton_max_abs": float(
            np.max(np.abs(arrays["packed_batch"] - arrays["packed_singletons"]))
        ),
        "padded_batch_vs_singleton_max_abs": float(
            np.max(np.abs(arrays["padded_batch"] - arrays["padded_singletons"]))
        ),
        "packed_batch_vs_padded_batch_max_abs": float(
            np.max(np.abs(arrays["packed_batch"] - arrays["padded_batch"]))
        ),
    }


def validate_packing_invariance_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the self-contained score and execution claims in an audit receipt."""

    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("status") != "complete":
        raise ValueError("Packing-invariance receipt is not complete")
    if payload.get("analysis_status") != "unplanned_post_failure_implementation_audit":
        raise ValueError("Packing-invariance analysis status changed")
    validation = payload.get("validation", {})
    if validation.get("control_indices") != list(CONTROL_INDICES):
        raise ValueError("Packing-invariance control indices changed")
    expected_execution = {
        "model_dtype": "float32",
        "forward_dtype": "bfloat16",
        "attention": "flash_attention_2",
        "packed_mode": "SentenceTransformers can_flatten_inputs=True",
        "padded_control": "SentenceTransformers can_flatten_inputs=False",
    }
    if payload.get("execution") != expected_execution:
        raise ValueError("Packing-invariance execution contract changed")
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("Packing-invariance scores are missing")
    recomputed = invariance_contrasts(scores)
    recorded = payload.get("contrasts")
    if not isinstance(recorded, dict) or set(recorded) != set(recomputed):
        raise ValueError("Packing-invariance contrasts are incomplete")
    for name, value in recomputed.items():
        if float(recorded[name]) != value:
            raise ValueError(f"Packing-invariance contrast changed: {name}")
    return payload


def audit_packing_invariance_report(
    report: str | Path,
    *,
    repository: str | Path = ".",
) -> dict[str, Any]:
    """Audit a receipt against its checkpoint and frozen validation inputs."""

    report = Path(report).resolve()
    repository = Path(repository).resolve()
    payload = validate_packing_invariance_payload(json.loads(report.read_text(encoding="utf-8")))

    def source_path(value: Any) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else repository / path

    checkpoint = source_path(payload.get("checkpoint", {}).get("path"))
    if payload["checkpoint"].get("inputs") != _checkpoint_inputs(checkpoint):
        raise ValueError("Packing-invariance checkpoint inputs changed")
    run_config = _validate_checkpoint_family(checkpoint, "dense")
    recorded_config = payload["checkpoint"].get("run_config")
    if run_config is None or not isinstance(recorded_config, dict):
        raise ValueError("Packing-invariance checkpoint run config is missing")
    if recorded_config.get("sha256") != run_config.get("sha256"):
        raise ValueError("Packing-invariance checkpoint run config changed")

    validation_root = source_path(payload.get("validation", {}).get("path"))
    manifest_path = validation_root / "manifest.json"
    if payload["validation"].get("manifest_bytes") != manifest_path.stat().st_size:
        raise ValueError("Packing-invariance validation manifest size changed")
    if payload["validation"].get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("Packing-invariance validation manifest changed")
    dataset = Dataset.load_from_disk(str(validation_root / "dataset"))
    sample_ids = [int(dataset[index]["sample_id"]) for index in CONTROL_INDICES]
    if payload["validation"].get("sample_ids") != sample_ids:
        raise ValueError("Packing-invariance validation control rows changed")
    return payload


def _score_rows(model: Any, rows: list[dict[str, Any]], device: str) -> np.ndarray:
    collator = DenseGroupCollator(model.preprocess)
    features = _collect_features(collator(rows), device)
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
        ),
    ):
        scores = group_scores(model, features, "dense")
    result = scores.float().cpu().numpy().astype(np.float32, copy=False)
    if result.shape != (len(rows), 8) or not np.isfinite(result).all():
        raise ValueError("Packing-invariance scorer returned invalid scores")
    return result


def run_packing_invariance_audit(
    checkpoint: str | Path,
    validation_root: str | Path,
    output: str | Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    validation_root = Path(validation_root).resolve()
    output = Path(output).resolve()
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("The formal packing-invariance audit requires a CUDA device")
    dataset_path = validation_root / "dataset"
    manifest_path = validation_root / "manifest.json"
    if not dataset_path.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError("Frozen validation data is incomplete")
    dataset = Dataset.load_from_disk(str(dataset_path))
    rows = [dict(dataset[index]) for index in CONTROL_INDICES]
    if len(rows) != 2 or any("sample_id" not in row for row in rows):
        raise ValueError("Packing-invariance control rows changed")

    model = _load_model(
        "dense",
        checkpoint,
        dtype=torch.float32,
        device=device,
        flash_attention=True,
    )
    first_module = model._first_module()
    if not hasattr(first_module, "can_flatten_inputs"):
        raise ValueError("Dense transformer does not expose the packing control")
    original = bool(first_module.can_flatten_inputs)
    if not original:
        raise ValueError("Legacy Dense execution is no longer configured to flatten inputs")
    try:
        first_module.can_flatten_inputs = True
        packed_batch = _score_rows(model, rows, device)
        packed_singletons = np.concatenate(
            [_score_rows(model, [row], device) for row in rows], axis=0
        )
        first_module.can_flatten_inputs = False
        padded_batch = _score_rows(model, rows, device)
        padded_singletons = np.concatenate(
            [_score_rows(model, [row], device) for row in rows], axis=0
        )
    finally:
        first_module.can_flatten_inputs = original
        del model
        torch.cuda.empty_cache()

    scores = {
        "packed_batch": packed_batch,
        "packed_singletons": packed_singletons,
        "padded_batch": padded_batch,
        "padded_singletons": padded_singletons,
    }
    contrasts = invariance_contrasts(scores)
    package_versions = {}
    for package in ("sentence-transformers", "transformers", "flash-attn"):
        package_versions[package] = importlib.metadata.version(package)
    run_config = _validate_checkpoint_family(checkpoint, "dense")
    if run_config is not None:
        run_config["path"] = _portable_path(Path(run_config["path"]))
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "analysis_status": "unplanned_post_failure_implementation_audit",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checkpoint": {
            "path": _portable_path(checkpoint),
            "inputs": _checkpoint_inputs(checkpoint),
            "run_config": run_config,
        },
        "validation": {
            "path": _portable_path(validation_root),
            "manifest_bytes": manifest_path.stat().st_size,
            "manifest_sha256": _sha256(manifest_path),
            "control_indices": list(CONTROL_INDICES),
            "sample_ids": [int(row["sample_id"]) for row in rows],
        },
        "execution": {
            "model_dtype": "float32",
            "forward_dtype": "bfloat16",
            "attention": "flash_attention_2",
            "packed_mode": "SentenceTransformers can_flatten_inputs=True",
            "padded_control": "SentenceTransformers can_flatten_inputs=False",
        },
        "scores": {name: value.tolist() for name, value in scores.items()},
        "contrasts": contrasts,
        "runtime": {
            "python_packages": package_versions,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": device,
            "gpu_name": torch.cuda.get_device_name(torch.device(device)),
        },
        "interpretation": (
            "A model without cross-example state should not materially change one example's "
            "scores when a second example is added. This unplanned control diagnoses execution "
            "semantics only; its thresholds and rows were chosen after the candidate width-7 "
            "reproduction failure was observed."
        ),
        "claim_boundary": (
            "The audit can identify batch non-invariance in this pinned software stack. It cannot "
            "estimate optimizer effects, repair historical training, or turn the post-hoc "
            "candidate-breadth analysis into confirmatory evidence."
        ),
    }
    _atomic_json(output, result)
    return validate_packing_invariance_payload(result)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/dense/adamw-lr1e-6/checkpoint-3907"),
    )
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=Path("data/validation-4096-seed20260826"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/candidate-breadth/packing_invariance.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.audit_only:
        result = audit_packing_invariance_report(args.output)
    else:
        result = run_packing_invariance_audit(
            args.checkpoint,
            args.validation_root,
            args.output,
            device=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
