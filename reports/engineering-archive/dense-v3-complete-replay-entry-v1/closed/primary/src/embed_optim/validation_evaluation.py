from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .collators import DenseGroupCollator, LateGroupCollator
from .functional_intervention import (
    InterventionCondition,
    _evaluate_condition,
)
from .geometry import SCHEMA_VERSION, _atomic_json, _atomic_jsonl, _sha256
from .gradient_probe import _temperature_from_checkpoint
from .probe_export import (
    ModelFamily,
    _checkpoint_inputs,
    _load_model,
    _load_probe,
    _validate_checkpoint_family,
    _validate_probe_spec,
)
from .validation_data import audit_validation_data, load_validation_spec

METRICS = (
    "contrastive_loss",
    "positive_score",
    "hardest_negative_score",
    "positive_margin",
    "reciprocal_rank",
    "top1_accuracy",
)


def _identity(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _verify_file(root: Path, item: dict[str, Any]) -> None:
    path = root / item["path"]
    if (
        not path.is_file()
        or path.stat().st_size != item.get("bytes")
        or _sha256(path) != item.get("sha256")
    ):
        raise ValueError(f"Validation evaluation output differs from its manifest: {path}")


def _group_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped["__all__"].extend(records)
    for record in records:
        grouped[str(record["group"])].append(record)
    summaries = []
    for group, rows in sorted(grouped.items()):
        summaries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "group": group,
                "samples": len(rows),
                **{
                    metric: sum(float(row[metric]) for row in rows) / len(rows)
                    for metric in METRICS
                },
            }
        )
    return summaries


def run_validation_evaluation(
    checkpoint: str | Path,
    probe_root: str | Path,
    output_dir: str | Path,
    *,
    family: ModelFamily,
    validation_spec: str | Path = "configs/validation_probe.json",
    device: str = "cuda",
    audit_probe: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    probe_root = Path(probe_root).resolve()
    output_dir = Path(output_dir).resolve()
    spec_path, spec = load_validation_spec(validation_spec)
    evaluation = spec["evaluation"]
    if family not in {"dense", "late"}:
        raise ValueError(f"Unsupported family {family!r}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation evaluation was requested but CUDA is unavailable")
    if audit_probe:
        audit_validation_data(
            probe_root,
            spec["source"]["training_data"],
            spec_path=spec_path,
        )
    dataset, probe_manifest, probe_manifest_sha256 = _load_probe(probe_root)
    probe_spec_identity = _validate_probe_spec(spec_path, probe_manifest_sha256)
    if len(dataset) != evaluation["expected_sample_records_per_job"]:
        raise ValueError("Validation probe row count differs from the evaluation protocol")
    checkpoint_inputs = _checkpoint_inputs(checkpoint)
    checkpoint_run_config = _validate_checkpoint_family(checkpoint, family)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "family": family,
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
        "validation_spec": {
            "path": str(spec_path),
            "bytes": spec_path.stat().st_size,
            "sha256": _sha256(spec_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    sample_path = output_dir / "sample_metrics.jsonl"
    group_path = output_dir / "group_metrics.jsonl"
    if manifest_path.is_file() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if {key: manifest.get(key) for key in identity} != identity:
            raise ValueError("Existing validation evaluation has different inputs")
        if manifest.get("status") != "complete":
            raise ValueError("Existing validation evaluation is incomplete")
        for item in manifest.get("outputs", {}).values():
            _verify_file(output_dir, item)
        return manifest
    if not overwrite and any(path.exists() for path in (sample_path, group_path)):
        raise FileExistsError(f"Partial validation evaluation exists under {output_dir}")
    if overwrite:
        for path in (manifest_path, sample_path, group_path):
            if path.is_file():
                path.unlink()

    model = _load_model(
        family,
        checkpoint,
        dtype=torch.float32,
        device=device,
        flash_attention=bool(evaluation["flash_attention"]),
    )
    model.eval()
    collator = (
        DenseGroupCollator(model.preprocess) if family == "dense" else LateGroupCollator(model)
    )
    temperature = _temperature_from_checkpoint(checkpoint, family, None)
    baseline = InterventionCondition("baseline", None, "baseline", 0.0, 0.0)
    try:
        records, overall = _evaluate_condition(
            model,
            dataset,
            family=family,
            collator=collator,
            batch_size=int(evaluation[f"{family}_batch_size"]),
            device=device,
            forward_dtype=evaluation["forward_dtype"],
            temperature=temperature,
            condition=baseline,
        )
    finally:
        del collator, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if len(records) != evaluation["expected_sample_records_per_job"]:
        raise AssertionError("Validation evaluation record count changed")
    groups = _group_summaries(records)
    if len(groups) != 8 or groups[0]["group"] != "__all__":
        raise ValueError("Validation evaluation lost a source group")
    for metric in METRICS:
        if abs(float(groups[0][metric]) - float(overall[metric])) > 1e-9:
            raise AssertionError("Overall validation aggregation is inconsistent")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(sample_path, records)
    _atomic_jsonl(group_path, groups)
    manifest = {
        **identity,
        "status": "complete",
        "temperature": temperature,
        "sample_records": len(records),
        "group_records": len(groups),
        "outputs": {
            "sample_metrics": _identity(sample_path, output_dir),
            "group_metrics": _identity(group_path, output_dir),
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
        description="Evaluate one final checkpoint on the frozen query-disjoint validation probe"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=Path("data/validation-4096-seed20260826"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family", choices=("dense", "late"), required=True)
    parser.add_argument(
        "--validation-spec", type=Path, default=Path("configs/validation_probe.json")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-probe-audit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = run_validation_evaluation(
        args.checkpoint,
        args.probe,
        args.output_dir,
        family=args.family,
        validation_spec=args.validation_spec,
        device=args.device,
        audit_probe=not args.skip_probe_audit,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
