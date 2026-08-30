from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .common_state_matrix import (
    CommonStateJob,
    _load_protocol,
    _resolve_reference,
    build_common_state_jobs,
    common_state_job_complete,
    resolve_common_state_spec,
)
from .common_state_summary import ExpectedCommonStateMetric, expected_common_state_metrics
from .config import ModelFamily, load_matrix, resolve_matrix_path
from .geometry import SCHEMA_VERSION, _atomic_json, _atomic_jsonl, _sha256
from .geometry_summary import _atomic_csv
from .scope import ALL_FAMILIES, resolve_scope
from .update_geometry import ALGORITHMS


@dataclass(frozen=True)
class CommonStateSpectrumJob:
    common_state: CommonStateJob
    output_dir: Path


@dataclass
class RunningSpectrumJob:
    job: CommonStateSpectrumJob
    process: subprocess.Popen
    log_handle: Any
    attempts: int


def resolve_spectrum_spec(path: str | Path, prefix: Path | None = None) -> Path:
    path = Path(path)
    if path.is_file() or path.is_absolute() or path.parent != Path("configs"):
        return path
    prefix = Path(sys.prefix) if prefix is None else prefix
    installed = prefix / "share" / "embedding-optimizer-study" / "configs" / path.name
    return installed if installed.is_file() else path


def load_spectrum_spec(path: Path, common_state_spec: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported common-state spectrum schema: {path}")
    if payload.get("common_state_spec_sha256") != _sha256(common_state_spec):
        raise ValueError("Spectrum protocol is bound to a different common-state specification")
    if set(payload) != {
        "schema_version",
        "common_state_spec_sha256",
        "freeze_context",
        "selection",
        "analysis",
    }:
        raise ValueError(f"Unexpected common-state spectrum protocol fields: {path}")
    freeze = payload["freeze_context"]
    expected_freeze = {
        "frozen_at_utc",
        "strict_beir_valid_units",
        "strict_beir_expected_units",
        "partial_beir_results_already_observed",
        "weight_trajectory_results_already_observed",
        "formal_common_state_outputs_already_observed",
        "formal_representation_outputs_already_observed",
        "exploratory_representation_smoke_outputs_already_observed",
        "selection_note",
    }
    if (
        not isinstance(freeze, dict)
        or set(freeze) != expected_freeze
        or not isinstance(freeze["frozen_at_utc"], str)
        or not isinstance(freeze["selection_note"], str)
        or not freeze["selection_note"]
        or freeze["partial_beir_results_already_observed"] is not True
        or freeze["weight_trajectory_results_already_observed"] is not True
        or freeze["formal_common_state_outputs_already_observed"] is not False
        or freeze["formal_representation_outputs_already_observed"] is not False
        or freeze["exploratory_representation_smoke_outputs_already_observed"] is not True
        or not isinstance(freeze["strict_beir_valid_units"], int)
        or not isinstance(freeze["strict_beir_expected_units"], int)
        or not 0 <= freeze["strict_beir_valid_units"] < freeze["strict_beir_expected_units"]
    ):
        raise ValueError("Invalid spectrum protocol freeze disclosure")
    selection = payload["selection"]
    if not isinstance(selection, dict) or set(selection) != {
        "anchor_scope",
        "expected_anchors",
        "families",
        "update_operators",
        "tensor_names",
        "expected_tensors_per_anchor",
        "expected_spectra",
        "rationale",
    }:
        raise ValueError("Invalid spectrum selection fields")
    tensors = selection["tensor_names"]
    if (
        selection["anchor_scope"] != "all-frozen-common-state-anchors"
        or selection["families"] != ["dense", "late"]
        or selection["update_operators"] != list(ALGORITHMS)
        or not isinstance(tensors, list)
        or not tensors
        or len(tensors) != len(set(tensors))
        or not all(isinstance(name, str) and name for name in tensors)
        or selection["expected_tensors_per_anchor"] != len(tensors)
        or selection["expected_spectra"]
        != selection["expected_anchors"] * len(tensors) * len(ALGORITHMS)
        or not isinstance(selection["rationale"], str)
        or not selection["rationale"]
    ):
        raise ValueError("Invalid frozen spectrum selection")
    _, anchor = _load_protocol(common_state_spec)
    if selection["expected_anchors"] != anchor["expected_total_anchors"]:
        raise ValueError("Spectrum anchor count differs from the common-state protocol")
    analysis = payload["analysis"]
    if not isinstance(analysis, dict) or set(analysis) != {
        "device",
        "input",
        "input_storage_dtype",
        "compute_dtype",
        "algorithm",
        "stored_values",
        "derived_normalizations",
        "weight_decay_included",
    }:
        raise ValueError("Invalid spectrum analysis fields")
    expected_analysis = {
        "input": "per-tensor-frobenius-matched-update-directions",
        "input_storage_dtype": "float16",
        "compute_dtype": "float32",
        "algorithm": "torch.linalg.svdvals-exact",
        "stored_values": "raw-singular-values-descending",
        "derived_normalizations": ["frobenius", "spectral"],
        "weight_decay_included": False,
    }
    if analysis.get("device") not in {"cpu", "cuda"} or any(
        analysis.get(key) != value for key, value in expected_analysis.items()
    ):
        raise ValueError("Unsupported common-state spectrum analysis protocol")
    return payload


def build_spectrum_jobs(
    common_state_jobs: list[CommonStateJob], output_root: Path
) -> list[CommonStateSpectrumJob]:
    root = output_root.resolve()
    return [
        CommonStateSpectrumJob(common_state=job, output_dir=root / job.label)
        for job in common_state_jobs
    ]


def _declared(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _verify_declared(root: Path, item: Any, *, verify_hash: bool) -> bool:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        return False
    path = root / item["path"]
    return (
        path.is_file()
        and path.stat().st_size == item.get("bytes")
        and (not verify_hash or _sha256(path) == item.get("sha256"))
    )


def spectrum_job_complete(
    job: CommonStateSpectrumJob,
    spectrum_spec: Path,
    common_state_spec: Path,
    *,
    verify_hashes: bool = False,
) -> bool:
    manifest_path = job.output_dir / "manifest.json"
    if not manifest_path.is_file() or not common_state_job_complete(
        job.common_state, common_state_spec, verify_hashes=verify_hashes
    ):
        return False
    try:
        protocol = load_spectrum_spec(spectrum_spec, common_state_spec)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_manifest = job.common_state.update_dir / "manifest.json"
        output = manifest["output"]
        return (
            manifest.get("schema_version") == SCHEMA_VERSION
            and manifest.get("status") == "complete"
            and manifest.get("family") == job.common_state.family
            and manifest.get("label") == job.common_state.label
            and Path(manifest["checkpoint"]).resolve() == job.common_state.checkpoint
            and manifest["spectrum_spec"]["sha256"] == _sha256(spectrum_spec)
            and manifest["common_state_spec"]["sha256"] == _sha256(common_state_spec)
            and Path(manifest["source_update_manifest"]["path"]).resolve()
            == source_manifest.resolve()
            and manifest["source_update_manifest"]["bytes"] == source_manifest.stat().st_size
            and manifest["source_update_manifest"]["sha256"] == _sha256(source_manifest)
            and manifest["records"] == len(protocol["selection"]["tensor_names"]) * len(ALGORITHMS)
            and _verify_declared(job.output_dir, output, verify_hash=verify_hashes)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _spectrum_record(
    matrix: torch.Tensor,
    *,
    family: str,
    label: str,
    algorithm: str,
    tensor_name: str,
    source_dtype: str,
) -> dict[str, Any]:
    singular = torch.linalg.svdvals(matrix).float().cpu()
    if singular.numel() != min(matrix.shape) or not torch.isfinite(singular).all():
        raise ValueError(f"Invalid singular values for {label}/{algorithm}/{tensor_name}")
    if singular.numel() > 1 and bool((singular[:-1] < singular[1:]).any()):
        raise ValueError(
            f"Singular values are not descending for {label}/{algorithm}/{tensor_name}"
        )
    spectral = singular[0]
    if spectral <= 0:
        raise ValueError(f"Zero spectrum for {label}/{algorithm}/{tensor_name}")
    frobenius_sq = singular.square().sum()
    nuclear = singular.sum()
    probabilities = singular / nuclear
    nonzero = probabilities > 0
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    threshold = torch.finfo(singular.dtype).eps * max(matrix.shape) * spectral
    usable = singular[singular > threshold]
    condition = float((usable[0] / usable[-1]).item()) if usable.numel() else None
    return {
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "label": label,
        "update_operator": algorithm,
        "tensor": tensor_name,
        "shape": list(matrix.shape),
        "source_dtype": source_dtype,
        "compute_dtype": "float32",
        "rank": int(singular.numel()),
        "frobenius_norm": float(frobenius_sq.sqrt().item()),
        "spectral_norm": float(spectral.item()),
        "stable_rank": float((frobenius_sq / spectral.square()).item()),
        "nuclear_norm": float(nuclear.item()),
        "entropy_effective_rank": float(entropy.exp().item()),
        "condition_number": condition,
        "singular_values": [float(value) for value in singular.tolist()],
    }


def analyze_common_state_spectra(
    job: CommonStateSpectrumJob,
    *,
    spectrum_spec: Path,
    common_state_spec: Path,
    device: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    spectrum_spec = spectrum_spec.resolve()
    common_state_spec = common_state_spec.resolve()
    protocol = load_spectrum_spec(spectrum_spec, common_state_spec)
    if device != protocol["analysis"]["device"]:
        raise ValueError("Requested spectrum device differs from the frozen protocol")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA spectrum analysis was requested but CUDA is unavailable")
    if not common_state_job_complete(job.common_state, common_state_spec, verify_hashes=True):
        raise ValueError(f"Common-state source is incomplete or invalid: {job.common_state.label}")
    source_manifest_path = job.common_state.update_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    identity = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "family": job.common_state.family,
        "label": job.common_state.label,
        "checkpoint": str(job.common_state.checkpoint),
        "spectrum_spec": {
            "path": str(spectrum_spec),
            "bytes": spectrum_spec.stat().st_size,
            "sha256": _sha256(spectrum_spec),
        },
        "common_state_spec": {
            "path": str(common_state_spec),
            "bytes": common_state_spec.stat().st_size,
            "sha256": _sha256(common_state_spec),
        },
        "source_update_manifest": {
            "path": str(source_manifest_path),
            "bytes": source_manifest_path.stat().st_size,
            "sha256": _sha256(source_manifest_path),
        },
        "analysis": protocol["analysis"],
    }
    manifest_path = job.output_dir / "manifest.json"
    output_path = job.output_dir / "spectra.jsonl"
    if manifest_path.is_file() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if {key: existing.get(key) for key in identity} != identity:
            raise ValueError(f"Existing spectrum identity differs for {job.common_state.label}")
        if not spectrum_job_complete(job, spectrum_spec, common_state_spec, verify_hashes=True):
            raise ValueError(f"Existing spectrum output is invalid for {job.common_state.label}")
        return existing
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Partial spectrum output exists: {output_path}")
    if overwrite:
        for path in (output_path, manifest_path):
            if path.is_file():
                path.unlink()

    selected = protocol["selection"]["tensor_names"]
    records: list[dict[str, Any]] = []
    source_files = []
    for algorithm in ALGORITHMS:
        metadata = source_manifest["outputs"][f"{algorithm}_matched"]
        source_path = job.common_state.update_dir / metadata["path"]
        if not _verify_declared(job.common_state.update_dir, metadata, verify_hash=True):
            raise ValueError(f"Matched update differs from its manifest: {source_path}")
        with safe_open(str(source_path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            missing = sorted(set(selected) - available)
            if missing:
                raise ValueError(f"Selected tensors are absent from {source_path}: {missing}")
            for tensor_name in selected:
                source = handle.get_tensor(tensor_name)
                if source.ndim != 2 or not torch.isfinite(source).all():
                    raise ValueError(f"Invalid matched update {algorithm}/{tensor_name}")
                matrix = source.to(device=target, dtype=torch.float32)
                records.append(
                    _spectrum_record(
                        matrix,
                        family=job.common_state.family,
                        label=job.common_state.label,
                        algorithm=algorithm,
                        tensor_name=tensor_name,
                        source_dtype=str(source.dtype).removeprefix("torch."),
                    )
                )
                del matrix
        source_files.append(
            {
                "update_operator": algorithm,
                "path": str(source_path),
                "bytes": source_path.stat().st_size,
                "sha256": _sha256(source_path),
            }
        )
    expected_records = len(selected) * len(ALGORITHMS)
    if len(records) != expected_records:
        raise AssertionError(f"Produced {len(records)} spectra, expected {expected_records}")
    _atomic_jsonl(output_path, records)
    manifest = {
        **identity,
        "records": len(records),
        "singular_values": sum(record["rank"] for record in records),
        "selected_tensors": selected,
        "source_files": source_files,
        "output": _declared(output_path, job.output_dir),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": device,
            "gpu_name": torch.cuda.get_device_name(target) if target.type == "cuda" else None,
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _summary_identity(expected: ExpectedCommonStateMetric) -> dict[str, Any]:
    return {
        "family": expected.job.family,
        "anchor_kind": expected.anchor_kind,
        "source_optimizer": expected.source_optimizer,
        "learning_rate": expected.learning_rate,
        "run_id": expected.run_id,
        "stage": expected.stage,
        "fraction": expected.fraction,
        "step": expected.step,
        "label": expected.job.label,
    }


def _read_spectrum_records(
    job: CommonStateSpectrumJob,
    *,
    spectrum_spec: Path,
    common_state_spec: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not spectrum_job_complete(job, spectrum_spec, common_state_spec, verify_hashes=True):
        raise ValueError(f"Spectrum job is missing or invalid: {job.common_state.label}")
    protocol = load_spectrum_spec(spectrum_spec, common_state_spec)
    manifest_path = job.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest["output"]
    path = job.output_dir / metadata["path"]
    raw = path.read_bytes()
    if len(raw) != metadata["bytes"] or _sha256(path) != metadata["sha256"]:
        raise ValueError(f"Spectrum records differ from their manifest: {path}")
    records = [json.loads(line) for line in raw.splitlines() if line]
    tensors = protocol["selection"]["tensor_names"]
    expected_pairs = {(algorithm, tensor) for algorithm in ALGORITHMS for tensor in tensors}
    observed_pairs: set[tuple[str, str]] = set()
    expected_fields = {
        "schema_version",
        "family",
        "label",
        "update_operator",
        "tensor",
        "shape",
        "source_dtype",
        "compute_dtype",
        "rank",
        "frobenius_norm",
        "spectral_norm",
        "stable_rank",
        "nuclear_norm",
        "entropy_effective_rank",
        "condition_number",
        "singular_values",
    }
    for record in records:
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ValueError(f"Invalid spectrum record under {path}")
        pair = (record["update_operator"], record["tensor"])
        shape = record["shape"]
        singular = record["singular_values"]
        context = f"{job.common_state.label}/{pair[0]}/{pair[1]}"
        scalar_fields = (
            "frobenius_norm",
            "spectral_norm",
            "stable_rank",
            "nuclear_norm",
            "entropy_effective_rank",
        )
        if (
            record["schema_version"] != SCHEMA_VERSION
            or record["family"] != job.common_state.family
            or record["label"] != job.common_state.label
            or pair not in expected_pairs
            or pair in observed_pairs
            or not isinstance(shape, list)
            or len(shape) != 2
            or not all(isinstance(value, int) and value > 0 for value in shape)
            or record["source_dtype"] != protocol["analysis"]["input_storage_dtype"]
            or record["compute_dtype"] != protocol["analysis"]["compute_dtype"]
            or record["rank"] != min(shape)
            or not isinstance(singular, list)
            or len(singular) != record["rank"]
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in singular
            )
            or any(singular[index] < singular[index + 1] for index in range(len(singular) - 1))
            or any(
                isinstance(record[name], bool)
                or not isinstance(record[name], (int, float))
                or not math.isfinite(record[name])
                or record[name] <= 0
                for name in scalar_fields
            )
            or (
                record["condition_number"] is not None
                and (
                    isinstance(record["condition_number"], bool)
                    or not isinstance(record["condition_number"], (int, float))
                    or not math.isfinite(record["condition_number"])
                    or record["condition_number"] < 1
                )
            )
        ):
            raise ValueError(f"Invalid exact spectrum values in {context}")
        frobenius = math.sqrt(sum(float(value) ** 2 for value in singular))
        if (
            not math.isclose(frobenius, record["frobenius_norm"], rel_tol=2e-6, abs_tol=2e-6)
            or not math.isclose(singular[0], record["spectral_norm"], rel_tol=2e-6, abs_tol=2e-6)
            or not math.isclose(sum(singular), record["nuclear_norm"], rel_tol=2e-6, abs_tol=2e-6)
            or not math.isclose(
                frobenius**2 / singular[0] ** 2,
                record["stable_rank"],
                rel_tol=2e-6,
                abs_tol=2e-6,
            )
        ):
            raise ValueError(f"Spectrum summary does not reproduce from singular values: {context}")
        observed_pairs.add(pair)
    if observed_pairs != expected_pairs:
        raise ValueError(f"Spectrum operator/tensor coverage differs under {path}")
    return manifest, sorted(records, key=lambda value: (value["update_operator"], value["tensor"]))


def summarize_spectrum_matrix(
    jobs: list[CommonStateSpectrumJob],
    expected: list[ExpectedCommonStateMetric],
    result_root: Path,
    output_dir: Path,
    *,
    spectrum_spec: Path,
    common_state_spec: Path,
    allow_partial: bool = False,
    families: tuple[str, ...] = ALL_FAMILIES,
    scope_amendment: str | Path | None = None,
) -> dict[str, Any]:
    result_root = result_root.resolve()
    output_dir = output_dir.resolve()
    spectrum_spec = spectrum_spec.resolve()
    common_state_spec = common_state_spec.resolve()
    protocol = load_spectrum_spec(spectrum_spec, common_state_spec)
    families, scope = resolve_scope(families, scope_amendment)
    declared_families = tuple(protocol["selection"]["families"])
    if declared_families != ALL_FAMILIES:
        raise ValueError("Frozen spectrum protocol no longer covers the original two-family scope")
    expected_per_family, remainder = divmod(
        int(protocol["selection"]["expected_anchors"]), len(declared_families)
    )
    if remainder:
        raise ValueError("Frozen spectrum anchor count is not divisible by its family count")
    expected_anchors = expected_per_family * len(families)
    if len(jobs) != expected_anchors or len(expected) != expected_anchors:
        raise ValueError(
            f"Frozen spectrum summary requires {expected_anchors} anchors, "
            f"received {len(jobs)} jobs and {len(expected)} identities"
        )
    if {job.common_state.family for job in jobs} != set(families):
        raise ValueError("Spectrum jobs do not match the requested family scope")
    identity_by_label = {item.job.label: item for item in expected}
    if len(identity_by_label) != expected_anchors:
        raise ValueError("Duplicate or missing common-state spectrum identities")
    job_labels = {job.common_state.label for job in jobs}
    if job_labels != set(identity_by_label):
        raise ValueError("Spectrum jobs and common-state identities differ")
    expected_paths = {(job.output_dir / "spectra.jsonl").resolve() for job in jobs}
    observed_paths = set()
    for path in result_root.rglob("spectra.jsonl"):
        relative = path.relative_to(result_root)
        if relative.parts and relative.parts[0] in set(ALL_FAMILIES) - set(families):
            continue
        observed_paths.add(path.resolve())
    unexpected = observed_paths.difference(expected_paths)
    if unexpected:
        raise ValueError(f"Unexpected common-state spectrum files: {sorted(unexpected)[:5]}")

    metric_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    inputs = []
    missing = []
    for job in jobs:
        if not spectrum_job_complete(job, spectrum_spec, common_state_spec, verify_hashes=True):
            missing.append(job.common_state.label)
            continue
        manifest, records = _read_spectrum_records(
            job, spectrum_spec=spectrum_spec, common_state_spec=common_state_spec
        )
        identity = _summary_identity(identity_by_label[job.common_state.label])
        for record in records:
            spectrum_identity = {
                **identity,
                "update_operator": record["update_operator"],
                "tensor": record["tensor"],
                "rows": record["shape"][0],
                "columns": record["shape"][1],
                "rank": record["rank"],
            }
            metric_rows.append(
                {
                    **spectrum_identity,
                    "frobenius_norm": record["frobenius_norm"],
                    "spectral_norm": record["spectral_norm"],
                    "stable_rank": record["stable_rank"],
                    "nuclear_norm": record["nuclear_norm"],
                    "entropy_effective_rank": record["entropy_effective_rank"],
                    "condition_number": (
                        "" if record["condition_number"] is None else record["condition_number"]
                    ),
                }
            )
            cumulative_energy = 0.0
            frobenius_sq = record["frobenius_norm"] ** 2
            for index, value in enumerate(record["singular_values"], start=1):
                energy = value**2 / frobenius_sq
                cumulative_energy += energy
                value_rows.append(
                    {
                        **spectrum_identity,
                        "singular_index": index,
                        "normalized_index": index / record["rank"],
                        "singular_value": value,
                        "frobenius_normalized_value": value / record["frobenius_norm"],
                        "spectral_normalized_value": value / record["spectral_norm"],
                        "energy_fraction": energy,
                        "cumulative_energy_fraction": min(cumulative_energy, 1.0),
                    }
                )
        manifest_path = job.output_dir / "manifest.json"
        inputs.append(
            {
                "label": job.common_state.label,
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "spectra_sha256": manifest["output"]["sha256"],
            }
        )
    if missing and not allow_partial:
        raise ValueError(
            f"Common-state spectrum matrix is incomplete: {len(missing)}/{len(jobs)} missing; "
            f"first={missing[:5]}"
        )
    if not metric_rows:
        raise ValueError("No valid common-state spectra were found")

    identity_fields = [
        "family",
        "anchor_kind",
        "source_optimizer",
        "learning_rate",
        "run_id",
        "stage",
        "fraction",
        "step",
        "label",
        "update_operator",
        "tensor",
        "rows",
        "columns",
        "rank",
    ]
    metric_path = output_dir / "spectrum_metrics.csv"
    values_path = output_dir / "singular_values.csv"
    _atomic_csv(
        metric_path,
        metric_rows,
        [
            *identity_fields,
            "frobenius_norm",
            "spectral_norm",
            "stable_rank",
            "nuclear_norm",
            "entropy_effective_rank",
            "condition_number",
        ],
    )
    _atomic_csv(
        values_path,
        value_rows,
        [
            *identity_fields,
            "singular_index",
            "normalized_index",
            "singular_value",
            "frobenius_normalized_value",
            "spectral_normalized_value",
            "energy_fraction",
            "cumulative_energy_fraction",
        ],
    )
    complete = not missing and len(inputs) == expected_anchors
    expected_spectra = (
        expected_anchors * len(protocol["selection"]["tensor_names"]) * len(ALGORITHMS)
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": complete,
        "allow_partial": allow_partial,
        "families": list(families),
        "scope_amendment": scope,
        "expected_anchors": expected_anchors,
        "valid_anchors": len(inputs),
        "expected_spectra": expected_spectra,
        "valid_spectra": len(metric_rows),
        "singular_values": len(value_rows),
        "missing_labels": missing,
        "spectrum_spec": {
            "path": str(spectrum_spec),
            "sha256": _sha256(spectrum_spec),
            "freeze_context": protocol["freeze_context"],
        },
        "common_state_spec": {
            "path": str(common_state_spec),
            "sha256": _sha256(common_state_spec),
        },
        "inputs": inputs,
        "outputs": {
            "spectrum_metrics": {
                "path": str(metric_path),
                "rows": len(metric_rows),
                "bytes": metric_path.stat().st_size,
                "sha256": _sha256(metric_path),
            },
            "singular_values": {
                "path": str(values_path),
                "rows": len(value_rows),
                "bytes": values_path.stat().st_size,
                "sha256": _sha256(values_path),
            },
        },
        "interpretation": (
            "Exact spectra are computed from float16 per-tensor Frobenius-matched directions; "
            "normalized spectrum shape is comparable across operators, while raw singular-value "
            "scale reflects the declared matched intervention rather than native learning rates."
        ),
    }
    _atomic_json(output_dir / "summary_manifest.json", manifest)
    return manifest


def _job_cli(job: CommonStateSpectrumJob, args: argparse.Namespace) -> list[str]:
    common = job.common_state
    return [
        sys.executable,
        "-m",
        "embed_optim.common_state_spectra",
        "--worker",
        "--family",
        common.family,
        "--label",
        common.label,
        "--checkpoint",
        str(common.checkpoint),
        "--gradient-dir",
        str(common.gradient_dir),
        "--update-dir",
        str(common.update_dir),
        "--output-dir",
        str(job.output_dir),
        "--common-state-spec",
        str(args.common_state_spec.resolve()),
        "--spectrum-spec",
        str(args.spectrum_spec.resolve()),
        "--device",
        "cuda",
    ]


def _launch(
    job: CommonStateSpectrumJob, gpu: str, args: argparse.Namespace, attempts: int
) -> RunningSpectrumJob:
    log_path = args.log_dir.resolve() / f"{job.common_state.label.replace('/', '__')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    command = _job_cli(job, args)
    print(f"[{gpu}] {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=Path.cwd(),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"},
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return RunningSpectrumJob(job=job, process=process, log_handle=log_handle, attempts=attempts)


def run_spectrum_matrix(jobs: list[CommonStateSpectrumJob], args: argparse.Namespace) -> int:
    blocked = [
        job
        for job in jobs
        if not common_state_job_complete(
            job.common_state, args.common_state_spec, verify_hashes=args.verify_hashes
        )
    ]
    ready = [job for job in jobs if job not in blocked]
    pending = [
        job
        for job in ready
        if not spectrum_job_complete(
            job,
            args.spectrum_spec,
            args.common_state_spec,
            verify_hashes=args.verify_hashes,
        )
    ]
    print(
        json.dumps(
            {
                "complete": len(ready) - len(pending),
                "expected": len(jobs),
                "pending": len(pending),
                "blocked_common_state_inputs": len(blocked),
                "verify_hashes": args.verify_hashes,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.audit_only:
        return len(pending) + len(blocked)
    if args.dry_run:
        for job in pending:
            print(job.common_state.label)
        for job in blocked:
            print(f"[blocked] {job.common_state.label}")
        return 0
    if blocked:
        raise ValueError(
            f"{len(blocked)} spectrum jobs lack complete common-state inputs; "
            f"first={[job.common_state.label for job in blocked[:5]]}"
        )
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError(f"--gpus must contain unique comma-separated IDs, got {args.gpus!r}")
    running: dict[str, RunningSpectrumJob] = {}
    attempts: dict[str, int] = {}
    failures = 0
    while pending or running:
        for gpu, running_job in list(running.items()):
            return_code = running_job.process.poll()
            if return_code is None:
                continue
            running_job.log_handle.close()
            del running[gpu]
            if return_code == 0 and spectrum_job_complete(
                running_job.job, args.spectrum_spec, args.common_state_spec
            ):
                print(f"completed {running_job.job.common_state.label}", flush=True)
                continue
            if running_job.attempts <= args.max_retries:
                pending.insert(0, running_job.job)
                print(
                    f"retrying {running_job.job.common_state.label} after exit {return_code} "
                    f"(attempt {running_job.attempts})",
                    flush=True,
                )
            else:
                failures += 1
                print(
                    f"failed {running_job.job.common_state.label} after "
                    f"{running_job.attempts} attempts",
                    flush=True,
                )
                if args.fail_fast:
                    for other in running.values():
                        other.process.terminate()
                    return failures
        for gpu in gpus:
            if gpu in running or not pending:
                continue
            job = pending.pop(0)
            attempt = attempts.get(job.common_state.label, 0) + 1
            attempts[job.common_state.label] = attempt
            running[gpu] = _launch(job, gpu, args, attempt)
        if running:
            time.sleep(1)
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a frozen exact-spectrum tier over common-state update directions"
    )
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--families", nargs="+", choices=("dense", "late"), default=["dense", "late"]
    )
    parser.add_argument("--scope-amendment", type=Path)
    parser.add_argument("--common-state-root", type=Path, default=Path("results/common-state"))
    parser.add_argument(
        "--common-state-spec", type=Path, default=Path("configs/common_state_probe.json")
    )
    parser.add_argument(
        "--spectrum-spec", type=Path, default=Path("configs/common_state_spectrum_probe.json")
    )
    parser.add_argument("--output-root", type=Path, default=Path("results/common-state-spectra"))
    parser.add_argument("--dense-reference-checkpoint", type=Path)
    parser.add_argument("--late-reference-checkpoint", type=Path)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/common-state-spectra"))
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--allow-partial-summary", action="store_true")
    parser.add_argument("--summary-dir", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--family", choices=("dense", "late"), help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--gradient-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--update-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative")
    if args.summarize_only and (args.dry_run or args.audit_only):
        raise ValueError("--summarize-only cannot be combined with --dry-run or --audit-only")
    if args.allow_partial_summary and not args.summarize_only:
        raise ValueError("--allow-partial-summary requires --summarize-only")
    args.common_state_spec = resolve_common_state_spec(args.common_state_spec).resolve()
    args.spectrum_spec = resolve_spectrum_spec(args.spectrum_spec).resolve()
    common_protocol, anchor = _load_protocol(args.common_state_spec)
    load_spectrum_spec(args.spectrum_spec, args.common_state_spec)
    if args.worker:
        required = (
            args.family,
            args.label,
            args.checkpoint,
            args.gradient_dir,
            args.update_dir,
            args.output_dir,
        )
        if any(value is None for value in required):
            raise ValueError("Spectrum worker invocation is missing required job fields")
        partition = anchor["expected_hidden_partition"]
        common = CommonStateJob(
            family=args.family,
            label=args.label,
            checkpoint=args.checkpoint.resolve(),
            gradient_dir=args.gradient_dir.resolve(),
            update_dir=args.update_dir.resolve(),
            gradient_steps=int(common_protocol["selection"]["gradient_steps"]),
            hidden_tensors=int(partition["tensors"]),
            hidden_parameters=int(partition["parameters"]),
        )
        analyze_common_state_spectra(
            CommonStateSpectrumJob(common_state=common, output_dir=args.output_dir.resolve()),
            spectrum_spec=args.spectrum_spec,
            common_state_spec=args.common_state_spec,
            device=args.device,
            overwrite=args.overwrite,
        )
        return

    families, _ = resolve_scope(args.families, args.scope_amendment)
    matrix_path = resolve_matrix_path(args.matrix).resolve()
    all_configs = load_matrix(matrix_path)
    configs = [config for config in all_configs if config.model_family in families]
    if {config.model_family for config in configs} != set(families):
        raise ValueError("Training matrix does not cover every requested model family")
    by_family = {config.model_family: config for config in configs}
    references: dict[ModelFamily, Path] = {}
    for family in families:
        explicit = (
            args.dense_reference_checkpoint if family == "dense" else args.late_reference_checkpoint
        )
        if args.dry_run and explicit is None:
            references[family] = Path(by_family[family].model_name).resolve()
        else:
            references[family] = _resolve_reference(by_family[family], explicit)
    common_jobs = build_common_state_jobs(
        configs, references, common_protocol, args.common_state_root
    )
    jobs = build_spectrum_jobs(common_jobs, args.output_root)
    expected = expected_common_state_metrics(common_jobs, configs)
    summary_dir = args.summary_dir or (args.output_root / "summary")
    if args.summarize_only:
        summary = summarize_spectrum_matrix(
            jobs,
            expected,
            args.output_root,
            summary_dir,
            spectrum_spec=args.spectrum_spec,
            common_state_spec=args.common_state_spec,
            allow_partial=args.allow_partial_summary,
            families=families,
            scope_amendment=args.scope_amendment,
        )
        print(
            f"Aggregated {summary['valid_spectra']}/{summary['expected_spectra']} exact spectra "
            f"into {summary_dir.resolve()}"
        )
        return
    failures = run_spectrum_matrix(jobs, args)
    if failures:
        raise SystemExit(1)
    if not args.dry_run and not args.audit_only:
        summarize_spectrum_matrix(
            jobs,
            expected,
            args.output_root,
            summary_dir,
            spectrum_spec=args.spectrum_spec,
            common_state_spec=args.common_state_spec,
            families=families,
            scope_amendment=args.scope_amendment,
        )


if __name__ == "__main__":
    main()
