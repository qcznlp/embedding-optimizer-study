"""Reconstruct exact geometry from retained spectra, not from checkpoint weights.

The stored SVD residual and original FP64 weight norms remain authenticated
measurements. Spectrum-defined quantities and all downstream aggregates are
recomputed. An archived orthonormal basis is not a fresh SVD of model weights.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from safetensors import safe_open

from . import reconstruction_files as files
from .geometry_robustness import REFERENCE_ATOL, REFERENCE_RTOL, spectrum_summary
from .primary_completion import integer, number
from .primary_contract import digest, read_json, require_same, verify_file
from .primary_exact_geometry_kernels import KINDS, checkpoint_row
from .primary_v3_exact_geometry import SCOPE
from .primary_v3_geometry_io import descriptor
from .primary_v3_geometry_primitives import stage_records as original_stage


def spectrum_metrics(singular, shape, settings, residual):
    """Same spectral definitions as exact_matrix; do not repeat its SVD claim."""
    if (
        not isinstance(singular, np.ndarray)
        or singular.dtype != np.float64
        or singular.shape != (min(shape),)
    ):
        raise ValueError("Require the complete native-shape FP64 exact spectrum")
    rank = min(integer(settings["subspace_rank"], minimum=1), min(shape))
    tolerance = number(settings["boundary_gap_to_leading_tolerance"])
    if not 0 <= tolerance < 1:
        raise ValueError("Invalid exact boundary resolution tolerance")
    summary = spectrum_summary(singular)
    if summary["zero"]:
        require_same(residual, None)
        return {
            "basis_status": "zero",
            "retained_rank": rank,
            "numerical_rank": 0,
            "numerical_threshold": 0.0,
            "frobenius_norm": 0.0,
            "spectral_norm": 0.0,
            "stable_rank": None,
            "full_entropy_effective_rank": None,
            "top_rank_energy": None,
            "boundary_gap_to_leading": None,
            "relative_boundary_gap": None,
            "relative_reconstruction_residual": None,
        }
    if number(residual) < 0:
        raise ValueError("Invalid recorded SVD reconstruction residual")
    threshold = float(np.finfo(np.float64).eps * max(shape) * singular[0])
    numerical_rank = int(np.sum(singular > threshold))
    difference = float(singular[rank - 1] - singular[rank]) if rank < len(singular) else None
    gap = difference / float(singular[0]) if difference is not None else None
    status = "resolved"
    if numerical_rank < rank:
        status = "insufficient_signal_rank"
    elif gap is not None and gap <= tolerance:
        status = "unresolved_boundary"
    relative = singular / singular[0]
    return {
        "basis_status": status,
        "retained_rank": rank,
        "numerical_rank": numerical_rank,
        "numerical_threshold": threshold,
        **{key: value for key, value in summary.items() if key != "zero"},
        "relative_reconstruction_residual": residual,
        "top_rank_energy": float(np.sum(relative[:rank] ** 2) / np.sum(relative**2)),
        "boundary_gap_to_leading": gap,
        "relative_boundary_gap": difference / float(singular[rank - 1])
        if difference is not None and singular[rank - 1] > 0
        else None,
    }


def stage_arrays(value, path, expected, checked, reference, settings, stage):
    """Check every raw row and array; reproduce metrics before reading derived CSVs."""
    if set(value) != {
        "step",
        "previous_step",
        "cumulative_anchor_step",
        "records",
        "checkpoint_row",
    }:
        raise ValueError("Incomplete archived exact geometry stage")
    steps, recipe, shapes = checked["steps"], expected["recipe"], reference["hidden_shapes"]
    step = steps[stage - 1]
    require_same(
        [value["step"], value["previous_step"], value["cumulative_anchor_step"]],
        [step, steps[stage - 2] if stage > 1 else 0, 0],
    )
    rows = value["records"]
    if not isinstance(rows, list):
        raise ValueError("Exact matrix records must be a complete ordered list")
    require_same(
        [[row["tensor"], row["displacement_kind"]] for row in rows],
        [[name, kind] for name in sorted(shapes) for kind in KINDS],
    )
    fresh, names = [], set()
    with safe_open(files.ordinary(path), framework="pt", device="cpu") as handle:
        require_same(handle.metadata(), {"scope": SCOPE, "run_identity_sha256": digest(expected)})
        for row in rows:
            name, kind = row["tensor"], row["displacement_kind"]
            shape = shapes[name]
            prefix = f"{kind}|{name}|"
            spectrum_name = prefix + "singular"
            names.add(spectrum_name)
            if spectrum_name not in handle.keys():
                raise ValueError("Missing complete exact spectrum")
            singular = handle.get_tensor(spectrum_name)
            if singular.dtype != torch.float64:
                raise ValueError("Archived exact singular values must be FP64")
            metrics = spectrum_metrics(
                singular.numpy(), shape, settings, row["relative_reconstruction_residual"]
            )
            metadata = {
                "run_id": recipe["run_id"],
                "optimizer": recipe["optimizer"]["name"],
                "learning_rate": recipe["optimizer"]["lr"],
                "stage": stage,
                "step": step,
                "displacement_kind": kind,
                "tensor": name,
                "shape": shape,
                "parameters": math.prod(shape),
            }
            require_same(row, {**metadata, **metrics})
            fresh.append({**metadata, **metrics})
            if metrics["basis_status"] == "resolved":
                rank = metrics["retained_rank"]
                for side, size in zip(("left", "right"), shape, strict=True):
                    key = prefix + side
                    names.add(key)
                    if key not in handle.keys():
                        raise ValueError("Missing resolved exact basis")
                    basis = handle.get_tensor(key)
                    if (
                        basis.dtype != torch.float64
                        or list(basis.shape) != [size, rank]
                        or not bool(torch.isfinite(basis).all())
                    ):
                        raise ValueError("Exact basis dtype, native shape or finiteness differs")
                    array = basis.numpy()
                    if np.max(np.abs(array.T @ array - np.eye(rank))) > 1e-10:
                        raise ValueError("Archived exact basis is not FP64 orthonormal")
        require_same(sorted(handle.keys()), sorted(names))
    return fresh


def inspect_run_primitives(root, original_root, expected, checked, reference, contract, original):
    """Recompute exact checkpoint rows with the same authenticated weight-norm inputs."""
    settings = contract.payload["settings"]
    if torch.get_num_threads() != settings["cpu_threads"]:
        raise ValueError("Use the declared exact geometry CPU thread count")
    manifest_path = files.ordinary(root / "manifest.json")
    manifest = read_json(manifest_path)
    if set(manifest) != {"plan", "outputs"}:
        raise ValueError("Incomplete archived exact geometry manifest")
    require_same(
        manifest["plan"], descriptor(expected, checked, reference, settings, SCOPE, contract.sha256)
    )
    require_same([row["step"] for row in manifest["outputs"]], checked["steps"])
    names, checkpoints, stored, matrices = {"manifest.json"}, [], [], []
    for stage, row in enumerate(manifest["outputs"], 1):
        if set(row) != {"step", "records", "arrays"}:
            raise ValueError("Incomplete exact stage binding schema")
        step = checked["steps"][stage - 1]
        for key, name in (
            ("records", f"checkpoint-{step}.json"),
            ("arrays", f"checkpoint-{step}-exact.safetensors"),
        ):
            if set(row[key]) != {"path", "bytes", "sha256"}:
                raise ValueError("Incomplete exact primitive file identity")
            require_same(row[key]["path"], name)
            files.identity(row[key])
            verify_file(files.ordinary(root / name), row[key])
            names.add(name)
        value = read_json(root / row["records"]["path"])
        records = stage_arrays(
            value, root / row["arrays"]["path"], expected, checked, reference, settings, stage
        )
        old = read_json(files.ordinary(original_root / f"checkpoint-{step}.json"))
        old_row, _ = original_stage(
            old, expected, checked, reference, original.payload["settings"], stage
        )
        require_same(old["checkpoint_row"], old_row)
        original_records = {row["tensor"]: row for row in old["records"]}
        for row in records:
            field = (
                "delta_from_previous"
                if row["displacement_kind"] == "saved_segment" and stage > 1
                else "delta_from_reference"
            )
            norm = original_records[row["tensor"]][field]["frobenius_norm"]
            if (row["basis_status"] == "zero") != (norm == 0) or abs(
                row["frobenius_norm"] - norm
            ) > REFERENCE_ATOL + REFERENCE_RTOL * norm:
                raise ValueError(
                    "Original and exact measurements disagree on the same displacement norm"
                )
        weight_norms = {row["tensor"]: row["weight"]["frobenius_norm"] for row in old["records"]}
        checkpoints.append(
            checkpoint_row(
                expected["recipe"],
                stage,
                checked["steps"],
                records,
                reference["hidden_shapes"],
                weight_norms,
            )
        )
        stored.append(value["checkpoint_row"])
        matrices.extend(records)
    require_same(sorted(path.name for path in root.iterdir()), sorted(names))
    return {
        "manifest": manifest,
        "checkpoint_rows": checkpoints,
        "stored_checkpoint_rows": stored,
        "matrix_rows": matrices,
    }
