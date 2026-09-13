"""Typed original geometry primitives, not fresh measurement of checkpoint weights.

Previously authenticated records/bases remain measurements from their authoring
run. This module checks complete schemas and consistent identities before using
them to reconstruct checkpoint/pair aggregates. It never loads a model tensor.
"""

import math

import torch
from safetensors import safe_open

from . import reconstruction_files as files
from .primary_completion import integer, number
from .primary_contract import digest, read_json, require_same, verify_file
from .primary_geometry_kernels import KINDS, checkpoint_row
from .primary_v3_geometry import SCOPE
from .primary_v3_geometry_io import descriptor

SPECTRAL_FIELDS = {
    "algorithm",
    "rank",
    "spectral_norm",
    "approx_stable_rank",
    "sketched_nuclear_norm",
    "sketched_entropy_effective_rank",
    "sketched_condition_number",
    "captured_frobenius_energy",
}
MATRIX_FIELDS = {
    "frobenius_norm",
    "row_norms",
    "column_norms",
    "top_1pct_row_energy",
    "top_10pct_row_energy",
    *SPECTRAL_FIELDS,
}
HEALTH_FIELDS = {
    "zero_displacement",
    "retained_rank",
    "projected_numerical_rank",
    "retained_signal_supported",
    "numerical_threshold",
    "smallest_retained_singular_value",
    "next_singular_value",
    "relative_boundary_gap",
    "retained_frobenius_energy",
}


def nonnegative(value):
    value = number(value)
    if value < 0:
        raise ValueError("Negative archived geometry measurement")
    return value


def metric(value, shape, settings, *, displacement=False):
    fields = MATRIX_FIELDS | ({"cosine_with_weight"} if displacement else set())
    if set(value) != fields:
        raise ValueError("Incomplete archived matrix-metric schema")
    norm = nonnegative(value["frobenius_norm"])
    for key in ("row_norms", "column_norms"):
        if set(value[key]) != {"mean", "cv", "gini", "max_to_median"}:
            raise ValueError("Incomplete archived norm distribution")
        for scalar in value[key].values():
            nonnegative(scalar)
    for key in (
        "top_1pct_row_energy",
        "top_10pct_row_energy",
        "spectral_norm",
        "approx_stable_rank",
        "sketched_nuclear_norm",
        "sketched_entropy_effective_rank",
        "captured_frobenius_energy",
    ):
        nonnegative(value[key])
    if value["sketched_condition_number"] is not None:
        nonnegative(value["sketched_condition_number"])
    if displacement and value["cosine_with_weight"] is not None:
        number(value["cosine_with_weight"])
    # Do not impose new FP32-derived rank/energy rounding tolerances or reinterpret the sketch.
    if norm == 0:
        wanted = {
            "algorithm": "zero",
            "rank": 0,
            "sketched_condition_number": None,
            **{
                key: 0.0
                for key in SPECTRAL_FIELDS - {"algorithm", "rank", "sketched_condition_number"}
            },
        }
        require_same({key: value[key] for key in SPECTRAL_FIELDS}, wanted)
        if any(
            scalar != 0 for key in ("row_norms", "column_norms") for scalar in value[key].values()
        ) or any(value[key] != 0 for key in ("top_1pct_row_energy", "top_10pct_row_energy")):
            raise ValueError(
                "An exactly zero matrix must have zero norm distributions and row energy"
            )
        if displacement and value["cosine_with_weight"] is not None:
            raise ValueError("A zero displacement has no weight cosine")
    else:
        rank = min(settings["sketch_rank"], min(shape))
        if integer(value["rank"]) != rank:
            raise ValueError("Archived sketch rank differs from frozen settings")
        require_same(value["algorithm"], "exact" if rank == min(shape) else "randomized")
        if value["spectral_norm"] <= 0:
            raise ValueError("Nonzero archived metric lacks a positive spectral norm")
    return norm


def health(value, *, recipe, stage, steps, name, shape, kind, nonzero, settings):
    metadata = {
        "run_id": recipe["run_id"],
        "stage": stage,
        "step": steps[stage - 1],
        "displacement_kind": kind,
        "tensor": name,
        "parameters": math.prod(shape),
    }
    if set(value) != set(metadata) | HEALTH_FIELDS:
        raise ValueError("Incomplete archived subspace-health schema")
    require_same({key: value[key] for key in metadata}, metadata)
    if not nonzero:
        require_same(
            {key: value[key] for key in HEALTH_FIELDS},
            {
                "zero_displacement": True,
                "retained_rank": 0,
                "projected_numerical_rank": 0,
                "retained_signal_supported": None,
                "numerical_threshold": 0.0,
                "smallest_retained_singular_value": None,
                "next_singular_value": None,
                "relative_boundary_gap": None,
                "retained_frobenius_energy": 0.0,
            },
        )
        return 0
    rank = min(settings["subspace_rank"], min(shape))
    width = min(min(shape), rank + settings["oversample"])
    if (
        value["zero_displacement"] is not False
        or value["retained_signal_supported"] is not True
        or integer(value["retained_rank"]) != rank
        or not rank <= integer(value["projected_numerical_rank"]) <= width
    ):
        raise ValueError("Unsupported recorded nonzero subspace rank")
    threshold = nonnegative(value["numerical_threshold"])
    smallest = number(value["smallest_retained_singular_value"], positive=True)
    energy = number(value["retained_frobenius_energy"], positive=True)
    if smallest <= threshold or energy > 1:
        raise ValueError("Invalid recorded signal support or retained energy")
    next_value = value["next_singular_value"]
    if width == rank:
        require_same([next_value, value["relative_boundary_gap"]], [None, None])
    else:
        next_value = nonnegative(next_value)
        if next_value > smallest:
            raise ValueError("Recorded boundary singular values are not ordered")
        require_same(value["relative_boundary_gap"], (smallest - next_value) / smallest)
    return rank


def stage_records(value, expected, checked, reference, settings, stage):
    """Rebuild the derived row while retaining raw measurements as authenticated inputs."""
    required = {
        "step",
        "previous_step",
        "cumulative_anchor_step",
        "records",
        "entry_records",
        "subspace_health",
        "checkpoint_row",
    }
    if set(value) != required:
        raise ValueError("Incomplete archived geometry stage")
    steps, recipe = checked["steps"], expected["recipe"]
    step, previous = steps[stage - 1], steps[stage - 2] if stage > 1 else 0
    require_same(
        [value["step"], value["previous_step"], value["cumulative_anchor_step"]],
        [step, previous, 0],
    )
    shapes = reference["hidden_shapes"]
    names = sorted(shapes)
    records, entries, recorded_health = (
        value[k] for k in ("records", "entry_records", "subspace_health")
    )
    if any(not isinstance(rows, list) for rows in (records, entries, recorded_health)):
        raise ValueError("Archived geometry populations must be lists")
    require_same([r["tensor"] for r in records], names)
    require_same([r["tensor"] for r in entries], names)
    require_same(
        [[r["tensor"], r["displacement_kind"]] for r in recorded_health],
        [[name, kind] for name in names for kind in KINDS],
    )
    ranks = {}
    for index, (name, record, entry) in enumerate(zip(names, records, entries, strict=True)):
        shape = shapes[name]
        metadata = {
            "schema_version": 1,
            "step": step,
            "tensor": name,
            "shape": shape,
            "parameters": math.prod(shape),
            "partition": "hidden",
        }
        metric_names = {"weight", "delta_from_reference"} | (
            {"delta_from_previous"} if stage > 1 else set()
        )
        if set(record) != set(metadata) | metric_names:
            raise ValueError("Incomplete archived raw matrix record")
        require_same({key: record[key] for key in metadata}, metadata)
        entry_metadata = {key: metadata[key] for key in ("tensor", "shape", "parameters")}
        if set(entry) != set(entry_metadata) | {
            "saved_segment_nonzero_parameters",
            "cumulative_nonzero_parameters",
        }:
            raise ValueError("Incomplete archived exact-entry record")
        require_same({key: entry[key] for key in entry_metadata}, entry_metadata)
        norms = {
            key: metric(record[key], shape, settings, displacement=key != "weight")
            for key in metric_names
        }
        for offset, kind in enumerate(KINDS):
            count = integer(entry[kind + "_nonzero_parameters"])
            if count > math.prod(shape):
                raise ValueError("Changed entries exceed their actual hidden-matrix denominator")
            column = (
                "delta_from_previous"
                if kind == "saved_segment" and stage > 1
                else "delta_from_reference"
            )
            if (norms[column] > 0) != (count > 0):
                raise ValueError("Recorded displacement norm and exact-entry presence disagree")
            ranks[kind, name] = health(
                recorded_health[2 * index + offset],
                recipe=recipe,
                stage=stage,
                steps=steps,
                name=name,
                shape=shape,
                kind=kind,
                nonzero=count > 0,
                settings=settings,
            )
        if stage == 1:
            require_same(
                entry["saved_segment_nonzero_parameters"], entry["cumulative_nonzero_parameters"]
            )
    fresh = checkpoint_row(recipe, stage, steps, records, entries, shapes)
    return fresh, ranks


def stage_bases(path, expected, reference, ranks):
    shapes = reference["hidden_shapes"]
    names = {
        f"{kind}|{name}|{side}"
        for (kind, name), rank in ranks.items()
        if rank
        for side in ("left", "right")
    }
    with safe_open(files.ordinary(path), framework="pt", device="cpu") as handle:
        require_same(sorted(handle.keys()), sorted(names))
        require_same(handle.metadata(), {"scope": SCOPE, "run_identity_sha256": digest(expected)})
        for (kind, name), rank in ranks.items():
            if not rank:
                continue
            for axis, side in enumerate(("left", "right")):
                tensor = handle.get_tensor(f"{kind}|{name}|{side}")
                if (
                    tensor.dtype != torch.float32
                    or list(tensor.shape) != [shapes[name][axis], rank]
                    or not bool(torch.isfinite(tensor).all())
                ):
                    raise ValueError("Archived basis dtype, native shape or finiteness differs")


def inspect_run_primitives(root, expected, checked, reference, contract):
    if torch.get_num_threads() != contract.payload["settings"]["cpu_threads"]:
        raise ValueError("Use the frozen geometry CPU thread count")
    manifest_path = files.ordinary(root / "manifest.json")
    manifest = read_json(manifest_path)
    if set(manifest) != {"plan", "outputs"}:
        raise ValueError("Incomplete archived geometry run manifest")
    settings = contract.payload["settings"]
    require_same(
        manifest["plan"], descriptor(expected, checked, reference, settings, SCOPE, contract.sha256)
    )
    require_same([row["step"] for row in manifest["outputs"]], checked["steps"])
    names, fresh_rows, health_rows, stored_rows = {"manifest.json"}, [], [], []
    for stage, row in enumerate(manifest["outputs"], 1):
        step = checked["steps"][stage - 1]
        if set(row) != {"step", "records", "bases"}:
            raise ValueError("Incomplete archived stage bindings")
        for key, name in (
            ("records", f"checkpoint-{step}.json"),
            ("bases", f"checkpoint-{step}-bases.safetensors"),
        ):
            if set(row[key]) != {"path", "bytes", "sha256"}:
                raise ValueError("Incomplete archived geometry file identity")
            require_same(row[key]["path"], name)
            files.identity(row[key])
            names.add(name)
            verify_file(files.ordinary(root / name), row[key])
        value = read_json(root / row["records"]["path"])
        fresh, ranks = stage_records(value, expected, checked, reference, settings, stage)
        stage_bases(root / row["bases"]["path"], expected, reference, ranks)
        fresh_rows.append(fresh)
        stored_rows.append(value["checkpoint_row"])
        health_rows.extend(value["subspace_health"])
    require_same(files.inventory(root), sorted(names))
    return {
        "checkpoint_rows": fresh_rows,
        "subspace_health": health_rows,
        "stored_checkpoint_rows": stored_rows,
        "manifest": manifest,
    }
