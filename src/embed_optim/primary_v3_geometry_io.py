"""New-only, identity-bound geometry production and fresh numeric readback."""

from __future__ import annotations

import math
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .corrected_geometry_summary import _basis_seed
from .geometry import TensorStore
from .primary_contract import digest, file_identity, read_json, require_same, verify_file
from .primary_geometry_kernels import KINDS, audited_top_bases, checkpoint_row
from .primary_geometry_reductions import global_cosine, matrix_metrics
from .primary_v3_validation_io import write_new
from .primary_weight_entries import changed_entries


def verify_inputs(run, expected, checked, reference, reference_bound):
    if (
        checked.get("whole_run_artifacts_verified") is not True
        or checked.get("run_identity_sha256") != digest(expected)
        or [row["step"] for row in checked["checkpoints"]] != checked["steps"]
        or not checked["steps"]
    ):
        raise ValueError("Geometry needs complete identity-bound run admission")
    for checkpoint in checked["checkpoints"]:
        for row in checkpoint["files"]:
            verify_file(run / f"checkpoint-{checkpoint['step']}" / row["path"], row)
    for row in reference_bound["files"]:
        verify_file(reference / row["path"], row)


def descriptor(expected, checked, reference_bound, settings, scope, contract_sha):
    return {
        "scope": scope,
        "geometry_protocol_sha256": contract_sha,
        "run_identity_sha256": digest(expected),
        "recipe": expected["recipe"],
        "steps": checked["steps"],
        "checkpoint_files": [
            {"step": row["step"], "files": row["files"]} for row in checked["checkpoints"]
        ],
        "reference": reference_bound,
        "settings": settings,
        "scientific_completion": False,
    }


def compute_stage(run, recipe, steps, stage, reference, reference_bound, settings):
    if torch.get_num_threads() != settings["cpu_threads"]:
        raise ValueError("Use the declared CPU thread count for same-runtime numeric replay")
    step = steps[stage - 1]
    previous_step = steps[stage - 2] if stage > 1 else 0
    previous_path = run / f"checkpoint-{previous_step}" if previous_step else reference
    records, entries, health, bases = [], [], [], {}
    metric_settings = {
        key: settings[key] for key in ("sketch_rank", "oversample", "power_iterations")
    }
    with ExitStack() as stack:
        current = stack.enter_context(TensorStore(run / f"checkpoint-{step}"))
        initial = stack.enter_context(TensorStore(reference))
        previous = stack.enter_context(TensorStore(previous_path)) if previous_step else initial
        for store in (current, initial, previous):
            require_same(
                {name: list(store.shape(name)) for name in store.keys()},
                reference_bound["all_shapes"],
            )
        for tensor_index, name in enumerate(current.keys()):
            if name not in reference_bound["hidden_shapes"]:
                continue
            shape = reference_bound["hidden_shapes"][name]
            weight = current.tensor(name)
            prior, base = previous.tensor(name), initial.tensor(name)
            if not all(bool(torch.isfinite(value).all()) for value in (weight, prior, base)):
                raise ValueError("A saved geometry input is non-finite")
            segment, cumulative = weight.float() - prior.float(), weight.float() - base.float()
            metric_seed = settings["seed"] + step * 1009 + tensor_index
            record = {
                "schema_version": 1,
                "step": step,
                "tensor": name,
                "shape": shape,
                "parameters": math.prod(shape),
                "partition": "hidden",
                "weight": matrix_metrics(weight, **metric_settings, seed=metric_seed),
                "delta_from_reference": {
                    **matrix_metrics(cumulative, **metric_settings, seed=metric_seed + 2),
                    "cosine_with_weight": global_cosine(cumulative, weight),
                },
            }
            if previous_step:
                record["delta_from_previous"] = {
                    **matrix_metrics(segment, **metric_settings, seed=metric_seed + 1),
                    "cosine_with_weight": global_cosine(segment, weight),
                }
            records.append(record)
            entries.append(
                {
                    "tensor": name,
                    "shape": shape,
                    "parameters": math.prod(shape),
                    "saved_segment_nonzero_parameters": changed_entries(weight, prior),
                    "cumulative_nonzero_parameters": changed_entries(weight, base),
                }
            )
            for kind, delta in zip(KINDS, (segment, cumulative), strict=True):
                pair, checks = audited_top_bases(
                    delta,
                    rank=settings["subspace_rank"],
                    oversample=settings["oversample"],
                    power_iterations=settings["power_iterations"],
                    seed=_basis_seed(kind, stage, name, settings["seed"]),
                )
                health.append(
                    {
                        "run_id": recipe["run_id"],
                        "stage": stage,
                        "step": step,
                        "displacement_kind": kind,
                        "tensor": name,
                        "parameters": math.prod(shape),
                        **checks,
                    }
                )
                if pair is not None:
                    for side, tensor in zip(("left", "right"), pair, strict=True):
                        bases[f"{kind}|{name}|{side}"] = tensor.contiguous()
    return {
        "step": step,
        "previous_step": previous_step,
        "cumulative_anchor_step": 0,
        "records": records,
        "entry_records": entries,
        "subspace_health": health,
        "checkpoint_row": checkpoint_row(
            recipe, stage, steps, records, entries, reference_bound["hidden_shapes"]
        ),
    }, bases


def produce_run(
    run, expected, checked, reference, reference_bound, settings, output, *, scope, contract_sha
):
    run, reference, output = Path(run), Path(reference), Path(output)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("Use a new geometry run directory; never adopt partial legacy outputs")
    verify_inputs(run, expected, checked, reference, reference_bound)
    plan = descriptor(expected, checked, reference_bound, settings, scope, contract_sha)
    output.mkdir()
    outputs = []
    for stage, step in enumerate(checked["steps"], start=1):
        payload, bases = compute_stage(
            run, expected["recipe"], checked["steps"], stage, reference, reference_bound, settings
        )
        record_path = output / f"checkpoint-{step}.json"
        basis_path = output / f"checkpoint-{step}-bases.safetensors"
        write_new(record_path, payload)
        save_file(
            bases, basis_path, metadata={"scope": scope, "run_identity_sha256": digest(expected)}
        )
        outputs.append(
            {
                "step": step,
                "records": {"path": record_path.name, **file_identity(record_path)},
                "bases": {"path": basis_path.name, **file_identity(basis_path)},
            }
        )
        print(
            {
                "geometry_run": expected["recipe"]["run_id"],
                "step": step,
                "scientific_completion": False,
            },
            flush=True,
        )
    verify_inputs(run, expected, checked, reference, reference_bound)
    result = {"plan": plan, "outputs": outputs}
    write_new(output / "manifest.json", result)
    return result


def inspect_run(
    output, run, expected, checked, reference, reference_bound, settings, *, scope, contract_sha
):
    """Recompute every raw metric and basis from saved weights, not a self-declared hash."""
    output, run, reference = Path(output), Path(run), Path(reference)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("Require an ordinary retained geometry directory")
    verify_inputs(run, expected, checked, reference, reference_bound)
    manifest_path = output / "manifest.json"
    manifest_binding = file_identity(manifest_path)
    manifest = read_json(manifest_path)
    require_same(
        manifest["plan"],
        descriptor(expected, checked, reference_bound, settings, scope, contract_sha),
    )
    require_same([row["step"] for row in manifest["outputs"]], checked["steps"])
    names = {"manifest.json"}
    checkpoint_rows, health = [], []
    for stage, row in enumerate(manifest["outputs"], start=1):
        step = row["step"]
        expected_record, expected_bases = compute_stage(
            run, expected["recipe"], checked["steps"], stage, reference, reference_bound, settings
        )
        require_same(row["records"]["path"], f"checkpoint-{step}.json")
        require_same(row["bases"]["path"], f"checkpoint-{step}-bases.safetensors")
        for key in ("records", "bases"):
            names.add(row[key]["path"])
            verify_file(output / row[key]["path"], row[key])
        require_same(read_json(output / row["records"]["path"]), expected_record)
        with safe_open(output / row["bases"]["path"], framework="pt", device="cpu") as handle:
            require_same(sorted(handle.keys()), sorted(expected_bases))
            require_same(
                handle.metadata(), {"scope": scope, "run_identity_sha256": digest(expected)}
            )
            for name, value in expected_bases.items():
                actual = handle.get_tensor(name)
                if (
                    actual.dtype != value.dtype
                    or actual.shape != value.shape
                    or not torch.equal(actual, value)
                ):
                    raise ValueError("Stored subspace differs from fresh saved-weight computation")
        checkpoint_rows.append(expected_record["checkpoint_row"])
        health.extend(expected_record["subspace_health"])
    if {path.name for path in output.iterdir()} != names:
        raise ValueError("Geometry run directory contains undeclared outputs")
    verify_inputs(run, expected, checked, reference, reference_bound)
    verify_file(manifest_path, manifest_binding)
    for row in manifest["outputs"]:
        for key in ("records", "bases"):
            verify_file(output / row[key]["path"], row[key])
    return {
        "checkpoint_rows": checkpoint_rows,
        "subspace_health": health,
        "manifest": manifest_binding,
    }


def read_stage_bases(output, step, shapes, kind):
    """Use only after the complete numeric run reader accepted the bound files."""
    path = Path(output) / f"checkpoint-{step}-bases.safetensors"
    result = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for name in shapes:
            pair = [f"{kind}|{name}|{side}" for side in ("left", "right")]
            if any(key in keys for key in pair) and not all(key in keys for key in pair):
                raise ValueError("Subspace file has an incomplete left/right pair")
            result[name] = (
                tuple(handle.get_tensor(key) for key in pair)
                if all(key in keys for key in pair)
                else None
            )
    return result
