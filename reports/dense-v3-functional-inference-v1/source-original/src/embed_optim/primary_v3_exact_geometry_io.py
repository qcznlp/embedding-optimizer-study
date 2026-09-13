"""New-only complete-spectrum outputs, verified from saved weights on every read."""

from __future__ import annotations

import math
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .geometry import TensorStore
from .primary_contract import digest, file_identity, read_json, require_same, verify_file
from .primary_exact_geometry_kernels import KINDS, checkpoint_row, exact_matrix
from .primary_v3_geometry_io import descriptor, verify_inputs
from .primary_v3_validation_io import write_new


def compute_stage(run, recipe, steps, stage, reference, reference_bound, settings):
    if torch.get_num_threads() != settings["cpu_threads"]:
        raise ValueError("Use the declared exact-geometry CPU thread count")
    step = steps[stage - 1]
    previous_step = steps[stage - 2] if stage > 1 else 0
    previous_path = run / f"checkpoint-{previous_step}" if previous_step else reference
    records, arrays, weight_norms = [], {}, {}
    with ExitStack() as stack:
        current = stack.enter_context(TensorStore(run / f"checkpoint-{step}"))
        initial = stack.enter_context(TensorStore(reference))
        previous = stack.enter_context(TensorStore(previous_path)) if previous_step else initial
        for store in (current, initial, previous):
            require_same(
                {name: list(store.shape(name)) for name in store.keys()},
                reference_bound["all_shapes"],
            )
        for name, shape in sorted(reference_bound["hidden_shapes"].items()):
            weight, before, base = (
                current.tensor(name).float(),
                previous.tensor(name).float(),
                initial.tensor(name).float(),
            )
            if not all(bool(torch.isfinite(value).all()) for value in (weight, before, base)):
                raise ValueError("An exact geometry input is non-finite")
            weight_norms[name] = float(torch.linalg.vector_norm(weight, dtype=torch.float64))
            segment, cumulative = weight - before, weight - base
            # Exact value equality permits reuse, not a norm/threshold cache hit.
            same = torch.equal(segment, cumulative)
            first_result = None
            for kind, delta in zip(KINDS, (segment, cumulative), strict=True):
                result = (
                    first_result
                    if kind == "cumulative" and same
                    else exact_matrix(delta.numpy().astype(np.float64), settings)
                )
                if kind == "saved_segment":
                    first_result = result
                metrics, singular, bases = result
                records.append(
                    {
                        "run_id": recipe["run_id"],
                        "optimizer": recipe["optimizer"]["name"],
                        "learning_rate": recipe["optimizer"]["lr"],
                        "stage": stage,
                        "step": step,
                        "displacement_kind": kind,
                        "tensor": name,
                        "shape": shape,
                        "parameters": math.prod(shape),
                        **metrics,
                    }
                )
                arrays[f"{kind}|{name}|singular"] = torch.from_numpy(singular.copy())
                if bases is not None:
                    for side, value in zip(("left", "right"), bases, strict=True):
                        arrays[f"{kind}|{name}|{side}"] = torch.from_numpy(value.copy())
    return {
        "step": step,
        "previous_step": previous_step,
        "cumulative_anchor_step": 0,
        "records": records,
        "checkpoint_row": checkpoint_row(
            recipe, stage, steps, records, reference_bound["hidden_shapes"], weight_norms
        ),
    }, arrays


def produce_run(
    run, expected, checked, reference, reference_bound, settings, output, *, scope, contract_sha
):
    run, reference, output = Path(run), Path(reference), Path(output)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("Require a new exact geometry directory; never adopt partial outputs")
    verify_inputs(run, expected, checked, reference, reference_bound)
    plan = descriptor(expected, checked, reference_bound, settings, scope, contract_sha)
    output.mkdir()
    outputs = []
    for stage, step in enumerate(checked["steps"], start=1):
        record, arrays = compute_stage(
            run, expected["recipe"], checked["steps"], stage, reference, reference_bound, settings
        )
        record_path, array_path = (
            output / f"checkpoint-{step}.json",
            output / f"checkpoint-{step}-exact.safetensors",
        )
        write_new(record_path, record)
        save_file(
            arrays, array_path, metadata={"scope": scope, "run_identity_sha256": digest(expected)}
        )
        outputs.append(
            {
                "step": step,
                "records": {"path": record_path.name, **file_identity(record_path)},
                "arrays": {"path": array_path.name, **file_identity(array_path)},
            }
        )
        print(
            {
                "exact_geometry_run": expected["recipe"]["run_id"],
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
    output, run, reference = Path(output), Path(run), Path(reference)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("Require an ordinary exact geometry output directory")
    verify_inputs(run, expected, checked, reference, reference_bound)
    manifest_path = output / "manifest.json"
    manifest_binding = file_identity(manifest_path)
    manifest = read_json(manifest_path)
    require_same(
        manifest["plan"],
        descriptor(expected, checked, reference_bound, settings, scope, contract_sha),
    )
    require_same([row["step"] for row in manifest["outputs"]], checked["steps"])
    names, checkpoint_rows, matrix_rows = {"manifest.json"}, [], []
    for stage, row in enumerate(manifest["outputs"], start=1):
        step = row["step"]
        expected_record, expected_arrays = compute_stage(
            run, expected["recipe"], checked["steps"], stage, reference, reference_bound, settings
        )
        require_same(row["records"]["path"], f"checkpoint-{step}.json")
        require_same(row["arrays"]["path"], f"checkpoint-{step}-exact.safetensors")
        for key in ("records", "arrays"):
            names.add(row[key]["path"])
            verify_file(output / row[key]["path"], row[key])
        require_same(read_json(output / row["records"]["path"]), expected_record)
        with safe_open(output / row["arrays"]["path"], framework="pt", device="cpu") as handle:
            require_same(sorted(handle.keys()), sorted(expected_arrays))
            require_same(
                handle.metadata(), {"scope": scope, "run_identity_sha256": digest(expected)}
            )
            for name, value in expected_arrays.items():
                actual = handle.get_tensor(name)
                if (
                    actual.dtype != torch.float64
                    or actual.shape != value.shape
                    or not torch.equal(actual, value)
                ):
                    raise ValueError(
                        "Exact spectrum/basis differs from fresh checkpoint computation"
                    )
        checkpoint_rows.append(expected_record["checkpoint_row"])
        matrix_rows.extend(expected_record["records"])
    if {path.name for path in output.iterdir()} != names:
        raise ValueError("Exact geometry run contains undeclared artifacts")
    verify_inputs(run, expected, checked, reference, reference_bound)
    verify_file(manifest_path, manifest_binding)
    for row in manifest["outputs"]:
        for key in ("records", "arrays"):
            verify_file(output / row[key]["path"], row[key])
    return {
        "checkpoint_rows": checkpoint_rows,
        "matrix_rows": matrix_rows,
        "manifest": manifest_binding,
    }


def read_stage_views(output, step, shapes, kind):
    """Only after full numerical admission; later verify that its manifest/files remain unchanged."""
    output = Path(output)
    records = [
        row
        for row in read_json(output / f"checkpoint-{step}.json")["records"]
        if row["displacement_kind"] == kind
    ]
    require_same([row["tensor"] for row in records], sorted(shapes))
    result = {}
    with safe_open(
        output / f"checkpoint-{step}-exact.safetensors", framework="pt", device="cpu"
    ) as handle:
        keys = set(handle.keys())
        for row in records:
            names = [f"{kind}|{row['tensor']}|{side}" for side in ("left", "right")]
            resolved = row["basis_status"] == "resolved"
            if any((name in keys) != resolved for name in names):
                raise ValueError("Exact basis inventory disagrees with its defined status")
            result[row["tensor"]] = {
                "status": row["basis_status"],
                "bases": tuple(handle.get_tensor(name).numpy() for name in names)
                if resolved
                else None,
            }
    return result
