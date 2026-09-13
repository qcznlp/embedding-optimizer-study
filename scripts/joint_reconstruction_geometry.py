"""Hypothetical native-shape geometry with valid positive bridge log inputs.

The standalone geometry parent's all-zero first stage intentionally fails the
original bridge's positive-log/finite-overlap domain. This separate fixture has
nonzero first-stage displacements. No production domain rule is relaxed.
"""

import torch
from safetensors.torch import save_file

from embed_optim.corrected_geometry_summary import OPTIMIZER_ORDER
from embed_optim.primary_contract import digest, file_identity, require_same
from embed_optim.primary_geometry_kernels import KINDS, checkpoint_row
from embed_optim.primary_v3_geometry import SCOPE, TABLE_COUNTS, tables_from_verified_runs
from embed_optim.primary_v3_geometry_io import descriptor
from embed_optim.primary_v3_geometry_reconstruction import BOUNDARY
from embed_optim.primary_v3_outcomes import csv_bytes
from embed_optim.primary_v3_validation_io import write_new
from scripts.geometry_reconstruction_fixture import (
    coordinate_indices,
    synthetic_health,
    synthetic_metric,
    synthetic_stage,
)


def joint_stage(recipe, run_index, steps, stage, reference):
    data, bases = synthetic_stage(recipe, run_index, steps, stage, reference)
    if stage != 1:
        return data, bases
    # The first saved segment and cumulative displacement have the same anchor.
    health = []
    for index, (record, entry) in enumerate(
        zip(data["records"], data["entry_records"], strict=True)
    ):
        name, shape = record["tensor"], record["shape"]
        magnitude = (run_index + 1) * (index + 1) / 256
        record["delta_from_reference"] = synthetic_metric(shape, magnitude, displacement=True)
        for kind in KINDS:
            entry[kind + "_nonzero_parameters"] = 16 + (index + 1) * 2 * (run_index + 1) % 128
            health.append(synthetic_health(recipe, stage, steps, name, shape, kind, True))
            for axis, side in enumerate(("left", "right")):
                basis = torch.zeros((shape[axis], 16), dtype=torch.float32)
                indices = coordinate_indices(
                    shape[axis], run_index, stage, index, "cumulative", axis
                )
                basis[indices, range(16)] = 1.0
                bases[f"{kind}|{name}|{side}"] = basis
    data["subspace_health"] = health
    data["checkpoint_row"] = checkpoint_row(
        recipe, stage, steps, data["records"], data["entry_records"], reference["hidden_shapes"]
    )
    return data, bases


def make(root, contract, admitted, *, progress=None):
    root.mkdir()
    (root / "runs").mkdir()
    primary, geometry = contract.primary, contract.bridge.geometry
    torch.set_num_threads(geometry.payload["settings"]["cpu_threads"])
    reference = admitted["reference"]
    order = sorted(
        admitted["complete_runs"],
        key=lambda run_id: (
            OPTIMIZER_ORDER[primary.expected_identity(run_id)["recipe"]["optimizer"]["name"]],
            primary.expected_identity(run_id)["recipe"]["optimizer"]["lr"],
            run_id,
        ),
    )
    checked_rows, inputs, raw_bindings = [], [], []
    for run_index, run_id in enumerate(order):
        path = root / "runs" / run_id
        path.mkdir()
        checked = admitted["complete_runs"][run_id]
        expected = primary.expected_identity(run_id)
        outputs, checkpoint_rows, health = [], [], []
        for stage, step in enumerate(checked["steps"], 1):
            data, bases = joint_stage(
                expected["recipe"], run_index, checked["steps"], stage, reference
            )
            record_path = path / f"checkpoint-{step}.json"
            basis_path = path / f"checkpoint-{step}-bases.safetensors"
            write_new(record_path, data)
            save_file(
                bases,
                basis_path,
                metadata={"scope": SCOPE, "run_identity_sha256": digest(expected)},
            )
            outputs.append(
                {
                    "step": step,
                    "records": {"path": record_path.name, **file_identity(record_path)},
                    "bases": {"path": basis_path.name, **file_identity(basis_path)},
                }
            )
            checkpoint_rows.append(data["checkpoint_row"])
            health.extend(data["subspace_health"])
            for file in (record_path, basis_path):
                raw_bindings.append(
                    {"path": file.relative_to(root).as_posix(), **file_identity(file)}
                )
        write_new(
            path / "manifest.json",
            {
                "plan": descriptor(
                    expected,
                    checked,
                    reference,
                    geometry.payload["settings"],
                    SCOPE,
                    geometry.sha256,
                ),
                "outputs": outputs,
            },
        )
        raw_bindings.append(
            {
                "path": (path / "manifest.json").relative_to(root).as_posix(),
                **file_identity(path / "manifest.json"),
            }
        )
        checked_rows.append({"checkpoint_rows": checkpoint_rows, "subspace_health": health})
        inputs.append((path, expected, checked))
        if progress:
            progress({"joint_fixture_geometry_runs": run_index + 1})
    tables = tables_from_verified_runs(inputs, root / "runs", reference, checked_rows)
    require_same({name: len(rows) for name, rows in tables.items()}, TABLE_COUNTS)
    for name, rows in tables.items():
        with (root / (name + ".csv")).open("xb") as stream:
            stream.write(csv_bytes(rows))
    summary = {
        "scope": SCOPE,
        "primary_protocol_sha256": primary.sha256,
        "geometry_protocol_sha256": geometry.sha256,
        "raw_bindings": raw_bindings,
        "table_counts": TABLE_COUNTS,
        "scientific_completion": False,
        "boundary": BOUNDARY,
    }
    write_new(root / "summary.json", summary)
    return summary, tables
