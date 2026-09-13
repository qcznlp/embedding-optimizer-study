"""Complete native-shape geometry fixture; every weight measurement is simulated.

The source contracts, reference shapes and probe identities are authentic. The
sixty checkpoint admissions, entry counts and spectral measurements are not model
results. Native-axis rank-16 coordinate bases have analytically known overlaps;
all-zero stages and heterogeneous later zeros exercise undefined denominators.
"""

import copy
import math
from pathlib import Path

import torch
from safetensors.torch import save_file

from embed_optim import reconstruction_files as files
from embed_optim.corrected_geometry_summary import OPTIMIZER_ORDER
from embed_optim.primary_contract import digest, file_identity, require_same
from embed_optim.primary_geometry_kernels import KINDS, checkpoint_row
from embed_optim.primary_v3_contract import SCOPE as PRIMARY_SCOPE
from embed_optim.primary_v3_dimension_contract import planned_states
from embed_optim.primary_v3_dimension_inference import SCOPE as INFERENCE_SCOPE
from embed_optim.primary_v3_geometry import SCOPE, TABLE_COUNTS, tables_from_verified_runs
from embed_optim.primary_v3_geometry_io import descriptor
from embed_optim.primary_v3_outcomes import csv_bytes
from embed_optim.primary_v3_reconstruction_authoring import ACCEPTANCE, BOUNDARY, require_parent
from embed_optim.primary_v3_reconstruction_inputs import vector_admission
from embed_optim.primary_v3_reconstruction_sources import source_selection
from embed_optim.primary_v3_validation_io import write_new
from embed_optim.runtime import runtime_snapshot
from scripts.outcome_reconstruction_fixture import upgraded_runs
from scripts.vector_reconstruction_fixture import admitted_fixture

GEOMETRY_BOUNDARY = "Complete geometry only; retrieval bridge, useful dimensions, factorial, publication and formal release remain separate acceptance gates."


def complete_admission(root, contract):
    """Reuse the unchanged sealed-metadata fixture, not any checkpoint payload."""
    primary = contract.primary
    admitted, _ = admitted_fixture(contract)
    for index, run in enumerate(admitted["complete_runs"].values()):
        run.update(
            {
                "scope": PRIMARY_SCOPE,
                "protocol_sha256": primary.sha256,
                "run_id": primary.inputs["runs"][index]["run_id"],
                "dataset_fingerprint": primary.inputs["data_linkage_audit"][
                    "training_view_fingerprint"
                ],
                "accepted_timing": {
                    "segments": 5,
                    "total_wall_time_seconds_max_rank": 5000.0 + index,
                },
                "system_metrics": {
                    "world_size": 4,
                    "gpu_name": "EXPLICIT SYNTHETIC geometry fixture hardware",
                    "trainer": {"train_samples_per_second": 100.0, "train_steps_per_second": 0.8},
                    "peak_allocated_bytes_max_rank": 2**30,
                    "peak_reserved_bytes_max_rank": 2**31,
                },
            }
        )
    write_new(root / "synthetic_complete_runs.json", admitted["complete_runs"])
    runs = upgraded_runs(primary, root)
    common = {
        key: value
        for key, value in admitted.items()
        if key not in {"states", "reference", "complete_runs"}
    }
    states = []
    for state in planned_states(primary):
        if state["cell"] == "pretrained":
            model = {"kind": "immutable_pretrained", "reference": admitted["reference"]}
        else:
            run = runs[state["meta"]["run_id"]]
            model = {
                "kind": "complete_primary_checkpoint",
                "run_identity_sha256": run["run_identity_sha256"],
                "complete_run_sha256": digest(run),
                "checkpoint": run["checkpoints"][state["stage"] - 1],
            }
        states.append({**common, "state": state, "model": model})
    admitted.update(complete_runs=runs, states=states)
    vector_admission(contract.dimensions, admitted)
    return admitted


def nonzero(run_index, stage, tensor_index, kind):
    # Stage 1 is entirely zero for every run; later matrices have varied presence.
    return stage > 1 and (run_index + 2 * stage + 3 * tensor_index + KINDS.index(kind)) % 7 != 0


def synthetic_metric(shape, value, *, displacement=False):
    """Schema-complete hypothetical measurements, NOT a fresh matrix/SVD calculation."""
    present = value > 0
    distribution = {
        "mean": value / 8,
        "cv": 0.5 if present else 0.0,
        "gini": 0.25 if present else 0.0,
        "max_to_median": 2.0 if present else 0.0,
    }
    row = {
        "frobenius_norm": value,
        "row_norms": distribution,
        "column_norms": copy.deepcopy(distribution),
        "top_1pct_row_energy": 0.125 if present else 0.0,
        "top_10pct_row_energy": 0.5 if present else 0.0,
        "algorithm": "randomized" if present else "zero",
        "rank": min(64, min(shape)) if present else 0,
        "spectral_norm": value / 8,
        "approx_stable_rank": 64.0 if present else 0.0,
        "sketched_nuclear_norm": value * 8,
        "sketched_entropy_effective_rank": 64.0 if present else 0.0,
        "sketched_condition_number": 1.0 if present else None,
        "captured_frobenius_energy": 1.0 if present else 0.0,
    }
    if displacement:
        row["cosine_with_weight"] = 0.5 if present else None
    return row


def synthetic_health(recipe, stage, steps, name, shape, kind, present):
    return {
        "run_id": recipe["run_id"],
        "stage": stage,
        "step": steps[stage - 1],
        "displacement_kind": kind,
        "tensor": name,
        "parameters": math.prod(shape),
        "zero_displacement": not present,
        "retained_rank": 16 if present else 0,
        "projected_numerical_rank": 24 if present else 0,
        "retained_signal_supported": True if present else None,
        "numerical_threshold": 0.0001 if present else 0.0,
        "smallest_retained_singular_value": 2.0 if present else None,
        "next_singular_value": 1.0 if present else None,
        "relative_boundary_gap": 0.5 if present else None,
        "retained_frobenius_energy": 0.75 if present else 0.0,
    }


def coordinate_indices(dimension, run_index, stage, tensor_index, kind, axis):
    shift = (5 * run_index + 3 * stage + 7 * tensor_index + 11 * KINDS.index(kind) + 13 * axis) % (
        dimension - 16 + 1
    )
    return list(range(shift, shift + 16))


def synthetic_stage(recipe, run_index, steps, stage, reference):
    shapes = reference["hidden_shapes"]
    step = steps[stage - 1]
    records, entries, health, bases = [], [], [], {}
    for tensor_index, name in enumerate(sorted(shapes)):
        shape = shapes[name]
        present = {kind: nonzero(run_index, stage, tensor_index, kind) for kind in KINDS}
        magnitude = (run_index + 1) * (tensor_index + 1) * stage / 256
        entry = {"tensor": name, "shape": shape, "parameters": math.prod(shape)}
        for kind in KINDS:
            count = 16 + ((tensor_index + 1) * (stage + 1) * (run_index + 1)) % 128
            entry[kind + "_nonzero_parameters"] = count if present[kind] else 0
            health.append(synthetic_health(recipe, stage, steps, name, shape, kind, present[kind]))
            if present[kind]:
                for axis, side in enumerate(("left", "right")):
                    basis = torch.zeros((shape[axis], 16), dtype=torch.float32)
                    indices = coordinate_indices(
                        shape[axis], run_index, stage, tensor_index, kind, axis
                    )
                    basis[indices, range(16)] = 1.0
                    bases[f"{kind}|{name}|{side}"] = basis
        record = {
            "schema_version": 1,
            "step": step,
            "tensor": name,
            "shape": shape,
            "parameters": math.prod(shape),
            "partition": "hidden",
            "weight": synthetic_metric(shape, 8 + magnitude),
            "delta_from_reference": synthetic_metric(
                shape, magnitude if present["cumulative"] else 0.0, displacement=True
            ),
        }
        if stage > 1:
            record["delta_from_previous"] = synthetic_metric(
                shape, magnitude / 2 if present["saved_segment"] else 0.0, displacement=True
            )
        records.append(record)
        entries.append(entry)
    return {
        "step": step,
        "previous_step": steps[stage - 2] if stage > 1 else 0,
        "cumulative_anchor_step": 0,
        "records": records,
        "entry_records": entries,
        "subspace_health": health,
        "checkpoint_row": checkpoint_row(recipe, stage, steps, records, entries, shapes),
    }, bases


def make_geometry(root, contract, admitted, progress=None):
    root.mkdir()
    (root / "runs").mkdir()
    primary, geometry = contract.primary, contract.bridge.geometry
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
            data, bases = synthetic_stage(
                expected["recipe"], run_index, checked["steps"], stage, reference
            )
            record_path, basis_path = (
                path / f"checkpoint-{step}.json",
                path / f"checkpoint-{step}-bases.safetensors",
            )
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
            progress({"synthetic_geometry_runs": run_index + 1, "checkpoints": 5 * (run_index + 1)})
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
        "boundary": GEOMETRY_BOUNDARY,
    }
    write_new(root / "summary.json", summary)
    return summary


def make(root, contract, *, progress=None):
    root = Path(root)
    root.mkdir()
    torch.set_num_threads(contract.bridge.geometry.payload["settings"]["cpu_threads"])
    producer = root / "producer"
    producer.mkdir()
    admitted = complete_admission(root, contract)
    summary = make_geometry(producer / "geometry", contract, admitted, progress)
    (producer / "vectors").mkdir()
    write_new(producer / "vectors/admission.json", admitted)
    evidence = {
        "upstream_primary_admission_simulated": True,
        "raw_weight_spectra_and_entry_counts_simulated": True,
        "vector_outcome_and_inference_payloads_simulated": True,
        "native_axis_coordinate_bases_synthetic": True,
        "producer_path_must_not_be_read": str(producer),
        "original_bridge_evidence": {
            "geometry_reader": summary,
            "source_bindings": [
                {
                    "path": str(producer / "geometry" / name),
                    **file_identity(producer / "geometry" / name),
                }
                for name in files.inventory(producer / "geometry")
            ],
        },
        "scientific_completion": False,
    }
    plan = {
        "scope": INFERENCE_SCOPE,
        "primary_protocol_sha256": contract.primary.sha256,
        "inference_protocol_sha256": contract.sha256,
        "evidence_sha256": digest(evidence),
        "scientific_completion": False,
        "manuscript_installation_authorized": False,
    }
    (producer / "inference").mkdir()
    write_new(producer / "inference/evidence.json", evidence)
    write_new(producer / "inference/manifest.json", {"plan": plan})
    selected = source_selection(contract, {})
    parent = require_parent(contract)
    files.select(selected, "source", ACCEPTANCE[0], parent, file_identity(parent))
    for role in ("geometry", "vectors", "inference"):
        for name in files.inventory(producer / role):
            path = producer / role / name
            files.select(selected, role, name, path, file_identity(path))
    metadata = {
        "scope": "dense_primary_v3_checkpoint_backed_reconstruction_selection",
        "primary_protocol_sha256": contract.primary.sha256,
        "inference_protocol_sha256": contract.sha256,
        "inference_acceptance_sha256": ACCEPTANCE[1],
        "dimension_protocol_sha256": contract.dimensions.sha256,
        "trusted_vector_manifest_sha256": "0" * 64,
        "authoring_plan": plan,
        "authoring_evidence_sha256": digest(evidence),
        "original_location_fields_preserved": True,
        "boundary": BOUNDARY,
        "authoring_runtime": runtime_snapshot(
            [
                "torch",
                "numpy",
                "scipy",
                "sympy",
                "sentence-transformers",
                "transformers",
                "accelerate",
                "mteb",
            ]
        ),
        "full_raw_vector_reconstruction_repeated_by_transport": False,
        "manuscript_installed": False,
        "scientific_completion": False,
    }
    result = files.write(root / "archive", selected, metadata)
    producer.rename(root / "preserved-producer-unavailable-at-original-location")
    return root / "archive", result["manifest_sha256"]
