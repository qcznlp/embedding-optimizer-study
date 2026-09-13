"""Full-native-shape synthetic exact branch; never call actual primary authoring."""

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from threadpoolctl import threadpool_limits

from embed_optim import primary_v3_exact_bridge as exact
from embed_optim import primary_v3_exact_reconstruction as reconstruction
from embed_optim import primary_v3_exact_reconstruction_archive as archive
from embed_optim import primary_v3_publication as publication
from embed_optim import primary_v3_publication_archive as parent
from embed_optim import reconstruction_files as files
from embed_optim.corrected_geometry_summary import OPTIMIZER_ORDER
from embed_optim.primary_contract import digest, file_identity, read_json, require_same
from embed_optim.primary_exact_geometry_kernels import KINDS, checkpoint_row
from embed_optim.primary_v3_exact_geometry_primitives import spectrum_metrics
from embed_optim.primary_v3_geometry_io import descriptor
from embed_optim.primary_v3_outcomes import csv_bytes
from embed_optim.primary_v3_reconstruction_sources import recheck_selection, source_selection
from embed_optim.primary_v3_validation_io import write_new

ORIGINAL = Path("/tmp/dense-v3-publication-cold.PvOmLl/relocated-payload")
ANCHOR = "485969aa42e0afaa3f46c8ff4ac777b034dc94f9603de2c33f5b6ee6cbf75d21"


def synthetic_exact_geometry(root, contract, admitted):
    """Native 88-matrix tensors/spectra, no tiny-array broadcast or SVD/model admission."""
    root.mkdir()
    (root / "runs").mkdir()
    primary, reference = contract.primary, admitted["reference"]
    steps, settings = primary.payload["checkpoint_steps"], contract.exact.payload["settings"]
    recipes = sorted(
        (primary.expected_identity(row["run_id"]) for row in primary.inputs["runs"]),
        key=lambda row: (
            OPTIMIZER_ORDER[row["recipe"]["optimizer"]["name"]],
            row["recipe"]["optimizer"]["lr"],
        ),
    )
    all_runs, results, bindings = [], [], []
    for run_index, expected in enumerate(recipes):
        recipe = expected["recipe"]
        run_id = recipe["run_id"]
        path = root / "runs" / run_id
        path.mkdir()
        checked = admitted["complete_runs"][run_id]
        outputs, checkpoints, matrix_rows = [], [], []
        for stage, step in enumerate(steps, 1):
            old = read_json(ORIGINAL / "geometry/runs" / run_id / f"checkpoint-{step}.json")
            norms = {row["tensor"]: row["weight"]["frobenius_norm"] for row in old["records"]}
            old_records = {row["tensor"]: row for row in old["records"]}
            records, arrays = [], {}
            for index, (name, shape) in enumerate(sorted(reference["hidden_shapes"].items())):
                for kind_index, kind in enumerate(KINDS):
                    # Every generated array has its actual native width. The spectra
                    # and coordinate subspaces describe explicit hypothetical inputs.
                    active = 17 + (run_index * stage + index + kind_index * (stage - 1)) % 15
                    singular = np.zeros(min(shape), dtype=np.float64)
                    singular[:active] = np.arange(active, 0, -1, dtype=np.float64) / 64
                    field = (
                        "delta_from_previous"
                        if kind == "saved_segment" and stage > 1
                        else "delta_from_reference"
                    )
                    norm = old_records[name][field]["frobenius_norm"]
                    zero = norm == 0
                    if zero:
                        singular.fill(0)
                    else:
                        singular *= norm / np.linalg.norm(singular)
                    metrics = spectrum_metrics(singular, shape, settings, None if zero else 0.0)
                    records.append(
                        {
                            "run_id": run_id,
                            "optimizer": recipe["optimizer"]["name"],
                            "learning_rate": recipe["optimizer"]["lr"],
                            "stage": stage,
                            "step": step,
                            "displacement_kind": kind,
                            "tensor": name,
                            "shape": shape,
                            "parameters": int(np.prod(shape)),
                            **metrics,
                        }
                    )
                    arrays[f"{kind}|{name}|singular"] = torch.from_numpy(singular)
                    if not zero:
                        for side, size in zip(("left", "right"), shape, strict=True):
                            basis = torch.zeros(size, 16, dtype=torch.float64)
                            offset = (
                                run_index * (stage + 1) + index + kind_index * (stage - 1)
                            ) % (size - 16)
                            basis[torch.arange(16) + offset, torch.arange(16)] = 1
                            arrays[f"{kind}|{name}|{side}"] = basis
            row = checkpoint_row(recipe, stage, steps, records, reference["hidden_shapes"], norms)
            record = {
                "step": step,
                "previous_step": steps[stage - 2] if stage > 1 else 0,
                "cumulative_anchor_step": 0,
                "records": records,
                "checkpoint_row": row,
            }
            record_file = path / f"checkpoint-{step}.json"
            array_file = path / f"checkpoint-{step}-exact.safetensors"
            write_new(record_file, record)
            save_file(
                arrays,
                array_file,
                metadata={
                    "scope": exact.exact_geometry.SCOPE,
                    "run_identity_sha256": digest(expected),
                },
            )
            item = {
                "step": step,
                "records": {"path": record_file.name, **file_identity(record_file)},
                "arrays": {"path": array_file.name, **file_identity(array_file)},
            }
            outputs.append(item)
            for file in (record_file, array_file):
                bindings.append({"path": file.relative_to(root).as_posix(), **file_identity(file)})
            checkpoints.append(row)
            matrix_rows.extend(records)
        write_new(
            path / "manifest.json",
            {
                "plan": descriptor(
                    expected,
                    checked,
                    reference,
                    settings,
                    exact.exact_geometry.SCOPE,
                    contract.exact.sha256,
                ),
                "outputs": outputs,
            },
        )
        bindings.append(
            {
                "path": (path / "manifest.json").relative_to(root).as_posix(),
                **file_identity(path / "manifest.json"),
            }
        )
        all_runs.append((path, expected, checked))
        results.append({"checkpoint_rows": checkpoints, "matrix_rows": matrix_rows})
        print({"synthetic_exact_run_written": run_id, "scientific_completion": False}, flush=True)
    with threadpool_limits(limits=settings["cpu_threads"]):
        tables = exact.exact_geometry.tables_from_verified_runs(
            all_runs, root / "runs", reference, results, settings
        )
    require_same(
        {name: len(rows) for name, rows in tables.items()}, exact.exact_geometry.TABLE_COUNTS
    )
    summary = {
        "scope": exact.exact_geometry.SCOPE,
        "primary_protocol_sha256": primary.sha256,
        "exact_geometry_protocol_sha256": contract.exact.sha256,
        "raw_bindings": bindings,
        "table_counts": exact.exact_geometry.TABLE_COUNTS,
        "scientific_completion": False,
        "boundary": reconstruction.GEOMETRY_BOUNDARY,
    }
    write_new(root / "summary.json", summary)
    for name, rows in tables.items():
        with (root / (name + ".csv")).open("xb") as stream:
            stream.write(csv_bytes(rows))
    return summary, tables


def make(work, publication_contract):
    work = Path(work)
    if work.exists():
        raise ValueError("Require a new exact synthetic fixture directory")
    previous = files.inspect(ORIGINAL, ANCHOR)
    work.mkdir()
    contract = exact.ExactBridgeContract.load(
        publication_contract.primary.repository / archive.PROTOCOL, publication_contract.primary
    )
    torch.set_num_threads(contract.exact.payload["settings"]["cpu_threads"])
    admitted = read_json(ORIGINAL / "vectors/admission.json")
    functional = read_json(ORIGINAL / "inference/evidence.json")
    assert functional["upstream_primary_admission_simulated"] is True
    geo_root = work / "synthetic-exact-geometry"
    summary, geometry_tables = synthetic_exact_geometry(geo_root, contract, admitted)
    original_tables = publication.read_tables(ORIGINAL / "bridge", exact.original.TABLE_COUNTS)
    tables, diagnostics = exact.sensitivity_tables(
        contract.primary,
        original_tables,
        geometry_tables["checkpoint_exact_geometry"],
        geometry_tables["run_pair_exact_subspace_overlap"],
    )
    evidence = {
        "source_bindings": [
            *exact.original.snapshot(
                ORIGINAL / "bridge", reconstruction.joint.bundle_names(exact.original.TABLE_COUNTS)
            ),
            *exact.original.snapshot(
                geo_root,
                ["summary.json", *(key + ".csv" for key in exact.exact_geometry.TABLE_COUNTS)],
            ),
            *functional["original_bridge_evidence"]["source_bindings"],
            *[{**row, "path": str(geo_root / row["path"])} for row in summary["raw_bindings"]],
        ],
        "original_bridge_reader": functional["original_bridge_reader"],
        "original_bridge_evidence": functional["original_bridge_evidence"],
        "exact_geometry_reader": summary,
        **diagnostics,
    }
    plan = reconstruction.exact_plan(contract, evidence)
    bridge_root = work / "synthetic-exact-bridge"
    exact.save_bundle(bridge_root, plan, evidence, tables)
    selected = {}
    for name, row in previous["files"].items():
        role, relative = name.split("/", 1)
        if role not in {"source", "training-source"}:
            files.select(selected, role, relative, ORIGINAL / name, row)
    source_selection(publication_contract.inference, selected)
    acceptance = parent.authoring.require_parent(publication_contract.inference)
    files.select(
        selected, "source", parent.authoring.ACCEPTANCE[0], acceptance, file_identity(acceptance)
    )
    parent.add_publication_sources(publication_contract, selected)
    archive.add_sources(contract, selected)
    for role, source in (("exact-geometry", geo_root), ("exact-bridge", bridge_root)):
        for name in files.inventory(source):
            files.select(
                selected,
                "provenance",
                role + "/" + name,
                source / name,
                file_identity(source / name),
            )
    metadata = {
        **previous["metadata"],
        archive.META: archive.extension(contract, plan, evidence),
        "exact_fixture": {
            "upstream_primary_admission_simulated": True,
            "original_raw_archive_sha256": ANCHOR,
            "exact_spectra_and_bases_are_hypothetical": True,
            "actual_primary_authoring_called": False,
            "scientific_completion": False,
        },
    }
    result = files.write(work / "archive", selected, metadata)
    recheck_selection(selected)
    require_same(files.inspect(ORIGINAL, ANCHOR), previous)
    write_new(
        work / "fixture.json",
        {
            "scope": "engineering_full_shape_synthetic_exact_branch",
            "archive": str(work / "archive"),
            **result,
            "original_archive_sha256": ANCHOR,
            "upstream_primary_admission_simulated": True,
            "scientific_completion": False,
        },
    )
    return work / "archive", result["manifest_sha256"]
