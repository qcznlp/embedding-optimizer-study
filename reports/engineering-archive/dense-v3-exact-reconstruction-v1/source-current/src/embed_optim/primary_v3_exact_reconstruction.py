"""Reconstruct every original/functional output and the separately named exact sensitivity."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from . import primary_v3_exact_bridge as exact
from . import primary_v3_exact_reconstruction_archive as archive
from . import primary_v3_joint_reconstruction as joint
from . import primary_v3_publication_reconstruction as publication
from . import reconstruction_files as files
from .corrected_geometry_summary import OPTIMIZER_ORDER
from .primary_contract import digest, file_identity, read_json, require_same, verify_file
from .primary_v3_exact_geometry_primitives import inspect_run_primitives
from .primary_v3_outcome_primitives import producer_path, recorded_runs
from .primary_v3_outcomes import csv_bytes, inspect_bundle, save_bundle
from .primary_v3_publication import read_tables
from .primary_v3_reconstruction_inputs import ReconstructionInput
from .primary_v3_validation_io import write_new

SCOPE = "dense_primary_v3_complete_exact_measurement_reconstruction"
GEOMETRY_BOUNDARY = "Complete exact measurement layer only; valid primary runs, retrieval/dimension/factorial consumers, explicit release transition and scientific acceptance remain separate gates."


def exact_plan(contract, evidence):
    return {
        "scope": exact.SCOPE,
        "exact_bridge_protocol_sha256": contract.sha256,
        "primary_protocol_sha256": contract.primary.sha256,
        "original_bridge_protocol_sha256": contract.bridge.sha256,
        "exact_geometry_protocol_sha256": contract.exact.sha256,
        "evidence_sha256": digest(evidence),
        "scientific_completion": False,
        "boundary": exact.AMENDMENT["boundary"],
    }


def provenance(inputs, evidence):
    original = inputs.evidence["original_bridge_evidence"]
    require_same(evidence["original_bridge_evidence"], original)
    require_same(evidence["original_bridge_reader"], inputs.evidence["original_bridge_reader"])
    bridge_names = joint.bundle_names(exact.original.TABLE_COUNTS)
    geometry_names = [
        "summary.json",
        *(name + ".csv" for name in exact.exact_geometry.TABLE_COUNTS),
    ]
    raw_names = [row["path"] for row in evidence["exact_geometry_reader"]["raw_bindings"]]
    rows, suffix = evidence["source_bindings"], original["source_bindings"]
    a, b = len(bridge_names), len(bridge_names) + len(geometry_names)
    if not isinstance(rows, list) or len(rows) != b + len(suffix) + len(raw_names):
        raise ValueError("Incomplete exact sensitivity provenance population")
    require_same(rows[b : b + len(suffix)], suffix)
    _, bridge_addresses = joint.local_bindings(inputs, rows[:a], "bridge", bridge_names)
    root, geometry_addresses = joint.local_bindings(
        inputs,
        rows[a:b] + rows[b + len(suffix) :],
        archive.GEOMETRY_ROLE,
        geometry_names + raw_names,
    )
    old_addresses = joint.provenance(inputs, original["geometry_reader"])
    _, _, pub_evidence, pub_addresses = publication.admitted_structure(inputs)
    del pub_evidence
    for row in (*old_addresses, *pub_addresses):
        local = producer_path(row["local"])
        old_root = producer_path(row["original"]).parents[len(local.parts) - 2]
        if root.is_relative_to(old_root) or old_root.is_relative_to(root):
            raise ValueError("Exact geometry producer role overlaps another original analysis role")
    return bridge_addresses + geometry_addresses


def admitted_structure(inputs):
    contract = exact.ExactBridgeContract.load(
        inputs.root / "source" / archive.PROTOCOL, inputs.contract.primary
    )
    require_same(contract.sha256, archive.PROTOCOL_SHA)
    require_same(contract.bridge.sha256, inputs.contract.bridge.sha256)
    evidence = read_json(inputs.root / archive.BRIDGE_ROLE / "evidence.json")
    if set(evidence) != {
        "source_bindings",
        "original_bridge_reader",
        "original_bridge_evidence",
        "exact_geometry_reader",
        "numerical_diagnostics",
        "measurement_coverage",
    }:
        raise ValueError("Incomplete exact sensitivity authoring evidence")
    plan = exact_plan(contract, evidence)
    require_same(
        inputs.manifest["metadata"][archive.META], archive.extension(contract, plan, evidence)
    )
    require_same(
        files.inventory(inputs.root / archive.BRIDGE_ROLE),
        sorted(joint.bundle_names(exact.TABLE_COUNTS)),
    )
    require_same(read_json(inputs.root / archive.BRIDGE_ROLE / "manifest.json")["plan"], plan)
    addresses = provenance(inputs, evidence)
    publication.simulation((inputs.manifest["metadata"], evidence))
    return contract, plan, evidence, addresses


def reconstruct_geometry(inputs, contract, output):
    """Complete spectra/status, checkpoint and projector aggregates, without tensor loading."""
    root, reference = inputs.root / archive.GEOMETRY_ROLE, inputs.admitted["reference"]
    runs = recorded_runs(contract.primary, inputs.admitted)
    run_root = root / "runs"
    require_same(sorted(path.name for path in run_root.iterdir()), sorted(runs))
    admitted = [
        (run_root / name, contract.primary.expected_identity(name), checked)
        for name, checked in runs.items()
    ]
    admitted.sort(
        key=lambda row: (
            OPTIMIZER_ORDER[row[1]["recipe"]["optimizer"]["name"]],
            row[1]["recipe"]["optimizer"]["lr"],
            row[1]["recipe"]["run_id"],
        )
    )
    settings = contract.exact.payload["settings"]
    torch.set_num_threads(settings["cpu_threads"])
    results, bindings = [], []
    with threadpool_limits(limits=settings["cpu_threads"]):
        for path, expected, checked in admitted:
            result = inspect_run_primitives(
                path,
                inputs.root / "geometry/runs" / expected["recipe"]["run_id"],
                expected,
                checked,
                reference,
                contract.exact,
                contract.bridge.geometry,
            )
            results.append(result)
            for row in result["manifest"]["outputs"]:
                for key in ("records", "arrays"):
                    file = path / row[key]["path"]
                    bindings.append(
                        {"path": file.relative_to(root).as_posix(), **file_identity(file)}
                    )
            file = path / "manifest.json"
            bindings.append({"path": file.relative_to(root).as_posix(), **file_identity(file)})
        tables = exact.exact_geometry.tables_from_verified_runs(
            admitted, run_root, reference, results, settings
        )
    require_same(
        {name: len(rows) for name, rows in tables.items()}, exact.exact_geometry.TABLE_COUNTS
    )
    summary = {
        "scope": exact.exact_geometry.SCOPE,
        "primary_protocol_sha256": contract.primary.sha256,
        "exact_geometry_protocol_sha256": contract.exact.sha256,
        "raw_bindings": bindings,
        "table_counts": exact.exact_geometry.TABLE_COUNTS,
        "scientific_completion": False,
        "boundary": GEOMETRY_BOUNDARY,
    }
    require_same(
        sorted(path.name for path in root.iterdir()),
        sorted(["runs", "summary.json", *(key + ".csv" for key in tables)]),
    )
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "summary.json", summary)
    for name, rows in tables.items():
        with (output / (name + ".csv")).open("xb") as stream:
            stream.write(csv_bytes(rows))
    for result in results:
        require_same(result["checkpoint_rows"], result["stored_checkpoint_rows"])
    for name in ("summary.json", *(key + ".csv" for key in tables)):
        if (root / name).read_bytes() != (output / name).read_bytes():
            raise ValueError(
                "Stored exact geometry differs from fresh spectrum/projector aggregates"
            )
    return summary, tables


def reconstruct_bridge(inputs, contract, evidence, plan, parent_root, geometry, output):
    """Only the current full publication invocation supplies the original predictors/outcomes."""
    fresh = read_json(parent_root / "reconstruction.json")
    require_same(fresh["external_archive_sha256"], inputs.anchor)
    if fresh["all_joint_raw_branches_and_inference_recomputed"] is not True:
        raise ValueError("Exact sensitivity requires the current complete original reconstruction")
    for name, row in fresh["outputs"].items():
        verify_file(files.ordinary(parent_root / name), row)
    old_root = parent_root / "joint/bridge"
    old_tables = read_tables(old_root, exact.original.TABLE_COUNTS)
    old_evidence = read_json(old_root / "evidence.json")
    old_plan = read_json(old_root / "manifest.json")["plan"]
    old_reader = inspect_bundle(old_root, old_plan, old_evidence, old_tables)
    summary, geo_tables = geometry
    tables, diagnostics = exact.sensitivity_tables(
        contract.primary,
        old_tables,
        geo_tables["checkpoint_exact_geometry"],
        geo_tables["run_pair_exact_subspace_overlap"],
    )
    fresh_evidence = {
        "source_bindings": evidence["source_bindings"],
        "original_bridge_reader": old_reader,
        "original_bridge_evidence": old_evidence,
        "exact_geometry_reader": summary,
        **diagnostics,
    }
    require_same(fresh_evidence, evidence)
    require_same(exact_plan(contract, fresh_evidence), plan)
    checked = inspect_bundle(inputs.root / archive.BRIDGE_ROLE, plan, fresh_evidence, tables)
    require_same(save_bundle(output, plan, fresh_evidence, tables), checked)
    return checked


def reconstruct(inputs, output, *, progress=None):
    output = Path(output).absolute()
    if (
        output.exists()
        or output.is_symlink()
        or output.parent.resolve() != output.parent
        or output == inputs.root
        or inputs.root in output.parents
    ):
        raise ValueError("Use a new exact reconstruction directory outside the immutable archive")
    inputs.recheck()
    contract, plan, evidence, addresses = admitted_structure(inputs)
    output.mkdir(parents=True, exist_ok=False)
    parent_result = publication.reconstruct(
        inputs, output / "original-functional", progress=progress
    )
    require_same(read_json(output / "original-functional/reconstruction.json"), parent_result)
    geometry = reconstruct_geometry(inputs, contract, output / "exact-geometry")
    bridge = reconstruct_bridge(
        inputs,
        contract,
        evidence,
        plan,
        output / "original-functional",
        geometry,
        output / "exact-bridge",
    )
    write_new(output / "exact_addresses.json", addresses)
    inputs.recheck()
    contract.recheck()
    receipt = {
        "scope": SCOPE,
        "external_archive_sha256": inputs.anchor,
        "exact_bridge_protocol_sha256": contract.sha256,
        "original_functional_reconstruction": file_identity(
            output / "original-functional/reconstruction.json"
        ),
        "exact_bridge_reader": bridge,
        "exact_geometry_table_counts": exact.exact_geometry.TABLE_COUNTS,
        "exact_bridge_table_counts": exact.TABLE_COUNTS,
        "all_original_functional_and_publication_outputs_recomputed": True,
        "all_full_spectrum_defined_metrics_and_projector_aggregates_recomputed": True,
        "all_five_exact_features_and_comparisons_recomputed": True,
        "all_nine_original_features_preserved": True,
        "upstream_admission_simulated": publication.simulation(
            (inputs.manifest["metadata"], evidence)
        ),
        "checkpoint_tensors_revalidated": False,
        "saved_singular_spectra_or_bases_remeasured": False,
        "svd_residuals_or_weight_norms_remeasured": False,
        "model_encoding_repeated": False,
        "retrieval_repeated": False,
        "physical_cross_host_execution_verified": False,
        "primary_scientific_admission": False,
        "manuscript_installed": False,
        "strict_manuscript_consumer_integrated": False,
        "reviewed_source_runtime_release_verified": False,
        "scientific_completion": False,
        "outputs": {name: file_identity(output / name) for name in files.inventory(output)},
    }
    write_new(output / "reconstruction.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Exact reconstruction is CPU only")
    inputs = ReconstructionInput.load(args.archive_root, args.expected_manifest_sha256)
    verify_file(
        Path(__file__),
        inputs.manifest["files"]["source/src/embed_optim/primary_v3_exact_reconstruction.py"],
    )
    reconstruct(inputs, args.output, progress=lambda row: print(row, flush=True))
    print(
        {"complete_exact_sensitivity_reconstructed": True, "scientific_completion": False},
        flush=True,
    )


if __name__ == "__main__":
    main()
