"""Reconstruct original and functional inference from all authenticated raw roles.

This is not primary admission, model encoding, corpus retrieval or publication.
Recorded weight measurements are authenticated, not remeasured without weights.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import primary_v3_bridge as bridge
from . import primary_v3_dimension_inference as inference
from . import primary_v3_geometry_reconstruction as geometry
from . import primary_v3_outcome_reconstruction as outcomes
from . import primary_v3_vector_reconstruction as vectors
from . import reconstruction_files as files
from .dimension_inference import summarize
from .dimension_inference_render import render
from .primary_contract import digest, file_identity, read_json, require_same, verify_file
from .primary_v3_dimension_contract import TABLE_COUNTS as FEATURE_COUNTS
from .primary_v3_dimensions import feature_plan
from .primary_v3_exact_bridge import typed_csv
from .primary_v3_outcome_primitives import producer_path
from .primary_v3_outcomes import TABLE_COUNTS as OUTCOME_COUNTS
from .primary_v3_outcomes import inspect_bundle, save_bundle
from .primary_v3_reconstruction_inputs import ReconstructionInput
from .primary_v3_validation_io import write_new

SCOPE = "dense_primary_v3_joint_numerical_reconstruction"


def bundle_names(counts):
    return ["manifest.json", "evidence.json", *(name + ".csv" for name in counts)]


def local_bindings(inputs, rows, role, names):
    """Require an ordered, complete lexical subtree, never hash/basename matching."""
    if not isinstance(rows, list) or len(rows) != len(names) or not names:
        raise ValueError("Incomplete joint provenance binding population")
    require_same(sorted(names), files.inventory(inputs.root / role))
    first = producer_path(rows[0]["path"])
    suffix = producer_path(names[0])
    original_root = first.parents[len(suffix.parts) - 1]
    addresses = []
    for row, name in zip(rows, names, strict=True):
        if set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("Invalid joint provenance binding schema")
        files.identity(row)
        if producer_path(row["path"]) != original_root / name:
            raise ValueError("Joint provenance order or original subtree differs")
        local = files.safe_name(role + "/" + name)
        verify_file(files.ordinary(inputs.root / local), row)
        addresses.append({"original": row["path"], "local": local})
    return original_root, addresses


def provenance(inputs, geometry_summary):
    """Close all five authoring subtrees and the original bridge's ordered suffix."""
    evidence = inputs.evidence
    old = evidence["original_bridge_evidence"]["source_bindings"]
    full = evidence["source_bindings"]
    out_names = bundle_names(OUTCOME_COUNTS)
    geo_names = [
        "summary.json",
        *(n + ".csv" for n in inputs.contract.bridge.geometry.payload["table_counts"]),
    ]
    geo_names += [row["path"] for row in geometry_summary["raw_bindings"]]
    if not isinstance(old, list) or len(old) != len(out_names) + len(geo_names):
        raise ValueError("The complete original bridge provenance is required")
    bridge_names = bundle_names(bridge.TABLE_COUNTS)
    vector_names = files.inventory(inputs.root / "vectors")
    feature_names = files.inventory(inputs.root / "features")
    prefix_count = len(bridge_names) + len(vector_names) + len(feature_names)
    if not isinstance(full, list) or len(full) != prefix_count + len(old):
        raise ValueError("The complete functional provenance is required")
    require_same(full[prefix_count:], old)
    cuts = [len(bridge_names), len(bridge_names) + len(vector_names)]
    blocks = [
        (old[: len(out_names)], "outcomes", out_names),
        (old[len(out_names) :], "geometry", geo_names),
        (full[: cuts[0]], "bridge", bridge_names),
        (full[cuts[0] : cuts[1]], "vectors", vector_names),
        (full[cuts[1] : prefix_count], "features", feature_names),
    ]
    roots, addresses = [], []
    for rows, role, names in blocks:
        root, current = local_bindings(inputs, rows, role, names)
        if any(root.is_relative_to(other) or other.is_relative_to(root) for other in roots):
            raise ValueError("Original analysis roles overlap or alias one another")
        roots.append(root)
        addresses.extend(current)
    if len({row["original"] for row in addresses}) != len(addresses):
        raise ValueError("Duplicated original joint provenance address")
    return addresses


def verified_reader(output, receipt):
    """Recover an old reader shape only from newly written, bound branch outputs."""
    for name, binding in receipt["outputs"].items():
        verify_file(output / name, binding)
    manifest = read_json(output / "outcomes/manifest.json")
    return {
        "plan": manifest["plan"],
        "outputs": manifest["outputs"],
        "recomputed_bytes_verified": True,
        "scientific_completion": False,
    }


def original_bridge(inputs, output, outcome_root, outcome_receipt, geometry_root):
    """Only fresh branch outputs enter the original nine-feature numerical kernel."""
    summary = read_json(geometry_root / "geometry/summary.json")
    addresses = provenance(inputs, summary)
    tables, numerical = bridge.bridge_tables(
        inputs.contract.primary,
        typed_csv(geometry_root / "geometry/checkpoint_geometry.csv"),
        typed_csv(geometry_root / "geometry/run_pair_subspace_overlap.csv"),
        typed_csv(outcome_root / "outcomes/run_stage_scores.csv"),
    )
    evidence = {
        "source_bindings": inputs.evidence["original_bridge_evidence"]["source_bindings"],
        "outcome_reader": verified_reader(outcome_root, outcome_receipt),
        "outcome_evidence": read_json(outcome_root / "outcomes/evidence.json"),
        "geometry_reader": summary,
        "numerical_diagnostics": numerical,
    }
    contract = inputs.contract.bridge
    plan = {
        "scope": bridge.SCOPE,
        "bridge_protocol_sha256": contract.sha256,
        "primary_protocol_sha256": contract.primary.sha256,
        "outcome_protocol_sha256": contract.outcomes.sha256,
        "geometry_protocol_sha256": contract.geometry.sha256,
        "evidence_sha256": digest(evidence),
        "scientific_completion": False,
        "boundary": bridge.AMENDMENT["boundary"],
    }
    fresh = save_bundle(output, plan, evidence, tables)
    require_same(evidence, inputs.evidence["original_bridge_evidence"])
    original = inspect_bundle(inputs.root / "bridge", plan, evidence, tables)
    require_same(original, fresh)
    require_same(original, inputs.evidence["original_bridge_reader"])
    return fresh, evidence, tables, addresses


def functional_inference(inputs, output, vector_root, old_reader, old_evidence, old_tables):
    tables = {name: typed_csv(vector_root / (name + ".csv")) for name in FEATURE_COUNTS}
    tasks = sorted({row["source"] for row in inputs.admitted["probe"]["row_identities"]})
    computed, decisions = summarize(
        inputs.contract.primary,
        tables,
        old_tables["bridge_rows"],
        inputs.contract.dimensions.scientific,
        tasks,
    )
    checked_features = {
        "status": "complete",
        "manifest_sha256": file_identity(inputs.root / "features/manifest.json")["sha256"],
        "plan": feature_plan(
            inputs.contract.dimensions,
            inputs.admitted,
            inputs.manifest["metadata"]["trusted_vector_manifest_sha256"],
        ),
        "table_counts": FEATURE_COUNTS,
        "raw_vector_states_recomputed": 61,
        "model_encoding_repeated": False,
        "scientific_completion": False,
    }
    evidence = {
        "source_bindings": inputs.evidence["source_bindings"],
        "original_bridge_reader": old_reader,
        "original_bridge_evidence": old_evidence,
        "dimension_feature_reader": checked_features,
        "decisions": decisions,
        "rendered_latex": render(computed, decisions),
        "manuscript_installed": False,
        "scientific_completion": False,
    }
    # A test-only provenance annotation must remain explicit in every derived bundle.
    if "upstream_primary_admission_simulated" in inputs.evidence:
        require_same(inputs.evidence["upstream_primary_admission_simulated"], True)
        evidence["upstream_primary_admission_simulated"] = True
    plan = {
        "scope": inference.SCOPE,
        "inference_protocol_sha256": inputs.contract.sha256,
        "primary_protocol_sha256": inputs.contract.primary.sha256,
        "evidence_sha256": digest(evidence),
        "scientific_completion": False,
        "manuscript_installation_authorized": False,
    }
    fresh = save_bundle(output, plan, evidence, computed)
    require_same(evidence, inputs.evidence)
    require_same(inspect_bundle(inputs.root / "inference", plan, evidence, computed), fresh)
    return fresh


def reconstruct(inputs, output, *, progress=None):
    output = Path(output).absolute()
    if (
        output.exists()
        or output.is_symlink()
        or output.parent.resolve() != output.parent
        or output == inputs.root
        or inputs.root in output.parents
    ):
        raise ValueError("Use a new joint reconstruction directory outside the immutable archive")
    inputs.recheck()
    # Cheap structural provenance rejection precedes expensive raw numerics/output.
    # The same mapping is checked again against freshly reconstructed geometry.
    provenance(inputs, inputs.evidence["original_bridge_evidence"]["geometry_reader"])
    output.mkdir(parents=True, exist_ok=False)
    write_new(
        output / "reconstruction_plan.json",
        {
            "scope": SCOPE,
            "external_archive_sha256": inputs.anchor,
            "scientific_completion": False,
        },
    )
    branch = {}
    for name, module in (("outcomes", outcomes), ("geometry", geometry), ("vectors", vectors)):
        kwargs = {"progress": progress} if name == "vectors" else {}
        branch[name] = module.reconstruct(inputs, output / name, **kwargs)
        if progress:
            progress({"raw_branch_completed": name})
    old_reader, old_evidence, old_tables, addresses = original_bridge(
        inputs, output / "bridge", output / "outcomes", branch["outcomes"], output / "geometry"
    )
    current = functional_inference(
        inputs, output / "inference", output / "vectors", old_reader, old_evidence, old_tables
    )
    write_new(output / "local_addresses.json", addresses)
    inputs.recheck()
    receipt = {
        "scope": SCOPE,
        "external_archive_sha256": inputs.anchor,
        "raw_branch_receipts": {
            name: file_identity(output / name / "reconstruction.json") for name in branch
        },
        "original_bridge_table_counts": bridge.TABLE_COUNTS,
        "functional_inference_table_counts": inference.TABLE_COUNTS,
        "original_bridge_reader": old_reader,
        "functional_inference_reader": current,
        "all_three_raw_branches_reconstructed": True,
        "original_and_functional_inference_recomputed": True,
        "original_nine_geometry_features_retained": True,
        "exact_rendered_latex_verified": True,
        "upstream_admission_simulated": inputs.evidence.get("upstream_primary_admission_simulated")
        is True,
        "checkpoint_tensors_revalidated": False,
        "weight_spectra_or_health_remeasured": False,
        "model_encoding_repeated": False,
        "retrieval_repeated": False,
        "physical_cross_host_execution_verified": False,
        "primary_scientific_admission": False,
        "manuscript_installed": False,
        "complete_primary_publication_pipeline_verified": False,
        "scientific_completion": False,
        "outputs": {name: file_identity(output / name) for name in files.inventory(output)},
    }
    write_new(output / "reconstruction.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Joint reconstruction is CPU only")
    inputs = ReconstructionInput.load(args.archive_root, args.expected_manifest_sha256)
    verify_file(
        Path(__file__),
        inputs.manifest["files"]["source/src/embed_optim/primary_v3_joint_reconstruction.py"],
    )
    reconstruct(inputs, args.output, progress=lambda row: print(row, flush=True))
    print(
        {"joint_numerical_reconstruction_verified": True, "scientific_completion": False},
        flush=True,
    )


if __name__ == "__main__":
    main()
