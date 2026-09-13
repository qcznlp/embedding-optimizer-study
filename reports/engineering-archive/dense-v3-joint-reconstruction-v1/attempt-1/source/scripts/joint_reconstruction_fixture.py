"""One full synthetic population for all three raw branches and both inferences.

Checkpoint admission, embeddings, weight measurements, scores and timing are all
simulated. Actual source/probe identities do not make this primary evidence.
"""

import shutil
from pathlib import Path

from embed_optim import primary_v3_bridge as bridge
from embed_optim import primary_v3_dimension_inference as inference
from embed_optim import reconstruction_files as files
from embed_optim.dimension_inference import summarize
from embed_optim.dimension_inference_render import render
from embed_optim.primary_contract import digest, file_identity, read_json, require_same
from embed_optim.primary_v3_outcomes import TABLE_COUNTS as OUTCOME_COUNTS
from embed_optim.primary_v3_outcomes import inspect_bundle, save_bundle, system_rows
from embed_optim.primary_v3_outcomes import outcome_tables as compute_outcome_tables
from embed_optim.primary_v3_reconstruction_authoring import ACCEPTANCE, BOUNDARY, require_parent
from embed_optim.primary_v3_reconstruction_inputs import vector_admission
from embed_optim.primary_v3_reconstruction_sources import source_selection
from scripts import joint_reconstruction_geometry, joint_reconstruction_vectors
from scripts import outcome_reconstruction_fixture as outcome_fixture
from scripts.vector_reconstruction_fixture import admitted_fixture


def make(root, contract, *, raw=None, progress=None):
    root = Path(root)
    root.mkdir()
    # This new parent supplies all complete runs, validation and task plans once.
    parent, anchor = outcome_fixture.make(root / "outcome-input", contract, raw=raw)
    parent_manifest = files.inspect(parent, anchor)
    admitted = read_json(parent / "vectors/admission.json")
    vector_admission(contract.dimensions, admitted)
    reference_admission, observed = admitted_fixture(contract)
    require_same(reference_admission["reference"], admitted["reference"])
    producer = root / "producer"
    producer.mkdir()
    for role in ("validation", "beir", "outcomes"):
        shutil.copytree(parent / role, producer / role)
    outcome_manifest = read_json(producer / "outcomes/manifest.json")
    outcome_evidence = read_json(producer / "outcomes/evidence.json")
    # Use actual typed fresh outcome tables, not a hand-invented 60-row panel.
    outcome_tables = compute_outcome_tables(
        contract.primary,
        outcome_evidence["grid"]["score_rows"],
        outcome_evidence["validation_selection"],
    )
    outcome_tables["system_metrics"] = system_rows(
        contract.primary, outcome_evidence["grid"]["runs"]
    )
    outcome_reader = inspect_bundle(
        producer / "outcomes", outcome_manifest["plan"], outcome_evidence, outcome_tables
    )
    geometry_summary, geometry_tables = joint_reconstruction_geometry.make(
        producer / "geometry", contract, admitted, progress=progress
    )
    old_tables, numerical = bridge.bridge_tables(
        contract.primary,
        geometry_tables["checkpoint_geometry"],
        geometry_tables["run_pair_subspace_overlap"],
        outcome_tables["run_stage_scores"],
    )
    old_bindings = bridge.snapshot(
        producer / "outcomes",
        [
            "manifest.json",
            "evidence.json",
            *(name + ".csv" for name in OUTCOME_COUNTS),
        ],
    )
    old_bindings += bridge.snapshot(
        producer / "geometry",
        [
            "summary.json",
            *(name + ".csv" for name in geometry_tables),
        ],
    )
    old_bindings += [
        {**row, "path": str(producer / "geometry" / row["path"])}
        for row in geometry_summary["raw_bindings"]
    ]
    old_evidence = {
        "source_bindings": old_bindings,
        "outcome_reader": outcome_reader,
        "outcome_evidence": outcome_evidence,
        "geometry_reader": geometry_summary,
        "numerical_diagnostics": numerical,
    }
    old_plan = {
        "scope": bridge.SCOPE,
        "bridge_protocol_sha256": contract.bridge.sha256,
        "primary_protocol_sha256": contract.primary.sha256,
        "outcome_protocol_sha256": contract.bridge.outcomes.sha256,
        "geometry_protocol_sha256": contract.bridge.geometry.sha256,
        "evidence_sha256": digest(old_evidence),
        "scientific_completion": False,
        "boundary": bridge.AMENDMENT["boundary"],
    }
    old_reader = save_bundle(producer / "bridge", old_plan, old_evidence, old_tables)
    if progress:
        progress({"joint_fixture_original_bridge_complete": True})
    vector_anchor, feature_reader, feature_tables = joint_reconstruction_vectors.make(
        producer, contract, admitted, observed, progress=progress
    )
    computed, decisions = summarize(
        contract.primary,
        feature_tables,
        old_tables["bridge_rows"],
        contract.dimensions.scientific,
        sorted({row["source"] for row in admitted["probe"]["row_identities"]}),
    )
    bindings = bridge.snapshot(
        producer / "bridge",
        [
            "manifest.json",
            "evidence.json",
            *(name + ".csv" for name in bridge.TABLE_COUNTS),
        ],
    )
    for role in ("vectors", "features"):
        bindings += bridge.snapshot(producer / role, files.inventory(producer / role))
    bindings += old_bindings
    bridge.verify_snapshot(bindings)
    evidence = {
        "source_bindings": bindings,
        "original_bridge_reader": old_reader,
        "original_bridge_evidence": old_evidence,
        "dimension_feature_reader": feature_reader,
        "decisions": decisions,
        "rendered_latex": render(computed, decisions),
        "manuscript_installed": False,
        "scientific_completion": False,
        "upstream_primary_admission_simulated": True,
    }
    plan = {
        "scope": inference.SCOPE,
        "primary_protocol_sha256": contract.primary.sha256,
        "inference_protocol_sha256": contract.sha256,
        "evidence_sha256": digest(evidence),
        "scientific_completion": False,
        "manuscript_installation_authorized": False,
    }
    save_bundle(producer / "inference", plan, evidence, computed)
    selected = source_selection(contract, {})
    acceptance = require_parent(contract)
    files.select(selected, "source", ACCEPTANCE[0], acceptance, file_identity(acceptance))
    for role in (
        "validation",
        "beir",
        "outcomes",
        "geometry",
        "vectors",
        "features",
        "bridge",
        "inference",
    ):
        for name in files.inventory(producer / role):
            path = producer / role / name
            files.select(selected, role, name, path, file_identity(path))
    metadata = {
        **parent_manifest["metadata"],
        "trusted_vector_manifest_sha256": vector_anchor,
        "authoring_plan": plan,
        "authoring_evidence_sha256": digest(evidence),
        "boundary": BOUNDARY,
        "joint_fixture": {
            "all_upstream_measurements_and_primary_admissions_simulated": True,
            "one_complete_run_population": True,
            "vector_dimension": 768,
            "vector_states": 61,
            "geometry_states": 60,
            "beir_task_units": 840,
            "valid_positive_first_stage_geometry": True,
            "first_stage_segment_and_cumulative_anchors_equal": True,
        },
    }
    result = files.write(root / "archive", selected, metadata)
    producer.rename(root / "preserved-producer-unavailable-at-original-location")
    return root / "archive", result["manifest_sha256"]
