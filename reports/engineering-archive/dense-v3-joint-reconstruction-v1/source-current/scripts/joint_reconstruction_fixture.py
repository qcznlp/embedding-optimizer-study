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
from embed_optim.primary_v3_dimension_contract import TABLE_COUNTS as FEATURE_COUNTS
from embed_optim.primary_v3_dimensions import _feature_manifest, feature_plan
from embed_optim.primary_v3_exact_bridge import typed_csv
from embed_optim.primary_v3_outcomes import TABLE_COUNTS as OUTCOME_COUNTS
from embed_optim.primary_v3_outcomes import inspect_bundle, save_bundle, system_rows
from embed_optim.primary_v3_outcomes import outcome_tables as compute_outcome_tables
from embed_optim.primary_v3_reconstruction_authoring import ACCEPTANCE, BOUNDARY, require_parent
from embed_optim.primary_v3_reconstruction_inputs import vector_admission
from embed_optim.primary_v3_reconstruction_sources import source_selection
from scripts import joint_reconstruction_geometry, joint_reconstruction_vectors
from scripts import outcome_reconstruction_fixture as outcome_fixture
from scripts.vector_reconstruction_fixture import admitted_fixture


def make(root, contract, *, raw=None, progress=None, components=None, component_anchor=None):
    root = Path(root)
    root.mkdir()
    if (components is None) != (component_anchor is None):
        raise ValueError("Synthetic component reuse needs an explicit external anchor")
    if components is not None and raw is not None:
        raise ValueError("Do not mix independently supplied synthetic component populations")
    if components is None:
        parent, anchor = outcome_fixture.make(root / "outcome-input", contract, raw=raw)
        parent_metadata = files.inspect(parent, anchor)["metadata"]
        roles = ("validation", "beir", "outcomes")
    else:
        parent = Path(components)
        preserved = files.inspect(parent, component_anchor)
        if (
            preserved["metadata"].get("scope")
            != "engineering_complete_synthetic_joint_raw_components"
        ):
            raise ValueError("Only the explicitly simulated raw-component fixture may be reused")
        require_same(preserved["metadata"]["scientific_completion"], False)
        require_same(
            preserved["metadata"]["all_primary_admissions_and_measurements_simulated"], True
        )
        parent_metadata = preserved["metadata"]["outcome_parent_metadata"]
        roles = ("validation", "beir", "outcomes", "geometry", "vectors", "features")
        require_same(sorted({name.split("/")[0] for name in preserved["files"]}), sorted(roles))
    admitted = read_json(parent / "vectors/admission.json")
    if not all(
        run.get("simulated_admission") is True for run in admitted["complete_runs"].values()
    ):
        raise ValueError("Joint fixtures never accept an actual primary checkpoint admission")
    vector_admission(contract.dimensions, admitted)
    reference_admission, observed = admitted_fixture(contract)
    require_same(reference_admission["reference"], admitted["reference"])
    producer = root / "producer"
    producer.mkdir()
    for role in roles:
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
    if components is None:
        geometry_summary, geometry_tables = joint_reconstruction_geometry.make(
            producer / "geometry", contract, admitted, progress=progress
        )
    else:
        geometry_summary = read_json(producer / "geometry/summary.json")
        geometry_tables = {
            name: typed_csv(producer / "geometry" / (name + ".csv"))
            for name in contract.bridge.geometry.payload["table_counts"]
        }
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
            *(name + ".csv" for name in contract.bridge.geometry.payload["table_counts"]),
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
    if components is None:
        vector_anchor, feature_reader, _ = joint_reconstruction_vectors.make(
            producer, contract, admitted, observed, progress=progress
        )
    else:
        vector_anchor = file_identity(producer / "vectors/manifest.json")["sha256"]
        matrix_plan = feature_plan(contract.dimensions, admitted, vector_anchor)
        _feature_manifest(
            producer / "features", matrix_plan, vector_admission(contract.dimensions, admitted)
        )
        feature_reader = {
            "status": "complete",
            "manifest_sha256": file_identity(producer / "features/manifest.json")["sha256"],
            "plan": matrix_plan,
            "table_counts": FEATURE_COUNTS,
            "raw_vector_states_recomputed": 61,
            "model_encoding_repeated": False,
            "scientific_completion": False,
        }
    # Actual authoring decodes checked CSV bytes. Kernel rows serialize rates as text.
    feature_tables = {
        name: typed_csv(producer / "features" / (name + ".csv")) for name in FEATURE_COUNTS
    }
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
        **parent_metadata,
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
            "raw_components_reused": components is not None,
            "raw_components_manifest_sha256": component_anchor,
            "feature_kernel_repeated_during_fixture_assembly": components is None,
            "complete_raw_reconstruction_required_after_assembly": True,
        },
    }
    result = files.write(root / "archive", selected, metadata)
    producer.rename(root / "preserved-producer-unavailable-at-original-location")
    return root / "archive", result["manifest_sha256"]
