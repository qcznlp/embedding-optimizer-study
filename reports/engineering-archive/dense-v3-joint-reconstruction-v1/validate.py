"""Verify this bounded synthetic milestone; never release a primary experiment."""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim import reconstruction_files as files
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_bridge import TABLE_COUNTS as ORIGINAL_COUNTS
from embed_optim.primary_v3_dimension_inference import TABLE_COUNTS as FUNCTIONAL_COUNTS
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.audit_dense_v3_joint_reconstruction import SOURCES
from scripts.joint_reconstruction_controls import CASES
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-v3-joint-reconstruction-v1")
PARENT = (
    "reports/engineering-archive/dense-v3-geometry-reconstruction-v1/validation.json",
    "0d2843d57e98289dc6dda89d8a92a25e8cd1a59f3f5cbbfd4c3a527b34ecf460",
)
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")
FALSE_FLAGS = (
    "checkpoint_tensors_revalidated",
    "weight_spectra_or_health_remeasured",
    "model_encoding_repeated",
    "retrieval_repeated",
    "physical_cross_host_execution_verified",
    "primary_scientific_admission",
    "manuscript_installed",
    "complete_primary_publication_pipeline_verified",
    "scientific_completion",
)


def tests(path, count, failures=0):
    suites = list(ET.parse(path).iter("testsuite"))
    assert len(suites) == 1
    value = suites[0].attrib
    require_same(
        {k: int(value[k]) for k in ("tests", "failures", "errors", "skipped")},
        {"tests": count, "failures": failures, "errors": 0, "skipped": 0},
    )
    return value


def check_audit(value, anchor, before):
    assert value["scope"] == "engineering_complete_joint_reconstruction_audit"
    assert value["passed"] is True and value["scientific_completion"] is False
    assert value["upstream_primary_admission_simulated"] is True
    assert value["final_portable_publication_verified"] is False
    assert value["external_manifest_sha256"] == anchor
    child = value["cold_child"]
    assert child["exit_code"] == 0
    child = child["result"]
    assert child["network_and_original_paths_blocked"] is True
    assert all(
        not Path(p).is_absolute() and p.startswith("embed_optim/")
        for p in child["imported_modules"].values()
    )
    assert "embed_optim.primary_v3_reconstruction_inputs" in child["imported_modules"]
    assert "/root/embedding-optimizer-study/src" in child["removed_search_paths"]
    receipt = child["receipt"]
    assert receipt["scope"] == "dense_primary_v3_joint_numerical_reconstruction"
    assert receipt["external_archive_sha256"] == anchor
    for key in (
        "all_three_raw_branches_reconstructed",
        "original_and_functional_inference_recomputed",
        "original_nine_geometry_features_retained",
        "exact_rendered_latex_verified",
        "upstream_admission_simulated",
    ):
        assert receipt[key] is True
    for key in FALSE_FLAGS:
        assert receipt[key] is False
    require_same(receipt["original_bridge_table_counts"], ORIGINAL_COUNTS)
    require_same(receipt["functional_inference_table_counts"], FUNCTIONAL_COUNTS)
    assert len(receipt["outputs"]) == 477
    require_same(sorted(receipt["raw_branch_receipts"]), ["geometry", "outcomes", "vectors"])
    changes = value["semantic_alterations_rejected"]
    require_same([r["case"] for r in changes], list(CASES))
    for row in changes:
        assert row["exit_code"] != 0 and row["success_receipt_absent"] is True
        mode = CASES[row["case"]]
        assert row["consumer_scope"] == mode
        assert row["full_public_entrypoint_invoked"] is (mode == "full")
        assert row["same_cold_run_fresh_branches_reused"] is (mode != "full")
        assert row["complete_fresh_raw_reconstruction_verified_by_refusal"] is False
        for forbidden in (
            "SHA-256 mismatch",
            "Original producer/source path forbidden",
            "Network access forbidden",
            "incorrectly accepted",
        ):
            assert forbidden not in row["stderr"]
        assert any(s.startswith(("ValueError:", "KeyError:")) for s in row["stderr"].splitlines())
    assert "ordinary retained run" in value["actual_primary_authoring_refusal"]
    require_same(value["post_execution_dispatchers"], before)


def check_oracle(value):
    assert value["passed"] is True and value["scientific_completion"] is False
    assert value["primary_admission_supplied"] is False
    require_same(
        value["alignment"],
        {
            "paired_states": 60,
            "geometry_feature_cells": 540,
            "functional_feature_cells": 240,
            "raw_beir_task_cells": 840,
        },
    )
    for name, predictions, folds, decisions in (
        ("original_bridge", 540, 36, 9),
        ("functional_bridge", 240, 16, 4),
    ):
        result = value[name]
        assert result["exact_predictions"] == predictions
        assert result["exact_fold_mse_comparisons"] == folds
        assert result["exact_pooled_decisions"] == decisions
        assert len(result["residual_associations"]) == decisions
    task = value["functional_task_family"]
    assert len(task["contrasts"]) == 9 and len(task["rotations"]) == 27
    assert task["samples"] == 50000 and task["figure_points_checked"] == 68


def validate(args):
    root = args.repository.resolve()
    archive = root / ARCHIVE
    parent_path = root / PARENT[0]
    assert file_identity(parent_path)["sha256"] == PARENT[1]
    previous = read_json(parent_path)
    preserved = [
        compare_binding(
            row,
            archive / "before" / row["path"] if row["path"] in DOCS else root / row["path"],
            root,
        )
        for row in previous["bindings"]
    ]
    for name in SOURCES:
        verify_file(root / name, file_identity(archive / "source-current" / name))
    records = {}
    for name in ("audit", "replay"):
        path = getattr(args, name)
        assert file_identity(path)["sha256"] == getattr(args, name + "_sha256")
        verify_file(archive / ("complete-" + name + ".json"), file_identity(path))
        records[name] = read_json(path)
    first, second = records["audit"], records["replay"]
    fixture_binding = first["fixture"]
    assert (
        fixture_binding["sha256"]
        == "7991b502cc7b9b7ab1a3199ee9396bb911b0ed25263ab9a42924af0ccf7784c4"
    )
    verify_file(fixture_binding["path"], fixture_binding)
    verify_file(archive / "complete-fixture.json", fixture_binding)
    fixture = read_json(fixture_binding["path"])
    assert fixture["one_complete_run_population"] is True
    for key, count in {
        "vector_dimension": 768,
        "vector_states": 61,
        "geometry_states": 60,
        "beir_task_units": 840,
    }.items():
        assert fixture[key] == count
    assert fixture["fixture_raw_components_reused"] is True
    assert fixture["upstream_primary_admission_simulated"] is True
    assert fixture["scientific_completion"] is False
    assert first["fresh_replay"] is False and second["fresh_replay"] is True
    assert second["replay_parent"]["sha256"] == args.audit_sha256
    require_same(first["sources"], second["sources"])
    require_same(
        first["sources"],
        [{"path": str(root / name), **file_identity(root / name)} for name in SOURCES],
    )
    require_same(first["cold_child"]["result"], second["cold_child"]["result"])
    require_same(first["independent_oracle"], second["independent_oracle"])
    before = handoff()
    for value in (first, second):
        check_audit(value, fixture["manifest_sha256"], before)
        check_oracle(value["independent_oracle"])
        require_same(value["fixture"], fixture_binding)
        payload = files.inspect(Path(value["payload_root"]), fixture["manifest_sha256"])
        assert (
            payload["metadata"]["joint_fixture"][
                "all_upstream_measurements_and_primary_admissions_simulated"
            ]
            is True
        )
    suites = {
        "focused-final.xml": tests(archive / "focused-final.xml", 25),
        "full-final.xml": tests(archive / "full-final.xml", 2573),
    }
    failure = tests(archive / "attempt-1/mapping.xml", 12, 1)
    failed_path = archive / "attempt-1/failed.json"
    assert (
        file_identity(failed_path)["sha256"]
        == "ea3e6b17bab4fb04bad7fb1bca7585d71c0ca203dce7679f9368c6d431b6ce10"
    )
    failed = read_json(failed_path)
    assert failed["passed"] is False and failed["scientific_completion"] is False
    assert "Artifact belongs to a different primary identity" in failed["traceback"]
    for row in failed["sources"]:
        verify_file(archive / "attempt-1/source" / Path(row["path"]).relative_to(root), row)
    typed = read_json(archive / "attempt-1/typed-metadata-check.json")
    assert all(row["typed_mismatches"] == 0 for row in typed.values())
    return finish(
        args, root, archive, previous, preserved, records, fixture_binding, suites, failure, before
    )


def finish(args, root, archive, previous, preserved, records, fixture, suites, failure, before):
    external = {}

    def add(row):
        if row["path"] in external:
            require_same(external[row["path"]], row)
        external[row["path"]] = row

    add(fixture)
    for name, value in records.items():
        path = getattr(args, name)
        add({"path": str(path), **file_identity(path)})
        for row in (*value["sources"], *value["artifacts"]):
            add(row)
        output = path.parent / "cold-reconstruction/recomputed"
        require_same(
            read_json(output / "reconstruction.json"), value["cold_child"]["result"]["receipt"]
        )
    component = read_json(archive / "raw-components.json")
    assert component["component_transport_verified"] is True
    assert component["joint_numerical_reconstruction_verified"] is False
    assert component["scientific_completion"] is False and component["files"] == 1774
    assert (
        component["manifest_sha256"]
        == "331b9b00b6f78b0a020be9fda2032614ad732fdaeef9a8b7e574804000ab4d2a"
    )
    payload = Path(component["payload"])
    files.inspect(payload, component["manifest_sha256"])
    for name in files.inventory(payload):
        path = payload / name
        add({"path": str(path), **file_identity(path)})
    checks = [compare_binding(row, Path(path), root) for path, row in sorted(external.items())]
    for row in previous["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            verify_file(base / row["path"], row)
    verify_file(root / "paper/main.tex", previous["unchanged_manuscript_source"])
    verify_file(previous["unchanged_main_ledger"]["path"], previous["unchanged_main_ledger"])
    require_same(handoff(), before)
    return {
        "scope": "engineering_synthetic_joint_reconstruction_acceptance",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "complete_synthetic_raw_to_original_and_functional_inference_reconstruction_verified": True,
        "independent_join_and_statistical_oracles_verified": True,
        "fresh_process_same_host_relocated_replay_verified": True,
        "network_and_original_producer_paths_blocked": True,
        "numerical_outputs_exactly_matched": 477,
        "semantic_controls_per_audit": {
            "full_entrypoint": 9,
            "bridge_consumer": 4,
            "functional_consumer": 8,
        },
        "late_controls_reuse_same_cold_run_fresh_raw_outputs": True,
        "early_refusal_is_not_full_raw_replay": True,
        "first_fixture_failed_sources_and_full_raw_components_preserved": True,
        "raw_components_reused_during_fixture_assembly": True,
        "all_three_raw_branches_recomputed_on_both_positive_audits": True,
        "upstream_primary_admission_simulated": True,
        "checkpoint_tensors_revalidated": False,
        "weight_spectra_or_health_remeasured": False,
        "model_encoding_repeated": False,
        "retrieval_repeated": False,
        "physical_cross_host_execution_verified": False,
        "primary_scientific_admission": False,
        "complete_primary_publication_pipeline_verified": False,
        "formal_replication_ready": False,
        "scientific_completion": False,
        "accepted_receipts": {
            name: {"path": str(getattr(args, name)), "sha256": getattr(args, name + "_sha256")}
            for name in records
        },
        "tests": suites,
        "preserved_failed_mapping_suite": failure,
        "prior_acceptance": {"path": PARENT[0], "sha256": PARENT[1]},
        "prior_bindings_preserved": preserved,
        "prior_external_payloads_rechecked": False,
        "external_payload_checks": checks,
        "unchanged_live_core": previous["unchanged_live_core"],
        "unchanged_manuscript_source": previous["unchanged_manuscript_source"],
        "unchanged_main_ledger": previous["unchanged_main_ledger"],
        "post_execution_dispatchers": before,
        "bindings": [
            identity(p, root)
            for p in sorted(archive.rglob("*"))
            if p.is_file() and p.name != "validation.json"
        ]
        + [identity(root / name, root) for name in DOCS],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "audit", "replay", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("audit-sha256", "replay-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if (
        sys.flags.optimize
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or args.output.exists()
        or not args.output.parent.is_dir()
    ):
        raise ValueError("Require CPU-only assertions and a new acceptance receipt")
    write_new(args.output, validate(args))
    print({"artifact_validation_passed": True, **file_identity(args.output)}, flush=True)


if __name__ == "__main__":
    main()
