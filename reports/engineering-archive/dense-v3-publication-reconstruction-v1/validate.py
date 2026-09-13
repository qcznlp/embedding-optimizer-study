"""Verify completed synthetic publication replay evidence; never authorize a release."""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim import primary_v3_publication_archive as authoring
from embed_optim import reconstruction_files as files
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_publication_contract import OUTPUTS
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff

ARCHIVE = Path(__file__).resolve().parent
ROOT = ARCHIVE.parents[2]


def binding(path):
    name = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    return {"path": name, **file_identity(path)}


def tests(path, count):
    values = list(ET.parse(path).iter("testsuite"))
    assert len(values) == 1
    row = values[0].attrib
    assert int(row["tests"]) == count
    assert int(row["failures"]) == int(row["errors"]) == int(row["skipped"]) == 0
    return row


def check(value, before):
    assert value["passed"] is True and value["upstream_primary_admission_simulated"] is True
    for key in (
        "actual_checkpoint_backed_positive_authoring_verified",
        "strict_manuscript_consumer_integrated",
        "physical_cross_host_execution_verified",
        "reviewed_source_runtime_release_verified",
        "manuscript_installed",
        "scientific_completion",
    ):
        assert value[key] is False
    child = value["cold_child"]
    assert child["exit_code"] == 0
    result = child["result"]
    assert result["network_and_original_paths_blocked"] is True
    assert all(name.startswith("embed_optim/") for name in result["imported_modules"].values())
    receipt = result["receipt"]
    assert receipt["external_archive_sha256"] == value["external_manifest_sha256"]
    for key in (
        "all_joint_raw_branches_and_inference_recomputed",
        "all_seven_publication_outputs_recomputed",
        "raw_functional_latex_and_label_only_include_preserved",
        "upstream_admission_simulated",
    ):
        assert receipt[key] is True
    for key in (
        "checkpoint_tensors_revalidated",
        "weight_spectra_or_health_remeasured",
        "model_encoding_repeated",
        "retrieval_repeated",
        "physical_cross_host_execution_verified",
        "primary_scientific_admission",
        "manuscript_installed",
        "reviewed_source_runtime_release_verified",
        "complete_primary_publication_pipeline_verified",
        "scientific_completion",
    ):
        assert receipt[key] is False
    require_same(sorted(receipt["publication_reader"]["manifest"]["outputs"]), sorted(OUTPUTS))
    controls = value["altered_publication_output_controls"]
    require_same([row["file"] for row in controls], list(OUTPUTS))
    for row in controls:
        assert row["refused"] is True and row["same_cold_run_expected_bytes_reused"] is True
        assert row["consumer_scope"] == "exact_publication_output_inspection"
        assert row["raw_reconstruction_repeated"] is False and row["fresh_process_control"] is False
    assert "ordinary retained run" in value["actual_missing_primary_authoring_refusal"]
    require_same(value["post_execution_dispatchers"], before)


def validate(args):
    parent_path = ROOT / authoring.ACCEPTANCE
    assert file_identity(parent_path)["sha256"] == authoring.ACCEPTANCE_SHA
    previous = read_json(parent_path)
    assert (
        previous["artifact_validation_passed"] is True
        and previous["scientific_completion"] is False
    )
    before = handoff()
    prior = []
    for row in previous["bindings"]:
        path = ROOT / row["path"]
        verify_file(path, row)
        prior.append(binding(path))
    records, checks = {}, []
    for name in ("first", "replay"):
        path = getattr(args, name)
        assert file_identity(path)["sha256"] == getattr(args, name + "_sha256")
        value = records[name] = read_json(path)
        check(value, before)
        for row in (*value["sources"], *value["artifacts"]):
            verify_file(Path(row["path"]), row)
            checks.append(binding(Path(row["path"])))
        files.inspect(Path(value["payload_root"]), value["external_manifest_sha256"])
    first, replay = records["first"], records["replay"]
    assert first["fresh_replay"] is False and replay["fresh_replay"] is True
    assert replay["replay_parent"]["sha256"] == args.first_sha256
    require_same(first["cold_child"]["result"], replay["cold_child"]["result"])
    require_same(first["sources"], replay["sources"])
    for row in first["sources"]:
        original = Path(row["path"])
        verify_file(ARCHIVE / "source-initial" / original.relative_to(ROOT), row)
    for row in previous["unchanged_live_core"]:
        for root in (ROOT, Path("/root/embedding-optimizer-study")):
            verify_file(root / row["path"], row)
    verify_file(ROOT / "paper/main.tex", previous["unchanged_manuscript_source"])
    verify_file(Path(previous["unchanged_main_ledger"]["path"]), previous["unchanged_main_ledger"])
    suites = {
        "focused": tests(ARCHIVE / "focused-first.xml", 27),
        "whole_development": tests(args.full_xml, 2663),
    }
    # The new archive carries richer authoring provenance, not changed scientific text.
    old_generation = read_json(parent_path.parent / "complete-final.json")
    prior_outputs = {
        Path(row["path"]).name: row
        for row in old_generation["generated_artifacts"]
        if "/synthetic-evidence/" in row["path"]
    }
    for name in OUTPUTS:
        if name == "evidence.json":
            continue
        old = prior_outputs[name]
        verify_file(Path(old["path"]), old)
        new = Path(first["payload_root"]) / authoring.ROLE / name
        verify_file(new, old)
    require_same(handoff(), before)
    return {
        "scope": "engineering_synthetic_full_publication_reconstruction_acceptance",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "two_complete_source_isolated_numerical_reconstructions_verified": True,
        "all_six_non_provenance_outputs_match_prior_generation": True,
        "seven_rehashed_output_refusals_per_complete_run": True,
        "upstream_primary_admission_simulated": True,
        "actual_checkpoint_backed_positive_authoring_verified": False,
        "strict_manuscript_consumer_integrated": False,
        "physical_cross_host_execution_verified": False,
        "formal_replication_ready": False,
        "manuscript_installed": False,
        "scientific_completion": False,
        "tests": suites,
        "test_binding": binding(args.full_xml),
        "accepted_receipts": {name: binding(getattr(args, name)) for name in records},
        "prior_acceptance": binding(parent_path),
        "prior_bindings_preserved": prior,
        "prior_external_payloads_recursively_rechecked": False,
        "source_and_artifact_checks": checks,
        "unchanged_live_core": previous["unchanged_live_core"],
        "unchanged_manuscript_source": previous["unchanged_manuscript_source"],
        "unchanged_main_ledger": previous["unchanged_main_ledger"],
        "post_execution_dispatchers": before,
        "bindings": [
            binding(path)
            for path in sorted(ARCHIVE.rglob("*"))
            if path.is_file() and path.name != "validation.json"
        ],
        "mutable_handoff_documents_are_not_acceptance_inputs": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "replay", "full-xml", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("first-sha256", "replay-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if (
        sys.flags.optimize
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or args.output.exists()
        or args.output.parent.resolve() != ARCHIVE
    ):
        raise ValueError("Require assertions and a new receipt inside this bounded archive")
    write_new(args.output, validate(args))
    print({"artifact_validation_passed": True, **file_identity(args.output)}, flush=True)
