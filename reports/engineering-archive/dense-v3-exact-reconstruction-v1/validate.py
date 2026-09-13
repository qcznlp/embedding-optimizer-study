"""Accept completed exact reconstruction engineering evidence, never release a study."""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim import primary_v3_exact_bridge as exact
from embed_optim import reconstruction_files as files
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_publication_contract import OUTPUTS
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff

ARCHIVE = Path(__file__).resolve().parent
ROOT = ARCHIVE.parents[2]
PARENT = ROOT / "reports/engineering-archive/dense-v3-publication-reconstruction-v1/validation.json"
PARENT_SHA = "1b9044c4506b483d1dddadb54819c0a7ae5809f15103b743ae6d98b5e0ad2e62"
REAL_SHA = "1568596c508cdff5a2c1ed5156741adf1403349989dd808ec9eed0de1b6e9372"
ORACLE_SHA = "dc1eb081419a91db467674fbd7330066472f3c4600057fe37a2d8062e8aaed8b"


def binding(path):
    path = Path(path).absolute()
    return {
        "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        **file_identity(path),
    }


def suite(path, count, failures=0):
    rows = list(ET.parse(path).iter("testsuite"))
    assert len(rows) == 1
    row = rows[0].attrib
    assert int(row["tests"]) == count and int(row["failures"]) == failures
    assert int(row["errors"]) == int(row["skipped"]) == 0
    return row


def check(value, dispatchers):
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
    result, receipt = child["result"], child["result"]["receipt"]
    assert result["network_and_original_paths_blocked"] is True
    assert all(name.startswith("embed_optim/") for name in result["imported_modules"].values())
    assert receipt["external_archive_sha256"] == value["external_manifest_sha256"]
    for key in (
        "all_original_functional_and_publication_outputs_recomputed",
        "all_full_spectrum_defined_metrics_and_projector_aggregates_recomputed",
        "all_five_exact_features_and_comparisons_recomputed",
        "all_nine_original_features_preserved",
        "upstream_admission_simulated",
    ):
        assert receipt[key] is True
    for key in (
        "checkpoint_tensors_revalidated",
        "saved_singular_spectra_or_bases_remeasured",
        "svd_residuals_or_weight_norms_remeasured",
        "model_encoding_repeated",
        "retrieval_repeated",
        "physical_cross_host_execution_verified",
        "primary_scientific_admission",
        "manuscript_installed",
        "strict_manuscript_consumer_integrated",
        "reviewed_source_runtime_release_verified",
        "scientific_completion",
    ):
        assert receipt[key] is False
    require_same(receipt["exact_geometry_table_counts"], exact.exact_geometry.TABLE_COUNTS)
    require_same(receipt["exact_bridge_table_counts"], exact.TABLE_COUNTS)
    for key, names, scope in (
        ("publication_output_controls", list(OUTPUTS), "exact_publication_output_inspection"),
        (
            "exact_output_controls",
            ["evidence.json", *(key + ".csv" for key in exact.TABLE_COUNTS)],
            "exact_bridge_output_inspection",
        ),
    ):
        require_same([row["file"] for row in value[key]], names)
        for row in value[key]:
            assert row["refused"] is True and row["same_cold_run_expected_bytes_reused"] is True
            assert row["consumer_scope"] == scope
            assert (
                row["raw_reconstruction_repeated"] is False
                and row["fresh_process_control"] is False
            )
    assert "ordinary retained run" in value["actual_missing_primary_authoring_refusal"]
    require_same(value["post_execution_dispatchers"], dispatchers)


def validate(args):
    assert file_identity(PARENT)["sha256"] == PARENT_SHA
    parent = read_json(PARENT)
    assert parent["artifact_validation_passed"] is True and parent["scientific_completion"] is False
    before, prior = handoff(), []
    for row in parent["bindings"]:
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
            verify_file(row["path"], row)
            checks.append(binding(row["path"]))
        files.inspect(value["payload_root"], value["external_manifest_sha256"])
    first, replay = records["first"], records["replay"]
    assert first["fresh_replay"] is False and replay["fresh_replay"] is True
    assert replay["replay_parent"]["sha256"] == args.first_sha256
    require_same(first["cold_child"]["result"], replay["cold_child"]["result"])
    require_same(first["sources"], replay["sources"])
    for row in first["sources"]:
        verify_file(ARCHIVE / "source-current" / Path(row["path"]).relative_to(ROOT), row)
    for row in parent["unchanged_live_core"]:
        for root in (ROOT, Path("/root/embedding-optimizer-study")):
            verify_file(root / row["path"], row)
    verify_file(ROOT / "paper/main.tex", parent["unchanged_manuscript_source"])
    verify_file(parent["unchanged_main_ledger"]["path"], parent["unchanged_main_ledger"])
    real_file, oracle_file = (
        ARCHIVE / "real-spectrum-readback.json",
        ARCHIVE / "full-oracle-first.json",
    )
    assert file_identity(real_file)["sha256"] == REAL_SHA
    assert file_identity(oracle_file)["sha256"] == ORACLE_SHA
    real, oracle = read_json(real_file), read_json(oracle_file)
    assert real["passed"] is True and oracle["passed"] is True
    assert real["all_1584_spectrum_defined_records_match_exactly"] is True
    require_same(real["basis_status_counts"], {"zero": 528, "resolved": 1056})
    for key in (
        "saved_weights_loaded",
        "full_svd_repeated",
        "reconstruction_residual_remeasured",
        "complete_primary_admission",
        "scientific_completion",
    ):
        assert real[key] is False
    for row in (
        *real["inputs"],
        real["parent_acceptance"],
        real["original_diagnostic"],
        real["new_primitive"],
        real["audit_source"],
        oracle["oracle_source"],
        oracle["reused_sympy_source"],
    ):
        verify_file(row["path"], row)
        checks.append(binding(row["path"]))
    assert oracle["external_manifest_sha256"] == first["external_manifest_sha256"]
    assert oracle["spectrum_rows"] == 10560 and oracle["checkpoint_rows"] == 60
    assert oracle["run_pair_rows"] == 660 and oracle["optimizer_pair_rows"] == 60
    independent = oracle["independent_sympy_systems"]
    assert (
        independent["exact_predictions"] == 300 and independent["exact_fold_mse_comparisons"] == 20
    )
    assert independent["exact_pooled_decisions"] == 5
    assert oracle["unaltered_exact_bundle_inspection_passed"] is True
    assert (
        oracle["upstream_primary_admission_simulated"] is True
        and oracle["scientific_completion"] is False
    )
    suites = {
        "first_primitives": suite(ARCHIVE / "focused-first.xml", 44),
        "initial_integration_failure_preserved": suite(
            ARCHIVE / "focused-integration-first.xml", 50, 2
        ),
        "focused_retry": suite(ARCHIVE / "focused-final.xml", 50),
        "whole_development": suite(args.full_xml, 2713),
    }
    require_same(handoff(), before)
    return {
        "scope": "engineering_complete_exact_reconstruction_acceptance",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "two_complete_source_isolated_reconstructions_verified": True,
        "independent_complete_synthetic_oracles_verified": True,
        "real_retained_diagnostic_spectrum_readback_verified": True,
        "upstream_primary_admission_simulated": True,
        "actual_checkpoint_backed_positive_authoring_verified": False,
        "strict_manuscript_consumer_integrated": False,
        "physical_cross_host_execution_verified": False,
        "formal_replication_ready": False,
        "manuscript_installed": False,
        "scientific_completion": False,
        "accepted_results": {name: binding(getattr(args, name)) for name in records},
        "tests": suites,
        "full_test_binding": binding(args.full_xml),
        "parent_acceptance": binding(PARENT),
        "prior_archive_bindings_preserved": prior,
        "prior_external_payloads_recursively_rechecked": False,
        "source_and_artifact_checks": checks,
        "unchanged_live_core": parent["unchanged_live_core"],
        "unchanged_manuscript_source": parent["unchanged_manuscript_source"],
        "unchanged_main_ledger": parent["unchanged_main_ledger"],
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
        or args.output.name != "validation.json"
    ):
        raise ValueError(
            "Require assertions, CPU-only operation and a new bounded acceptance receipt"
        )
    write_new(args.output, validate(args))
    print({"artifact_validation_passed": True, **file_identity(args.output)}, flush=True)
