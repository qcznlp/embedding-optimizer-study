"""Accept the bounded synthetic original-geometry reconstruction, not a primary release."""

import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim import reconstruction_files as files
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_geometry import TABLE_COUNTS
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.geometry_reconstruction_controls import CASES
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-v3-geometry-reconstruction-v1")
PARENT = (
    "reports/engineering-archive/dense-v3-outcome-reconstruction-v1/validation.json",
    "41eb6c7cd643ef68071914ac1f3a793f6efa4542005ff9754ed1b0f625e203b9",
)
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")
FALSE_FLAGS = (
    "raw_weight_metrics_recomputed",
    "spectral_health_remeasured",
    "basis_orthogonality_remeasured",
    "checkpoint_tensors_revalidated",
    "model_encoding_repeated",
    "retrieval_repeated",
    "original_outcomes_recomputed",
    "raw_vector_features_recomputed",
    "bridge_and_functional_inference_recomputed",
    "primary_scientific_admission",
    "manuscript_installed",
    "scientific_completion",
)


def tests(path, expected, failures=0):
    value = next(ET.parse(path).iter("testsuite")).attrib
    assert int(value["tests"]) == expected and int(value["failures"]) == failures
    assert int(value["errors"]) == int(value["skipped"]) == 0
    return value


def check_audit(value, anchor, before):
    assert value["passed"] is True and value["scientific_completion"] is False
    assert value["upstream_primary_admission_simulated"] is True
    assert value["final_portable_publication_verified"] is False
    assert value["external_manifest_sha256"] == anchor
    child = value["cold_child"]
    assert child["exit_code"] == 0
    child = child["result"]
    assert child["network_and_original_paths_blocked"] is True
    assert len(child["imported_modules"]) >= 65
    assert all(
        not Path(path).is_absolute() and path.startswith("embed_optim/")
        for path in child["imported_modules"].values()
    )
    assert "/root/embedding-optimizer-study/src" in child["removed_search_paths"]
    receipt = child["receipt"]
    assert receipt["original_geometry_aggregate_numerics_verified"] is True
    assert receipt["upstream_admission_simulated"] is True
    require_same(receipt["table_counts"], TABLE_COUNTS)
    for key, count in {
        "checkpoints_reconstructed": 60,
        "raw_matrix_records_authenticated": 5280,
        "spectral_health_records_authenticated": 10560,
        "run_pairs_recomputed": 660,
    }.items():
        assert receipt[key] == count
    assert len(receipt["outputs"]) == 8
    for key in FALSE_FLAGS:
        assert receipt[key] is False
    oracle = value["independent_oracle"]
    assert oracle["passed"] is True and oracle["scientific_completion"] is False
    assert oracle["exact_set_overlap_pairs"] == 660 and oracle["undefined_pairs"] == 132
    assert oracle["native_coordinate_bases_checked"] == 14480
    changes = value["semantic_alterations_rejected"]
    require_same([r["case"] for r in changes], list(CASES))
    for row in changes:
        assert row["exit_code"] != 0 and row["success_receipt_absent"] is True
        assert "SHA-256 mismatch" not in row["stderr"]
        assert "Original producer/source path forbidden" not in row["stderr"]
        assert "Network access forbidden" not in row["stderr"]
        if row["case"] in {"geometry_table", "geometry_summary", "checkpoint_row", "basis_values"}:
            assert row["partial_numerical_output_retained"] is True
    assert "ordinary retained run" in value["actual_primary_authoring_refusal"]
    require_same(value["post_execution_dispatchers"], before)


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
    current_source = archive / "source-current"
    for path in current_source.rglob("*"):
        if path.is_file():
            verify_file(root / path.relative_to(current_source), file_identity(path))
    records = {}
    for name in ("fixture", "audit", "replay"):
        path, sha = getattr(args, name), getattr(args, name + "_sha256")
        assert file_identity(path)["sha256"] == sha
        verify_file(archive / ("complete-" + name + ".json"), file_identity(path))
        records[name] = read_json(path)
    fixture, first, second = (records[name] for name in ("fixture", "audit", "replay"))
    assert fixture["checkpoints"] == 60 and fixture["hidden_matrices_per_checkpoint"] == 88
    assert fixture["native_shapes_retained"] is True and fixture["basis_rank"] == 16
    assert fixture["raw_weight_spectra_and_entry_counts_simulated"] is True
    assert (
        fixture["upstream_primary_admission_simulated"] is True
        and fixture["scientific_completion"] is False
    )
    assert first["fresh_replay"] is False and second["fresh_replay"] is True
    assert second["replay_parent"]["sha256"] == args.audit_sha256
    require_same(first["sources"], second["sources"])
    require_same(first["cold_child"]["result"], second["cold_child"]["result"])
    before = handoff()
    for result in (first, second):
        check_audit(result, fixture["manifest_sha256"], before)
        assert result["fixture"]["sha256"] == args.fixture_sha256
    suites = {
        "focused-final.xml": tests(archive / "focused-final.xml", 67),
        "full-final.xml": tests(archive / "full-final.xml", 2548),
    }
    failures = {
        "zero-initial.xml": tests(archive / "attempt-1/zero-initial.xml", 10, 10),
        "focused-first.xml": tests(archive / "attempt-2/focused-failed.xml", 63, 1),
    }
    real = read_json(archive / "real-record-check-retry/result.json")
    assert real["passed"] is True and real["scientific_completion"] is False
    assert len(real["checks"]) == 9 and all(r["exact_checkpoint_row_match"] for r in real["checks"])
    assert sum(r["records"] for r in real["checks"]) == 792
    assert sum(r["health"] for r in real["checks"]) == 1584
    smoke = read_json(archive / "first-smoke/result.json")
    assert smoke["passed"] is True and smoke["cold_reconstruction_verified"] is False
    for row in smoke["sources"]:
        verify_file(archive / "source-initial" / Path(row["path"]).name, row)
    return finish(args, root, archive, previous, preserved, records, real, suites, failures, before)


def finish(args, root, archive, previous, preserved, records, real, suites, failures, before):
    external = {}
    for value in (*records.values(), real):
        for row in (
            *value.get("sources", []),
            *value.get("artifacts", []),
            *value.get("inputs", []),
        ):
            external[row["path"]] = row
    for name in records:
        path = getattr(args, name)
        external[str(path)] = {"path": str(path), **file_identity(path)}
    failed = Path("/tmp/dense-v3-geometry-failed-tests.myLpUE/focused-fixture")
    files.inspect(
        failed / "fixture/archive",
        "7785d0bdba2fe406066facad11ba1a9100e004c41cb721ab1ee8d92bbb3ec9db",
    )
    for path in failed.rglob("*"):
        if path.is_file():
            external[str(path)] = {"path": str(path), **file_identity(path)}
    checks = [compare_binding(row, Path(path), root) for path, row in sorted(external.items())]
    for row in previous["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            verify_file(base / row["path"], row)
    verify_file(root / "paper/main.tex", previous["unchanged_manuscript_source"])
    verify_file(Path(previous["unchanged_main_ledger"]["path"]), previous["unchanged_main_ledger"])
    require_same(handoff(), before)
    return {
        "scope": "engineering_synthetic_original_geometry_reconstruction_acceptance",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "synthetic_native_shape_geometry_reconstruction_verified": True,
        "independent_coordinate_basis_and_rational_aggregate_oracle_verified": True,
        "cold_relocated_network_and_producer_blocked_replay_verified": True,
        "recorded_spectral_health_authenticated_not_remeasured": True,
        "upstream_primary_admission_simulated": True,
        "physical_cross_host_execution_verified": False,
        "original_bridge_and_functional_inference_reconstruction_verified": False,
        "complete_primary_publication_pipeline_verified": False,
        "formal_replication_ready": False,
        "scientific_completion": False,
        "accepted_receipts": {
            name: {"path": str(getattr(args, name)), "sha256": getattr(args, name + "_sha256")}
            for name in records
        },
        "tests": suites,
        "preserved_failed_test_attempts": failures,
        "zero_statistic_consistency_check_added_without_kernel_change": True,
        "real_diagnostic_checkpoint_rows_rechecked": 9,
        "prior_acceptance": {"path": PARENT[0], "sha256": PARENT[1]},
        "prior_bindings_preserved": preserved,
        # Preserve the accepted parent and its direct bindings. Do not present a
        # recursive reread of old unrelated payloads as new numerical progress.
        "prior_external_payloads_rechecked": False,
        "external_payload_checks": checks,
        "unchanged_live_core": previous["unchanged_live_core"],
        "unchanged_manuscript_source": previous["unchanged_manuscript_source"],
        "unchanged_main_ledger": previous["unchanged_main_ledger"],
        "post_execution_dispatchers": before,
        "bindings": [
            identity(path, root)
            for path in sorted(archive.rglob("*"))
            if path.is_file() and path.name != "validation.json"
        ]
        + [identity(root / name, root) for name in DOCS],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "fixture", "audit", "replay", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("fixture-sha256", "audit-sha256", "replay-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if sys.flags.optimize or args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("Require assertions and a new bounded acceptance receipt")
    write_new(args.output, validate(args))
    print({"artifact_validation_passed": True, **file_identity(args.output)}, flush=True)


if __name__ == "__main__":
    main()
