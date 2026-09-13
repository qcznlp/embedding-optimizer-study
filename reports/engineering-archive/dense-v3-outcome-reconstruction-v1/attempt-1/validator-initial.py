"""Accept only the bounded synthetic local raw-outcome reconstruction milestone."""

import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_outcomes import TABLE_COUNTS
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.outcome_reconstruction_controls import CASES
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-v3-outcome-reconstruction-v1")
PARENT = (
    "reports/engineering-archive/dense-v3-vector-reconstruction-v1/validation.json",
    "a5ec82bafa467462dc17600ae2c041e0cf5f67f988bc028a0983d5e47c82aed1",
)
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")


def test_record(path, count, failures=0):
    value = next(ET.parse(path).iter("testsuite")).attrib
    assert int(value["tests"]) == count and int(value["failures"]) == failures
    assert int(value["errors"]) == 0 and int(value["skipped"]) == 0
    return value


def check_audit(result, fixture, before):
    assert result["passed"] is True
    assert result["fixture"]["sha256"] == file_identity(fixture)["sha256"]
    value = read_json(fixture)
    assert result["external_manifest_sha256"] == value["manifest_sha256"]
    assert result["upstream_primary_admission_simulated"] is True
    assert result["scientific_completion"] is False
    assert result["final_portable_publication_verified"] is False
    assert result["cold_child"]["exit_code"] == 0
    child = result["cold_child"]["result"]
    assert child["network_and_original_paths_blocked"] is True
    assert len(child["imported_modules"]) >= 65
    assert all(
        not Path(path).is_absolute() and path.startswith("embed_optim/")
        for path in child["imported_modules"].values()
    )
    assert "/root/embedding-optimizer-study/src" in child["removed_search_paths"]
    numerical = child["receipt"]
    require_same(numerical["table_counts"], TABLE_COUNTS)
    assert numerical["original_outcome_numerics_verified"] is True
    assert numerical["validation_jobs"] == 12
    assert numerical["validation_records_replayed"] == 49152
    assert numerical["validation_selection_recomputed"] is True
    assert numerical["beir_task_results_reparsed"] == 840
    assert numerical["upstream_admission_simulated"] is True
    assert len(numerical["outputs"]) == 14
    for key in (
        "validation_replay_changes_recorded_metrics",
        "checkpoint_tensors_revalidated",
        "model_encoding_repeated",
        "retrieval_repeated",
        "raw_validation_text_revalidated",
        "timing_remeasured",
        "raw_vector_features_recomputed",
        "geometry_primitives_recomputed",
        "functional_inference_recomputed",
        "primary_scientific_admission",
        "manuscript_installed",
        "scientific_completion",
    ):
        assert numerical[key] is False
    changes = result["semantic_alterations_rejected"]
    require_same([row["case"] for row in changes], list(CASES))
    for row in changes:
        assert row["exit_code"] != 0 and row["success_receipt_absent"] is True
        assert "Network access forbidden" not in row["stderr"]
        assert "Original producer/source path forbidden" not in row["stderr"]
        if row["case"] == "outcome_table":
            assert row["partial_numerical_output_retained"] is True
            assert "differs from fresh evidence/statistical recomputation" in row["stderr"]
    assert "ordinary retained run" in result["actual_primary_authoring_refusal"]
    require_same(result["post_execution_dispatchers"], before)


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
    assert fixture["actual_validation_row_identities_retained"] is True
    assert fixture["validation_records"] == 49152 and fixture["beir_cells"] == 840
    assert fixture["upstream_primary_admission_simulated"] is True
    assert fixture["scientific_completion"] is False
    assert first["fresh_replay"] is False and second["fresh_replay"] is True
    assert second["replay_parent"]["sha256"] == args.audit_sha256
    require_same(first["sources"], second["sources"])
    require_same(first["cold_child"]["result"], second["cold_child"]["result"])
    before = handoff()
    for result in (first, second):
        check_audit(result, args.fixture, before)
    tests = {
        "focused-final.xml": test_record(archive / "focused-final.xml", 62),
        "full-final.xml": test_record(archive / "full-final.xml", 2481),
    }
    failures = {
        "focused": test_record(archive / "attempt-1/focused-failed.xml", 62, 1),
        "full": test_record(archive / "attempt-1/full-failed.xml", 2481, 1),
    }
    error = (archive / "attempt-1/cold-stderr.txt").read_text()
    assert "Original producer/source path forbidden" in error
    failed_fixture = read_json(archive / "attempt-1/fixture.json")
    for row in failed_fixture["sources"]:
        local = Path(row["path"]).relative_to(root)
        verify_file(archive / "attempt-1/source" / local, row)
    # Only the test and cold-audit launcher changed after their first failed attempt.
    for name in ("primary_v3_outcome_primitives.py", "primary_v3_outcome_reconstruction.py"):
        for snapshot in ("source-initial", "attempt-1/source", "source-current"):
            verify_file(
                archive / snapshot / "src/embed_optim" / name,
                file_identity(root / "src/embed_optim" / name),
            )
    return finish(args, root, archive, previous, preserved, records, tests, failures, before)


def finish(args, root, archive, previous, preserved, records, tests, failures, before):
    external = {row["path"]: row for row in previous["external_payload_checks"]}
    raw = read_json(archive / "fixture.json")
    for value in (raw, *records.values()):
        for row in (*value.get("sources", []), *value.get("artifacts", [])):
            external[row["path"]] = row
    for name in records:
        path = getattr(args, name)
        external[str(path)] = {"path": str(path), **file_identity(path)}
    failed_root = Path("/tmp/dense-v3-outcome-reconstruction-cold.EDmj0q")
    for path in failed_root.rglob("*"):
        if path.is_file():
            external[str(path)] = {"path": str(path), **file_identity(path)}
    checked = [compare_binding(row, Path(path), root) for path, row in sorted(external.items())]
    for row in previous["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            verify_file(base / row["path"], row)
    verify_file(root / "paper/main.tex", previous["unchanged_manuscript_source"])
    verify_file(Path(previous["unchanged_main_ledger"]["path"]), previous["unchanged_main_ledger"])
    require_same(handoff(), before)
    return {
        "scope": "engineering_synthetic_original_outcome_reconstruction_acceptance",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "synthetic_raw_outcome_reconstruction_verified": True,
        "all_original_validation_records_and_beir_cells_reparsed": True,
        "cold_relocated_network_and_producer_blocked_replay_verified": True,
        "upstream_primary_admission_simulated": True,
        "physical_cross_host_execution_verified": False,
        "original_geometry_and_functional_inference_reconstruction_verified": False,
        "complete_primary_publication_pipeline_verified": False,
        "formal_replication_ready": False,
        "scientific_completion": False,
        "accepted_receipts": {
            name: {"path": str(getattr(args, name)), "sha256": getattr(args, name + "_sha256")}
            for name in records
        },
        "tests": tests,
        "preserved_failed_test_attempts": failures,
        "preserved_first_cold_import_refusal": True,
        "production_outcome_readers_unchanged_since_first_smoke": True,
        "prior_bindings_preserved": preserved,
        "external_payload_checks": checked,
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
