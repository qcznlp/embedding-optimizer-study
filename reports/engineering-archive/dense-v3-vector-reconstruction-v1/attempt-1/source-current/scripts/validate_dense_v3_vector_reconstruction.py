"""Accept bounded raw-vector reconstruction evidence; never primary publication."""

import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_dimension_contract import TABLE_COUNTS
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-v3-vector-reconstruction-v1")
PARENT = (
    "reports/engineering-archive/dense-v3-portable-reconstruction-v1/validation.json",
    "bc18daa331507bc95481f7e99430157906d1c8ed4e3eebb5848915f872f283f0",
)
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")


def validate(args):
    root = args.repository.resolve()
    archive = root / ARCHIVE
    path = root / PARENT[0]
    assert file_identity(path)["sha256"] == PARENT[1]
    previous = read_json(path)
    preserved = [
        compare_binding(
            row,
            archive / "before" / row["path"] if row["path"] in DOCS else root / row["path"],
            root,
        )
        for row in previous["bindings"]
    ]
    for path in (archive / "source-current").rglob("*"):
        if path.is_file():
            verify_file(root / path.relative_to(archive / "source-current"), file_identity(path))
    records = {}
    for name in ("fixture", "audit", "replay"):
        path, sha = getattr(args, name), getattr(args, name + "_sha256")
        assert file_identity(path)["sha256"] == sha
        verify_file(archive / (name + ".json"), file_identity(path))
        records[name] = read_json(path)
    fixture, first, second = (records[name] for name in ("fixture", "audit", "replay"))
    assert fixture["full_dimension"] == 768 and fixture["states"] == 61
    assert (
        fixture["upstream_admission_simulated"] is True
        and fixture["scientific_completion"] is False
    )
    assert first["fresh_replay"] is False and second["fresh_replay"] is True
    assert second["replay_parent"]["sha256"] == args.audit_sha256
    require_same(first["sources"], second["sources"])
    require_same(first["cold_child"]["result"], second["cold_child"]["result"])
    for result in (first, second):
        assert result["passed"] is True
        assert result["fixture"]["sha256"] == args.fixture_sha256
        assert result["external_manifest_sha256"] == fixture["manifest_sha256"]
        assert result["full_width"] == 768 and result["raw_states_recomputed"] == 61
        assert result["upstream_admission_simulated"] is True
        assert (
            result["scientific_completion"] is False
            and result["final_portable_publication_verified"] is False
        )
        assert result["cold_child"]["exit_code"] == 0
        child = result["cold_child"]["result"]
        assert (
            child["network_and_original_paths_blocked"] is True
            and len(child["imported_modules"]) >= 65
        )
        numerical = child["receipt"]
        assert numerical["raw_vector_states_recomputed"] == 61
        require_same(numerical["table_counts"], TABLE_COUNTS)
        assert len(numerical["state_checks"]) == 61 and len(numerical["outputs"]) == 432
        assert (
            numerical["exact_numerics_verified"] is True
            and numerical["comparison_tolerance"] is None
        )
        assert numerical["upstream_admission_simulated"] is True
        for key in (
            "model_encoding_repeated",
            "checkpoint_tensors_revalidated",
            "raw_probe_text_revalidated",
            "outcome_geometry_primitives_recomputed",
            "publication_inference_recomputed",
            "primary_scientific_admission",
            "manuscript_installed",
            "scientific_completion",
        ):
            assert numerical[key] is False
        changes = result["numerical_alterations_rejected"]
        assert [row["case"] for row in changes] == ["table", "attribution"]
        for row in changes:
            assert row["exit_code"] != 0 and row["success_receipt_absent"] is True
            assert row["partial_numerical_output_retained"] is True
            assert "differs from fresh numerical computation" in row["stderr"]
        assert "ordinary retained run" in result["actual_primary_authoring_refusal"]
        require_same(result["post_execution_dispatchers"], handoff())
    tests = {}
    for filename, count in (("focused-final.xml", 37), ("full-final.xml", 2418)):
        summary = next(ET.parse(archive / filename).iter("testsuite")).attrib
        assert int(summary["tests"]) == count
        assert all(int(summary[key]) == 0 for key in ("failures", "errors", "skipped"))
        tests[filename] = summary
    external = {row["path"]: row for row in previous["external_payload_checks"]}
    for name, result in records.items():
        for row in (*result.get("sources", []), *result.get("artifacts", [])):
            external[row["path"]] = row
        path = getattr(args, name)
        external[str(path)] = {"path": str(path), **file_identity(path)}
    checked = [compare_binding(row, Path(path), root) for path, row in sorted(external.items())]
    for row in previous["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            verify_file(base / row["path"], row)
    verify_file(root / "paper/main.tex", previous["unchanged_manuscript_source"])
    verify_file(Path(previous["unchanged_main_ledger"]["path"]), previous["unchanged_main_ledger"])
    return {
        "scope": "engineering_full_768_v3_raw_vector_reconstruction_acceptance",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "synthetic_full_768_raw_vector_reconstruction_verified": True,
        "cold_relocated_network_and_producer_blocked_replay_verified": True,
        "upstream_primary_admission_simulated": True,
        "original_outcome_geometry_and_inference_reconstruction_verified": False,
        "complete_primary_dimension_publication_pipeline_verified": False,
        "formal_replication_ready": False,
        "scientific_completion": False,
        "accepted_receipts": {
            name: {"path": str(getattr(args, name)), "sha256": getattr(args, name + "_sha256")}
            for name in records
        },
        "tests": tests,
        "prior_bindings_preserved": preserved,
        "external_payload_checks": checked,
        "unchanged_live_core": previous["unchanged_live_core"],
        "unchanged_manuscript_source": previous["unchanged_manuscript_source"],
        "unchanged_main_ledger": previous["unchanged_main_ledger"],
        "post_execution_dispatchers": handoff(),
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
        raise ValueError("Require assertions and a new acceptance receipt")
    write_new(args.output, validate(args))
    print({"artifact_validation_passed": True, **file_identity(args.output)}, flush=True)


if __name__ == "__main__":
    main()
