"""Accept source/transport preparation only, retaining all missing numerical/release gates."""

import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_reconstruction_authoring import ACCEPTANCE
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-v3-portable-reconstruction-v1")
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")


def validate(args):
    root = args.repository.resolve()
    archive = root / ARCHIVE
    previous_path = root / ACCEPTANCE[0]
    assert file_identity(previous_path)["sha256"] == ACCEPTANCE[1]
    previous = read_json(previous_path)
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
    for name in ("audit", "replay"):
        path, sha = getattr(args, name), getattr(args, name + "_sha256")
        assert file_identity(path)["sha256"] == sha
        verify_file(archive / (name + ".json"), file_identity(path))
        records[name] = read_json(path)
    first, second = records["audit"], records["replay"]
    assert first["fresh_replay"] is False and second["fresh_replay"] is True
    assert second["replay_parent"]["sha256"] == args.audit_sha256
    for key in ("sources", "actual_payload", "synthetic_payload"):
        require_same(first[key], second[key])
    require_same(first["fresh_child"]["result"], second["fresh_child"]["result"])
    for result in records.values():
        assert result["passed"] is True
        assert result["primary_runs_or_vectors_admitted"] == 0
        assert result["authoring_upstream_admission_simulated_in_positive_fixture"] is True
        for key in (
            "raw_vector_numerical_reconstruction_verified",
            "final_portable_publication_verified",
            "manuscript_installed",
            "scientific_completion",
        ):
            assert result[key] is False
        assert result["actual_payload"]["files"] == 238
        assert result["synthetic_payload"]["files"] == 251
        assert result["fresh_child"]["exit_code"] == 0
        child = result["fresh_child"]["result"]
        assert len(child["run_identities"]) == 12 and len(child["imported_modules"]) == 69
        assert child["output_absent"] is True and child["scientific_completion"] is False
        assert "ordinary retained run" in child["actual_primary_refusal"]
        assert "ordinary retained run" in result["actual_local_primary_refusal"]
        assert len(result["alterations"]) == 4 and all(
            row["rejected"] for row in result["alterations"]
        )
        require_same(result["post_execution_dispatchers"], handoff())
    tests = {}
    for filename, count in (("focused-final.xml", 47), ("full-final.xml", 2381)):
        summary = next(ET.parse(archive / filename).iter("testsuite")).attrib
        assert int(summary["tests"]) == count
        assert all(int(summary[key]) == 0 for key in ("errors", "failures", "skipped"))
        tests[filename] = summary
    external = {row["path"]: row for row in previous["external_payload_checks"]}
    for name, result in records.items():
        for row in (*result["sources"], *result["artifacts"]):
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
        "scope": "engineering_v3_source_closure_authoring_transport_acceptance",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "actual_source_only_relocated_contract_loading_verified": True,
        "synthetic_local_authoring_transport_verified": True,
        "raw_vector_numerical_reconstruction_verified": False,
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
    for name in ("repository", "audit", "replay", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("audit-sha256", "replay-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if sys.flags.optimize or args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("Require assertions and a new acceptance receipt")
    write_new(args.output, validate(args))
    print({"artifact_validation_passed": True, **file_identity(args.output)}, flush=True)


if __name__ == "__main__":
    main()
