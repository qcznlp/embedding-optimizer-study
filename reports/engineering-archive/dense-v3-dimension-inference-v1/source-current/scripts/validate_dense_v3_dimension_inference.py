"""Accept independent preparation evidence, never optimizer effects or release authority."""

import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_dimension_inference import PARENTS
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-v3-dimension-inference-v1")
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")


def validate(args):
    root = args.repository.resolve()
    archive = root / ARCHIVE
    parent_path, parent_sha = PARENTS["feature_acceptance"]
    assert file_identity(root / parent_path)["sha256"] == parent_sha
    parent = read_json(root / parent_path)
    preserved = [
        compare_binding(
            row,
            archive / "before" / row["path"] if row["path"] in DOCS else root / row["path"],
            root,
        )
        for row in parent["bindings"]
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
    assert second["relocated_replay"]["recomputed_bytes_verified"] is True
    for key in (
        "sources",
        "protocol",
        "task_reference",
        "bridge_reference",
        "synthetic_bundle",
        "actual_runtime",
    ):
        require_same(first[key], second[key])
    verify_file(
        root / "configs/dense_primary_v3_dimension_inference_protocol.json", first["protocol"]
    )
    for result in records.values():
        assert result["passed"] is True and result["primary_runs_admitted"] == 0
        for key in (
            "raw_primary_vectors_used",
            "upstream_primary_sources_independently_reconstructed",
            "full_primary_publication_pipeline_verified",
            "manuscript_installed",
            "scientific_completion",
        ):
            assert result[key] is False
        assert "ordinary retained run" in result["actual_primary_refusal"]
        assert len(result["actual_cli_refusals"]) == 2
        for row in result["actual_cli_refusals"]:
            assert row["exit_code"] == 1 and row["output_absent"] is True
            assert "ordinary retained run" in row["stderr"]
            target = Path(row["command"][row["command"].index("--output") + 1])
            assert not target.exists()
        task = result["task_reference"]
        assert len(task["contrasts"]) == 9 and len(task["rotations"]) == 27
        assert task["samples"] == 50000 and task["figure_points_checked"] == 68
        assert all(row["passed"] for row in (*task["contrasts"], *task["rotations"]))
        bridge = result["bridge_reference"]
        assert bridge["exact_predictions"] == 240
        assert bridge["exact_fold_mse_comparisons"] == len(bridge["systems"]) == 16
        assert bridge["exact_pooled_decisions"] == len(bridge["residual_associations"]) == 4
        assert all(row["passed"] for row in (*bridge["systems"], *bridge["residual_associations"]))
        assert len(result["altered_cases"]) == 6 and all(
            row["rejected"] for row in result["altered_cases"]
        )
        layout = result["synthetic_layout"]
        assert layout["exit_code"] == layout["fonts_exit_code"] == 0
        assert layout["type_3_fonts"] is layout["full_naacl_layout_acceptance"] is False
        require_same(result["post_execution_dispatchers"], handoff())
    tests = {}
    for filename, expected in (("focused-final.xml", 64), ("full-final.xml", 2334)):
        summary = next(ET.parse(archive / filename).iter("testsuite")).attrib
        assert int(summary["tests"]) == expected
        assert all(int(summary[key]) == 0 for key in ("errors", "failures", "skipped"))
        tests[filename] = summary
    external = {row["path"]: row for row in parent["external_payload_checks"]}
    for name, result in records.items():
        for row in (*result["sources"], result["protocol"], *result["artifacts"]):
            external[row["path"]] = row
        path = getattr(args, name)
        external[str(path)] = {"path": str(path), **file_identity(path)}
    checked = [compare_binding(row, Path(path), root) for path, row in sorted(external.items())]
    for row in parent["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            verify_file(base / row["path"], row)
    verify_file(root / "paper/main.tex", parent["unchanged_manuscript_source"])
    verify_file(Path(parent["unchanged_main_ledger"]["path"]), parent["unchanged_main_ledger"])
    return {
        "scope": "engineering_dense_v3_functional_inference_acceptance",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "synthetic_task_inference_and_named_bridge_independently_verified": True,
        "relocated_synthetic_numerical_replay_verified": True,
        "actual_primary_missing_population_refusals_verified": True,
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
        "unchanged_live_core": parent["unchanged_live_core"],
        "unchanged_manuscript_source": parent["unchanged_manuscript_source"],
        "unchanged_main_ledger": parent["unchanged_main_ledger"],
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
