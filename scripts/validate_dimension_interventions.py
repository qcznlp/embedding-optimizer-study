"""Accept bounded intervention/input evidence without admitting primary model results."""

import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.audit_dimension_interventions import PARENT, PARENT_SHA
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-v3-dimension-interventions-v1")
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")


def validate(args):
    root = args.repository.resolve()
    archive = root / ARCHIVE
    assert file_identity(root / PARENT)["sha256"] == PARENT_SHA
    previous = read_json(root / PARENT)
    preserved = [
        compare_binding(
            row,
            archive / "before" / row["path"] if row["path"] in DOCS else root / row["path"],
            root,
        )
        for row in previous["bindings"]
    ]
    for path in (archive / "source").rglob("*"):
        if path.is_file():
            verify_file(root / path.relative_to(archive / "source"), file_identity(path))
    records = {}
    for name in ("audit", "replay", "probe", "probe_replay"):
        path, sha = getattr(args, name), getattr(args, name + "_sha256")
        assert file_identity(path)["sha256"] == sha
        verify_file(archive / f"{name}.json", file_identity(path))
        records[name] = read_json(path)
    first, second = records["audit"], records["replay"]
    for result in (first, second):
        assert result["rehearsal_passed"] is True
        assert result["primary_pipeline_integrated"] is result["scientific_completion"] is False
        assert result["model_updates"] == result["gpu_workers"] == 0
        assert result["plan"]["historical_float16_input_is_primary"] is False
        independent = result["independent"]
        assert independent["coordinate_query_rank_checks"] == 688128
        assert independent["shared_mask_query_rank_checks"] == 17920
        assert independent["all_rank_checks_passed"] is True
        assert set(independent["bases"]) == {
            "native",
            "rotation_314159",
            "rotation_271828",
            "rotation_161803",
        }
        assert all(
            row["all_positive_ranks_identical"] and row["coordinate_query_rank_checks"] == 172032
            for row in independent["bases"].values()
        )
        assert len(result["altered_cases"]) == 5 and all(
            row["rejected"] for row in result["altered_cases"]
        )
        assert result["legacy_comparison"]["native_ndcg_changed_cells"] == 0
        assert "zero-norm" in result["new_zero_remainder_refusal"]
        require_same(handoff(), result["post_execution_dispatchers"])
    assert first["fresh_replay"] is False and second["fresh_replay"] is True
    assert second["replay_parent"]["sha256"] == args.audit_sha256
    for key in ("plan", "sources", "inputs", "independent", "legacy_boundary", "legacy_comparison"):
        require_same(first[key], second[key])
    probe, probe_replay = records["probe"], records["probe_replay"]
    for result in (probe, probe_replay):
        assert result["input_validation_passed"] is True
        assert result["verified_text_fields"] == 2016 and len(result["row_identities"]) == 224
        assert result["negative_priority_regenerated"] is result["scientific_completion"] is False
        assert result["model_updates"] == result["gpu_workers"] == 0
        assert len(result["tasks"]) == 14
        assert all(
            task["rows"] == 16 and task["verified_text_fields"] == 144 and len(task["files"]) == 3
            for task in result["tasks"]
        )
        assert all(
            row["sha256"] == row["immutable_hf_lfs_sha256"]
            for task in result["tasks"]
            for row in task["files"]
        )
        require_same(handoff(), result["post_execution_dispatchers"])
    assert probe["fresh_replay"] is False and probe_replay["fresh_replay"] is True
    assert probe_replay["replay_parent"]["sha256"] == args.probe_sha256
    for key in ("source", "inputs", "tasks", "row_identities", "row_identities_sha256"):
        require_same(probe[key], probe_replay[key])
    old_wrapper = archive / "probe-attempt-1/source/scripts/audit_dimension_probe_inputs.py"
    assert (
        file_identity(old_wrapper)["sha256"]
        == "4db503595afe8c65496159e2375f869273eaf2e595448da22b87e5b55584e7b3"
    )
    corrected = old_wrapper.read_text().replace(
        "        actual = binding(root / name)",
        "        # HF snapshots legitimately link to content-addressed cache blobs. Resolve\n"
        "        # that one upstream cache role, then retain the ordinary-file guard and\n"
        "        # independently compare its actual bytes with the pinned remote digest.\n"
        "        actual = binding((root / name).resolve(strict=True))",
    )
    assert corrected == (root / "scripts/audit_dimension_probe_inputs.py").read_text()
    assert not Path("/tmp/dense-v3-dimension-probe.2d7A6Y/result.json").exists()
    boundary = read_json(archive / "deletion-boundary.json")
    require_same(boundary["controls"], first["legacy_boundary"])
    for name, bound in boundary["sources"].items():
        verify_file(root / name, bound)
    tests = {}
    for name, count in (
        ("first-focused.xml", 28),
        ("focused-final.xml", 42),
        ("full-tests.xml", 2195),
        ("probe-focused.xml", 20),
        ("full-final.xml", 2215),
    ):
        result = next(ET.parse(archive / name).iter("testsuite")).attrib
        assert int(result["tests"]) == count
        assert all(int(result[k]) == 0 for k in ("errors", "failures", "skipped"))
        tests[name] = result
    external = {row["path"]: row for row in previous["external_payload_checks"]}
    for result in (first, second):
        for row in (*result["sources"], *result["inputs"], *result["artifacts"]):
            external[row["path"]] = row
    for result in (probe, probe_replay):
        for row in (
            *result["source"],
            *result["inputs"],
            *(record for task in result["tasks"] for record in task["files"]),
        ):
            external[row["path"]] = row
    for name in records:
        path = getattr(args, name)
        external[str(path)] = {"path": str(path), **file_identity(path)}
    checked = [compare_binding(row, Path(path), root) for path, row in sorted(external.items())]
    for row in previous["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            verify_file(base / row["path"], row)
    verify_file(root / "paper/main.tex", previous["unchanged_manuscript_source"])
    verify_file(Path(previous["unchanged_main_ledger"]["path"]), previous["unchanged_main_ledger"])
    return {
        "scope": "engineering_dense_dimension_intervention_and_probe_acceptance",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "literal_intervention_kernel_verified": True,
        "fixed_probe_text_and_order_verified": True,
        "primary_dimension_pipeline_integrated": False,
        "formal_replication_ready": False,
        "scientific_completion": False,
        "accepted_receipts": {
            name: {"path": str(getattr(args, name)), "sha256": getattr(args, name + "_sha256")}
            for name in records
        },
        "prior_bindings_preserved": preserved,
        "external_payload_checks": checked,
        "tests": tests,
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
    for name in ("repository", "output", "audit", "replay", "probe", "probe-replay"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("audit", "replay", "probe", "probe-replay"):
        parser.add_argument(f"--{name}-sha256", required=True)
    args = parser.parse_args()
    if sys.flags.optimize or args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("Require a fresh exact acceptance receipt")
    result = validate(args)
    write_new(args.output, result)
    print(
        {
            "artifact_validation_passed": True,
            "primary_dimension_pipeline_integrated": False,
            **file_identity(args.output),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
