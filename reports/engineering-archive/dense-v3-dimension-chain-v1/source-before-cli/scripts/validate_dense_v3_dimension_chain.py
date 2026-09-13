"""Accept the bounded real-input and synthetic full-chain evidence, not primary findings."""

import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_dimension_contract import PARENTS, TABLE_COUNTS
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-v3-dimension-chain-v1")
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")


def validate(args):
    root = args.repository.resolve()
    archive = root / ARCHIVE
    parent_path, parent_sha = PARENTS["kernel_acceptance"]
    assert file_identity(root / parent_path)["sha256"] == parent_sha
    previous = read_json(root / parent_path)
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
        verify_file(archive / f"{name}.json", file_identity(path))
        records[name] = read_json(path)
    first, second = records["audit"], records["replay"]
    verify_file(root / "configs/dense_primary_v3_dimensions_protocol.json", first["inputs"][0])
    for result in records.values():
        assert result["rehearsal_passed"] is True
        assert (
            result["primary_publication_pipeline_integrated"]
            is result["scientific_completion"]
            is False
        )
        assert (
            result["primary_models_encoded"]
            == result["gpu_workers"]
            == result["model_updates"]
            == 0
        )
        loading = result["actual_loading"]
        require_same(
            result["actual_runtime"]["packages"],
            {
                "torch": "2.9.1+cu129",
                "numpy": "2.5.2",
                "scipy": "1.17.1",
                "sentence-transformers": "5.7.0",
                "transformers": "5.3.0",
                "accelerate": "1.13.0",
            },
        )
        assert len(result["pinned_stack_bindings"]) == 4
        for row in result["pinned_stack_bindings"]:
            assert file_identity(Path(row["path"]))["sha256"] == row["sha256"]
        assert loading["actual_saved_tensor_equalities"] == 134
        assert loading["formal_flash_attention_execution_verified"] is False
        assert len(loading["observation"]["state_tensors"]) == 134
        assert len(result["actual_probe"]["row_identities"]) == 224
        assert len(result["actual_probe"]["files"]) == 5
        assert (
            result["actual_probe"]["row_identities_sha256"]
            == "c30be6bd9b86f34c160170168014d904446a857a9eda65cbd05080b9ab1d4344"
        )
        assert len(result["actual_primary_refusals"]) == 3 and all(
            row["rejected"] for row in result["actual_primary_refusals"]
        )
        scope = result["synthetic_scope"]
        assert scope["upstream_run_reference_probe_admission_simulated"] is True
        assert scope["kernel_fixture_dimension"] == 8
        assert scope["attribution_schema_expansion_is_not_a_768D_numerical_check"] is True
        assert scope["real_768D_numerical_parent"] == parent_sha
        assert len(result["synthetic_export"]["states"]) == 61
        assert result["synthetic_export"]["model_encoding_repeated"] is False
        feature = result["synthetic_features"]
        assert feature["raw_vector_states_recomputed"] == 61
        assert feature["plan"]["publication_inference_verified"] is False
        require_same(feature["table_counts"], TABLE_COUNTS)
        assert len(result["altered_cases"]) == 2 and all(
            row["rejected"] for row in result["altered_cases"]
        )
        require_same(handoff(), result["post_execution_dispatchers"])
    assert first["fresh_replay"] is False and second["fresh_replay"] is True
    assert second["replay_parent"]["sha256"] == args.audit_sha256
    for key in (
        "sources",
        "inputs",
        "actual_probe",
        "actual_loading",
        "synthetic_scope",
        "synthetic_export",
        "synthetic_features",
        "actual_runtime",
        "pinned_stack_bindings",
    ):
        require_same(first[key], second[key])
    assert not Path("/tmp/dense-v3-dimension-chain-audit.JzeFAi/result.json").exists()
    assert (
        file_identity(archive / "attempt-1/protocol.json")["sha256"]
        == "42ee1128fe376ead2ecccdcb387ca2bee3f62dedb57b401cf17d85d435843444"
    )
    tests = {}
    for name, count in (
        ("focused-final.xml", args.focused_count),
        ("full-final.xml", args.full_count),
    ):
        result = next(ET.parse(archive / name).iter("testsuite")).attrib
        assert int(result["tests"]) == count
        assert all(int(result[k]) == 0 for k in ("errors", "failures", "skipped"))
        tests[name] = result
    first_suite = next(ET.parse(archive / "attempt-1/full.xml").iter("testsuite")).attrib
    assert (
        int(first_suite["tests"]),
        int(first_suite["failures"]),
        int(first_suite["errors"]),
    ) == (2268, 3, 1)
    regressions = next(ET.parse(archive / "final-failure-regressions.xml").iter("testsuite")).attrib
    assert int(regressions["tests"]) == 4 and all(
        int(regressions[k]) == 0 for k in ("failures", "errors", "skipped")
    )
    tests["initial_full_attempt_preserved"] = first_suite
    tests["final_compatibility_regressions"] = regressions
    assert args.full_count == 2215 + args.focused_count
    external = {row["path"]: row for row in previous["external_payload_checks"]}
    for name, result in records.items():
        for row in (*result["sources"], *result["inputs"], *result["artifacts"]):
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
        "scope": "engineering_dense_v3_dimension_export_feature_acceptance",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "actual_pretrained_loading_and_probe_verified": True,
        "complete_synthetic_export_feature_wiring_verified": True,
        "complete_primary_dimension_publication_pipeline_verified": False,
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
    for name in ("repository", "output", "audit", "replay"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("audit", "replay"):
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--focused-count", type=int, required=True)
    parser.add_argument("--full-count", type=int, required=True)
    args = parser.parse_args()
    if sys.flags.optimize or args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("Require a new exact acceptance receipt")
    result = validate(args)
    write_new(args.output, result)
    print({"artifact_validation_passed": True, **file_identity(args.output)}, flush=True)


if __name__ == "__main__":
    main()
