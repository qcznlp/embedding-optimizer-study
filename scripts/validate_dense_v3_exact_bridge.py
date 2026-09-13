"""Validate exact-feature sensitivity preparation without promoting synthetic panels."""

import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim import primary_v3_exact_bridge as exact
from embed_optim.bridge_numerics import evaluate
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_outcomes import inspect_bundle
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-v3-exact-bridge-v1")
PRIOR = Path("reports/engineering-archive/dense-v3-bridge-v1/validation.json")
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")
PROTOCOL_SHA = "270a908d4242a831ce239f6d6258a19b3884329646f076c0af43ab968e5c9fdc"


def validate(root, audit_path, audit_sha, replay_path, replay_sha):
    if sys.flags.optimize:
        raise ValueError("Evidence assertions must not be disabled")
    archive = root / ARCHIVE
    assert (
        file_identity(root / PRIOR)["sha256"]
        == "f8a36ff600ba431fb71dee4c82edfa5aa87dae0513bd1e3b3ec186f81250110f"
    )
    prior = read_json(root / PRIOR)
    preserved = [
        compare_binding(
            row,
            archive / "before" / row["path"] if row["path"] in DOCS else root / row["path"],
            root,
        )
        for row in prior["bindings"]
    ]
    for path in (archive / "source").rglob("*"):
        if path.is_file():
            verify_file(root / path.relative_to(archive / "source"), file_identity(path))
    primary = PrimaryV3Contract.load(
        root / exact.PARENTS["primary"][0], root, "/tmp/dense-identity-source.8rUgLF"
    )
    contract = exact.ExactBridgeContract.load(
        root / "configs/dense_primary_v3_exact_bridge_protocol.json", primary
    )
    assert contract.sha256 == PROTOCOL_SHA
    verify_file(archive / "protocol.json", file_identity(contract.path))
    assert (
        file_identity(audit_path)["sha256"] == audit_sha
        and file_identity(replay_path)["sha256"] == replay_sha
    )
    verify_file(archive / "rehearsal.json", file_identity(audit_path))
    verify_file(archive / "fresh-replay.json", file_identity(replay_path))
    audit, replay = read_json(audit_path), read_json(replay_path)
    for result in (audit, replay):
        assert (
            result["rehearsal_passed"]
            is result["all_positive_retrieval_panels_are_synthetic"]
            is True
        )
        assert result["formal_exact_bridge_produced"] is result["scientific_completion"] is False
        assert result["model_updates"] == result["gpu_workers"] == 0
        assert result["protocol"]["sha256"] == PROTOCOL_SHA
        assert len(result["altered_bundle_cases"]) == 4 and all(
            row["rejected"] for row in result["altered_bundle_cases"]
        )
        assert len(result["altered_protocol_cases"]) == 10 and all(
            row["rejected"] for row in result["altered_protocol_cases"]
        )
        assert {row["action"] for row in result["actual_missing_primary_cli"]} == {
            "build",
            "inspect",
        }
        assert all(
            row["returncode"] != 0
            and not row["output_created"]
            and "ordinary retained run" in row["stderr"]
            for row in result["actual_missing_primary_cli"]
        )
        require_same(handoff(), result["post_execution_dispatchers"])
        mapped = result["real_diagnostic_geometry_mapping"]
        assert mapped["retrieval_outcomes_added"] is mapped["primary_identity_assigned"] is False
        assert (
            len(mapped["comparisons"]) == 27
            and mapped["undefined"] == 9
            and mapped["defined"] == 18
        )
        assert all(row["verified"] for row in mapped["comparisons"])
        assert "twelve-run primary" in mapped["diagnostic_pair_population_rejected"]
        dimension = result["dimension_numeric_control"]
        assert dimension["scope"] == "engineering_synthetic_dimension_numeric_control"
        assert dimension["primary_dimension_pipeline_integrated"] is False
        assert dimension["four_feature_slots_repeat_one_counterexample"] is True
        assert len(dimension["legacy_summary"]) == 4 and all(
            row["predictively_useful"] for row in dimension["legacy_summary"]
        )
        assert len(dimension["named_tables"]["feature_prediction_summary"]) == 4
        assert all(
            row["predictively_useful"] is False and row["pooled_mse_reduction_exact"] == "0"
            for row in dimension["named_tables"]["feature_prediction_summary"]
        )
    assert audit["fresh_replay"] is False and replay["fresh_replay"] is True
    assert replay["replay_parent"]["sha256"] == audit_sha
    for key in (
        "complete_bundle_readback",
        "dimension_numeric_control",
        "real_diagnostic_geometry_mapping",
    ):
        require_same(audit[key], replay[key])
    independent = audit["independent_full_system_controls"]
    assert (
        independent["exact_predictions"] == 300 and independent["exact_fold_mse_comparisons"] == 20
    )
    assert independent["exact_pooled_decisions"] == 5
    assert len(independent["systems"]) == 20 and all(
        row["exact_equal"] for row in independent["systems"]
    )
    assert len(independent["residual_associations"]) == 5 and all(
        row["passed"] for row in independent["residual_associations"]
    )
    fixture = read_json(audit_path.parent / "fixture-input.json")
    assert fixture["not_primary_evidence"] is True
    old_tables, _ = evaluate(fixture["original_rows"])
    tables, metadata = exact.sensitivity_tables(
        primary, old_tables, fixture["checkpoints"], fixture["pairs"]
    )
    actual = inspect_bundle(
        audit_path.parent / "fixture-bundle",
        audit["complete_bundle_readback"]["plan"],
        {"fixture": fixture, **metadata},
        tables,
    )
    require_same(actual, audit["complete_bundle_readback"])
    for name in exact.TABLE_COUNTS:
        verify_file(
            archive / f"{name}.csv",
            file_identity(audit_path.parent / "fixture-bundle" / f"{name}.csv"),
        )
    tests = {}
    for name, count in (
        ("first-focused.xml", 23),
        ("focused-final.xml", 48),
        ("full-tests.xml", 2153),
    ):
        attributes = next(ET.parse(archive / name).iter("testsuite")).attrib
        assert int(attributes["tests"]) == count
        assert all(int(attributes[key]) == 0 for key in ("failures", "errors", "skipped"))
        tests[name] = attributes
    checks = {row["path"]: row for row in prior["external_payload_checks"]}
    for result in (audit, replay):
        for row in (*result["inputs"], *result["artifacts"]):
            checks[row["path"]] = row
    for path in (audit_path, replay_path):
        checks[str(path)] = {"path": str(path), **file_identity(path)}
    external = [compare_binding(row, Path(path), root) for path, row in sorted(checks.items())]
    for row in prior["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            verify_file(base / row["path"], row)
    verify_file(root / "paper/main.tex", prior["unchanged_manuscript_source"])
    verify_file(Path(prior["unchanged_main_ledger"]["path"]), prior["unchanged_main_ledger"])
    return {
        "scope": "engineering_dense_v3_exact_bridge_evidence_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "exact_feature_sensitivity_consumer_prepared": True,
        "named_family_numerics_verified": True,
        "dimension_numeric_counterexample_corrected_in_named_helper": True,
        "primary_dimension_pipeline_integrated": False,
        "all_positive_retrieval_panels_are_synthetic": True,
        "formal_exact_bridge_produced": False,
        "formal_replication_ready": False,
        "scientific_completion": False,
        "accepted_audit_sha256": audit_sha,
        "accepted_replay_sha256": replay_sha,
        "prior_bindings_preserved": preserved,
        "external_payload_checks": external,
        "tests": tests,
        "unchanged_live_core": prior["unchanged_live_core"],
        "unchanged_manuscript_source": prior["unchanged_manuscript_source"],
        "unchanged_main_ledger": prior["unchanged_main_ledger"],
        "post_execution_dispatchers": handoff(),
        "bindings": [
            identity(path, root)
            for path in sorted(archive.rglob("*"))
            if path.is_file() and path.name != "validation.json"
        ]
        + [
            identity(root / name, root)
            for name in (
                *DOCS,
                "configs/dense_primary_v3_exact_bridge_protocol.json",
                "scripts/validate_dense_v3_exact_bridge.py",
            )
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "audit", "replay", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("audit-sha256", "replay-sha256"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("Require a new receipt under an explicit existing parent")
    result = validate(
        args.repository.resolve(),
        args.audit.resolve(),
        args.audit_sha256,
        args.replay.resolve(),
        args.replay_sha256,
    )
    write_new(args.output, result)
    print(
        {
            key: result[key]
            for key in (
                "artifact_validation_passed",
                "exact_feature_sensitivity_consumer_prepared",
                "scientific_completion",
            )
        }
    )


if __name__ == "__main__":
    main()
