"""Accept bounded bridge preparation evidence, never synthetic primary results."""

import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim import primary_v3_bridge as bridge
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_outcomes import inspect_bundle
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-v3-bridge-v1")
PRIOR = Path("reports/engineering-archive/dense-v3-exact-geometry-v1/validation.json")
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")
PROTOCOL_SHA = "540c3c3041ea13d9029fb5534283187b4f4141dfc66eba668938a7dc7d8ae803"


def validate(root, audit_path, audit_sha, replay_path, replay_sha):
    if sys.flags.optimize:
        raise ValueError("Evidence assertions must not be disabled")
    archive = root / ARCHIVE
    assert (
        file_identity(root / PRIOR)["sha256"]
        == "63076c3fcabbca857601b72f5dd55f83698d25134980b54f71304dbf517fa993"
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
        root / bridge.PARENTS["primary"][0], root, "/tmp/dense-identity-source.8rUgLF"
    )
    contract = bridge.BridgeContract.load(
        root / "configs/dense_primary_v3_bridge_protocol.json", primary
    )
    assert contract.sha256 == PROTOCOL_SHA
    verify_file(archive / "protocol.json", file_identity(contract.path))
    assert file_identity(audit_path)["sha256"] == audit_sha
    assert file_identity(replay_path)["sha256"] == replay_sha
    verify_file(archive / "rehearsal.json", file_identity(audit_path))
    verify_file(archive / "fresh-replay.json", file_identity(replay_path))
    audit, replay = read_json(audit_path), read_json(replay_path)
    for result in (audit, replay):
        assert result["rehearsal_passed"] is result["all_positive_panels_are_synthetic"] is True
        assert result["formal_bridge_produced"] is result["scientific_completion"] is False
        assert result["model_updates"] == result["gpu_workers"] == 0
        assert result["protocol"]["sha256"] == PROTOCOL_SHA
        assert len(result["altered_bundle_cases"]) == 4 and all(
            r["rejected"] for r in result["altered_bundle_cases"]
        )
        assert len(result["altered_protocol_cases"]) == 8 and all(
            r["rejected"] for r in result["altered_protocol_cases"]
        )
        assert {r["action"] for r in result["actual_missing_primary_cli"]} == {"build", "inspect"}
        assert all(
            r["returncode"] != 0
            and not r["output_created"]
            and "ordinary retained run" in r["stderr"]
            for r in result["actual_missing_primary_cli"]
        )
        require_same(handoff(), result["post_execution_dispatchers"])
        assert len(result["null_counterexample_corrections"]) == 2
        for case in result["null_counterexample_corrections"]:
            assert all(
                r["pooled_mse_reduction_exact"] == "0" and r["predictively_useful"] is False
                for r in case["tables"]["feature_prediction_summary"]
            )
            assert all(
                r["pearson_residual_association"] is None
                and r["spearman_residual_association"] is None
                for r in case["tables"]["residual_associations"]
            )
    assert audit["fresh_replay"] is False and replay["fresh_replay"] is True
    assert replay["replay_parent"]["sha256"] == audit_sha
    require_same(audit["complete_bundle_readback"], replay["complete_bundle_readback"])
    require_same(
        audit["null_counterexample_corrections"], replay["null_counterexample_corrections"]
    )
    oracle = audit["independent_full_system_controls"]
    assert oracle["independent_exact_rank_controls"] == 3
    assert oracle["exact_predictions"] == 540 and oracle["exact_fold_mse_comparisons"] == 36
    assert oracle["exact_pooled_decisions"] == 9 and len(oracle["residual_associations"]) == 9
    assert len(oracle["full_systems"]) == 36 and all(
        r["exact_equal"] for r in oracle["full_systems"]
    )
    assert all(r["passed"] for r in oracle["residual_associations"])
    fixture = read_json(audit_path.parent / "fixture-input.json")
    assert (
        fixture["not_primary_evidence"] is True
        and fixture["scope"] == "engineering_synthetic_bridge_inputs"
    )
    tables, numerical = bridge.bridge_tables(
        primary, fixture["checkpoints"], fixture["pairs"], fixture["scores"]
    )
    plan = audit["complete_bundle_readback"]["plan"]
    actual = inspect_bundle(
        audit_path.parent / "fixture-bundle",
        plan,
        {"fixture": fixture, "numerical_diagnostics": numerical},
        tables,
    )
    require_same(actual, audit["complete_bundle_readback"])
    failure = read_json(archive / "attempt-1/failure.json")
    assert failure["exit_code"] == 1 and failure["failure_overridden"] is False
    assert failure["analysis_source_changed"] is failure["protocol_or_tolerance_changed"] is False
    assert failure["protocol_sha256"] == PROTOCOL_SHA
    assert file_identity(archive / "attempt-1/protocol.json")["sha256"] == PROTOCOL_SHA
    assert (
        "sum(int(v < x)"
        in (archive / "attempt-1/source/scripts/audit_dense_v3_bridge.py").read_text()
    )
    failed_work = Path(failure["workdir"])
    require_same(read_json(failed_work / "fixture-input.json"), fixture)
    assert not (failed_work / "result.json").exists()
    tests = {}
    for name, count in (
        ("first-focused.xml", 28),
        ("focused-adapter.xml", 52),
        ("focused-final.xml", 54),
        ("full-tests.xml", 2105),
    ):
        attributes = next(ET.parse(archive / name).iter("testsuite")).attrib
        assert int(attributes["tests"]) == count
        assert all(int(attributes[key]) == 0 for key in ("failures", "errors", "skipped"))
        tests[name] = attributes
    checks = {row["path"]: row for row in prior["external_payload_checks"]}
    for result in (audit, replay):
        for row in (*result["inputs"], *result["artifacts"]):
            checks[row["path"]] = row
    for path in (audit_path, replay_path, *[p for p in failed_work.rglob("*") if p.is_file()]):
        checks[str(path)] = {"path": str(path), **file_identity(path)}
    external = [compare_binding(row, Path(path), root) for path, row in sorted(checks.items())]
    for row in prior["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            verify_file(base / row["path"], row)
    verify_file(root / "paper/main.tex", prior["unchanged_manuscript_source"])
    verify_file(Path(prior["unchanged_main_ledger"]["path"]), prior["unchanged_main_ledger"])
    return {
        "scope": "engineering_dense_v3_bridge_evidence_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "bridge_numerical_amendment_verified": True,
        "v3_bridge_consumer_prepared": True,
        "old_counterexample_corrected_in_new_consumer": True,
        "legacy_bridge_source_changed": False,
        "all_positive_panels_are_synthetic": True,
        "independent_reference_failure_preserved": True,
        "formal_bridge_produced": False,
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
                "configs/dense_primary_v3_bridge_protocol.json",
                "scripts/validate_dense_v3_bridge.py",
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
        raise ValueError("Require a new evidence receipt under an existing parent")
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
                "bridge_numerical_amendment_verified",
                "scientific_completion",
            )
        }
    )


if __name__ == "__main__":
    main()
