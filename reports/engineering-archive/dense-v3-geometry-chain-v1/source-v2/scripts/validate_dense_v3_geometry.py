"""Validate complete geometry preparation and bounded real diagnostic evidence."""

import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_geometry import IMPLEMENTATION_REVISION, GeometryContract
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-v3-geometry-chain-v1")
PRIOR = Path("reports/engineering-archive/dense-v3-geometry-v1/validation.json")
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")
PROTOCOL_SHA = "9b6dad688072c08e8994e45f80c897a7889ab5fbcbccab27b643944c85ae625b"


def validate(root, rehearsal_sha, replay_sha):
    if sys.flags.optimize:
        raise ValueError("Do not disable evidence assertions")
    archive = root / ARCHIVE
    assert (
        file_identity(root / PRIOR)["sha256"]
        == "3b8575f2818c2b4d9f35f93610b4a0100078bb854941af6e0b61da7f76236df9"
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
    for path in (archive / "source-v2").rglob("*"):
        if path.is_file():
            verify_file(root / path.relative_to(archive / "source-v2"), file_identity(path))
    old_path = root / "configs/dense_primary_v3_geometry_protocol.json"
    assert (
        file_identity(old_path)["sha256"]
        == "9f11cbcdfa8b75593cc3ebd22336e5a34073531426431da88cf904f6e470f358"
    )
    require_same(read_json(old_path), read_json(archive / "attempt-1/protocol.json"))
    for name, binding in read_json(old_path)["sources"].items():
        saved = archive / "attempt-1/source" / name
        verify_file(saved if saved.exists() else root / name, binding)
    assert file_identity(archive / "protocol-v2.json")["sha256"] == PROTOCOL_SHA
    primary = PrimaryV3Contract.load(
        root / "configs/dense_primary_v3_protocol.json", root, "/tmp/dense-identity-source.8rUgLF"
    )
    contract = GeometryContract.load(
        root / "configs/dense_primary_v3_geometry_protocol_v2.json", primary
    )
    require_same(contract.payload, read_json(archive / "protocol-v2.json"))
    assert contract.payload["implementation_revision"] == IMPLEMENTATION_REVISION
    failure = read_json(archive / "attempt-1/failure.json")
    assert failure["audit_passed"] is failure["scientific_completion"] is False
    assert failure["error_type"] == "AssertionError"
    verify_file(archive / "attempt-1/source/scripts/audit_dense_v3_geometry.py", failure["source"])
    old_work = Path("/tmp/dense-v3-geometry.75bnhK")
    require_same(read_json(old_work / "failure.json"), failure)
    assert not (old_work / "result.json").exists()
    localization = read_json(archive / "reduction-localization.json")
    assert (
        file_identity(archive / "reduction-localization.json")["sha256"]
        == "3fd2ebb1c9fbef543c32b214b39733dcac91554b69e29110864aff64c2592ca2"
    )
    assert localization["selected_cases"] == len(localization["rows"]) == 12
    assert sum(not row["original_gate_passed"] for row in localization["rows"]) == 3
    assert max(row["fp64_relative_error"] for row in localization["rows"]) < 1e-12
    assert localization["tolerance_changed"] is False
    verify_file(
        archive / "attempt-1/source/scripts/audit_dense_geometry_reductions.py",
        localization["source"],
    )
    assert file_identity(archive / "rehearsal-v2.json")["sha256"] == rehearsal_sha
    assert file_identity(archive / "fresh-cpu-replay-v2.json")["sha256"] == replay_sha
    rehearsal = read_json(archive / "rehearsal-v2.json")
    replay = read_json(archive / "fresh-cpu-replay-v2.json")
    assert rehearsal["actual_geometry_audit_passed"] is True
    assert replay["fresh_geometry_replay_passed"] is True
    for result in (rehearsal, replay):
        assert result["models"] == 3 and result["checkpoints"] == 9
        assert result["formal_geometry_produced"] is result["scientific_completion"] is False
        assert result["model_updates"] == result["gpu_workers"] == 0
        require_same(
            result["table_counts"],
            {
                "checkpoint_geometry": 9,
                "run_pair_subspace_overlap": 18,
                "optimizer_pair_subspace_summary": 18,
                "subspace_health": 1584,
            },
        )
    assert rehearsal["raw_hidden_matrix_records"] == 792
    assert rehearsal["old_raw_records_unchanged_except_declared_global_scalar_precision"] == 792
    assert rehearsal["entry_amendment_record_comparisons"] == 792
    assert rehearsal["nonzero_rank16_supported_cases"] == 1056
    assert rehearsal["zero_undefined_cases"] == 528
    assert rehearsal["independent_norm_comparisons"] == 2112
    assert rehearsal["independent_defined_cosine_comparisons"] == 1056
    assert rehearsal["independent_undefined_cosine_checks"] == 264
    assert (
        rehearsal["independent_full_spectrum_cases"]
        == len(rehearsal["full_spectrum_controls"])
        == 12
    )
    assert rehearsal["scalar_rtol"] == localization["original_rtol"] == 2e-5
    assert rehearsal["scalar_atol"] == localization["original_atol"] == 1e-10
    assert len(rehearsal["changed_cases"]) == 8 and all(
        row["rejected"] for row in rehearsal["changed_cases"]
    )
    assert len(replay["changed_cases"]) == 2 and all(
        row["rejected"] for row in replay["changed_cases"]
    )
    for attempt in rehearsal["actual_missing_primary_cli"]:
        assert (
            attempt["returncode"] != 0
            and "Require an ordinary retained run directory" in attempt["stderr"]
        )
    assert {row["action"] for row in rehearsal["actual_missing_primary_cli"]} == {
        "produce",
        "inspect",
    }
    checks = {row["path"]: row for row in prior["external_payload_checks"]}
    for row in (
        *rehearsal["unchanged_inputs_and_sources"],
        *rehearsal["artifacts"],
        *replay["artifacts"],
        replay["rehearsal"],
        replay["source"],
    ):
        checks[row["path"]] = row
    for path in sorted(old_work.rglob("*")):
        if path.is_file():
            checks[str(path)] = {"path": str(path), **file_identity(path)}
    for path in (
        Path("/tmp/dense-v3-geometry-retry.Pysj3k/result.json"),
        Path("/tmp/dense-v3-geometry-replay.jd0MyI/result.json"),
    ):
        checks[str(path)] = {"path": str(path), **file_identity(path)}
    external = [compare_binding(row, Path(path), root) for path, row in sorted(checks.items())]
    tests = {}
    for name, count in (
        ("focused-tests.xml", 49),
        ("full-tests.xml", 1940),
        ("focused-tests-v2.xml", 64),
        ("full-tests-v2.xml", 1955),
    ):
        value = next(ET.parse(archive / name).iter("testsuite")).attrib
        assert int(value["tests"]) == count
        assert all(int(value[key]) == 0 for key in ("failures", "errors", "skipped"))
        tests[name] = value
    for row in prior["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            verify_file(base / row["path"], row)
    verify_file(root / "paper/main.tex", prior["unchanged_manuscript_source"])
    verify_file(Path(prior["unchanged_main_ledger"]["path"]), prior["unchanged_main_ledger"])
    require_same(handoff(), rehearsal["post_execution_dispatchers"])
    return {
        "scope": "engineering_dense_v3_complete_geometry_evidence_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "prepared_complete_geometry_consumer_verified": True,
        "actual_diagnostic_geometry_verified": True,
        "original_precision_failure_preserved": True,
        "independent_tolerances_unchanged": True,
        "formal_geometry_produced": False,
        "full_primary_geometry_observed": False,
        "full_downstream_integration_complete": False,
        "formal_replication_ready": False,
        "scientific_completion": False,
        "production_deployed": False,
        "source_published": False,
        "accepted_rehearsal_sha256": rehearsal_sha,
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
                "configs/dense_primary_v3_geometry_protocol.json",
                "configs/dense_primary_v3_geometry_protocol_v2.json",
                "scripts/validate_dense_v3_geometry.py",
            )
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rehearsal-sha256", required=True)
    parser.add_argument("--replay-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("Require a new validation receipt under an explicit existing parent")
    value = validate(args.repository.resolve(), args.rehearsal_sha256, args.replay_sha256)
    write_new(args.output, value)
    print(
        {
            key: value[key]
            for key in (
                "artifact_validation_passed",
                "prepared_complete_geometry_consumer_verified",
                "formal_geometry_produced",
                "scientific_completion",
            )
        }
    )


if __name__ == "__main__":
    main()
