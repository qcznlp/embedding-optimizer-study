"""Validate a bounded exact-spectrum/projector audit and fresh numerical replay."""

import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.audit_dense_probe_coupling import PLAN
from scripts.prepare_dense_geometry_robustness import PARENTS, load_protocol
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-geometry-robustness-v1")
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")
PROTOCOL_SHA = "e950a43ddf2f947d2e52993fd563e36fa2cffc7897ec00f71e25a051ea71da4b"
COUNTS = {
    "exact_cases": 12,
    "approximation_cases": 156,
    "same_matrix_seed_comparisons": 144,
    "cross_optimizer_pair_comparisons": 156,
}


def validate(root, audit_sha, replay_sha, coupling_sha, coupling_replay_sha):
    if sys.flags.optimize:
        raise ValueError("Evidence assertions must not be disabled")
    archive = root / ARCHIVE
    prior_path, prior_sha = PARENTS["geometry_acceptance"]
    assert file_identity(root / prior_path)["sha256"] == prior_sha
    prior = read_json(root / prior_path)
    preserved = [
        compare_binding(
            row,
            archive / "before" / row["path"] if row["path"] in DOCS else root / row["path"],
            root,
        )
        for row in prior["bindings"]
    ]
    path = root / "configs/dense_geometry_robustness_protocol.json"
    assert file_identity(path)["sha256"] == PROTOCOL_SHA
    plan = load_protocol(path, root)
    require_same(plan, read_json(archive / "protocol.json"))
    for path in (archive / "source").rglob("*"):
        if path.is_file():
            verify_file(root / path.relative_to(archive / "source"), file_identity(path))
    audit_path, replay_path = archive / "audit.json", archive / "fresh-replay.json"
    assert file_identity(audit_path)["sha256"] == audit_sha
    assert file_identity(replay_path)["sha256"] == replay_sha
    audit, replay = read_json(audit_path), read_json(replay_path)
    assert replay["replay_parent"]["sha256"] == audit_sha
    assert audit["fresh_numeric_replay_passed"] is None
    assert replay["fresh_numeric_replay_passed"] is True
    assert len(replay["changed_cases"]) == 4 and all(
        row["rejected"] for row in replay["changed_cases"]
    )
    for result in (audit, replay):
        require_same(result["counts"], COUNTS)
        assert result["numerical_controls_passed"] is True
        assert result["protocol"]["sha256"] == PROTOCOL_SHA
        assert result["model_updates"] == result["gpu_workers"] == 0
        for key in (
            "approximation_accuracy_pass_declared",
            "primary_features_changed",
            "formal_geometry_produced",
            "scientific_completion",
        ):
            assert result[key] is False
    require_same(audit["inputs"], replay["inputs"])
    details = read_json(archive / "details.json")
    require_same({key: len(rows) for key, rows in details.items()}, COUNTS)
    original_detail = next(
        row for row in audit["artifacts"] if row["path"].endswith("/details.json")
    )
    verify_file(archive / "details.json", original_detail)
    checks = {row["path"]: row for row in prior["external_payload_checks"]}
    for result in (audit, replay):
        for row in (*result["inputs"], *result["artifacts"]):
            checks[row["path"]] = row
    checks[replay["replay_parent"]["path"]] = replay["replay_parent"]
    coupling_path, coupling_replay_path = (
        archive / "coupling-audit.json",
        archive / "coupling-replay.json",
    )
    assert file_identity(coupling_path)["sha256"] == coupling_sha
    assert file_identity(coupling_replay_path)["sha256"] == coupling_replay_sha
    coupling, coupling_replay = read_json(coupling_path), read_json(coupling_replay_path)
    assert coupling_replay["replay_parent"]["sha256"] == coupling_sha
    require_same(coupling["results"], coupling_replay["results"])
    assert (
        coupling["fresh_replay_passed"] is None and coupling_replay["fresh_replay_passed"] is True
    )
    for result in (coupling, coupling_replay):
        require_same(result["plan"], PLAN)
        assert result["parent"]["sha256"] == audit_sha
        assert result["scientific_completion"] is False
        assert result["model_updates"] == result["gpu_workers"] == 0
        assert (
            result["same_seed_parent_agreements"] == 36 and result["exact_parent_agreements"] == 12
        )
        assert len(result["results"]["rows"]) == 108 and len(result["results"]["summaries"]) == 12
        assert sum(row["same_probe_seed"] for row in result["results"]["rows"]) == 36
        require_same(handoff(), result["post_execution_dispatchers"])
        for row in (*result["sources"], result["parent"]):
            verify_file(row["path"], row)
            checks[row["path"]] = row
    checks[coupling_replay["replay_parent"]["path"]] = coupling_replay["replay_parent"]
    external = [compare_binding(row, Path(path), root) for path, row in sorted(checks.items())]
    tests = {}
    for name, count, failures in (
        ("focused-tests.xml", 41, 1),
        ("focused-tests-v2.xml", 44, 0),
        ("full-tests.xml", 1999, 0),
        ("coupling-tests.xml", 7, 0),
        ("full-tests-final.xml", 2006, 0),
    ):
        values = next(ET.parse(archive / name).iter("testsuite")).attrib
        assert int(values["tests"]) == count and int(values["failures"]) == failures
        assert int(values["errors"]) == int(values["skipped"]) == 0
        tests[name] = values
    for row in prior["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            verify_file(base / row["path"], row)
    verify_file(root / "paper/main.tex", prior["unchanged_manuscript_source"])
    verify_file(Path(prior["unchanged_main_ledger"]["path"]), prior["unchanged_main_ledger"])
    require_same(handoff(), audit["post_execution_dispatchers"])
    require_same(handoff(), replay["post_execution_dispatchers"])
    return {
        "scope": "engineering_dense_geometry_robustness_evidence_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "exact_reference_and_fresh_replay_verified": True,
        "shared_probe_followup_and_fresh_replay_verified": True,
        "approximation_accuracy_pass_declared": False,
        "formal_primary_robustness_observed": False,
        "formal_replication_ready": False,
        "scientific_completion": False,
        "accepted_audit_sha256": audit_sha,
        "accepted_replay_sha256": replay_sha,
        "accepted_coupling_sha256": coupling_sha,
        "accepted_coupling_replay_sha256": coupling_replay_sha,
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
                "configs/dense_geometry_robustness_protocol.json",
                "scripts/validate_dense_geometry_robustness.py",
            )
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--replay-sha256", required=True)
    parser.add_argument("--coupling-sha256", required=True)
    parser.add_argument("--coupling-replay-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("Require a new receipt in an explicit existing parent")
    result = validate(
        args.repository.resolve(),
        args.audit_sha256,
        args.replay_sha256,
        args.coupling_sha256,
        args.coupling_replay_sha256,
    )
    write_new(args.output, result)
    print(
        {
            key: result[key]
            for key in (
                "artifact_validation_passed",
                "exact_reference_and_fresh_replay_verified",
                "scientific_completion",
            )
        }
    )


if __name__ == "__main__":
    main()
