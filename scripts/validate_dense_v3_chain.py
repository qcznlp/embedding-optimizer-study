"""Validate the prepared v3 core-chain rehearsal and preserve its preceding evidence."""

import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import NATURAL_SHA, PrimaryV3Contract
from scripts.audit_dense_natural_data import handoff, write_new
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-primary-v3-chain-v1")
PRIOR = Path("reports/engineering-archive/dense-revised-natural-v1/validation.json")
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")
PROTOCOL_SHA = "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b"
REHEARSAL_SHA = "3abd941dddf1686576a17ebf8ceffc7fe9ebc68cd610b1de1e965ecd77d536c1"


def validate(root):
    if sys.flags.optimize:
        raise ValueError("Do not disable audit assertions")
    archive = root / ARCHIVE
    assert file_identity(root / PRIOR)["sha256"] == NATURAL_SHA
    prior = read_json(root / PRIOR)
    preserved = [
        compare_binding(
            r, archive / "before" / r["path"] if r["path"] in DOCS else root / r["path"], root
        )
        for r in prior["bindings"]
    ]
    for path in (archive / "source").rglob("*"):
        if path.is_file():
            verify_file(root / path.relative_to(archive / "source"), file_identity(path))
    assert file_identity(archive / "protocol.json")["sha256"] == PROTOCOL_SHA
    assert file_identity(archive / "rehearsal.json")["sha256"] == REHEARSAL_SHA
    contract = PrimaryV3Contract.load(
        root / "configs/dense_primary_v3_protocol.json", root, "/tmp/dense-identity-source.8rUgLF"
    )
    assert contract.sha256 == PROTOCOL_SHA
    require_same(contract.payload, read_json(archive / "protocol.json"))
    rehearsal = read_json(archive / "rehearsal.json")
    assert set(rehearsal["actual_input_partitions_accepted"]) == {"training", "validation"}
    for partition, row in rehearsal["actual_input_partitions_accepted"].items():
        require_same(
            {k: row[k] for k in ("rows", "files")}, contract.payload["datasets"][partition]
        )
    assert len(rehearsal["actual_rejections"]) == 68
    assert all(row["rejected"] is True for row in rehearsal["actual_rejections"])
    assert all(
        "File content identity differs" in row["error"]
        for row in rehearsal["actual_rejections"][:2]
    )
    assert len(rehearsal["actual_inspection_cli"]) == 12
    assert {r["observed"]["run_id"] for r in rehearsal["actual_inspection_cli"]} == {
        r["run_id"] for r in contract.inputs["runs"]
    }
    assert all(row["returncode"] == 0 for row in rehearsal["actual_inspection_cli"])
    assert rehearsal["missing_primary_grid_cli"]["returncode"] != 0
    assert (
        "Require an ordinary retained run directory"
        in rehearsal["missing_primary_grid_cli"]["stderr"]
    )
    assert all(rehearsal[k] == 0 for k in ("model_updates", "gpu_workers", "network_uploads"))
    assert rehearsal["scientific_completion"] is False
    assert rehearsal["formal_replication_ready"] is False
    checks = {r["path"]: r for r in prior["external_payload_checks"]}
    for row in rehearsal["input_markers_preserved"] + rehearsal["source_bindings"]:
        checks[row["path"]] = row
    external = [compare_binding(r, Path(p), root) for p, r in sorted(checks.items())]
    tests = {}
    for name, count in (
        ("focused-tests.xml", 48),
        ("focused-tests-final.xml", 48),
        ("full-tests.xml", 1710),
    ):
        result = next(ET.parse(archive / name).iter("testsuite")).attrib
        assert int(result["tests"]) == count
        assert all(int(result[k]) == 0 for k in ("failures", "errors", "skipped"))
        tests[name] = result
    for row in prior["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            verify_file(base / row["path"], row)
    verify_file(root / "paper/main.tex", prior["unchanged_manuscript_source"])
    verify_file(Path(prior["unchanged_main_ledger"]["path"]), prior["unchanged_main_ledger"])
    return {
        "scope": "engineering_dense_v3_core_chain_evidence_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "prepared_v3_core_chain_rehearsal_passed": True,
        "scientific_completion": False,
        "formal_replication_ready": False,
        "full_downstream_integration_complete": False,
        "production_deployed": False,
        "source_published": False,
        "prior_bindings_preserved": preserved,
        "external_payload_checks": external,
        "tests": tests,
        "unchanged_live_core": prior["unchanged_live_core"],
        "unchanged_manuscript_source": prior["unchanged_manuscript_source"],
        "unchanged_main_ledger": prior["unchanged_main_ledger"],
        "post_execution_dispatchers": handoff(),
        "protocol_sha256": PROTOCOL_SHA,
        "actual_rehearsal_sha256": REHEARSAL_SHA,
        "bindings": [
            identity(p, root)
            for p in sorted(archive.rglob("*"))
            if p.is_file() and p.name != "validation.json"
        ]
        + [
            identity(root / p, root)
            for p in (
                *DOCS,
                "configs/dense_primary_v3_protocol.json",
                "scripts/validate_dense_v3_chain.py",
            )
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("Require a new receipt under an existing explicit parent")
    value = validate(args.repository.resolve())
    write_new(args.output, value)
    print(
        {
            k: value[k]
            for k in (
                "artifact_validation_passed",
                "prepared_v3_core_chain_rehearsal_passed",
                "formal_replication_ready",
                "full_downstream_integration_complete",
            )
        }
    )


if __name__ == "__main__":
    main()
