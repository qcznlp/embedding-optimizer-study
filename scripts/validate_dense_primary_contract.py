"""Validate the prepared primary contract and its read-only engineering evidence."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import PrimaryContract, file_identity, read_json, verify_file
from scripts.pause_dense_evaluation_for_audit import CHAIN, LEDGER, LEDGER_SHA
from scripts.pause_dense_evaluation_for_audit import inspect as inspect_process
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-primary-contract-v1")
PRIOR = Path("reports/engineering-archive/dense-full-identity-v1/validation.json")
PRIOR_SHA = "35779a029f357981e92c4a9c8c87e7a27118e791c88692fbcacf9ec2cd35b29c"
PROTOCOL_SHA = "e21a7226c09740d38d49855738267c080f615f8d4f60c2abf9dbf3ccd874edd2"
REHEARSAL_SHA = "af78df6b6f5cba499cf40cc43fdf923234e589b25062713cd671dc07ee5977cc"
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")


def check_rehearsal(value):
    assert value["scope"] == "engineering_dense_primary_consumer_rehearsal"
    assert value["read_only_rehearsal_passed"] is True
    assert value["scientific_completion"] is value["production_deployed"] is False
    assert value["model_or_pickle_loaded"] is False
    assert (
        value["model_updates_executed"]
        == value["hf_writes_executed"]
        == value["evaluation_workers_executed"]
        == 0
    )
    assert value["protocol"]["sha256"] == PROTOCOL_SHA
    real = value["real_diagnostic_checkpoints"]
    assert len(real) == len({r["checkpoint"] for r in real}) == 12
    for row in real:
        assert row["diagnostic_identity_accepted"] is row["producer_payloads_unchanged"] is True
        assert (
            row["primary_identity_rejected"] == "Artifact belongs to a different primary identity"
        )
        assert len(row["checked"]["files"]) == 20
    old = value["old_primary_checkpoints_rejected"]
    assert len(old) == len({r["checkpoint"] for r in old}) == 60
    assert all("Missing ordinary contract file" in r["rejected"] for r in old)
    corrupt = value["real_corrupt_optimizer_copies_rejected"]
    assert len(corrupt) == len({r["checkpoint"] for r in corrupt}) == 3
    assert all("File content identity differs: optimizer.pt" == r["rejected"] for r in corrupt)
    cli = value["actual_read_only_cli_plans"]
    assert len(cli) == len({r["plan"]["run_id"] for r in cli}) == 12
    assert all(r["returncode"] == 0 and r["plan"]["training_executed"] is False for r in cli)
    assert value["actual_draft_execution_rejection"]["returncode"] != 0
    assert "not execution authorized" in value["actual_draft_execution_rejection"]["stderr"]
    return {
        "real_diagnostic_checkpoints": 12,
        "old_checkpoints_rejected": 60,
        "real_corrupt_copies_rejected": 3,
        "actual_cli_plans": 12,
    }


def verify(root):
    if sys.flags.optimize:
        raise RuntimeError("Audit assertions must not be disabled")
    archive = root / ARCHIVE
    assert file_identity(root / PRIOR)["sha256"] == PRIOR_SHA
    prior = read_json(root / PRIOR)
    preserved = [
        compare_binding(
            row,
            archive / "before" / row["path"] if row["path"] in DOCS else root / row["path"],
            root,
        )
        for row in prior["bindings"]
    ]
    proposal = root / "configs/dense_primary_v2_protocol.json"
    assert file_identity(proposal)["sha256"] == PROTOCOL_SHA
    training = root / "reports/engineering-archive/dense-full-identity-v1/candidate-source"
    contract = PrimaryContract.load(proposal, root, training)
    assert contract.payload["status"] == "prepared_not_execution_authorized"
    inputs = contract.inputs
    assert inputs["model_loaded"] is inputs["protocol_activated"] is False
    assert inputs["model_updates_executed"] == 0
    assert inputs["data_linkage_audit"]["verified_rows"] == 500000
    assert inputs["data_linkage_audit"]["complete"] is True
    assert len(inputs["common_identity"]["data"]["files"]) == 16
    assert sum(r["bytes"] for r in inputs["common_identity"]["data"]["files"]) == 2147849132
    external = []
    for row in inputs["common_identity"]["data"]["files"]:
        path = Path(inputs["data_manifest"]["path"]).parent / "dataset" / row["path"]
        external.append({"path": str(path), **verify_file(path, row)})
    remote = inputs["initial_model"]["independent_hf_digest_checks"]
    assert len(remote) == 11 and all(r["local_digest"] == r["remote_digest"] for r in remote)
    cached = Path(inputs["initial_model"]["cached_root"])
    model_files = inputs["common_identity"]["model"]["files"]
    assert [r["path"] for r in remote] == [r["path"] for r in model_files]
    for row in model_files:
        path = cached / row["path"]
        resolved = path.resolve()
        assert resolved.is_relative_to(cached) or resolved.is_relative_to(
            cached.parent.parent / "blobs"
        )
        external.append({"path": str(path), **verify_file(resolved, row)})
    sources = [compare_binding(row, Path(row["path"]), root) for row in inputs["source_bindings"]]
    for key in ("original_scientific_protocol", "data_manifest", "row_manifest", "source_manifest"):
        row = inputs[key]
        sources.append(compare_binding(row, Path(row["path"]), root))
    prepared = read_json(Path(inputs["source_manifest"]["path"]))
    for row in prepared["candidate_sources"]:
        sources.append(compare_binding(row["identity"], Path(row["identity"]["path"]), root))
        verify_file(training / row["relative_path"], row["identity"])
    rehearsal_path = archive / "rehearsal/result.json"
    assert file_identity(rehearsal_path)["sha256"] == REHEARSAL_SHA
    rehearsal = read_json(rehearsal_path)
    counts = check_rehearsal(rehearsal)
    for name, expected in rehearsal["sources"].items():
        verify_file(root / name, expected)
    for path in sorted((archive / "consumer-source").rglob("*")):
        if path.is_file():
            relative = path.relative_to(archive / "consumer-source")
            verify_file(root / relative, file_identity(path))
    for row in rehearsal["real_diagnostic_checkpoints"]:
        for payload in row["checked"]["files"]:
            path = Path(row["checkpoint"]) / payload["path"]
            verify_file(path, payload)
            external.append({**payload, "path": str(path)})
    for row in prior["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            assert file_identity(base / row["path"])["sha256"] == row["sha256"]
    paper = identity(root / "paper/main.tex", root)
    assert paper["sha256"] == prior["unchanged_manuscript_source"]["sha256"]
    assert file_identity(LEDGER)["sha256"] == LEDGER_SHA
    dispatchers = [
        inspect_process(pid, start, name, CHAIN[i - 1][0] if i else 1)
        for i, (pid, start, name) in enumerate(CHAIN)
    ]
    assert all(row["state"] == "T" for row in dispatchers)
    tests = {}
    for filename, count in (("focused-final.xml", 51), ("isolated-full.xml", 1475)):
        suite = next(ET.parse(archive / "tests" / filename).iter("testsuite")).attrib
        assert int(suite["tests"]) == count and all(
            int(suite[k]) == 0 for k in ("failures", "errors", "skipped")
        )
        tests[filename] = suite
    files = [p for p in sorted(archive.rglob("*")) if p.is_file() and p.name != "validation.json"]
    return {
        "scope": "engineering_dense_primary_contract_evidence_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "primary_contract_prepared": True,
        "formal_replication_ready": False,
        "production_deployed": False,
        "source_published": False,
        "scientific_completion": False,
        "summary": {
            "primary_rows": 500000,
            "base_hf_digest_checks": 11,
            "primary_recipe_identities": 12,
            **counts,
        },
        "tests": tests,
        "prior_bindings_preserved": preserved,
        "input_producer_sources_unchanged": sources,
        "external_payload_checks": external,
        "unchanged_live_core": prior["unchanged_live_core"],
        "unchanged_manuscript_source": paper,
        "unchanged_main_ledger": identity(LEDGER, root),
        "post_execution_dispatchers": dispatchers,
        "bindings": [identity(p, root) for p in files]
        + [
            identity(root / p, root)
            for p in (
                *DOCS,
                "scripts/validate_dense_primary_contract.py",
                "configs/dense_primary_v2_protocol.json",
                "reports/dense-primary-v2/input-bindings.json",
            )
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("Use a new validation receipt path")
    result = verify(args.repository.resolve())
    with args.output.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("artifact_validation_passed", "formal_replication_ready", "summary")
            }
        )
    )


if __name__ == "__main__":
    main()
