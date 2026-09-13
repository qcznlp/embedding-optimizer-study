"""Read-only evidence validation for actual full-run/checkpoint identity integration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from scripts.pause_dense_evaluation_for_audit import CHAIN, LEDGER, LEDGER_SHA
from scripts.pause_dense_evaluation_for_audit import inspect as inspect_process
from scripts.validate_dense_correction_candidate import EXPECTED_FAILURES
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity, load

ARCHIVE = Path("reports/engineering-archive/dense-full-identity-v1")
PRIOR = Path("reports/engineering-archive/dense-predeployment-checks-v1/validation.json")
PRIOR_SHA = "b8f7b1fe0331fce06332e92f421bf30890b56e7386b55d7f4069628a169887a2"
MANIFEST_SHA = "1c6163c5aeee9d0735a6859c0e2f56a701c59d5111b553de6fb5c758da138dfd"
RUNS = ("verified-adamw-3e-5", "verified-muon-3e-4", "verified-normuon-3e-4")
POSITIVE = {"original", "identical_relocation"}
NEGATIVE = {
    "seed",
    "dataset_content",
    "epochs",
    "context_length",
    "temperature",
    "gradient_clipping",
    "global_batch_same_accumulation",
    "warmup",
    "declared_base_revision",
    "runtime_horizon_override",
    "optimizer_lr",
    "accumulation",
}
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")
RUN_RECEIPT = "dense_run_contract.json"
SEAL = "dense_checkpoint_seal.json"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_inventory(directory, exclude=()):
    values = []
    for path in sorted(directory.rglob("*")):
        assert not path.is_symlink()
        if path.is_file() and path.name not in exclude:
            row = identity(path, directory)
            values.append(row)
    return values


def check_admission(payload):
    assert payload["audit_execution_complete"] is payload["admission_checks_passed"] is True
    assert payload["scientific_completion"] is payload["production_deployed"] is False
    assert payload["model_updates_executed"] == 0 and payload["model_or_pickle_loaded"] is False
    assert payload["all_original_producer_files_unchanged"] is True
    assert payload["accepted_positive_controls"] == 6
    assert payload["rejected_changed_identities"] == 36
    assert payload["rejected_changed_real_optimizer_payloads"] == 3
    assert len(payload["records"]) == 45
    assert {(r["run_id"], r["case"]) for r in payload["records"]} == {
        (run, case)
        for run in RUNS
        for case in POSITIVE | NEGATIVE | {"changed_real_optimizer_payload"}
    }
    for row in payload["records"]:
        accepted = row["case"] in POSITIVE
        assert row["expected_accepted"] is row["observed"]["accepted"] is accepted
        if accepted:
            assert row["observed"]["stopped_before_setup"] is True
        else:
            assert row["observed"]["rejection"]
    return {
        "positive_controls": 6,
        "changed_identities_rejected": 36,
        "changed_real_payloads_rejected": 3,
    }


def check_tests(path):
    xml = ET.parse(path).getroot()
    suite = next(xml.iter("testsuite")).attrib
    assert int(suite["tests"]) == 963 and int(suite["failures"]) == 11
    assert int(suite["errors"]) == int(suite["skipped"]) == 0
    failures = set()
    for row in xml.iter("testcase"):
        failed = row.find("failure")
        if failed is not None:
            failures.add(f"{row.attrib['classname']}.{row.attrib['name']}")
            assert failed.attrib["message"].startswith("ValueError:")
            assert any(
                x in failed.attrib["message"]
                for x in ("source_bindings", "source binding", "source contract")
            )
    assert failures == EXPECTED_FAILURES
    return {**suite, "full_suite_passed": False, "unwaived_source_failures": sorted(failures)}


def verify(root):
    if sys.flags.optimize:
        raise RuntimeError("Audit assertions must not be disabled")
    archive = root / ARCHIVE
    assert identity(root / PRIOR, root)["sha256"] == PRIOR_SHA
    prior = load(root / PRIOR)
    preserved = [
        compare_binding(
            x, archive / "before" / x["path"] if x["path"] in DOCS else root / x["path"], root
        )
        for x in prior["bindings"]
    ]
    manifest_path = archive / "prepared-source.json"
    assert identity(manifest_path, root)["sha256"] == MANIFEST_SHA
    manifest = load(manifest_path)
    assert manifest["production_deployed"] is manifest["scientific_completion"] is False
    assert manifest["formal_execution_protocol_reviewed"] is False
    assert manifest["matrix_unchanged_from_parent"] is True
    compare_binding(manifest["parent_manifest"], Path(manifest["parent_manifest"]["path"]), root)
    sources = []
    for row in manifest["candidate_sources"]:
        sources.append(compare_binding(row["identity"], Path(row["identity"]["path"]), root))
        compare_binding(row["identity"], archive / "candidate-source" / row["relative_path"], root)
    parent_manifest = load(Path(manifest["parent_manifest"]["path"]))
    for row in parent_manifest["candidate_sources"]:
        sources.append(compare_binding(row["identity"], Path(row["identity"]["path"]), root))
    entry, admission = (
        load(archive / "entrypoint/result.json"),
        load(archive / "admission/result.json"),
    )
    assert entry["entrypoint_acceptance_passed"] is entry["audit_execution_complete"] is True
    assert entry["source_checkpoint_unchanged"] is True and entry["world_size"] == 4
    assert entry["scientific_completion"] is entry["production_deployed"] is False
    assert [(r["run_id"], r["resume_step"]) for r in entry["records"]] == [
        (r, s) for r in RUNS for s in (0, 2)
    ]
    compare_binding(entry["source_manifest"], manifest_path, root)
    compare_binding(admission["source_manifest"], manifest_path, root)
    compare_binding(admission["entry_result"], archive / "entrypoint/result.json", root)
    external, run_ids = [], {}
    for record in entry["records"]:
        done = record["completion"]
        assert (
            done["all_checkpoint_payload_seals_passed"]
            is done["final_equals_scheduled_checkpoint"]
            is True
        )
        for key in ("completion", "resolved_config", "full_run_identity"):
            item = done[key]
            external.append(compare_binding(item, Path(item["path"]), root))
        run_root = Path(done["completion"]["path"]).parent
        full = load(run_root / RUN_RECEIPT)
        full_hash = hashlib.sha256(canonical(full)).hexdigest()
        assert done["full_identity_sha256"] == full_hash
        assert full["recipe"]["run_id"] == record["run_id"]
        assert full["scope"] == "content_bound_corrected_dense_run" and full["schema_version"] == 1
        if record["resume_step"] == 0:
            run_ids[record["run_id"]] = full_hash
        else:
            assert run_ids[record["run_id"]] == full_hash
        completed = load(run_root / "completed.json")
        assert completed["global_step"] == 3 and completed["dataset_rows"] == 288
        assert completed["run_identity_sha256"] == full_hash
        assert canonical(load(run_root / "final" / RUN_RECEIPT)) == canonical(full)
        config = load(run_root / "run_config.json")
        assert full["data"]["files"] == content_inventory(Path(config["dataset_path"]))
        assert full["model"]["files"] == content_inventory(Path(config["model_name"]))
        steps = ("1", "2", "3") if record["resume_step"] == 0 else ("3",)
        assert tuple(done["checkpoint_files"]) == steps
        for step in steps:
            files = done["checkpoint_files"][step]
            assert len(files) == 20
            external.extend(compare_binding(x, Path(x["path"]), root) for x in files)
            checkpoint = run_root / f"checkpoint-{step}"
            seal = load(checkpoint / SEAL)
            assert seal["step"] == int(step) and seal["run_identity_sha256"] == full_hash
            assert canonical(load(checkpoint / RUN_RECEIPT)) == canonical(full)
            assert seal["files"] == content_inventory(checkpoint, exclude=(SEAL,))
            assert {f"rng_state_{r}.pth" for r in range(4)}.issubset(
                {x["path"] for x in seal["files"]}
            )
            assert load(checkpoint / "trainer_state.json")["global_step"] == int(step)
    checks = check_admission(admission)
    for row in admission["records"]:
        if row["case"] == "changed_real_optimizer_payload":
            copied = Path(row["diagnostic_copy"])
            sealed_rows = load(copied / SEAL)["files"]
            actual_rows = content_inventory(copied, exclude=(SEAL,))
            assert [x["path"] for x in actual_rows] == [x["path"] for x in sealed_rows]
            assert [a["path"] for a, b in zip(actual_rows, sealed_rows, strict=True) if a != b] == [
                "optimizer.pt"
            ]
            assert all(
                a["bytes"] == b["bytes"] for a, b in zip(actual_rows, sealed_rows, strict=True)
            )
            producer = next(
                r
                for r in entry["records"]
                if r["run_id"] == row["run_id"] and r["resume_step"] == 0
            )
            original = (
                Path(producer["completion"]["full_run_identity"]["path"]).parent
                / "checkpoint-2/optimizer.pt"
            )
            with original.open("rb") as left, (copied / "optimizer.pt").open("rb") as right:
                assert left.read(1)[0] ^ right.read(1)[0] == 1
                assert (
                    hashlib.file_digest(left, "sha256").digest()
                    == hashlib.file_digest(right, "sha256").digest()
                )
    assert admission["authenticated_files"] == [
        {**x, "path": str(root / x["path"]) if not Path(x["path"]).is_absolute() else x["path"]}
        for x in external
    ]
    for report in (entry, admission):
        sources.extend(compare_binding(x, Path(x["path"]), root) for x in report["source_bindings"])
    for item in prior["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            assert identity(base / item["path"], root)["sha256"] == item["sha256"]
    dispatchers = [
        inspect_process(pid, start, entry, CHAIN[i - 1][0] if i else 1)
        for i, (pid, start, entry) in enumerate(CHAIN)
    ]
    assert all(x["state"] == "T" for x in dispatchers)
    assert identity(LEDGER, root)["sha256"] == LEDGER_SHA
    paper = identity(root / "paper/main.tex", root)
    assert paper["sha256"] == prior["unchanged_manuscript_source"]["sha256"]
    isolated_suite = next(
        ET.parse(archive / "tests/isolated-full.xml").getroot().iter("testsuite")
    ).attrib
    assert int(isolated_suite["tests"]) == 1424 and all(
        int(isolated_suite[k]) == 0 for k in ("failures", "errors", "skipped")
    )
    return {
        "scope": "engineering_full_identity_evidence_validation",
        "artifact_validation_passed": True,
        "full_identity_integrated_in_prepared_candidate": True,
        "scientific_completion": False,
        "formal_replication_ready": False,
        "production_deployed": False,
        "source_published": False,
        "independent_process_bitwise_replay_of_this_source_tested": False,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "summary": {"actual_entrypoint_calls": 6, "sealed_checkpoints": 12, **checks},
        "candidate_test_suite": check_tests(archive / "tests/full-first.xml"),
        "isolated_test_suite": isolated_suite,
        "prior_bindings_preserved": preserved,
        "executed_source_checks": sources,
        "external_payload_checks": external,
        "external_payloads_uploaded": False,
        "unchanged_live_core": prior["unchanged_live_core"],
        "unchanged_manuscript_source": paper,
        "unchanged_main_ledger": identity(LEDGER, root),
        "post_execution_dispatchers": dispatchers,
        "bindings": [
            identity(p, root)
            for p in sorted(archive.rglob("*"))
            if p.is_file() and p.name != "validation.json"
        ]
        + [identity(root / p, root) for p in (*DOCS, "scripts/validate_dense_full_identity.py")],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("Use a new validation receipt path")
    payload = verify(args.repository.resolve())
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                k: payload[k]
                for k in ("artifact_validation_passed", "formal_replication_ready", "summary")
            }
        )
    )


if __name__ == "__main__":
    main()
