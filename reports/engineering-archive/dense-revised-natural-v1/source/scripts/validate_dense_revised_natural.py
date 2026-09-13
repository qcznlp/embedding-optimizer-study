"""Validate actual revised-input/natural-entrypoint evidence and preserve predecessor bindings."""

import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.gpu_lease import acquire_gpu_lease
from embed_optim.primary_contract import digest, file_identity, read_json, require_same, verify_file
from embed_optim.primary_data_revision import load_amendment
from scripts.audit_dense_natural_data import handoff, write_new
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-revised-natural-v1")
PRIOR = Path("reports/engineering-archive/dense-data-candidate-v1/validation.json")
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")


def validate(root):
    if sys.flags.optimize:
        raise ValueError("Do not disable audit assertions")
    archive = root / ARCHIVE
    assert (
        file_identity(root / PRIOR)["sha256"]
        == "3b2cd1902e02a43242b16fee41127cea3c8671e389606d9363c8776a0ef6a52c"
    )
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
    for name, sha in (
        ("revised-inputs.json", "40b7d3e63b1c4a17b995507d5d96c042980acca96baa08f6c0c52dda66b1f1f9"),
        ("data-amendment.json", "5051ec40c7e44ba677980fec422ed4a49ef851ef3e9fff9c29cc0f6fc3d8e4e1"),
        ("natural-result.json", "bf6e893e3b9e244242871227e28a3729acc17477dd3c989f7c647308bf332d36"),
        (
            "amendment-admission.json",
            "e922e5688fdaae8a5db095b666ec8d22e1d24cce006a2535df813c6219c7b53d",
        ),
    ):
        assert file_identity(archive / name)["sha256"] == sha
    amendment, inputs = load_amendment(root / "configs/dense_primary_v3_data_amendment.json", root)
    require_same(inputs, read_json(archive / "revised-inputs.json"))
    require_same(amendment, read_json(archive / "data-amendment.json"))
    assert len(inputs["runs"]) == 12 and inputs["model_updates_executed"] == 0
    assert len(inputs["derived_subset_revised_parent_checks"]) == 3
    assert all(
        s["every_value_matches_revised_parent"]
        for s in inputs["derived_subset_revised_parent_checks"]
    )
    natural = read_json(archive / "natural-result.json")
    admission = read_json(archive / "amendment-admission.json")
    assert natural["natural_data_readiness_passed"] is True and natural["worker_returncode"] == 0
    assert (
        natural["scientific_completion"] is False and natural["primary_training_executed"] is False
    )
    assert [r["algorithm"] for r in natural["records"]] == ["adamw", "muon", "normuon"]
    assert len(natural["selection"]["selected_positions"]) == 288
    assert natural["selection"]["observed_max_truncated_token_length"] == 7679
    assert natural["selection"]["source_quotas"] == amendment["diagnostic"]["source_quotas"]
    assert set(natural["selection"]["mandatory_replacement_positions"]) == set(
        amendment["datasets"]["training"]["changed_sample_ids"]
    )
    first_logs = []
    for row in natural["records"]:
        assert (
            len(row["checked"]["checkpoints"]) == len(row["checked"]["deep_checkpoint_checks"]) == 3
        )
        assert row["checked"]["whole_run_artifacts_verified"] is True
        first_logs.append(row["first_step"])
    assert first_logs[0] == first_logs[1] == first_logs[2]
    assert len(admission["records"]) == 21 and admission["all_changed_files_rejected"] is True
    assert all(r["rejected"] for r in admission["records"])
    cpu = read_json(archive / "fresh-cpu-verification.json")
    assert cpu["exit_code"] == 0 and cpu["observed_result"]["verified"] is True
    assert (
        cpu["observed_result"]["receipt"]["sha256"]
        == file_identity(archive / "natural-result.json")["sha256"]
    )
    checks = {r["path"]: r for r in prior["external_payload_checks"]}
    for row in natural["source_bindings"] + natural["input_bindings"] + natural["artifacts"]:
        checks[row["path"]] = row
    for row in [
        inputs["source"],
        inputs["data_producer"],
        admission["source"],
        *[r["file"] for r in admission["records"]],
    ]:
        checks[row["path"]] = row
    external = [compare_binding(r, Path(p), root) for p, r in sorted(checks.items())]
    tests = {}
    for name, count in (("focused-tests.xml", 30), ("full-tests.xml", 1662)):
        result = next(ET.parse(archive / "tests" / name).iter("testsuite")).attrib
        assert int(result["tests"]) == count
        assert all(int(result[k]) == 0 for k in ("failures", "errors", "skipped"))
        tests[name] = result
    for row in prior["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            verify_file(base / row["path"], row)
    verify_file(root / "paper/main.tex", prior["unchanged_manuscript_source"])
    verify_file(Path(prior["unchanged_main_ledger"]["path"]), prior["unchanged_main_ledger"])
    with acquire_gpu_lease(
        ("4", "5", "6", "7"),
        lock_dir="/tmp/embedding-optimizer-primary-gpu-leases",
        timeout_seconds=1,
        purpose="readiness-post-exit-lease-check",
    ):
        dispatchers = handoff()
    return {
        "scope": "engineering_revised_natural_entrypoint_evidence_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "revised_actual_input_admission_passed": True,
        "natural_data_entrypoint_passed": True,
        "diagnostic_runs": 3,
        "diagnostic_checkpoints": 9,
        "post_exit_gpu_lease_reacquired_and_released": True,
        "scientific_completion": False,
        "formal_replication_ready": False,
        "production_deployed": False,
        "source_published": False,
        "prior_bindings_preserved": preserved,
        "external_payload_checks": external,
        "tests": tests,
        "unchanged_live_core": prior["unchanged_live_core"],
        "unchanged_manuscript_source": prior["unchanged_manuscript_source"],
        "unchanged_main_ledger": prior["unchanged_main_ledger"],
        "post_execution_dispatchers": dispatchers,
        "natural_worker_receipt_sha256": file_identity(archive / "natural-result.json")["sha256"],
        "all_first_step_logs_equal": True,
        "diagnostic_selection_sha256": digest(natural["selection"]),
        "bindings": [
            identity(p, root)
            for p in sorted(archive.rglob("*"))
            if p.is_file() and p.name != "validation.json"
        ]
        + [
            identity(root / p, root)
            for p in (
                *DOCS,
                "configs/dense_primary_v3_data_amendment.json",
                "reports/dense-primary-v3/input-bindings.json",
                "scripts/validate_dense_revised_natural.py",
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
                "diagnostic_runs",
                "diagnostic_checkpoints",
                "formal_replication_ready",
            )
        }
    )


if __name__ == "__main__":
    main()
