"""Read-only validation of candidate fresh-process replay and admission boundaries."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from scripts import dense_run_identity_proposal as proposal
from scripts.pause_dense_evaluation_for_audit import CHAIN, LEDGER, LEDGER_SHA
from scripts.pause_dense_evaluation_for_audit import inspect as inspect_process
from scripts.validate_dense_canonical_replay import check_observations
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity, load

ARCHIVE = Path("reports/engineering-archive/dense-predeployment-checks-v1")
PRIOR = Path("reports/engineering-archive/dense-correction-candidate-v1/validation.json")
PRIOR_SHA = "df1210f30ad57e8950050a6530eba755fe34267f88074389fb655a6b6e7aff7f"
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")
RUNS = ("verified-adamw-3e-5", "verified-muon-3e-4", "verified-normuon-3e-4")
CHANGES = {
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
}
NEGATIVE_CONTROLS = {"optimizer_lr_negative_control", "accumulation_negative_control"}


def check_admission(payload):
    assert payload["audit_execution_complete"] is True
    assert payload["full_recipe_resume_contract_passed"] is False
    assert payload["scientific_completion"] is payload["production_deployed"] is False
    assert payload["model_updates_executed"] == 0
    assert payload["proposal_integrated_into_trainer_or_launcher"] is False
    assert payload["source_metadata_and_data_unchanged"] is True
    assert payload["unchecked_change_cases"] == 30 and payload["negative_controls_rejected"] == 6
    assert len(payload["records"]) == 36
    assert {(x["run_id"], x["change"]) for x in payload["records"]} == {
        (run, change) for run in RUNS for change in CHANGES | NEGATIVE_CONTROLS
    }
    for row in payload["records"]:
        actual = row["actual_preload_gate"]
        assert actual["accepted_preload_gate"] is (row["change"] in CHANGES)
        if actual["accepted_preload_gate"]:
            assert actual["stopped_before_setup"] is True
        assert row["proposed_full_identity_rejected"] is True
    assert payload["proposal_comparison_checks_passed"] == 42
    assert len(payload["prototype_cases"]) == 42
    assert {(x["run_id"], x["case"]) for x in payload["prototype_cases"]} == {
        (run, change)
        for run in RUNS
        for change in CHANGES | NEGATIVE_CONTROLS | {"original", "identical_relocation"}
    }
    for row in payload["prototype_cases"]:
        expected = payload["expected_identities"][row["run_id"]]
        equal = proposal.canonical(expected) == proposal.canonical(row["requested_identity"])
        assert equal is row["should_equal"]
        assert equal is (row["case"] in {"original", "identical_relocation"})
    return {
        "unchecked_admission_changes": 30,
        "existing_negative_controls_rejected": 6,
        "prototype_exact_comparisons": 42,
        "prototype_integrated": False,
    }


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
    parent_root = root / PRIOR.parent
    baseline = load(parent_root / "resume/result.json")
    fresh = load(archive / "fresh-process/result.json")
    assert fresh["audit_execution_complete"] is fresh["strict_bitwise_replay_passed"] is True
    assert (
        fresh["scientific_completion"]
        is fresh["production_deployed"]
        is fresh["baseline_training_performed"]
        is False
    )
    assert fresh["all_source_checkpoints_unchanged"] is True
    compare_binding(fresh["baseline_control"], parent_root / "resume/result.json", root)
    compare_binding(fresh["source_manifest"], parent_root / "prepared-source-v2.json", root)
    assert fresh["baseline_torchrun_id"] == "9174d58f-4230-4e1b-add2-3a9f6f76ebcc"
    assert [x["rank"] for x in fresh["process_group"]] == [0, 1, 2, 3]
    assert len({x["torchrun_id"] for x in fresh["process_group"]}) == 1
    assert all(x["torchrun_id"] != fresh["baseline_torchrun_id"] for x in fresh["process_group"])
    assert [(x["run_id"], x["resume_step"]) for x in fresh["records"]] == [
        (run, step) for run in RUNS for step in (1, 2)
    ]
    exports = {x["run_id"]: x for x in baseline["baseline_exports"]}
    external, sources = [], []
    expected = {}
    for run, export in exports.items():
        for rank, item in enumerate(export["ranks"]):
            external.append(compare_binding(item, Path(item["path"]), root))
            expected[run, rank] = load(Path(item["path"]))["expected"]
        for rows in export["checkpoint_files"].values():
            external.extend(compare_binding(x, Path(x["path"]), root) for x in rows)
    for record in fresh["records"]:
        assert record["strict_bitwise_replay_all_ranks"] is True
        assert record["argument_overrides"]["dataloader_num_workers"] == 8
        assert [x["rank"] for x in record["ranks"]] == [0, 1, 2, 3]
        for rank in record["ranks"]:
            assert (
                rank["comparison"]["exact"] is True
                and rank["comparison"]["first_mismatch_paths"] == []
            )
            item = rank["fingerprint"]
            external.append(compare_binding(item, Path(item["path"]), root))
            actual = load(Path(item["path"]))
            assert actual["run_id"] == record["run_id"] and actual["rank"] == rank["rank"]
            assert actual["step"] == record["resume_step"]
            assert (
                actual["actual"]
                == expected[record["run_id"], rank["rank"]][str(record["resume_step"])]
            )
    check_observations(archive / "fresh-process", [2, 1] * 3, 6)
    sources.extend(compare_binding(x, Path(x["path"]), root) for x in fresh["source_bindings"])
    admission = load(archive / "admission/result.json")
    admission_summary = check_admission(admission)
    compare_binding(admission["parent_validation"], root / PRIOR, root)
    compare_binding(admission["source_manifest"], parent_root / "prepared-source-v2.json", root)
    sources.extend(compare_binding(x, Path(x["path"]), root) for x in admission["source_bindings"])
    external.extend(
        compare_binding(x, Path(x["path"]), root)
        for x in admission["authenticated_source_metadata"]
    )
    for entry in admission["diagnostic_data"]:
        assert proposal.dataset_files(Path(entry["root"])) == entry["identity"]["files"]
        assert entry["identity"]["rows"] == 288
    original, relocated, changed = [x["identity"] for x in admission["diagnostic_data"]]
    assert original == relocated and original != changed
    initial = load(archive / "admission-first/result.json")
    assert initial["full_recipe_resume_contract_passed"] is False
    assert (
        initial["unchecked_change_cases"] == 30
        and initial["proposal_comparison_checks_passed"] == 42
    )
    for item in initial["source_bindings"]:
        path = Path(item["path"])
        if path.name == "audit_prepared_resume_identity.py":
            path = archive / "admission-first/auditor.py"
        sources.append(compare_binding(item, path, root))
    for item in prior["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            assert identity(base / item["path"], root)["sha256"] == item["sha256"]
    candidate = Path(load(parent_root / "prepared-source-v2.json")["candidate_root"])
    for item in load(parent_root / "prepared-source-v2.json")["candidate_sources"]:
        sources.append(compare_binding(item["identity"], candidate / item["relative_path"], root))
    commands = load(archive / "commands.json")
    assert commands["fresh_process"]["observed_launcher_exit_code"] == 0
    assert commands["admission"]["observed_exit_code"] == 1
    dispatchers = [
        inspect_process(pid, start, entry, CHAIN[i - 1][0] if i else 1)
        for i, (pid, start, entry) in enumerate(CHAIN)
    ]
    assert all(x["state"] == "T" for x in dispatchers)
    assert identity(LEDGER, root)["sha256"] == LEDGER_SHA
    suite = next(ET.parse(archive / "tests/isolated-full.xml").getroot().iter("testsuite")).attrib
    assert int(suite["tests"]) == 1415 and all(
        int(suite[k]) == 0 for k in ("failures", "errors", "skipped")
    )
    paper = identity(root / "paper/main.tex", root)
    assert paper["sha256"] == prior["unchanged_manuscript_source"]["sha256"]
    return {
        "scope": "engineering_predeployment_checks_evidence_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "scientific_completion": False,
        "formal_replication_ready": False,
        "production_deployed": False,
        "source_published": False,
        "full_recipe_guard_integrated": False,
        "summary": {
            "new_process_group_exact_replay_checks": 24,
            "new_baseline_training": False,
            "admission": admission_summary,
        },
        "prior_validation": identity(root / PRIOR, root),
        "prior_bindings_preserved": preserved,
        "executed_source_checks": sources,
        "external_checkpoint_fingerprint_metadata_checks": external,
        "external_payloads_uploaded": False,
        "unchanged_live_core": prior["unchanged_live_core"],
        "post_execution_dispatchers": dispatchers,
        "unchanged_main_ledger": identity(LEDGER, root),
        "unchanged_manuscript_source": paper,
        "isolated_test_suite": suite,
        "bindings": [
            identity(p, root)
            for p in sorted(archive.rglob("*"))
            if p.is_file() and p.name != "validation.json"
        ]
        + [
            identity(root / p, root)
            for p in (
                *DOCS,
                "scripts/validate_dense_predeployment_checks.py",
                "scripts/dense_run_identity_proposal.py",
                "tests/test_dense_run_identity_proposal.py",
            )
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new validation receipt path")
    payload = verify(args.repository.resolve())
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "artifact_validation_passed",
                    "scientific_completion",
                    "formal_replication_ready",
                    "summary",
                    "isolated_test_suite",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
