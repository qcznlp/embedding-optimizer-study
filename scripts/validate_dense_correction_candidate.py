"""Read-only evidence validation for the undeployed Dense numerical candidate."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import yaml

from scripts.pause_dense_evaluation_for_audit import CHAIN, LEDGER, LEDGER_SHA
from scripts.pause_dense_evaluation_for_audit import inspect as inspect_process
from scripts.validate_dense_canonical_replay import check_observations
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity, load

ARCHIVE = Path("reports/engineering-archive/dense-correction-candidate-v1")
PRIOR = Path("reports/engineering-archive/dense-canonical-replay-v1/validation.json")
PRIOR_SHA = "a1e561251e0fc424a0d2674056b71435b4800573961971a3d0e5958e30195583"
SOURCE_SHA = "559c6df47af4bc8885665334865b1e87c36b18d6dfb4bb75b95faa41440b7a6f"
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")
RUNS = ("verified-adamw-3e-5", "verified-muon-3e-4", "verified-normuon-3e-4")
EXPECTED_FAILURES = {
    "tests.test_corrected_execution_sensitivity.test_checked_in_sensitivity_protocol_binds_sources_and_history",
    "tests.test_corrected_execution_sensitivity.test_build_report_audits_sources_and_writes_all_tables",
    "tests.test_corrected_geometry.test_checked_in_corrected_geometry_protocol_matches_current_sources",
    "tests.test_corrected_outcome_summary.test_checked_in_corrected_outcome_protocol_matches_current_sources",
    "tests.test_corrected_retrieval_bridge.test_checked_in_bridge_protocol_binds_current_sources",
    "tests.test_corrected_retrieval_bridge.test_build_report_audits_and_writes_complete_bridge",
    "tests.test_state_operator_factorial.test_generate_and_audit_six_two_run_matrices",
    "tests.test_state_operator_factorial.test_gradient_export_rejects_manifest_without_padded_receipt",
    "tests.test_state_operator_factorial_completion.test_checked_in_completion_contract_is_source_bound_and_exact",
    "tests.test_state_operator_factorial_completion.test_main_completion_gate_requires_the_entire_exact_parent",
    "tests.test_state_operator_factorial_protocol.test_state_operator_factorial_implementation_is_source_bound",
}


def scope(payload, key):
    assert payload["audit_execution_complete"] is payload[key] is True
    assert payload["scientific_completion"] is payload["production_deployed"] is False
    assert "failure" not in payload


def check_matrix(old, new):
    old, new = copy.deepcopy(old), copy.deepcopy(new)
    assert new["common"].pop("numerical_policy") == "single-normalization-deterministic-backward-v1"
    assert new["common"]["output_root"] == "outputs/dense-correctness-v2"
    new["common"]["output_root"] = old["common"]["output_root"]
    assert len(old["optimizers"]) == len(new["optimizers"]) == 12
    for left, right in zip(old["optimizers"], new["optimizers"], strict=True):
        assert right["id"] == left["id"].replace("padded-", "verified-", 1)
        right["id"] = left["id"]
        if right["name"] == "normuon":
            assert right.pop("ns_implementation") == "unfused-bfloat16-additive-eps-v2"
    assert old == new, "Undeclared recipe change"


def check_reference(payload, device):
    scope(payload, "candidate_reference_conformance_passed")
    assert payload["device"] == device
    assert payload["tested_updates"] == payload["candidate_exact_weight_state_checks"] == 96
    assert payload["muon_unchanged_update_state_checks"] == 96
    assert payload["legacy_nonconforming_updates"] == 44
    assert (
        payload["upstream"]["sha256"]
        == "706c1a35fb35342f6ff207f8310b814a1c8f55c6ffba0e843537db434a696141"
    )
    assert len(payload["records"]) == 24
    assert {(tuple(c["shape"]), c["scale"]) for c in payload["records"]} == {
        (shape, scale)
        for shape in ((8, 4), (4, 8), (8, 8), (64, 32))
        for scale in (1.0, 1e-3, 1e-5, 1e-7, 1e-9, 0.0)
    }
    legacy_failures = 0
    for case in payload["records"]:
        assert [x["step"] for x in case["steps"]] == [1, 2, 3, 4]
        for row in case["steps"]:
            assert row["candidate_weight_and_state_bitwise_equal_to_official"] is True
            assert row["muon_update_and_state_bitwise_unchanged"] is True
            assert row["incoming_gradient_unchanged"] is True
            legacy_failures += not row["legacy_normuon_conforms_to_reference_tolerance"]
    assert legacy_failures == 44


def check_gradients(payload, fixture):
    scope(payload, "normalization_acceptance_passed")
    assert payload["source_checkpoint_unchanged"] is True and payload["fixture"] == fixture
    assert payload["tolerances"]["gradient_atol"] == 5e-6
    assert payload["tolerances"]["gradient_rtol"] == 5e-4
    assert [x["run_id"] for x in payload["records"]] == list(RUNS)
    maxima = {"raw": 0.0, "clipped": 0.0}
    for run in payload["records"]:
        assert run["all_raw_and_clipped_checks_pass"] is run["final_weights_rank_identical"] is True
        assert [x["global_queries"] for x in run["records"]] == [128, 128, 32]
        for index, record in enumerate(run["records"]):
            count = (4, 4, 1)[index]
            assert record["trainer_divisors"] == [[count] * count] * 4
            assert record["accelerator_divisor"] == 1
            assert (
                record["raw_gradients_rank_identical"]
                is record["actual_reference_token_inputs_identical"]
                is True
            )
            for kind in maxima:
                assert record[kind]["all_parameters_match_prior_tolerance"] is True
                assert len(record[kind]["parameters"]) == 134
                assert all(x["matches_prior_tolerance"] for x in record[kind]["parameters"])
                maxima[kind] = max(
                    maxima[kind], max(x["max_absolute_error"] for x in record[kind]["parameters"])
                )
        if fixture == "max_context":
            assert [x["rank"] for x in run["rank_stress_coverage"]] == [0, 1, 2, 3]
            assert all(
                max(x["step_max_token_lengths"]) == 8192 and x["all_raw_and_clipped_checks_pass"]
                for x in run["rank_stress_coverage"]
            )
    return {"raw_and_clipped_checks": 9, "maximum_absolute_errors": maxima}


def check_candidate_tests(path):
    parsed = ET.parse(path).getroot()
    suite = next(parsed.iter("testsuite")).attrib
    assert int(suite["tests"]) == 923 and int(suite["failures"]) == 11
    assert int(suite["errors"]) == int(suite["skipped"]) == 0
    observed = set()
    for case in parsed.iter("testcase"):
        failure = case.find("failure")
        if failure is not None:
            observed.add(f"{case.attrib['classname']}.{case.attrib['name']}")
            message = failure.attrib["message"]
            assert message.startswith("ValueError:") and any(
                term in message for term in ("source_bindings", "source binding", "source contract")
            )
    assert observed == EXPECTED_FAILURES
    return {
        **suite,
        "full_suite_passed": False,
        "unwaived_source_contract_failures": sorted(observed),
    }


def verify(root):
    if sys.flags.optimize:
        raise RuntimeError("Audit assertions must not be disabled")
    archive = root / ARCHIVE
    assert identity(root / PRIOR, root)["sha256"] == PRIOR_SHA
    prior = load(root / PRIOR)
    preserved = [
        compare_binding(
            item,
            archive / "before" / item["path"] if item["path"] in DOCS else root / item["path"],
            root,
        )
        for item in prior["bindings"]
    ]
    assert identity(archive / "prepared-source-v2.json", root)["sha256"] == SOURCE_SHA
    source = load(archive / "prepared-source-v2.json")
    candidate = Path(source["candidate_root"])
    sources = []
    for name, directory in (
        ("prepared-source.json", "v1-source"),
        ("prepared-source-v2.json", "candidate-source"),
    ):
        manifest = load(archive / name)
        assert manifest["scientific_completion"] is manifest["production_deployed"] is False
        for item in manifest["candidate_sources"]:
            sources.append(
                compare_binding(item["identity"], archive / directory / item["relative_path"], root)
            )
            if directory == "candidate-source":
                sources.append(
                    compare_binding(item["identity"], candidate / item["relative_path"], root)
                )
    compare_binding(
        source["matrix_source"], candidate / "configs/dense_no_packing_retrain.yaml", root
    )
    check_matrix(
        yaml.safe_load((root / "configs/dense_no_packing_retrain.yaml").read_text()),
        yaml.safe_load(
            (archive / "candidate-source/configs/dense_correctness_candidate.yaml").read_text()
        ),
    )
    summary, external = {}, []
    for device, label in (("cpu", "cpu"), ("cuda", "gpu")):
        payload = load(archive / f"reference/optimizer-{label}-v2.json")
        check_reference(payload, device)
        sources.extend(
            compare_binding(x, Path(x["path"]), root) for x in payload["source_bindings"]
        )
        summary[f"reference_{device}"] = {
            "new_normuon_exact_updates": 96,
            "muon_unchanged_updates": 96,
            "legacy_nonconforming_updates_preserved": 44,
        }
    for label, fixture in (("short", "short"), ("max-context", "max_context")):
        payload = load(archive / label / "result.json")
        summary[label] = check_gradients(payload, fixture)
        compare_binding(payload["source_manifest"], archive / "prepared-source-v2.json", root)
        sources.extend(
            compare_binding(x, Path(x["path"]), root) for x in payload["source_bindings"]
        )
    payload = load(archive / "resume/result.json")
    scope(payload, "strict_bitwise_replay_passed")
    assert payload["source_checkpoint_unchanged"] is True
    assert payload["legacy_checkpoint_rejections"] == list(RUNS)
    assert [(x["run_id"], x["resume_step"]) for x in payload["records"]] == [
        (run, step) for run in RUNS for step in (1, 2)
    ]
    for record in payload["records"]:
        assert record["strict_bitwise_replay_all_ranks"] is True
        assert record["argument_overrides"]["dataloader_num_workers"] == 8
        assert [x["rank"] for x in record["ranks"]] == [0, 1, 2, 3]
        for rank in record["ranks"]:
            comparison = rank["comparison"]
            assert comparison["bitwise_replay"]["passed"] is True
            assert all(
                comparison[x]
                for x in (
                    "exact_loaded_entry_state",
                    "exact_row_order_and_rank_rng",
                    "exact_scheduler_and_final_progress",
                )
            )
    check_observations(archive / "resume", [3, 2, 1] * 3, 9)
    for export in payload["baseline_exports"]:
        for item in export["ranks"] + [
            i for rows in export["checkpoint_files"].values() for i in rows
        ]:
            external.append(compare_binding(item, Path(item["path"]), root))
    sources.extend(compare_binding(x, Path(x["path"]), root) for x in payload["source_bindings"])
    summary["controlled_replay"] = {
        "continuations": 6,
        "exact_rank_checks": 24,
        "new_process_group_tested": False,
    }
    payload = load(archive / "entrypoint/result.json")
    scope(payload, "entrypoint_acceptance_passed")
    assert payload["source_checkpoint_unchanged"] is True
    assert [(x["run_id"], x["resume_step"]) for x in payload["records"]] == [
        (run, step) for run in RUNS for step in (0, 2)
    ]
    for record in payload["records"]:
        assert record["completion"]["final_equals_scheduled_checkpoint"] is True
        for label in ("completion", "resolved_config"):
            item = record["completion"][label]
            external.append(compare_binding(item, Path(item["path"]), root))
        for items in record["completion"]["checkpoint_files"].values():
            external.extend(compare_binding(x, Path(x["path"]), root) for x in items)
    sources.extend(compare_binding(x, Path(x["path"]), root) for x in payload["source_bindings"])
    summary["actual_entrypoint"] = {
        "baseline_calls": 3,
        "continuation_calls": 3,
        "implementations_patched": False,
    }
    tests = check_candidate_tests(archive / "tests/candidate-base-full-v2.xml")
    for item in prior["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            assert identity(base / item["path"], root)["sha256"] == item["sha256"]
    dispatchers = [
        inspect_process(pid, start, entry, CHAIN[i - 1][0] if i else 1)
        for i, (pid, start, entry) in enumerate(CHAIN)
    ]
    assert all(x["state"] == "T" for x in dispatchers)
    assert identity(LEDGER, root)["sha256"] == LEDGER_SHA
    progress = load(root / "CURRENT_PROGRESS.json")
    assert progress["complete_runs"] == 12 and progress["resumable_checkpoints"] == 60
    paper = identity(root / "paper/main.tex", root)
    assert paper["sha256"] == "4d59c3219589204ec396d5b9d8c6269df1dfc89f29361ddf70ada26100dbbb2b"
    return {
        "scope": "engineering_prepared_dense_correction_evidence_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "scientific_completion": False,
        "whole_training_stack_accepted": False,
        "formal_replication_ready": False,
        "production_deployed": False,
        "source_published": False,
        "summary": summary,
        "candidate_test_suite": tests,
        "prior_validation": identity(root / PRIOR, root),
        "prior_bindings_preserved": preserved,
        "executed_source_checks": sources,
        "external_fingerprints_and_checkpoints": external,
        "external_payloads_uploaded": False,
        "unchanged_live_core": prior["unchanged_live_core"],
        "post_execution_dispatchers": dispatchers,
        "unchanged_main_ledger": identity(LEDGER, root),
        "unchanged_manuscript_source": paper,
        "bindings": [
            identity(p, root)
            for p in sorted(archive.rglob("*"))
            if p.is_file() and p.name != "validation.json"
        ]
        + [
            identity(root / p, root)
            for p in (*DOCS, "scripts/validate_dense_correction_candidate.py")
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
                k: payload[k]
                for k in (
                    "artifact_validation_passed",
                    "scientific_completion",
                    "formal_replication_ready",
                    "summary",
                    "candidate_test_suite",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
