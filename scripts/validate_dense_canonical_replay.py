"""Read-only source/evidence audit of the canonical-reduction GPU controls."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from scripts.pause_dense_evaluation_for_audit import CHAIN, LEDGER, LEDGER_SHA
from scripts.pause_dense_evaluation_for_audit import inspect as inspect_process
from scripts.validate_dense_gpu_replay_localization import RUNS, compare_binding, identity, load

ARCHIVE = Path("reports/engineering-archive/dense-canonical-replay-v1")
PRIOR = Path("reports/engineering-archive/dense-gradient-origin-v1/validation.json")
PRIOR_SHA = "8ebba68f9623660c3251a8596e3c72fc64d57cb67eaba6de3ce3bc44746a76ac"
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")


def check_observations(directory, counts, attention_count):
    result = []
    for rank in range(4):
        payload = load(directory / f"observations-rank-{rank}.json")
        assert payload["rank"] == rank
        observations = payload["observations"]
        assert len(observations["attention_controls"]) == attention_count
        assert all(
            x == {"attention_modules": 22, "before": False, "after": True}
            for x in observations["attention_controls"]
        )
        reducers = observations["reducers"]
        assert [x["completed_reductions"] for x in reducers] == counts
        for reducer, count in zip(reducers, counts, strict=True):
            assert reducer["world_size"] == 4 and len(reducer["topology"]) == 134
            assert reducer["canonical_elements"] == 149014272
            assert [x["name"] for x in reducer["topology"]] == sorted(
                x["name"] for x in reducer["topology"]
            )
            assert len(reducer["records"]) == count
            assert [x["update_index"] for x in reducer["records"]] == list(range(count))
            assert [x["incoming_ddp_buckets"] for x in reducer["records"]] == [1] + [16] * (
                count - 1
            )
            assert all(x["collectives"] == 1 and x["predivision"] == 4 for x in reducer["records"])
        result.append(payload)
    assert all(x["observations"] == result[0]["observations"] for x in result)
    return result[0]["observations"]


def check_gradient_parent(parent, long_inputs=False):
    assert parent["audit_execution_complete"] is True
    assert parent["normalization_acceptance_passed"] is True
    assert parent["scientific_completion"] is False
    assert parent["source_checkpoint_unchanged"] is True
    tolerances = parent["prior_gradient_tolerances" if long_inputs else "tolerances"]
    assert tolerances["gradient_atol"] == 5e-6 and tolerances["gradient_rtol"] == 5e-4
    assert [x["run_id"] for x in parent["records"]] == list(RUNS)
    max_raw, max_clipped, max_memory = 0.0, 0.0, 0
    for run in parent["records"]:
        assert run["all_raw_and_clipped_checks_pass"] is True
        assert run["final_weights_rank_identical"] is True
        assert len(run["records"]) == 3
        assert [x["global_queries"] for x in run["records"]] == [128, 128, 32]
        for step, record in enumerate(run["records"]):
            count = [4, 4, 1][step]
            assert record["trainer_divisors"] == [[count] * count] * 4
            assert record["accelerator_divisor"] == 1
            assert record["raw_gradients_rank_identical"] is True
            assert record["actual_reference_token_inputs_identical"] is True
            for key in ("raw", "clipped"):
                assert record[key]["all_parameters_match_prior_tolerance"] is True
                assert len(record[key]["parameters"]) == 134
            max_raw = max(
                max_raw, max(x["max_absolute_error"] for x in record["raw"]["parameters"])
            )
            max_clipped = max(
                max_clipped, max(x["max_absolute_error"] for x in record["clipped"]["parameters"])
            )
        if long_inputs:
            ranks = run["rank_stress_coverage"]
            assert len(ranks) == 4
            assert all(max(x["step_max_token_lengths"]) == 8192 for x in ranks)
            assert all(x["all_raw_and_clipped_checks_pass"] for x in ranks)
            max_memory = max(max_memory, max(x["peak_allocated_bytes"] for x in ranks))
    return {
        "raw_and_clipped_full_parameter_checks": 9,
        "max_raw_absolute_error": max_raw,
        "max_clipped_absolute_error": max_clipped,
        "max_allocated_bytes": max_memory if long_inputs else None,
    }


def verify(root):
    if sys.flags.optimize:
        raise RuntimeError("Audit assertions must not be disabled")
    archive = root / ARCHIVE
    assert identity(root / PRIOR, root)["sha256"] == PRIOR_SHA
    prior = load(root / PRIOR)
    preserved = []
    for item in prior["bindings"]:
        path = Path(item["path"])
        target = archive / "before" / path if str(path) in DOCS else root / path
        preserved.append(compare_binding(item, target, root))
    source_checks, summaries = [], {}
    for label in ("normalization", "same-process", "max-context"):
        directory = archive / label
        wrapper, parent = load(directory / "result.json"), load(directory / "parent.json")
        assert wrapper["audit_execution_complete"] is wrapper["acceptance_passed"] is True
        assert wrapper["scientific_completion"] is wrapper["production_deployed"] is False
        compare_binding(wrapper["parent_result"], directory / "parent.json", root)
        source_checks.append(
            compare_binding(wrapper["source"], Path(wrapper["source"]["path"]), root)
        )
        for record in wrapper["source_bindings"] + parent["source_bindings"]:
            source_checks.append(compare_binding(record, Path(record["path"]), root))
        if label == "same-process":
            assert wrapper["target"] == "resume"
            assert parent["strict_bitwise_replay_passed"] is True
            assert (
                parent["audit_execution_complete"] is parent["source_checkpoint_unchanged"] is True
            )
            assert len(parent["records"]) == 6
            assert [(x["run_id"], x["resume_step"]) for x in parent["records"]] == [
                (run_id, step) for run_id in RUNS for step in (1, 2)
            ]
            for record in parent["records"]:
                assert record["strict_bitwise_replay_all_ranks"] is True
                assert record["argument_overrides"]["dataloader_num_workers"] == 8
                assert len(record["ranks"]) == 4
                for row in record["ranks"]:
                    comparison = row["comparison"]
                    assert all(
                        comparison[key] is True
                        for key in (
                            "exact_loaded_entry_state",
                            "exact_row_order_and_rank_rng",
                            "exact_scheduler_and_final_progress",
                        )
                    )
                    assert comparison["bitwise_replay"]["passed"] is True
            check_observations(directory, [3, 2, 1] * 3, 9)
            summaries[label] = {"continuations": 6, "exact_rank_checks": 24}
        else:
            summaries[label] = check_gradient_parent(parent, label == "max-context")
            check_observations(directory, [3] * 3, 6)
    baseline = load(archive / "same-process/result.json")
    fresh = load(archive / "fresh-process/result.json")
    assert fresh["audit_execution_complete"] is fresh["strict_bitwise_replay_passed"] is True
    assert fresh["scientific_completion"] is fresh["production_deployed"] is False
    assert fresh["baseline_training_performed"] is False
    assert fresh["all_source_checkpoints_unchanged"] is True
    compare_binding(fresh["baseline_control"], archive / "same-process/result.json", root)
    commands = load(archive / "commands.json")
    baseline_group = commands["same_process"]["torchrun_id"]
    assert len(fresh["process_group"]) == 4
    assert all(x["torchrun_id"] != baseline_group for x in fresh["process_group"])
    assert len({x["torchrun_id"] for x in fresh["process_group"]}) == 1
    assert commands["same_process"]["observed_launcher_exit_code"] == 0
    assert commands["fresh_process"]["observed_launcher_exit_code"] == 0
    assert len(fresh["records"]) == 6
    assert [(x["run_id"], x["resume_step"]) for x in fresh["records"]] == [
        (run_id, step) for run_id in RUNS for step in (1, 2)
    ]
    exports = {x["run_id"]: x for x in baseline["baseline_exports"]}
    external = []
    expected_fingerprints = {}
    for run_id, export in exports.items():
        for rank, record in enumerate(export["ranks"]):
            external.append(compare_binding(record, Path(record["path"]), root))
            payload = load(Path(record["path"]))
            assert payload["run_id"] == run_id and payload["rank"] == rank
            expected_fingerprints[run_id, rank] = payload["expected"]
        for records in export["checkpoint_files"].values():
            for record in records:
                external.append(compare_binding(record, Path(record["path"]), root))
    for record in fresh["records"]:
        assert record["strict_bitwise_replay_all_ranks"] is True
        assert record["argument_overrides"]["dataloader_num_workers"] == 8
        assert len(record["ranks"]) == 4
        for rank, row in enumerate(record["ranks"]):
            assert row["rank"] == rank and row["comparison"]["exact"] is True
            assert row["comparison"]["first_mismatch_paths"] == []
            binding = row["fingerprint"]
            external.append(compare_binding(binding, Path(binding["path"]), root))
            actual = load(Path(binding["path"]))
            assert actual["rank"] == rank and actual["run_id"] == record["run_id"]
            assert actual["step"] == record["resume_step"]
            assert (
                actual["actual"]
                == expected_fingerprints[record["run_id"], rank][str(record["resume_step"])]
            )
    for record in fresh["source_bindings"]:
        source_checks.append(compare_binding(record, Path(record["path"]), root))
    check_observations(archive / "fresh-process", [2, 1] * 3, 6)
    summaries["fresh-process"] = {"continuations": 6, "exact_rank_checks": 24}
    suite = next(ET.parse(archive / "pytest-full.xml").getroot().iter("testsuite")).attrib
    assert int(suite["tests"]) == 1367 and all(
        int(suite[k]) == 0 for k in ("failures", "errors", "skipped")
    )
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
        "scope": "engineering_canonical_control_gpu_artifact_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "scientific_completion": False,
        "whole_training_stack_accepted": False,
        "production_deployed": False,
        "source_published": False,
        "summary": summaries,
        "boundary": "Exact controlled replay, including a new process group on the same host, and independently checked global-average gradients for short/8192-token fixtures. The control deliberately fixes owned backward and reduction order. Prior default-path failures and the production double-normalization defect remain unchanged. Not historical optimizer continuation, another host, long-horizon training, a performance recommendation or a scientific optimizer finding.",
        "prior_validation": identity(root / PRIOR, root),
        "prior_bindings_preserved": preserved,
        "executed_source_checks": source_checks,
        "external_fingerprints_and_checkpoints": external,
        "external_payloads_uploaded": False,
        "unchanged_live_core": prior["unchanged_live_core"],
        "post_execution_dispatchers": dispatchers,
        "unchanged_main_ledger": identity(LEDGER, root),
        "unchanged_manuscript_source": paper,
        "test_suite": suite,
        "bindings": [
            identity(p, root)
            for p in sorted(archive.rglob("*"))
            if p.is_file()
            and p.name
            not in ("validation.json", "issue-body-after.md", "issue-publication-receipt.json")
        ]
        + [identity(root / p, root) for p in (*DOCS, "scripts/validate_dense_canonical_replay.py")],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new audit receipt path")
    result = verify(args.repository.resolve())
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "artifact_validation_passed",
                    "scientific_completion",
                    "summary",
                    "test_suite",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
