"""Read-only integrity and scope audit of the GPU single-update localization."""

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

ARCHIVE = Path("reports/engineering-archive/dense-gpu-replay-localization-v1")
PRIOR = Path("reports/engineering-archive/dense-gpu-correctness-v1/validation.json")
PRIOR_SHA = "c470078d3e32017fd1640f9fbad9411f28de1a72d9ab7c9f04d3ada57b251ddd"
RUNS = ("padded-adamw-3e-5", "padded-muon-3e-4", "padded-normuon-3e-4")


def identity(path, root):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Require an ordinary evidence file: {path}")
    with path.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha,
    }


def load(path):
    return json.loads(path.read_text())


def compare_binding(item, path, root):
    actual = identity(path, root)
    if any(actual[k] != item[k] for k in ("bytes", "sha256")):
        raise ValueError(f"Evidence bytes changed: {path}")
    return actual


def summarize(case):
    assert len(case["ranks"]) == 4
    first = case["ranks"][0]
    for rank, item in enumerate(case["ranks"]):
        assert item["rank"] == rank
        for key in (
            "same_entry_exact",
            "baseline_direct_update_matches_actual_exactly",
            "resumed_direct_update_matches_actual_exactly",
            "same_gradient_repeat_exact",
        ):
            assert item[key] is True
        assert len(item["names"]) == 134 and len(set(item["names"])) == 134
        assert len(item["baseline_clipped_hashes"]) == len(item["resumed_clipped_hashes"]) == 134
        for key in (
            "names",
            "layout",
            "baseline_clipped_hashes",
            "resumed_clipped_hashes",
            "post_update_weight_pairs",
            "clipped_gradient_pairs",
        ):
            assert item[key] == first[key]
    traces = first["operator_trace_pairs"]
    assert len(traces) == (0 if case["run_id"] == RUNS[0] else 88)
    stages = {}
    if traces:
        assert len({t["name"] for t in traces}) == 88
        for trace in traces:
            assert trace["trace_matches_production"] is True
            assert trace["stages"]["denominator_bf16"]["bitwise_equal"] is True
            # Identical BF16 inputs must not acquire unexplained downstream differences.
            if trace["stages"]["cast_bf16"]["bitwise_equal"]:
                for key in ("normalized_bf16", "newton_schulz_5", "operator_update"):
                    assert trace["stages"][key]["bitwise_equal"] is True
        for stage in traces[0]["stages"]:
            rows = [t["stages"][stage] for t in traces]
            stages[stage] = {
                "different_matrices": sum(not x["bitwise_equal"] for x in rows),
                "different_elements": sum(x["unequal_elements"] for x in rows),
                "max_absolute_error": max(x["max_absolute_error"] for x in rows),
                "max_per_matrix_relative_l2_error": max(x["relative_l2_error"] for x in rows),
            }
    return {
        "run_id": case["run_id"],
        "exact_direct_reproductions": 8,
        "exact_same_gradient_repeats": 4,
        "clipped_gradient_max_absolute_error": max(
            x["max_absolute_error"] for x in first["clipped_gradient_pairs"].values()
        ),
        "immediate_weight_max_absolute_error": max(
            x["max_absolute_error"] for x in first["post_update_weight_pairs"].values()
        ),
        "hidden_matrix_stages": stages,
    }


def verify(root):
    if sys.flags.optimize:
        raise RuntimeError("Audit assertions must not be disabled")
    archive = root / ARCHIVE
    prior_identity = identity(root / PRIOR, root)
    assert prior_identity["sha256"] == PRIOR_SHA
    prior = load(root / PRIOR)
    previous_checks = []
    for item in prior["bindings"]:
        path = Path(item["path"])
        target = (
            archive / "before" / path
            if str(path) in ("AGENTS.md", "PROJECT_STATUS.md", "CURRENT_PROGRESS.json", "README.md")
            else root / path
        )
        previous_checks.append(compare_binding(item, target, root))
    report, parent = load(archive / "result.json"), load(archive / "gpu-resume.json")
    assert report["scope"] == "engineering_gpu_replay_update_localization"
    assert report["audit_execution_complete"] is True
    assert report["scientific_completion"] is report["production_deployed"] is False
    assert report["upstream_strict_replay_passed"] is False
    compare_binding(report["parent_result"], archive / "gpu-resume.json", root)
    compare_binding(report["source"], root / "scripts/audit_dense_gpu_replay_update.py", root)
    assert parent["audit_execution_complete"] is parent["source_checkpoint_unchanged"] is True
    assert parent["strict_bitwise_replay_passed"] is parent["scientific_completion"] is False
    assert len(parent["records"]) == 6
    for case in parent["records"]:
        assert case["argument_overrides"]["dataloader_num_workers"] == 8
        assert len(case["ranks"]) == 4
        for rank in case["ranks"]:
            comparison = rank["comparison"]
            assert comparison["exact_loaded_entry_state"] is True
            assert comparison["exact_row_order_and_rank_rng"] is True
            assert comparison["exact_scheduler_and_final_progress"] is True
            assert comparison["bitwise_replay"]["passed"] is False
    sources = [*report["source_bindings"], *parent["source_bindings"]]
    checked_sources = [compare_binding(x, Path(x["path"]), root) for x in sources]
    summaries, payloads = [], []
    assert len(report["records"]) == 3
    for run_id, binding in zip(RUNS, report["records"], strict=True):
        case_path = archive / f"{run_id}.json"
        compare_binding(binding, case_path, root)
        case = load(case_path)
        assert case["run_id"] == run_id
        summaries.append(summarize(case))
        item = case["ranks"][0]["captured_gradient_payload"]
        payloads.append(compare_binding(item, Path(item["path"]), root))
    failed_messages = []
    for number, fragment in ((1, "different types"), (2, "param_groups/0/lr")):
        attempt = archive / f"attempt-{number}"
        failed = load(attempt / "gpu-resume.json")
        wrapper = load(attempt / "result.json")
        assert not failed["audit_execution_complete"] and not wrapper["audit_execution_complete"]
        assert fragment in failed["failure"]["message"]
        compare_binding(wrapper["source"], attempt / "failed-source.py", root)
        failed_messages.append(failed["failure"])
    suites = list(ET.parse(archive / "pytest-full-absolute-path.xml").getroot().iter("testsuite"))
    assert len(suites) == 1
    suite = suites[0].attrib
    assert int(suite["tests"]) == 1337
    assert all(int(suite[k]) == 0 for k in ("failures", "errors", "skipped"))
    first_suite = next(
        ET.parse(archive / "pytest-full-relative-path.xml").getroot().iter("testsuite")
    )
    assert int(first_suite.attrib["tests"]) == 1337 and int(first_suite.attrib["failures"]) == 2
    assert {x.attrib["name"] for x in first_suite if x.find("failure") is not None} == {
        "test_complete_synthetic_findings_fit_the_naacl_paper_layout",
        "test_complete_synthetic_candidate_paper_compiles_within_layout_gate",
    }
    formal = load(root / "configs/dense_no_packing_execution_protocol.json")["source_bindings"][
        "formal_matrix"
    ]
    assert identity(root / formal["path"], root)["sha256"] == formal["sha256"]
    unchanged_core = []
    for item in prior["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            assert identity(base / item["path"], root)["sha256"] == item["sha256"]
        unchanged_core.append(item)
    dispatchers = [
        inspect_process(pid, start, entrypoint, CHAIN[i - 1][0] if i else 1)
        for i, (pid, start, entrypoint) in enumerate(CHAIN)
    ]
    assert all(p["state"] == "T" for p in dispatchers)
    assert identity(LEDGER, root)["sha256"] == LEDGER_SHA
    assert payloads[1]["sha256"] == payloads[2]["sha256"]
    return {
        "scope": "engineering_gpu_replay_localization_artifact_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "scientific_completion": False,
        "whole_training_stack_accepted": False,
        "runtime_deployed": False,
        "source_published": False,
        "first_gradient_difference_source_isolated": False,
        "update_amplification_localized_on_fixture": True,
        "summaries": summaries,
        "prior_validation": prior_identity,
        "prior_bindings_preserved": previous_checks,
        "executed_source_checks": checked_sources,
        "captured_external_gradient_payloads": payloads,
        "payload_boundary": "Gradient tensor payloads and freshly generated diagnostic checkpoints remain in named temporary directories, not in Git or uploaded to HF. The archived JSON measurements and exact executed source are portable; full numerical re-execution also needs those digest-bound inputs or a fresh explicitly labeled capture.",
        "unchanged_live_core": unchanged_core,
        "post_execution_dispatchers": dispatchers,
        "unchanged_main_ledger": identity(LEDGER, root),
        "test_suite": suite,
        "retained_harness_failures": failed_messages,
        "bindings": [
            identity(p, root)
            for p in sorted(archive.rglob("*"))
            if p.is_file()
            and p.name
            not in ("validation.json", "issue-publication-receipt.json", "issue-body-after.md")
        ]
        + [
            identity(root / p, root)
            for p in (
                "AGENTS.md",
                "PROJECT_STATUS.md",
                "README.md",
                "CURRENT_PROGRESS.json",
                "scripts/validate_dense_gpu_replay_localization.py",
                "tests/test_dense_gpu_replay_update.py",
            )
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new receipt path")
    result = verify(args.repository.resolve())
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "artifact_validation_passed": True,
                "scientific_completion": False,
                "runs": len(result["summaries"]),
                "prior_bindings_preserved": len(result["prior_bindings_preserved"]),
            }
        )
    )


if __name__ == "__main__":
    main()
