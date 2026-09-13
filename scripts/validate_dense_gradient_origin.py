"""Verify the paired first-gradient origin evidence without GPU execution."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from scripts.pause_dense_evaluation_for_audit import CHAIN, LEDGER, LEDGER_SHA
from scripts.pause_dense_evaluation_for_audit import inspect as inspect_process
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity, load

ARCHIVE = Path("reports/engineering-archive/dense-gradient-origin-v1")
PRIOR = Path("reports/engineering-archive/dense-gpu-replay-localization-v1/validation.json")
PRIOR_SHA = "829f29f1cc132d15fcbcfebf70c2b6c1fcf23bdb01dd6dda87d1ca95117574ed"


def verify(root):
    if sys.flags.optimize:
        raise RuntimeError("Audit assertions must not be disabled")
    archive = root / ARCHIVE
    assert identity(root / PRIOR, root)["sha256"] == PRIOR_SHA
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
    cases, payloads, source_checks = {}, [], []
    for variant, deterministic in (("default", False), ("deterministic", True)):
        case_dir = archive / variant
        wrapper, parent = load(case_dir / "result.json"), load(case_dir / "gpu-resume.json")
        assert wrapper["scope"] == "engineering_first_gradient_origin_bucket_layout_control"
        assert wrapper["scientific_completion"] is wrapper["production_deployed"] is False
        assert wrapper["audit_execution_complete"] is True
        assert wrapper["deterministic_backward_control"] is deterministic
        assert wrapper["run_ids"] == ["padded-muon-3e-4"] and len(wrapper["records"]) == 1
        assert len(wrapper["model_controls"]) == (3 if deterministic else 0)
        assert all(
            x == {"attention_modules": 22, "before": False, "after": True}
            for x in wrapper["model_controls"]
        )
        compare_binding(wrapper["parent_result"], case_dir / "gpu-resume.json", root)
        compare_binding(
            wrapper["source"], root / "scripts/audit_dense_gpu_gradient_origin.py", root
        )
        compare_binding(wrapper["records"][0], case_dir / "first-gradient.json", root)
        assert parent["audit_execution_complete"] is parent["source_checkpoint_unchanged"] is True
        assert parent["scientific_completion"] is parent["strict_bitwise_replay_passed"] is False
        assert len(parent["records"]) == 2 and {x["resume_step"] for x in parent["records"]} == {
            1,
            2,
        }
        for record in parent["records"]:
            assert record["run_id"] == "padded-muon-3e-4"
            assert record["argument_overrides"]["dataloader_num_workers"] == 8
            assert len(record["ranks"]) == 4
            for rank in record["ranks"]:
                comp = rank["comparison"]
                assert comp["exact_loaded_entry_state"] and comp["exact_row_order_and_rank_rng"]
                assert (
                    comp["exact_scheduler_and_final_progress"]
                    and not comp["bitwise_replay"]["passed"]
                )
        for item in wrapper["source_bindings"] + parent["source_bindings"]:
            source_checks.append(compare_binding(item, Path(item["path"]), root))
        case = load(case_dir / "first-gradient.json")
        assert len(case["ranks"]) == 4
        for rank, item in enumerate(case["ranks"]):
            assert item["rank"] == rank and item["exact_entry_state"] is True
            names = item["names"]
            assert len(names) == len(set(names)) == 134
            assert set(item["baseline_hook_counts"].values()) == {9}
            assert set(item["resumed_hook_counts"].values()) == {5}
            assert all(
                len(rows) == 4 and all(x["bitwise_equal"] for x in rows)
                for rows in item["local_microbatch_pairs"].values()
            )
            assert len(item["local_microbatch_pairs"]) == 134
            assert [len(item["baseline_layout"]), len(item["resumed_layout"])] == [16, 1]
            for label in ("baseline_layout", "resumed_layout"):
                flat = [name for bucket in item[label] for name in bucket["names"]]
                assert len(flat) == 134 and set(flat) == set(names)
            for key in (
                "all_local_microbatch_tensors_exact",
                "all_local_accumulated_tensors_exact",
                "both_actual_bucket_replays_exact",
                "same_gradient_same_layout_repeat_exact",
                "fixed_local_gradient_cross_layout_reproduces_resumed_exactly",
                "actual_raw_gradients_differ",
            ):
                assert item[key] is True
            for key in (
                "local_accumulated",
                "baseline_bucket_replay_vs_actual",
                "resumed_bucket_replay_vs_actual",
                "same_input_resumed_layout_vs_actual_resumed",
                "same_input_same_layout_repeat",
            ):
                rows = item["comparisons"][key]
                assert set(rows) == set(names) and all(
                    x["bitwise_equal"] and x["max_absolute_error"] == 0.0 for x in rows.values()
                )
            actual = item["comparisons"]["actual_ddp_raw"]
            assert sum(not x["bitwise_equal"] for x in actual.values()) == 123
            assert max(x["max_absolute_error"] for x in actual.values()) == 1.1920928955078125e-7
            assert item["comparisons"]["same_input_crossed_layout"] == actual
            payload = item["captured_local_sum_payload"]
            payloads.append(compare_binding(payload, Path(payload["path"]), root))
        first = case["ranks"][0]
        for label in ("baseline", "resumed"):
            rows = first["prior_uninstrumented_gradient_comparisons"][label]
            assert len(rows) == 134 and all(x["bitwise_equal"] for x in rows.values())
        compare_binding(
            first["prior_gradient_payload"], Path(first["prior_gradient_payload"]["path"]), root
        )
        cases[variant] = case
    for rank in range(4):
        left, right = cases["default"]["ranks"][rank], cases["deterministic"]["ranks"][rank]
        assert (
            left["captured_local_sum_payload"]["sha256"]
            == right["captured_local_sum_payload"]["sha256"]
        )
        for key in (
            "names",
            "baseline_layout",
            "resumed_layout",
            "local_microbatch_pairs",
            "comparisons",
        ):
            assert left[key] == right[key]
    suite = next(ET.parse(archive / "pytest-full.xml").getroot().iter("testsuite")).attrib
    assert int(suite["tests"]) == 1344 and all(
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
    return {
        "scope": "engineering_first_gradient_origin_artifact_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "scientific_completion": False,
        "whole_training_stack_accepted": False,
        "production_deployed": False,
        "source_published": False,
        "first_difference_origin_isolated_on_fixture": True,
        "summary": {
            "optimizer_recipes": 1,
            "kernel_modes": 2,
            "ranks_per_mode": 4,
            "local_microbatch_tensor_pairs_exact": 4288,
            "actual_bucket_replays_exact": 16,
            "fixed_gradient_layout_swap_reproductions_exact": 8,
            "same_gradient_same_layout_repeats_exact": 8,
            "uninterrupted_bucket_count": 16,
            "first_resumed_bucket_count": 1,
            "raw_parameter_tensors_differ": 123,
            "raw_gradient_max_absolute_error": 1.1920928955078125e-7,
            "full_replay_rank_checks_passed": 0,
            "full_replay_rank_checks_total": 16,
            "prior_uninstrumented_raw_gradients_exact": True,
            "deterministic_backward_did_not_remove_this_short_input_difference": True,
        },
        "scope_boundary": "One shared short-input Muon candidate checkpoint replay, same process group. Exact controlled bucket-layout reconstruction identifies the first gradient difference and links it to the prior uninstrumented capture. This is floating-point reduction-order variation, not a newly demonstrated optimizer/state-restoration defect, independent replication, long-input result, retrieval finding or acceptance of historical training. The separate duplicate-normalization defect and scientific hold remain.",
        "prior_validation": identity(root / PRIOR, root),
        "prior_bindings_preserved": previous_checks,
        "executed_source_checks": source_checks,
        "external_local_sum_payloads": payloads,
        "external_payloads_uploaded": False,
        "unchanged_live_core": prior["unchanged_live_core"],
        "post_execution_dispatchers": dispatchers,
        "unchanged_main_ledger": identity(LEDGER, root),
        "test_suite": suite,
        "bindings": [
            identity(p, root)
            for p in sorted(archive.rglob("*"))
            if p.is_file()
            and p.name
            not in ("validation.json", "issue-body-after.md", "issue-publication-receipt.json")
        ]
        + [
            identity(root / p, root)
            for p in (
                "AGENTS.md",
                "PROJECT_STATUS.md",
                "README.md",
                "CURRENT_PROGRESS.json",
                "scripts/validate_dense_gradient_origin.py",
                "tests/test_dense_gpu_gradient_origin.py",
            )
        ],
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
                "artifact_validation_passed": True,
                "scientific_completion": False,
                "first_difference_origin_isolated_on_fixture": True,
                "prior_bindings_preserved": len(result["prior_bindings_preserved"]),
            }
        )
    )


if __name__ == "__main__":
    main()
