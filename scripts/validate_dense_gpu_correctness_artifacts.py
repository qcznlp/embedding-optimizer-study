"""Verify and summarize this host-scoped engineering campaign without GPU work."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

LIVE = Path("/root/embedding-optimizer-study")
ARCHIVE = Path("reports/engineering-archive/dense-gpu-correctness-v1")


def identity(path, root):
    raw = path.read_bytes()
    return {
        "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def load(path):
    return json.loads(path.read_text())


def checks(report):
    rows = [x for case in report["records"] for x in case["records"]]
    assert report["audit_execution_complete"] is True and len(report["records"]) == 3
    assert len(rows) == 9 and report["scientific_completion"] is False
    for case in report["records"]:
        assert [x["global_queries"] for x in case["records"]] == [128, 128, 32]
        assert case["final_weights_rank_identical"] is True
        for row in case["records"]:
            assert len(row["raw"]["parameters"]) == len(row["clipped"]["parameters"]) == 134
            assert row["raw_gradients_rank_identical"] is True
            assert row["actual_reference_token_inputs_identical"] is True
    return rows


def verify(root):
    if sys.flags.optimize:
        raise RuntimeError("Audit assertions must not be disabled by Python optimization")
    archive = root / ARCHIVE
    reports = {
        name: load(archive / f"{name}.json")
        for name in (
            "short-live",
            "short-candidate",
            "max-context-default",
            "max-context-deterministic",
            "gpu-resume",
        )
    }
    live, short, default, deterministic = [
        checks(reports[n])
        for n in (
            "short-live",
            "short-candidate",
            "max-context-default",
            "max-context-deterministic",
        )
    ]
    assert all(not x["raw"]["all_parameters_match_prior_tolerance"] for x in live)
    assert all(abs(x["raw"]["norm_ratio"] - 0.25) < 1e-7 for x in live)
    assert all(
        x["compensated_raw_diagnostic"]["all_parameters_match_prior_tolerance"] for x in live
    )
    assert all(not x["raw"]["all_parameters_match_prior_tolerance"] for x in default)
    for rows in (short, deterministic):
        assert all(
            x[k]["all_parameters_match_prior_tolerance"] for x in rows for k in ("raw", "clipped")
        )
    assert reports["short-live"]["normalization_acceptance_passed"] is False
    assert reports["max-context-default"]["normalization_acceptance_passed"] is False
    assert reports["short-candidate"]["normalization_acceptance_passed"] is True
    assert reports["max-context-deterministic"]["normalization_acceptance_passed"] is True
    assert reports["max-context-default"]["probe"] == reports["max-context-deterministic"]["probe"]
    for name in ("max-context-default", "max-context-deterministic"):
        assert all(
            max(r["step_max_token_lengths"]) == 8192
            for c in reports[name]["records"]
            for r in c["rank_stress_coverage"]
        )
    control = load(archive / "deterministic-control.json")
    assert len(control["model_observations"]) == 6
    assert all(
        x == {"attention_modules": 22, "before": False, "after": True}
        for x in control["model_observations"]
    )
    assert (
        control["upstream_result"]["sha256"]
        == identity(archive / "max-context-deterministic.json", root)["sha256"]
    )
    for name, expected in control["source_hashes"].items():
        assert identity(Path(name), root)["sha256"] == expected

    replay = reports["gpu-resume"]
    assert (
        replay["audit_execution_complete"] is True
        and replay["strict_bitwise_replay_passed"] is False
    )
    assert len(replay["records"]) == 6
    ranks = [r["comparison"] for case in replay["records"] for r in case["ranks"]]
    assert len(ranks) == 24
    for rank in ranks:
        assert rank["exact_loaded_entry_state"] and rank["exact_row_order_and_rank_rng"]
        assert rank["exact_scheduler_and_final_progress"] and not rank["bitwise_replay"]["passed"]
    assert all(c["argument_overrides"]["dataloader_num_workers"] == 8 for c in replay["records"])

    source_checks = []
    for label, report in reports.items():
        assert report["source_checkpoint_unchanged"] is True
        for item in report["source_bindings"]:
            observed = identity(Path(item["path"]), root)
            assert observed["sha256"] == item["sha256"], item["path"]
            source_checks.append(
                {"receipt": label, "recorded_path": item["path"], "sha256": item["sha256"]}
            )

    preserved = load(archive / "handoff/preservation.json")
    assert len(preserved) == 124
    for item in preserved:
        for path in (LIVE / item["path"], archive / "handoff/preserved" / item["path"]):
            actual = identity(path, root)
            assert (actual["bytes"], actual["sha256"]) == (item["bytes"], item["sha256"])
    logs = sorted((archive / "handoff/worker-logs").glob("*.log"))
    assert len(logs) == 8
    assert all(
        identity(p, root)["sha256"]
        == identity(LIVE / "logs/dense-no-packing-beir" / p.name, root)["sha256"]
        for p in logs
    )
    contract = load(LIVE / "logs/dense-no-packing-finalization/pipeline-ledger.json")["contract"]
    for item in contract["sources"]:
        assert identity(LIVE / item["path"], root)["sha256"] == item["sha256"]
    prior = load(root / "reports/engineering-archive/normuon-full-reference-v1/validation.json")
    prior_checks = []
    for item in prior["bindings"]:
        path = (
            archive / "before" / item["path"]
            if item["path"]
            in ("AGENTS.md", "PROJECT_STATUS.md", "CURRENT_PROGRESS.json", "README.md")
            else root / item["path"]
        )
        observed = identity(path, root)
        assert (observed["bytes"], observed["sha256"]) == (item["bytes"], item["sha256"])
        prior_checks.append(
            {
                "recorded_path": item["path"],
                "verified_path": observed["path"],
                "sha256": item["sha256"],
            }
        )
    for item in prior["unchanged_live_core"]:
        assert (
            identity(root / item["path"], root)["sha256"]
            == identity(LIVE / item["path"], root)["sha256"]
            == item["sha256"]
        )
    suites = list(ET.parse(archive / "pytest-final.xml").getroot().iter("testsuite"))
    assert len(suites) == 1
    suite = suites[0].attrib
    assert int(suite["tests"]) == 1319 and all(
        int(suite[k]) == 0 for k in ("failures", "errors", "skipped")
    )
    return {
        "scope": "engineering_dense_gpu_correctness_artifact_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "whole_training_stack_accepted": False,
        "scientific_completion": False,
        "runtime_deployed": False,
        "source_published": False,
        "evaluation_paused": True,
        "gpu_audit_launchers_exited": True,
        "summary": {
            "unchanged_raw_gradient_failures": 9,
            "short_candidate_raw_clipped_passes": 9,
            "default_max_context_failures": 9,
            "paired_deterministic_max_context_passes": 9,
            "gpu_replay_cases": 6,
            "exact_replay_entry_row_rng_rank_checks": 24,
            "strict_bitwise_replay_passes": 0,
            "replay_amplification_cause_isolated": False,
        },
        "test_suite": suite,
        "executed_source_checks": source_checks,
        "prior_probe_bindings_preserved": prior_checks,
        "preserved_handoff_files": len(preserved),
        "preserved_worker_logs": len(logs),
        "unchanged_live_core": prior["unchanged_live_core"],
        "unchanged_main_contract_sources": len(contract["sources"]),
        "bindings": [
            identity(p, root)
            for p in sorted(archive.rglob("*"))
            if p.is_file()
            and p.name
            not in ("validation.json", "issue-publication-receipt.json", "issue-body-after.md")
        ]
        + [
            identity(root / p, root)
            for p in ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")
        ]
        + [identity(Path(__file__).resolve(), root)],
        "boundary": "Passing archival/source/regression validation preserves failed numerical outcomes; it is not acceptance of the whole training stack, scientific results, a deployed fix or a new primary run. Actual GPU checkpoints are diagnostic states, not historical optimizer resumption or independent-process restart. The original controller remains stopped and its lease retained.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refuse to overwrite an evidence receipt")
    result = verify(args.repository.resolve())
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "verified_artifacts": len(result["bindings"]),
                "source_checks": len(result["executed_source_checks"]),
                "prior_bindings_preserved": len(result["prior_probe_bindings_preserved"]),
                **result["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
