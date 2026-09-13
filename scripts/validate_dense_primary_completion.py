"""Read-only validation of whole-run/task/grid consumer evidence and retained failures."""

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from embed_optim.primary_completion import check_completion_lock, task_score
from embed_optim.primary_contract import PrimaryContract, file_identity, read_json, verify_file
from scripts.pause_dense_evaluation_for_audit import CHAIN, LEDGER, LEDGER_SHA
from scripts.pause_dense_evaluation_for_audit import inspect as inspect_process
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-primary-completion-v1")
PRIOR = Path("reports/engineering-archive/dense-primary-contract-v1/validation.json")
PRIOR_SHA = "ca6bc36159d1aba2dbe752c82ee50d740bbd5fd8815ddc44f483dbdbd5056e00"
WHOLE_SHA = "a3f941f0c1ebe592be2fe4f3f27d6306ca35528f9f24971002adb5b92732de5b"
TASK_SHA = "dce8fc0ac2faec0bdeb68cdeb6c8c030ff92c42883dc0e976ce9fada5c027e73"
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")


def validate(root):
    if sys.flags.optimize:
        raise RuntimeError("Do not disable audit assertions")
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
    contract = PrimaryContract.load(
        root / "configs/dense_primary_v2_protocol.json",
        root,
        root / "reports/engineering-archive/dense-full-identity-v1/candidate-source",
    )
    lock_path = root / "configs/dense_primary_v2_completion_protocol.json"
    lock = check_completion_lock(lock_path, contract)
    verify_file(archive / lock_path.name, lock)
    for path in sorted((archive / "source-current").rglob("*")):
        if path.is_file():
            verify_file(root / path.relative_to(archive / "source-current"), file_identity(path))
    assert file_identity(archive / "whole-run/final.json")["sha256"] == WHOLE_SHA
    whole = read_json(archive / "whole-run/final.json")
    assert whole["whole_run_rehearsal_passed"] is whole["producer_payloads_unchanged"] is True
    assert whole["scientific_completion"] is whole["production_deployed"] is False
    assert whole["model_updates_executed"] == whole["models_instantiated"] == 0
    assert len(whole["records"]) == 6 and all(r["correct_decision"] for r in whole["records"])
    positives = [r for r in whole["records"] if r["accepted"]]
    negatives = [r for r in whole["records"] if not r["accepted"]]
    assert len(positives) == len(negatives) == 3
    assert all(r["resume_step"] == 0 for r in positives)
    assert all(
        r["resume_step"] == 2 and "checkpoint prefix is incomplete" in r["error"] for r in negatives
    )
    for row in positives:
        value = row["result"]
        assert value["steps"] == [1, 2, 3] and value["whole_run_artifacts_verified"] is True
        assert len(value["checkpoints"]) == len(value["deep_checkpoint_checks"]) == 3
        assert [r["scheduler_step"] for r in value["deep_checkpoint_checks"]] == [1, 2, 3]
        assert all(r["parameter_states"] == 134 for r in value["deep_checkpoint_checks"])
        for name, expected in value["metadata"].items():
            verify_file(Path(row["run_root"]) / name, expected)
        for item in value["final_inference_files"]:
            verify_file(Path(row["run_root"]) / "final" / item["path"], item)
    external = [
        compare_binding(r, Path(r["path"]), root) for r in whole["authenticated_producer_files"]
    ]
    for row in whole["source_bindings"]:
        verify_file(Path(row["path"]), row)
    initial = read_json(archive / "whole-run/initial.json")
    assert initial["whole_run_rehearsal_passed"] is False
    assert all(
        r.get("error") == "KeyError: 'materialization_manifest'"
        for r in initial["records"]
        if not r["correct_decision"]
    )
    verify_file(
        archive / "source-initial/primary_completion.py",
        next(r for r in initial["source_bindings"] if r["path"].endswith("/primary_completion.py")),
    )
    producer = read_json(archive / "task/producer-first.json")
    assert producer["worker_returncode"] == 0 and producer["real_task_reader_passed"] is False
    assert (
        producer["primary_evaluation_executed"] is False
        and producer["diagnostic_evaluation_executed"] is True
    )
    assert "Out of range float values" in producer["error"]
    verify_file(
        archive / "source-initial/primary_completion.before-task-compatibility.py",
        producer["source_bindings"]["src/embed_optim/primary_completion.py"],
    )
    for row in producer["artifacts"]:
        external.append(compare_binding(row, Path(row["path"]), root))
        filename = Path(row["path"]).name
        if filename in {"SciFactDecontaminated.json", "model_meta.json", "run_settings.jsonl"}:
            verify_file(archive / "task/raw" / filename, row)
    assert read_json(archive / "task/lease.json")["status"] == "released"
    assert file_identity(archive / "task/reader-final.json")["sha256"] == TASK_SHA
    replay = read_json(archive / "task/reader-final.json")
    assert replay["real_task_reader_passed"] is replay["raw_task_artifacts_unchanged"] is True
    assert replay["evaluation_workers_executed"] == replay["model_updates_executed"] == 0
    assert replay["scientific_completion"] is False
    assert len(replay["inspected"]["undefined_auxiliary_metrics_not_used"]) == 6
    for name, expected in replay["source_bindings"].items():
        verify_file(root / name, expected)
    upstream = replay["upstream_control"]["source"]
    verify_file(Path(upstream["path"]), upstream)
    verify_file(archive / "upstream/retrieval_metrics.py", upstream)
    versions = read_json(root / "configs/formal_runtime.json")["packages"]
    actual_path = next(
        Path(r["path"])
        for r in replay["inspected"]["files"]
        if Path(r["path"]).name == "SciFactDecontaminated.json"
    )
    checked = task_score(
        actual_path,
        "SciFact",
        "verified-muon-3e-4",
        3,
        8192,
        "0729fa34af49875724d18ace64ce07f3e1dc0587",
        versions,
    )
    assert checked == replay["inspected"]
    # Independently verify the unchanged pre-existing scientific reader accepts
    # this exact finite nDCG result, including its unused auxiliary NaN values.
    from embed_optim.aggregate import _result_provenance

    diagnostic = next(r for r in positives if r["run_id"] == "verified-muon-3e-4")
    config = SimpleNamespace(
        run_id=diagnostic["run_id"],
        model_family="dense",
        max_length=8192,
        output_dir=Path(diagnostic["run_root"]),
    )
    original_acceptance = _result_provenance(
        actual_path, json.loads(actual_path.read_text()), config, 3, "SciFact", versions
    )
    assert original_acceptance == {k: versions[k] for k in original_acceptance}
    command = [
        sys.executable,
        "-B",
        "-m",
        "embed_optim.primary_completion",
        "matrix",
        "--completion-lock",
        str(lock_path),
        "--protocol",
        str(contract.path),
        "--repository",
        str(root),
        "--training-root",
        str(contract.training_root),
        "--experiment-root",
        "/root/embedding-optimizer-study",
        "--results-root",
        "/root/embedding-optimizer-study/results/dense-primary-v2",
    ]
    unavailable = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
    assert (
        unavailable.returncode != 0
        and "Require an ordinary retained run directory" in unavailable.stderr
    )
    assert not Path("/root/embedding-optimizer-study/outputs/dense-correctness-v2").exists()
    tests = {}
    for name, count in (("focused-final.xml", 63), ("full-final.xml", 1538)):
        suite = next(ET.parse(archive / "tests" / name).iter("testsuite")).attrib
        assert int(suite["tests"]) == count and all(
            int(suite[k]) == 0 for k in ("failures", "errors", "skipped")
        )
        tests[name] = suite
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
    assert all(r["state"] == "T" for r in dispatchers)
    return {
        "scope": "engineering_dense_primary_completion_evidence_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "scientific_completion": False,
        "formal_replication_ready": False,
        "production_deployed": False,
        "source_published": False,
        "summary": {
            "complete_diagnostic_runs_accepted": 3,
            "continuation_only_directories_rejected": 3,
            "deep_full_model_checkpoints": 9,
            "parameter_states_per_checkpoint": 134,
            "real_diagnostic_tasks": 1,
            "new_primary_results": 0,
        },
        "original_ndcg_acceptance_preserved": {
            "same_raw_file_accepted": True,
            "versions": original_acceptance,
        },
        "primary_matrix_readiness": {
            "available": False,
            "command": command,
            "returncode": unavailable.returncode,
            "stderr": unavailable.stderr,
        },
        "tests": tests,
        "prior_bindings_preserved": preserved,
        "external_payload_checks": external,
        "unchanged_live_core": prior["unchanged_live_core"],
        "unchanged_manuscript_source": paper,
        "unchanged_main_ledger": identity(LEDGER, root),
        "post_execution_dispatchers": dispatchers,
        "bindings": [
            identity(p, root)
            for p in sorted(archive.rglob("*"))
            if p.is_file() and p.name != "validation.json"
        ]
        + [
            identity(root / p, root)
            for p in (
                *DOCS,
                "src/embed_optim/primary_completion.py",
                "configs/dense_primary_v2_completion_protocol.json",
                "scripts/validate_dense_primary_completion.py",
            )
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("Use a new immutable validation receipt path")
    value = validate(args.repository.resolve())
    with args.output.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                k: value[k]
                for k in ("artifact_validation_passed", "formal_replication_ready", "summary")
            }
        )
    )


if __name__ == "__main__":
    main()
