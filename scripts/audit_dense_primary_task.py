"""One real full-corpus SciFact diagnostic for the new MTEB result reader."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.gpu_lease import acquire_gpu_lease
from embed_optim.primary_completion import task_score
from embed_optim.primary_contract import file_identity, read_json, verify_file
from scripts.pause_dense_evaluation_for_audit import CHAIN, LEDGER, LEDGER_SHA
from scripts.pause_dense_evaluation_for_audit import inspect as inspect_process

ENTRY_SHA = "0a773e0906a635ae513e8322a803bffcb1437372e873abae0aaf9a744f28f4e9"
RUN = "verified-muon-3e-4"
REVISION = "0729fa34af49875724d18ace64ce07f3e1dc0587"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    root, work = args.repository.resolve(), args.workdir.resolve()
    if (
        work.parent != Path("/tmp")
        or not work.name.startswith("dense-primary-task.")
        or not work.is_dir()
        or any(work.iterdir())
    ):
        raise ValueError("Require a fresh named engineering-only temporary namespace")
    source_names = (
        "scripts/audit_dense_primary_task.py",
        "src/embed_optim/primary_completion.py",
        "src/embed_optim/primary_contract.py",
        "src/embed_optim/gpu_lease.py",
        "scripts/eval/dense_no_packing_parallel.py",
        "scripts/eval/dense_parallel.py",
        "scripts/eval/dense_sequential.py",
        "src/embed_optim/evaluation_utils.py",
        "src/embed_optim/corrected_input_execution.py",
        "src/embed_optim/decontamination.py",
        "configs/formal_runtime.json",
    )
    sources = {name: file_identity(root / name) for name in source_names}
    entry_path = root / "reports/engineering-archive/dense-full-identity-v1/entrypoint/result.json"
    if (
        file_identity(entry_path)["sha256"] != ENTRY_SHA
        or file_identity(LEDGER)["sha256"] != LEDGER_SHA
    ):
        raise ValueError("Original producer/handoff changed")
    dispatchers = [
        inspect_process(pid, start, name, CHAIN[i - 1][0] if i else 1)
        for i, (pid, start, name) in enumerate(CHAIN)
    ]
    if any(row["state"] != "T" for row in dispatchers):
        raise ValueError("Original BEIR dispatch is no longer paused")
    entry = read_json(entry_path)
    producer = next(r for r in entry["records"] if r["run_id"] == RUN and r["resume_step"] == 0)
    files = producer["completion"]["checkpoint_files"]["3"]
    for row in files:
        verify_file(Path(row["path"]), row)
    checkpoint = Path(producer["completion"]["full_run_identity"]["path"]).parent / "checkpoint-3"
    # Validate the exact installed worker stack before acquiring a device lease.
    runtime = subprocess.run(
        [
            "/usr/bin/python3",
            "-B",
            "-m",
            "embed_optim.runtime",
            "--spec",
            str(root / "configs/formal_runtime.json"),
        ],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": str(root / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if runtime.returncode:
        raise RuntimeError(runtime.stderr or runtime.stdout)
    command = [
        "/usr/bin/python3",
        "-B",
        str(root / "scripts/eval/dense_no_packing_parallel.py"),
        "--worker",
        "SciFact",
        "--results_folder",
        str(work / "dense"),
        "--models",
        str(checkpoint),
        "--bf16",
        "--fa2",
        "--local",
        "--decontaminated",
    ]
    print(
        json.dumps(
            {
                "stage": "authenticated_diagnostic_worker",
                "gpu_token": "4",
                "task": "SciFact",
                "scientific_completion": False,
            }
        ),
        flush=True,
    )
    with acquire_gpu_lease(
        ("4",),
        lock_dir="/tmp/embedding-optimizer-primary-gpu-leases",
        timeout_seconds=30,
        purpose="engineering-primary-completion-scifact",
        ledger_path=work / "lease.json",
    ):
        with (work / "worker.log").open("x") as log:
            worker = subprocess.run(
                command,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "4",
                    "PYTHONPATH": str(root / "src"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "WANDB_MODE": "disabled",
                    "WANDB_DISABLED": "true",
                },
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
    failure, inspected = None, None
    try:
        if worker.returncode:
            raise RuntimeError(f"Actual diagnostic worker exited {worker.returncode}")
        matches = list((work / "dense").rglob("SciFactDecontaminated.json"))
        if len(matches) != 1:
            raise ValueError("Actual worker lacks one unambiguous SciFact result")
        versions = read_json(root / "configs/formal_runtime.json")["packages"]
        inspected = task_score(matches[0], "SciFact", RUN, 3, 8192, REVISION, versions)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    for row in files:
        verify_file(Path(row["path"]), row)
    for name, expected in sources.items():
        verify_file(root / name, expected)
    if file_identity(LEDGER)["sha256"] != LEDGER_SHA:
        raise ValueError("Original ledger changed during the diagnostic")
    result = {
        "scope": "engineering_dense_primary_real_task_reader",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "real_task_reader_passed": failure is None,
        "scientific_completion": False,
        "primary_evaluation_executed": False,
        "diagnostic_evaluation_executed": True,
        "production_deployed": False,
        "model_updates_executed": 0,
        "hf_writes_executed": 0,
        "command": command,
        "worker_returncode": worker.returncode,
        "error": failure,
        "diagnostic_result_inspection": inspected,
        "source_bindings": sources,
        "authenticated_checkpoint_files": files,
        "checkpoint_unchanged": True,
        "original_dispatchers": dispatchers,
        "original_ledger_sha256": LEDGER_SHA,
        "artifacts": [
            {"path": str(p), **file_identity(p)} for p in sorted(work.rglob("*")) if p.is_file()
        ],
        "boundary": "One actual pinned full-corpus SciFact task on a prior synthetic-training diagnostic checkpoint, on one exclusively leased GPU. This validates the real result schema/reader, not optimizer quality, primary acceptance, all 14 tasks, a full 840-unit grid, or the former BEIR controller's resumption.",
    }
    with (work / "result.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "passed": failure is None,
                "error": failure,
                "workdir": str(work),
                **file_identity(work / "result.json"),
            }
        )
    )
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
