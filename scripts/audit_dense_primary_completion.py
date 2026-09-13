"""Read-only whole-run rehearsal on the prior authenticated diagnostic producers."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path

from embed_optim import primary_completion as completion
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file

ENTRY_SHA = "0a773e0906a635ae513e8322a803bffcb1437372e873abae0aaf9a744f28f4e9"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Require an explicitly CPU-only diagnostic")
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("Preserve previous audit receipts; use a new output path")
    root = args.repository.resolve()
    producer_path = (
        root / "reports/engineering-archive/dense-full-identity-v1/entrypoint/result.json"
    )
    if file_identity(producer_path)["sha256"] != ENTRY_SHA:
        raise ValueError("Wrong trusted real producer receipt")
    producer = read_json(producer_path)
    source_paths = (
        root / "src/embed_optim/primary_completion.py",
        root / "src/embed_optim/primary_contract.py",
        root / "src/embed_optim/primary_io.py",
        Path(__file__).resolve(),
    )
    sources = [{"path": str(path), **file_identity(path)} for path in source_paths]
    records, authenticated = [], []
    for row in producer["records"]:
        done = row["completion"]
        for key in ("completion", "resolved_config", "full_run_identity"):
            item = done[key]
            verify_file(Path(item["path"]), item)
            authenticated.append(item)
        for files in done["checkpoint_files"].values():
            for item in files:
                verify_file(Path(item["path"]), item)
                authenticated.append(item)
    for row in producer["records"]:
        run_root = Path(row["completion"]["full_run_identity"]["path"]).parent
        expected = read_json(run_root / "dense_run_contract.json")
        should_accept = row["resume_step"] == 0
        record = {
            "run_id": row["run_id"],
            "run_root": str(run_root),
            "resume_step": row["resume_step"],
            "expected_whole_run_acceptance": should_accept,
        }
        try:
            result = completion.inspect_complete_run(run_root, expected, [1, 2, 3])
            record.update(accepted=True, result=result, correct_decision=should_accept)
        except Exception as error:
            record.update(
                accepted=False,
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
                correct_decision=not should_accept
                and "checkpoint prefix is incomplete" in str(error),
            )
        records.append(record)
        print(
            json.dumps(
                {k: record[k] for k in ("run_id", "resume_step", "accepted", "correct_decision")}
            ),
            flush=True,
        )
    for item in authenticated:
        verify_file(Path(item["path"]), item)
    require_same(sources, [{"path": str(path), **file_identity(path)} for path in source_paths])
    passed = all(row["correct_decision"] for row in records)
    result = {
        "scope": "engineering_dense_primary_whole_run_rehearsal",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "whole_run_rehearsal_passed": passed,
        "scientific_completion": False,
        "production_deployed": False,
        "model_updates_executed": 0,
        "models_instantiated": 0,
        "optimizer_and_scheduler_read_mode": "torch.load(weights_only=True, map_location=cpu), only after original producer digests and new complete seals",
        "producer_receipt": {"path": str(producer_path), **file_identity(producer_path)},
        "producer_payloads_unchanged": True,
        "authenticated_producer_files": authenticated,
        "source_bindings": sources,
        "records": records,
        "boundary": "Three complete three-step synthetic diagnostic baselines are positive controls. Three continuation-only directories ending at the same terminal step are negative whole-run controls: they lack the earlier checkpoint/timing prefix. No primary acceptance, training, new optimizer trial, GPU access, HF write, evaluation or scientific finding.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"passed": passed, "output": str(args.output), **file_identity(args.output)}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
