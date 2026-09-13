"""Read complete native checkpoint evaluations; no dispatch, selection, or GPU use."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

EXPERIMENT = Path("/root/embedding-optimizer-v3-experiment")
PRIMARY = Path("/root/embedding-optimizer-primary-v3")
HERE = EXPERIMENT / "launch/evaluation-handoff"
RESULTS = EXPERIMENT / "evaluations/dense-primary-v3"
SOURCE_SHA = "5b8a89bed5e0eaa02a12585ee3f6c883ca3e162b550898d68322321ee41d8427"
AUTH_SHA = "2351e225387f42fe607d7e002a0717de79a78cba9f3c93f66652510083f0be7c"
ASSEMBLY_SHA = "e603ef175c0f691a1dd83c4d8b474d90da2fc67c8c65f5a77f3c3180e78332b8"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Not a regular file: {path}")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), "Moving input")
    return {"path": str(path), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(), "text": raw.decode()}


def verify_identity(item):
    actual = snapshot(item["path"])
    require(all(actual[key] == item[key] for key in ("bytes", "sha256")), "Changed input")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Require CPU-only readback")
    require(not args.output.exists(), "Preserve all prior output files")
    # Freeze the observation's set before opening any score. Never select by score.
    paths = sorted(RESULTS.glob("*/all-fourteen-tasks-verified.json"))
    require(bool(paths), "No complete native fourteen-task checkpoint yet")
    source = snapshot(HERE / "dispatch.py")
    require(source["sha256"] == SOURCE_SHA, "Original dispatcher changed")
    assembly = snapshot(PRIMARY / "source-assembly.json")
    require(assembly["sha256"] == ASSEMBLY_SHA, "Original training assembly changed")
    for root in (PRIMARY, EXPERIMENT / "launch/source-snapshot"):
        for rel, item in json.loads(assembly["text"])["files"].items():
            raw = (root / rel).read_bytes()
            require(len(raw) == item["identity"]["bytes"] and
                    hashlib.sha256(raw).hexdigest() == item["identity"]["sha256"],
                    "Original source assembly differs")
    spec = importlib.util.spec_from_file_location("original_complete_evaluation_handoff", HERE / "dispatch.py")
    dispatch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dispatch)
    module, contract, authorization = dispatch.stack(SOURCE_SHA, AUTH_SHA)
    import torch
    from embed_optim.primary_v3_completion import inspect_evaluation

    records = []
    for path in paths:
        original_snapshot = snapshot(path)
        original = json.loads(original_snapshot["text"])
        plan = original["plan"]
        run_id, step = plan["checkpoint"]["run_id"], plan["checkpoint"]["step"]
        require(run_id in {run for queue in authorization["queues"].values() for run in queue}
                and step in authorization["checkpoint_steps"], "Undeclared checkpoint")
        require(path.parent == Path(plan["results_root"]), "Wrong result namespace")
        require(original["complete_task_files_verified"] is True and
                original["scientific_completion"] is False, "Wrong native acceptance scope")
        operational = snapshot(path.parent / "operational_admission.json")
        authority = json.loads(operational["text"])
        require(authority["source_sha256"] == SOURCE_SHA and
                authority["authorization_sha256"] == AUTH_SHA and
                authority["committed_source_release"] is False and
                authority["native_released_contract_guard_passed"] is False and
                authority["scientific_completion"] is False, "Changed operational boundaries")
        proof = dispatch.completion(module, contract, authorization, run_id)
        require(proof is not None, "Missing authentic training completion")
        dispatch.bind_plan_to_completion(plan, proof["proof"], step)
        require({key: proof["binding"][key] for key in ("path", "bytes", "sha256")} ==
                {key: authority["training_completion"][key] for key in ("path", "bytes", "sha256")},
                "Different training-completion parent")
        checkpoint = EXPERIMENT / contract.payload["output_root"] / "dense" / run_id / f"checkpoint-{step}"
        reread = inspect_evaluation(contract, checkpoint, run_id, step, RESULTS)
        require(reread == original, "Native complete evaluation readback differs")
        expected_tasks = contract.payload["evaluation"]["tasks"]
        require(len(expected_tasks) == 14 and len(set(expected_tasks)) == 14 and
                [row["task"] for row in reread["tasks"]] == expected_tasks,
                "Cannot summarize incomplete or duplicate task coverage")
        pool = next(pool for pool, queue in authorization["queues"].items() if run_id in queue)
        jobs = []
        raw_files = {}
        for row in reread["tasks"]:
            task = row["task"]
            prefix = HERE / f"pool-{pool}" / "jobs" / f"{run_id}-{step}-{task}"
            started, exited = snapshot(prefix.with_suffix(".started.json")), snapshot(prefix.with_suffix(".exited.json"))
            start, terminal = json.loads(started["text"]), json.loads(exited["text"])
            require(start["job"] == terminal["job"] == [run_id, step, task] and
                    start["pid"] == terminal["pid"] and terminal["exit_code"] == 0 and
                    start["source_sha256"] == SOURCE_SHA and start["authorization_sha256"] == AUTH_SHA and
                    start["command"] == dispatch.worker_command(checkpoint, plan, task),
                    "Missing or inconsistent original worker success")
            jobs.append({"started": started, "exited": exited})
            for item in row["files"]:
                actual = verify_identity(item)
                require(item["path"] not in raw_files or raw_files[item["path"]] == actual,
                        "Shared completed metadata changed")
                raw_files[item["path"]] = actual
        exact = sum((Fraction.from_float(row["ndcg_at_10"]) for row in reread["tasks"]), Fraction()) / 14
        macro = float(exact)
        require(math.isclose(macro, math.fsum(row["ndcg_at_10"] for row in reread["tasks"]) / 14,
                             rel_tol=0, abs_tol=1e-15), "Independent mean arithmetic differs")
        require(snapshot(path) == original_snapshot and
                snapshot(path.parent / "operational_admission.json") == operational,
                "Original completed receipt changed")
        for item in raw_files.values():
            require(verify_identity(item) == item, "Completed score inputs changed")
        records.append({
            "run_id": run_id, "step": step, "task_count": 14, "macro_ndcg_at_10": macro,
            "macro_score_0_to_100": float(exact * 100),
            "exact_binary64_input_mean": {"numerator": exact.numerator, "denominator": exact.denominator},
            "original_complete_receipt": original_snapshot, "original_operational_receipt": operational,
            "native_complete_reread": reread, "original_workers": jobs,
            "raw_score_and_metadata_snapshots": list(raw_files.values()),
        })
    require(not torch.cuda.is_initialized(), "Readback initialized CUDA")
    result = {
        "scope": "descriptive_complete_fourteen_task_checkpoint_readback",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": contract.sha256, "checkpoints": records,
        "source": snapshot(__file__), "dispatcher_source": source, "assembly": assembly,
        "available_case_mean": False, "optimizer_ranking_or_selection": False,
        "full_primary_grid_complete": False, "model_or_retrieval_recomputed": False,
        "cuda_initialized": False, "committed_source_release": False, "scientific_completion": False,
        "boundary": "Complete native checkpoint scores only. Not all-rate inference, validation selection, causal evidence, or a source release.",
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "observed_at_utc": result["observed_at_utc"],
                      "checkpoints": [{key: row[key] for key in
                                      ("run_id", "step", "task_count", "macro_ndcg_at_10", "macro_score_0_to_100")}
                                     for row in records], "scientific_completion": False}, allow_nan=False))


if __name__ == "__main__":
    main()
