"""Independent-process continuation of sealed canonical-control checkpoints.

No baseline training is performed here. A new torchrun group loads the previous
campaign's actual checkpoint files and compares every strict replay field with
its authenticated per-rank baseline fingerprints. This is engineering evidence,
not historical-optimizer, cross-host, long-horizon or retrieval acceptance.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from embed_optim import train
from embed_optim.config import load_matrix
from scripts import audit_dense_gpu_canonical_replay as control
from scripts import audit_dense_gpu_normalization as gpu
from scripts import audit_dense_trainer_resume as replay
from scripts.audit_checkpoint_download import select_download, verify_download
from scripts.audit_dense_gpu_replay_update import identity


def verify_identity(record):
    if set(record) != {"path", "bytes", "sha256"} or identity(record["path"]) != record:
        raise ValueError("Sealed file identity changed")


def verify_baseline(path, expected_sha):
    actual = identity(path)
    if actual["sha256"] != expected_sha:
        raise ValueError("Untrusted baseline-control receipt")
    receipt = json.loads(Path(path).read_text())
    if (
        receipt.get("scope") != "engineering_canonical_reduction_deterministic_backward_control"
        or receipt.get("target") != "resume"
        or receipt.get("scientific_completion") is not False
        or receipt.get("production_deployed") is not False
        or receipt.get("audit_execution_complete") is not True
        or receipt.get("acceptance_passed") is not True
    ):
        raise ValueError("Require the completed passing diagnostic baseline")
    for record in [receipt["source"], *receipt["source_bindings"], receipt["parent_result"]]:
        verify_identity(record)
    if receipt["source"]["path"] != str(Path(inspect.getsourcefile(control)).resolve()):
        raise ValueError("Baseline and current control source differ")
    parent = json.loads(Path(receipt["parent_result"]["path"]).read_text())
    if (
        parent.get("strict_bitwise_replay_passed") is not True
        or parent.get("audit_execution_complete") is not True
        or parent.get("source_checkpoint_unchanged") is not True
    ):
        raise ValueError("Parent strict replay is not accepted")
    for record in parent["source_bindings"]:
        verify_identity(record)
    exports = receipt["baseline_exports"]
    if [x["run_id"] for x in exports] != list(replay.previous.RUN_IDS):
        raise ValueError("Require the exact three optimizer recipes")
    for export in exports:
        checkpoint_root = Path(export["checkpoint_root"])
        if (
            checkpoint_root.parent.name != export["run_id"]
            or checkpoint_root.name != "uninterrupted"
            or checkpoint_root.parent.parent.parent != Path("/tmp")
            or not checkpoint_root.parent.parent.name.startswith("dense-gpu-resume.")
            or checkpoint_root.is_symlink()
        ):
            raise ValueError("Baseline checkpoint namespace mismatch")
        if len(export["ranks"]) != 4 or set(export["checkpoint_files"]) != {"1", "2", "3"}:
            raise ValueError("Missing baseline rank or checkpoint")
        for rank, record in enumerate(export["ranks"]):
            if Path(record["path"]) != Path(path).parent / (
                f"{export['run_id']}-baseline-rank-{rank}.json"
            ):
                raise ValueError("Unexpected per-rank fingerprint path")
            verify_identity(record)
            payload = json.loads(Path(record["path"]).read_text())
            if (
                payload["rank"] != rank
                or payload["run_id"] != export["run_id"]
                or set(payload["expected"]) != {"1", "2"}
            ):
                raise ValueError("Fingerprint identity differs")
    return receipt


def verify_checkpoints(receipt):
    # Recompute complete inventories, so added/removed files are also rejected.
    for export in receipt["baseline_exports"]:
        for step, expected in export["checkpoint_files"].items():
            actual = control.checkpoint_files(
                Path(export["checkpoint_root"]) / f"checkpoint-{step}"
            )
            if actual != expected:
                raise ValueError("Source diagnostic checkpoint changed")


def compare_fingerprints(expected, actual, limit=24):
    """Compact exact failure paths; never numerical tolerances or field exclusions."""
    mismatches = []

    def visit(a, b, path):
        if len(mismatches) >= limit:
            return
        if type(a) is not type(b):
            mismatches.append(path + ":type")
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                mismatches.append(path + ":keys")
                return
            for key in a:
                visit(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, list):
            if len(a) != len(b):
                mismatches.append(path + ":length")
                return
            for index, (left, right) in enumerate(zip(a, b, strict=True)):
                visit(left, right, f"{path}/{index}")
        elif a != b:
            mismatches.append(path)

    visit(expected, actual, "state")
    return {"exact": expected == actual, "first_mismatch_paths": mismatches, "path_limit": limit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--baseline-control", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    args = parser.parse_args()
    if any(
        os.environ.get(k) != v
        for k, v in {
            "WORLD_SIZE": "4",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        }.items()
    ):
        raise ValueError("Require four owned CUDA ranks and offline/disabled external services")
    root, work = args.repository.resolve(), args.workdir.absolute()
    if (
        work.parent != Path("/tmp")
        or not work.name.startswith("dense-gpu-fresh-replay.")
        or not work.is_dir()
        or work.is_symlink()
        or Path(inspect.getsourcefile(train)).resolve() != root / "src/embed_optim/train.py"
    ):
        raise ValueError("Require the isolated source and a fresh mktemp diagnostic namespace")
    baseline = verify_baseline(args.baseline_control, args.baseline_sha256)
    handoff = gpu.require_handoff(args.handoff)
    gpu.verify_stack()
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if rank == 0 and any(work.iterdir()):
        raise ValueError("Require empty output")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group("nccl", timeout=timedelta(seconds=600), device_id=device)
    report = {
        "scope": "engineering_full_denseon_independent_process_canonical_replay",
        "scientific_completion": False,
        "production_deployed": False,
        "audit_execution_complete": False,
        "strict_bitwise_replay_passed": False,
        "baseline_training_performed": False,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "baseline_control": identity(args.baseline_control),
        "handoff": handoff,
        "records": [],
        "observations": {"attention_controls": [], "reducers": []},
        "boundary": "A newly launched four-rank process group resumes the prior completed control's sealed real diagnostic checkpoints. Same physical host/GPU mapping, pinned candidate normalization, owned deterministic backward and canonical reduction. Native CUDA deserialization, eight persistent data-loader workers per rank, 288 synthetic rows and three total updates. Fingerprints preserve every field in the original exact replay test, including all later entry states and raw/clipped gradient tensors. No new baseline, tolerance change, production deployment, historical optimizer continuation, long-horizon or scientific result.",
    }
    try:
        rank_processes = [None] * 4
        dist.all_gather_object(
            rank_processes,
            {"rank": rank, "pid": os.getpid(), "torchrun_id": os.environ["TORCHELASTIC_RUN_ID"]},
        )
        report["process_group"] = rank_processes
        if rank == 0:
            verify_checkpoints(baseline)
        dist.barrier()
        selection = select_download(args.audit, args.audit_sha256, "padded-normuon-3e-4", 3126)
        before = verify_download(selection, args.download_root.absolute())
        report["source_checkpoint"] = before
        checkpoint = Path(before["checkpoint_root"])
        configs = {c.run_id: c for c in load_matrix(root / "configs/dense_no_packing_retrain.yaml")}
        original_rng = replay.rng_hashes

        def load_full(path):
            if Path(path) != checkpoint:
                raise ValueError("Unexpected full-model fixture")
            return gpu.load_model(checkpoint, device)

        def arguments(config, output):
            actual, overrides = gpu.gpu_arguments(config, output)
            actual.dataloader_num_workers = overrides["dataloader_num_workers"] = (
                config.dataloader_workers
            )
            actual.dataloader_persistent_workers = overrides["dataloader_persistent_workers"] = (
                config.dataloader_workers > 0
            )
            actual.dataloader_prefetch_factor = overrides["dataloader_prefetch_factor"] = (
                4 if config.dataloader_workers else None
            )
            return actual, overrides

        def rng():
            return {
                **original_rng(),
                "cuda_local": replay.tensor_hash(torch.cuda.get_rng_state(device)),
            }

        with (
            control.controls(report["observations"]),
            patch.object(replay.previous, "load_fixture", side_effect=load_full),
            patch.object(replay.previous, "cpu_arguments", side_effect=arguments),
            patch.object(replay, "rng_hashes", side_effect=rng),
        ):
            for export in baseline["baseline_exports"]:
                run_id = export["run_id"]
                expected = json.loads(Path(export["ranks"][rank]["path"]).read_text())
                for step in (1, 2):
                    restored, metadata = replay.train_segment(
                        configs[run_id],
                        checkpoint,
                        work / run_id / f"resume-{step}",
                        "candidate",
                        0.0,
                        Path(export["checkpoint_root"]) / f"checkpoint-{step}",
                    )
                    if metadata["cpu_transport_adapter"]:
                        raise ValueError("Must use native GPU deserialization")
                    actual = control.replay_fingerprint(restored, step)
                    comparison = compare_fingerprints(expected["expected"][str(step)], actual)
                    fingerprint_path = work / f"{run_id}-resume-{step}-rank-{rank}.json"
                    with fingerprint_path.open("x") as handle:
                        json.dump(
                            {"run_id": run_id, "step": step, "rank": rank, "actual": actual},
                            handle,
                            indent=2,
                            sort_keys=True,
                        )
                        handle.write("\n")
                    ranks = [None] * 4
                    dist.all_gather_object(
                        ranks,
                        {
                            "rank": rank,
                            "comparison": comparison,
                            "fingerprint": identity(fingerprint_path),
                        },
                    )
                    if rank == 0:
                        record = {
                            "run_id": run_id,
                            "resume_step": step,
                            "ranks": ranks,
                            "argument_overrides": metadata["argument_overrides"],
                            "strict_bitwise_replay_all_ranks": all(
                                x["comparison"]["exact"] for x in ranks
                            ),
                        }
                        report["records"].append(record)
                        print(
                            json.dumps(
                                {
                                    "run_id": run_id,
                                    "step": step,
                                    "exact": record["strict_bitwise_replay_all_ranks"],
                                }
                            ),
                            flush=True,
                        )
                    del restored, actual
                    gc.collect()
                    torch.cuda.empty_cache()
                    dist.barrier()
        if rank == 0:
            verify_checkpoints(baseline)
        dist.barrier()
        if verify_download(selection, args.download_root.absolute()) != before:
            raise ValueError("Authenticated primary anchor changed")
        verify_baseline(args.baseline_control, args.baseline_sha256)
        gpu.require_handoff(args.handoff)
        report["all_source_checkpoints_unchanged"] = True
        report["audit_execution_complete"] = True
        report["strict_bitwise_replay_passed"] = (
            len(report["records"]) == 6
            and all(x["strict_bitwise_replay_all_ranks"] for x in report["records"])
            if rank == 0
            else False
        )
    except BaseException as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        with (work / f"observations-rank-{rank}.json").open("x") as handle:
            json.dump({"rank": rank, "observations": report["observations"]}, handle, indent=2)
        if rank == 0:
            report.update(
                finished_at_utc=datetime.now(UTC).isoformat(),
                source_bindings=[
                    identity(p)
                    for p in (
                        Path(__file__).resolve(),
                        Path(inspect.getsourcefile(control)),
                        Path(inspect.getsourcefile(control.canonical)),
                        Path(inspect.getsourcefile(gpu)),
                        Path(inspect.getsourcefile(replay)),
                    )
                ],
            )
            with (work / "result.json").open("x") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")
        dist.destroy_process_group()
    if rank == 0 and not report["strict_bitwise_replay_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
