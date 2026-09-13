"""Actual prepared Trainer checkpoint contract with controlled full GPU replay.

The fixed-layout communication strategy is an explicit diagnostic control, not
part of the candidate production source. The inherited save/load implementation
and new per-checkpoint numerical receipt are exercised without a loading adapter.
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

from embed_optim import dense_numerical_contract as numerical
from embed_optim import train
from embed_optim.config import DENSE_NUMERICAL_POLICY, load_matrix
from scripts import audit_dense_gpu_canonical_replay as control
from scripts import audit_dense_gpu_normalization as gpu
from scripts import audit_dense_trainer_resume as replay
from scripts.audit_checkpoint_download import select_download, verify_download
from scripts.audit_dense_gpu_replay_update import identity
from scripts.audit_prepared_dense_gpu import RUNS, check_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    args = parser.parse_args()
    root, work = args.candidate_root.resolve(), args.workdir.absolute()
    if any(
        os.environ.get(k) != v
        for k, v in {
            "WORLD_SIZE": "4",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
        }.items()
    ):
        raise ValueError("Require the explicit four-rank offline diagnostic pool")
    if (
        work.parent != Path("/tmp")
        or not work.name.startswith("dense-prepared-resume.")
        or not work.is_dir()
        or work.is_symlink()
    ):
        raise ValueError("Require a new prepared-resume mktemp namespace")
    manifest = check_sources(args.source_manifest, args.source_sha256, root)
    handoff_path = Path("/tmp/dense-gpu-handoff.4MuYjC/receipt/result.json")
    handoff = gpu.require_handoff(handoff_path)
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group("nccl", timeout=timedelta(seconds=600), device_id=device)
    report = {
        "scope": "engineering_prepared_dense_checkpoint_contract_and_controlled_replay",
        "scientific_completion": False,
        "production_deployed": False,
        "audit_execution_complete": False,
        "strict_bitwise_replay_passed": False,
        "source_manifest": identity(args.source_manifest),
        "handoff": handoff,
        "observations": {"attention_controls": [], "reducers": []},
        "records": [],
        "baseline_exports": [],
        "legacy_checkpoint_rejections": [],
        "boundary": "Actual prepared OptimizerTrainer save/restore on four GPUs, authenticated full DenseOn weights and fresh AdamW/Muon/NorMuon, zero dropout, 288 synthetic rows, 128/128/32 global query groups, eight persistent workers per rank, ordinary callback and native GPU deserialization. Fixed lexical reduction and deterministic attention are explicit diagnostic controls. Baseline and restarts at steps 1 and 2 share a process group; not the full run_training entrypoint, an independent restart, long-horizon, natural-text quality or a scientific result. No relaxed old tolerance or relabeling of legacy checkpoints.",
    }
    try:
        if rank == 0 and any(work.iterdir()):
            raise ValueError("Output must be empty")
        selection = select_download(args.audit, args.audit_sha256, "padded-normuon-3e-4", 3126)
        before = verify_download(selection, args.download_root.absolute())
        report["source_checkpoint"] = before
        checkpoint = Path(before["checkpoint_root"])
        configs = {
            c.run_id: c for c in load_matrix(root / "configs/dense_correctness_candidate.yaml")
        }
        original_rng = replay.rng_hashes

        def load_full(path):
            if Path(path) != checkpoint:
                raise ValueError("Attempted to load an unauthenticated model fixture")
            model = gpu.load_model(checkpoint, device)
            numerical.require_backward(model)
            return model

        def arguments(config, output):
            actual, overrides = gpu.gpu_arguments(config, output)
            actual.dataloader_num_workers = overrides["dataloader_num_workers"] = 8
            actual.dataloader_persistent_workers = overrides["dataloader_persistent_workers"] = True
            actual.dataloader_prefetch_factor = overrides["dataloader_prefetch_factor"] = 4
            if config.dataloader_workers != 8:
                raise ValueError("Require the declared worker contract")
            return actual, overrides

        def rng():
            return {
                **original_rng(),
                "cuda_local": replay.tensor_hash(torch.cuda.get_rng_state(device)),
            }

        def trainer_factory(*a, **kw):
            return train.OptimizerTrainer(*a, numerical_policy=DENSE_NUMERICAL_POLICY, **kw)

        with (
            control.controls(report["observations"]),
            patch.object(replay.previous, "load_fixture", side_effect=load_full),
            patch.object(replay.previous, "cpu_arguments", side_effect=arguments),
            patch.object(replay, "rng_hashes", side_effect=rng),
            patch.object(
                replay, "SingleNormalizationTrainerCandidate", side_effect=trainer_factory
            ),
        ):
            for run_id in RUNS:
                config, case = configs[run_id], work / run_id
                expected_receipt = numerical.receipt(config.optimizer, 4)
                try:
                    numerical.require_resume_receipt(checkpoint, expected_receipt)
                except ValueError as error:
                    if "lacks the corrected" not in str(error):
                        raise
                    report["legacy_checkpoint_rejections"].append(run_id)
                else:
                    raise ValueError("Legacy checkpoint was incorrectly accepted for continuation")
                reference, metadata = replay.train_segment(
                    config, checkpoint, case / "uninterrupted", "candidate", 0.0
                )
                ids_by_rank = [None] * 4
                dist.all_gather_object(
                    ids_by_rank, [i for x in reference["trace"] for i in x["row_ids"]]
                )
                ids = [i for chunk in ids_by_rank for i in chunk]
                if len(ids) != 288 or set(ids) != set(range(288)):
                    raise ValueError("Baseline row coverage differs")
                baseline_files = {}
                for step in (1, 2, 3):
                    source = case / "uninterrupted" / f"checkpoint-{step}"
                    numerical.require_resume_receipt(source, expected_receipt)
                    if rank == 0:
                        baseline_files[str(step)] = control.checkpoint_files(source)
                cache = {}
                expected = {
                    str(step): control.replay_fingerprint(reference, step, cache) for step in (1, 2)
                }
                fingerprint_path = work / f"{run_id}-baseline-rank-{rank}.json"
                with fingerprint_path.open("x") as handle:
                    json.dump(
                        {
                            "run_id": run_id,
                            "rank": rank,
                            "expected": expected,
                            "metadata": metadata,
                        },
                        handle,
                        indent=2,
                        sort_keys=True,
                    )
                    handle.write("\n")
                exports = [None] * 4
                dist.all_gather_object(exports, identity(fingerprint_path))
                if rank == 0:
                    report["baseline_exports"].append(
                        {
                            "run_id": run_id,
                            "ranks": exports,
                            "checkpoint_root": str(case / "uninterrupted"),
                            "checkpoint_files": baseline_files,
                        }
                    )
                del expected, cache
                for step in (1, 2):
                    restored, restored_metadata = replay.train_segment(
                        config,
                        checkpoint,
                        case / f"resume-{step}",
                        "candidate",
                        0.0,
                        case / "uninterrupted" / f"checkpoint-{step}",
                    )
                    if restored_metadata["cpu_transport_adapter"]:
                        raise ValueError("Native GPU continuation was replaced")
                    comparison = replay.measure_replay(reference, restored, step)
                    ranks = [None] * 4
                    dist.all_gather_object(ranks, {"rank": rank, "comparison": comparison})
                    numerical.require_resume_receipt(
                        case / f"resume-{step}" / "checkpoint-3", expected_receipt
                    )
                    if rank == 0:
                        record = {
                            "run_id": run_id,
                            "resume_step": step,
                            "ranks": ranks,
                            "argument_overrides": metadata["argument_overrides"],
                            "strict_bitwise_replay_all_ranks": all(
                                x["comparison"]["bitwise_replay"]["passed"] for x in ranks
                            ),
                        }
                        report["records"].append(record)
                        with (work / f"{run_id}-resume-{step}.json").open("x") as handle:
                            json.dump(record, handle, indent=2, sort_keys=True)
                        print(
                            json.dumps(
                                {
                                    "run_id": run_id,
                                    "resume_step": step,
                                    "strict_bitwise_replay_all_ranks": record[
                                        "strict_bitwise_replay_all_ranks"
                                    ],
                                }
                            ),
                            flush=True,
                        )
                    del restored
                    gc.collect()
                    torch.cuda.empty_cache()
                    dist.barrier()
                if rank == 0:
                    for step, files in baseline_files.items():
                        if (
                            control.checkpoint_files(case / "uninterrupted" / f"checkpoint-{step}")
                            != files
                        ):
                            raise ValueError("Sealed baseline checkpoint changed during replay")
                del reference
                gc.collect()
                torch.cuda.empty_cache()
                dist.barrier()
        if verify_download(selection, args.download_root.absolute()) != before:
            raise ValueError("Original source checkpoint changed")
        if check_sources(args.source_manifest, args.source_sha256, root) != manifest:
            raise ValueError("Prepared numerical source changed")
        gpu.require_handoff(handoff_path)
        report["source_checkpoint_unchanged"] = True
        report["audit_execution_complete"] = True
        report["strict_bitwise_replay_passed"] = len(report["records"]) == 6 and all(
            x["strict_bitwise_replay_all_ranks"] for x in report["records"]
        )
    except BaseException as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        with (work / f"observations-rank-{rank}.json").open("x") as handle:
            json.dump(
                {"rank": rank, "observations": report["observations"]},
                handle,
                indent=2,
                sort_keys=True,
            )
        if rank == 0:
            report.update(
                observed_at_utc=datetime.now(UTC).isoformat(),
                source_bindings=[
                    identity(Path(inspect.getsourcefile(obj)).resolve())
                    for obj in (
                        train,
                        numerical,
                        gpu,
                        replay,
                        replay.previous,
                        replay.normalization,
                        control,
                        control.canonical,
                        control.change_owned_model,
                        control.Accelerator,
                        control.DistributedDataParallel,
                    )
                ]
                + [identity(Path(__file__).resolve())],
                device_name=torch.cuda.get_device_name(device),
                world_size=4,
            )
            with (work / "result.json").open("x") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")
        dist.destroy_process_group()
    if rank == 0 and not report["strict_bitwise_replay_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
