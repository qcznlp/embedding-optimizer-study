"""Four-GPU full-DenseOn candidate save/resume diagnostic, not formal training.

Reuses the CPU audit's actual Trainer/checkpoint instrumentation, replacing only
its diagnostic fixture loader/argument builder and adding local CUDA RNG hashes.
No production/library implementation or deserialization mapping is patched.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
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
from scripts import audit_dense_gpu_normalization as gpu
from scripts import audit_dense_trainer_resume as resume
from scripts.audit_checkpoint_download import select_download, verify_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
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
        raise ValueError("Require four owned CUDA ranks, offline inputs and disabled W&B")
    root, work = args.repository.resolve(), args.workdir.absolute()
    if (
        work.parent != Path("/tmp")
        or not work.name.startswith("dense-gpu-resume.")
        or not work.is_dir()
    ):
        raise ValueError("Require a fresh /tmp/dense-gpu-resume.XXXXXX mktemp directory")
    if Path(inspect.getsourcefile(train)).resolve() != root / "src/embed_optim/train.py":
        raise ValueError("Wrong audited source")
    handoff = gpu.require_handoff(args.handoff)
    gpu.verify_stack()
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group("nccl", timeout=timedelta(seconds=600), device_id=device)
    report = {
        "scope": "engineering_full_denseon_four_gpu_candidate_save_resume",
        "scientific_completion": False,
        "runtime_deployed": False,
        "audit_execution_complete": False,
        "strict_bitwise_replay_passed": False,
        "records": [],
        "handoff": handoff,
        "prior_gradient_tolerances": resume.previous.TOLERANCES,
        "boundary": "Authenticated real trained DenseOn weights, fresh candidate optimizers, synthetic 288-row epoch with groups 128/128/32, actual NCCL/BF16/FA2, zero dropout, ordinary checkpoint callbacks and GPU deserialization. Baseline then fresh Trainer resumes from steps 1 and 2 in the same process group; local CUDA plus CPU/Python/NumPy RNG and actual row order checked on every rank. Not independent process/host restart, historical optimizer continuation, maximum-length inputs, long-horizon or retrieval evidence. Numerical differences remain measurements, not relaxed bitwise acceptance.",
    }
    try:
        if rank == 0 and any(work.iterdir()):
            raise ValueError("Output directory must be empty")
        selection = select_download(args.audit, args.audit_sha256, "padded-normuon-3e-4", 3126)
        before = verify_download(selection, args.download_root.absolute())
        report["source_checkpoint"] = before
        checkpoint = Path(before["checkpoint_root"])
        configs = {c.run_id: c for c in load_matrix(root / "configs/dense_no_packing_retrain.yaml")}
        original_rng = resume.rng_hashes

        def load_full(path):
            if Path(path) != checkpoint:
                raise ValueError("Diagnostic attempted to load an unauthenticated fixture")
            return gpu.load_model(checkpoint, device)

        def arguments(config, output):
            actual, overrides = gpu.gpu_arguments(config, output)
            # Exercise the declared production worker/prefetch settings as well.
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
                "cuda_local": resume.tensor_hash(torch.cuda.get_rng_state(device)),
            }

        with (
            patch.object(resume.previous, "load_fixture", side_effect=load_full),
            patch.object(resume.previous, "cpu_arguments", side_effect=arguments),
            patch.object(resume, "rng_hashes", side_effect=rng),
        ):
            for run_id in resume.previous.RUN_IDS:
                case = work / run_id
                reference, metadata = resume.train_segment(
                    configs[run_id], checkpoint, case / "uninterrupted", "candidate", 0.0
                )
                gathered_ids = [None] * 4
                dist.all_gather_object(
                    gathered_ids, [i for x in reference["trace"] for i in x["row_ids"]]
                )
                ids = [i for chunk in gathered_ids for i in chunk]
                if len(ids) != 288 or set(ids) != set(range(288)):
                    raise ValueError("Baseline rows repeated or missing")
                for step in (1, 2):
                    restored, restored_metadata = resume.train_segment(
                        configs[run_id],
                        checkpoint,
                        case / f"resume-{step}",
                        "candidate",
                        0.0,
                        case / "uninterrupted" / f"checkpoint-{step}",
                    )
                    if restored_metadata["cpu_transport_adapter"]:
                        raise ValueError("GPU resume must use native deserialization")
                    comparison = resume.measure_replay(reference, restored, step)
                    ranks = [None] * 4
                    dist.all_gather_object(
                        ranks,
                        {
                            "rank": rank,
                            "comparison": comparison,
                            "trace": restored["trace"],
                            "gradients": restored["gradients"],
                            "rng": restored["rng"],
                        },
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
                                    "run": run_id,
                                    "resume_step": step,
                                    "exact_entry_rows_rng_scheduler": True,
                                    "strict_bitwise": record["strict_bitwise_replay_all_ranks"],
                                }
                            ),
                            flush=True,
                        )
                    del restored
                    gc.collect()
                    torch.cuda.empty_cache()
                    dist.barrier()
                del reference
                gc.collect()
                torch.cuda.empty_cache()
                dist.barrier()
        if verify_download(selection, args.download_root.absolute()) != before:
            raise ValueError("Source checkpoint changed")
        gpu.require_handoff(args.handoff)
        report["source_checkpoint_unchanged"] = True
        report["audit_execution_complete"] = True
        report["strict_bitwise_replay_passed"] = (
            all(x["strict_bitwise_replay_all_ranks"] for x in report["records"])
            if rank == 0
            else False
        )
    except BaseException as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if rank == 0:
            paths = [
                Path(__file__).resolve(),
                Path(inspect.getsourcefile(resume)),
                Path(inspect.getsourcefile(gpu)),
                Path(inspect.getsourcefile(resume.previous)),
                Path(inspect.getsourcefile(resume.normalization)),
                Path(inspect.getsourcefile(resume.SingleNormalizationTrainerCandidate)),
                *[
                    root / f"src/embed_optim/{n}.py"
                    for n in ("train", "callbacks", "losses", "collators", "optimizers", "config")
                ],
            ]
            report.update(
                observed_at_utc=datetime.now(UTC).isoformat(),
                source_bindings=[resume.previous.identity(p) for p in paths],
                versions={
                    p: importlib.metadata.version(p)
                    for p in (
                        "torch",
                        "transformers",
                        "accelerate",
                        "sentence-transformers",
                        "datasets",
                        "flash-attn",
                    )
                },
            )
            with (work / "result.json").open("x") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")
        dist.destroy_process_group()
    if rank == 0 and not report["strict_bitwise_replay_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
