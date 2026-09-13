"""Isolated four-GPU 8192-token full-model normalization/memory stress audit."""

from __future__ import annotations

import argparse
import gc
import hashlib
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
from embed_optim.collators import TEXT_COLUMNS
from embed_optim.config import load_matrix
from scripts import audit_dense_gpu_normalization as gpu
from scripts.audit_checkpoint_download import select_download, verify_download

SHORT_ROWS = gpu.previous.synthetic_rows
LONG_ROW_COUNT = 32
REPEATS = 1500


def stress_rows(count):
    if count != 288:
        raise ValueError("The full/tail stress epoch has exactly 288 distinct rows")
    rows = SHORT_ROWS(count)
    for row in rows[:LONG_ROW_COUNT]:
        for column in TEXT_COLUMNS:
            row[column] = (row[column] + " ") * REPEATS
        row["length"] = sum(len(row[c]) for c in TEXT_COLUMNS)
    return rows


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
            "CUDA_VISIBLE_DEVICES": "4,5,6,7",
        }.items()
    ):
        raise ValueError(
            "Require the disjoint four-GPU stress pool, offline inputs and disabled W&B"
        )
    root, work = args.repository.resolve(), args.workdir.absolute()
    if (
        work.parent != Path("/tmp")
        or not work.name.startswith("dense-gpu-max-context.")
        or not work.is_dir()
    ):
        raise ValueError("Require a fresh /tmp/dense-gpu-max-context.XXXXXX mktemp directory")
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
        "scope": "engineering_full_denseon_four_gpu_8192_token_candidate",
        "scientific_completion": False,
        "runtime_deployed": False,
        "audit_execution_complete": False,
        "normalization_acceptance_passed": False,
        "records": [],
        "handoff": handoff,
        "prior_gradient_tolerances": gpu.previous.TOLERANCES,
        "probe": {
            "rows": 288,
            "long_row_ids": list(range(LONG_ROW_COUNT)),
            "repetition_count": REPEATS,
            "definition": "First 32 deterministic synthetic rows repeat every text column 1500 times before the unchanged 8192-token collator truncation; remaining 256 rows are short. No quality-based selection.",
        },
        "boundary": "Actual four-GPU candidate Trainer, real authenticated trained DenseOn weights, fresh AdamW/Muon/NorMuon diagnostic optimizers, BF16/FA2 and groups 128/128/32. All nine columns of 32 predetermined synthetic rows exercise configured truncation and maximum length; not natural-text quality or a scientific optimizer comparison. Same-microbatch manual-autograd normalization oracle, no production deployment, worker replay or historical optimizer resume.",
    }
    try:
        if rank == 0 and any(work.iterdir()):
            raise ValueError("Require an empty diagnostic output directory")
        selection = select_download(args.audit, args.audit_sha256, "padded-normuon-3e-4", 3126)
        before = verify_download(selection, args.download_root.absolute())
        report["source_checkpoint"] = before
        rows = stress_rows(288)
        report["probe"]["all_rows_sha256"] = hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        configs = {c.run_id: c for c in load_matrix(root / "configs/dense_no_packing_retrain.yaml")}
        with patch.object(gpu.previous, "synthetic_rows", side_effect=stress_rows):
            for index, run_id in enumerate(gpu.previous.RUN_IDS):
                result = gpu.run_case(
                    configs[run_id],
                    Path(before["checkpoint_root"]),
                    work / run_id,
                    "candidate",
                    rank,
                    device,
                    profile=index == 0,
                )
                lengths = [x["local_token_length_max"] for x in result["records"]]
                if max(lengths) != 8192:
                    raise ValueError(
                        "The actual reference/Trainer tokens did not reach 8192 on this rank"
                    )
                local = {
                    "rank": rank,
                    "step_max_token_lengths": lengths,
                    "peak_allocated_bytes": result["peak_allocated_bytes"],
                    "peak_reserved_bytes": result["peak_reserved_bytes"],
                    "all_raw_and_clipped_checks_pass": result["all_raw_and_clipped_checks_pass"],
                }
                ranks = [None] * 4
                dist.all_gather_object(ranks, local)
                if rank == 0:
                    result["rank_stress_coverage"] = ranks
                    report["records"].append(result)
                    with (work / f"{run_id}-stress.json").open("x") as handle:
                        json.dump(result, handle, indent=2, sort_keys=True)
                gc.collect()
                torch.cuda.empty_cache()
                dist.barrier()
        if verify_download(selection, args.download_root.absolute()) != before:
            raise ValueError("Source checkpoint changed")
        gpu.require_handoff(args.handoff)
        report["source_checkpoint_unchanged"] = True
        report["audit_execution_complete"] = True
        report["normalization_acceptance_passed"] = (
            all(x["all_raw_and_clipped_checks_pass"] for x in report["records"])
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
                Path(inspect.getsourcefile(gpu)),
                Path(inspect.getsourcefile(gpu.previous)),
                Path(inspect.getsourcefile(gpu.normalization)),
                Path(inspect.getsourcefile(gpu.SingleNormalizationTrainerCandidate)),
                *[
                    root / f"src/embed_optim/{n}.py"
                    for n in ("train", "losses", "collators", "optimizers", "config")
                ],
            ]
            report.update(
                observed_at_utc=datetime.now(UTC).isoformat(),
                source_bindings=[gpu.previous.identity(p) for p in paths],
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
    if rank == 0 and not report["normalization_acceptance_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
