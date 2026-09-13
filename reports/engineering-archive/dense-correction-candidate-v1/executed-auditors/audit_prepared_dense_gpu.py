"""Actual full-model gradient oracle for the explicit prepared production Trainer.

This selects the new Trainer through an explicit factory, not by weakening the
old candidate's source guard. All old diagnostics/receipts remain unchanged.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from embed_optim import dense_numerical_contract as numerical
from embed_optim import train
from embed_optim.config import DENSE_NUMERICAL_POLICY, load_matrix
from scripts import audit_dense_gpu_max_context as stress
from scripts import audit_dense_gpu_normalization as gpu
from scripts.audit_checkpoint_download import select_download, verify_download
from scripts.audit_dense_gpu_replay_update import identity

RUNS = ("verified-adamw-3e-5", "verified-muon-3e-4", "verified-normuon-3e-4")


def check_sources(path, sha256, root):
    if identity(path)["sha256"] != sha256:
        raise ValueError("Untrusted prepared-source manifest")
    manifest = json.loads(path.read_text())
    if manifest["scope"] != "preparation_only_dense_numerical_correction" or manifest[
        "candidate_root"
    ] != str(root):
        raise ValueError("Wrong candidate source selection")
    for item in manifest["candidate_sources"]:
        if identity(root / item["relative_path"]) != item["identity"]:
            raise ValueError("Prepared source changed")
    if Path(inspect.getsourcefile(train)).resolve() != root / "src/embed_optim/train.py":
        raise ValueError("Wrong actual Trainer import")
    numerical.verify_stack(train.OptimizerTrainer)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--fixture", choices=("short", "max_context"), default="short")
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    args = parser.parse_args()
    root, work = args.candidate_root.resolve(), args.workdir.absolute()
    if any(
        os.environ.get(k) != v
        for k, v in {
            "WORLD_SIZE": "4",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3" if args.fixture == "short" else "4,5,6,7",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
        }.items()
    ):
        raise ValueError("Require the explicit four-rank offline diagnostic pool")
    if (
        work.parent != Path("/tmp")
        or not work.name.startswith("dense-prepared-gpu.")
        or not work.is_dir()
        or work.is_symlink()
    ):
        raise ValueError("Require a fresh prepared-GPU mktemp namespace")
    manifest = check_sources(args.source_manifest, args.source_sha256, root)
    handoff_path = Path("/tmp/dense-gpu-handoff.4MuYjC/receipt/result.json")
    handoff = gpu.require_handoff(handoff_path)
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group("nccl", timeout=timedelta(seconds=600), device_id=device)
    report = {
        "scope": "engineering_prepared_dense_trainer_full_gpu_normalization",
        "scientific_completion": False,
        "production_deployed": False,
        "audit_execution_complete": False,
        "normalization_acceptance_passed": False,
        "source_manifest": identity(args.source_manifest),
        "handoff": handoff,
        "records": [],
        "fixture": args.fixture,
        "owned_model_count": 0,
        "trainer_selection": "Prepared embed_optim.train.OptimizerTrainer with explicit corrected numerical policy",
        "tolerances": gpu.previous.TOLERANCES,
        "boundary": "Actual integrated Trainer loss-normalization and optimizer-factory paths, owned deterministic attention backward, default DDP communication, authenticated full trained DenseOn weights, fresh optimizers, 288 predetermined synthetic rows and global groups 128/128/32. Short or pre-existing maximum-context stress fixture selected explicitly. Unchanged manual-autograd gradient oracle and tolerance; no canonical communication hook, old source-guard override, primary launch, checkpoint save/resume, historical optimizer continuation or scientific claim.",
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
        rows = (stress.stress_rows if args.fixture == "max_context" else stress.SHORT_ROWS)(288)
        report["fixture_rows_sha256"] = hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        original_load = gpu.load_model

        def load(*a, **kw):
            model = original_load(*a, **kw)
            numerical.configure_backward(model)
            report["owned_model_count"] += 1
            return model

        def trainer_factory(*a, **kw):
            return train.OptimizerTrainer(*a, numerical_policy=DENSE_NUMERICAL_POLICY, **kw)

        with (
            patch.object(gpu, "load_model", side_effect=load),
            patch.object(gpu, "SingleNormalizationTrainerCandidate", side_effect=trainer_factory),
            (
                patch.object(gpu.previous, "synthetic_rows", side_effect=stress.stress_rows)
                if args.fixture == "max_context"
                else nullcontext()
            ),
        ):
            for index, run_id in enumerate(RUNS):
                config = configs[run_id]
                if config.numerical_policy != DENSE_NUMERICAL_POLICY:
                    raise ValueError("Matrix has not selected the corrected policy")
                numerical.require_optimizer_choice(config.optimizer)
                result = gpu.run_case(
                    config, checkpoint, work / run_id, "candidate", rank, device, profile=index == 0
                )
                if args.fixture == "max_context":
                    lengths = [x["local_token_length_max"] for x in result["records"]]
                    if max(lengths) != 8192:
                        raise ValueError("Maximum context not exercised on this rank")
                    ranks = [None] * 4
                    dist.all_gather_object(
                        ranks,
                        {
                            "rank": rank,
                            "step_max_token_lengths": lengths,
                            "peak_allocated_bytes": result["peak_allocated_bytes"],
                            "peak_reserved_bytes": result["peak_reserved_bytes"],
                            "all_raw_and_clipped_checks_pass": result[
                                "all_raw_and_clipped_checks_pass"
                            ],
                        },
                    )
                    result["rank_stress_coverage"] = ranks
                report["records"].append(result)
                if rank == 0:
                    with (work / f"{run_id}.json").open("x") as handle:
                        json.dump(result, handle, indent=2, sort_keys=True)
                gc.collect()
                torch.cuda.empty_cache()
                dist.barrier()
        if verify_download(selection, args.download_root.absolute()) != before:
            raise ValueError("Source checkpoint changed")
        if check_sources(args.source_manifest, args.source_sha256, root) != manifest:
            raise ValueError("Source manifest changed")
        gpu.require_handoff(handoff_path)
        report["source_checkpoint_unchanged"] = True
        report["audit_execution_complete"] = True
        report["normalization_acceptance_passed"] = all(
            x["all_raw_and_clipped_checks_pass"] for x in report["records"]
        )
    except BaseException as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if rank == 0:
            report.update(
                observed_at_utc=datetime.now(UTC).isoformat(),
                source_bindings=[
                    identity(Path(inspect.getsourcefile(obj)).resolve())
                    for obj in (gpu, gpu.previous, gpu.normalization, stress, train, numerical)
                ]
                + [identity(Path(__file__).resolve())],
                device_name=torch.cuda.get_device_name(device),
                world_size=4,
            )
            with (work / "result.json").open("x") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")
        dist.destroy_process_group()
    if rank == 0 and not report["normalization_acceptance_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
