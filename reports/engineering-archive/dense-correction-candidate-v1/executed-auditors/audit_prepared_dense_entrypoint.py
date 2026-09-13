"""Offline full-model smoke audit of the actual prepared run_training function.

No Trainer/model/collator/checkpoint implementation is patched. Only the declared
diagnostic configuration changes the input checkpoint, 288-row synthetic dataset,
three-step horizon, output namespace and checkpoint fractions. W&B is disabled.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset
from safetensors import safe_open

from embed_optim import dense_numerical_contract as numerical
from embed_optim import train
from embed_optim.config import load_matrix
from scripts import audit_dense_gpu_canonical_replay as checkpoint_tools
from scripts import audit_dense_gpu_normalization as gpu
from scripts.audit_checkpoint_download import select_download, verify_download
from scripts.audit_dense_gpu_replay_update import identity
from scripts.audit_prepared_dense_gpu import RUNS, check_sources


def check_completion(config, resume_step):
    root = config.output_dir
    payload = json.loads((root / "completed.json").read_text())
    contract = numerical.receipt(config.optimizer, 4)
    if (
        payload["global_step"] != 3
        or payload["dataset_rows"] != 288
        or payload["checkpoints"] != [1, 2, 3]
        or payload["numerical_contract"] != contract
        or payload["system_metrics"]["world_size"] != 4
        or payload["input_execution"]["sentence_transformers_can_flatten_inputs"] is not False
    ):
        raise ValueError("Real run_training completion receipt differs")
    observed = json.loads((root / "run_config.json").read_text())
    expected = json.loads(json.dumps(config.as_dict()))
    if observed != expected:
        raise ValueError("Real resolved run configuration differs")
    checkpoint_files = {}
    for step in range(resume_step + 1, 4):
        checkpoint = root / f"checkpoint-{step}"
        numerical.require_resume_receipt(checkpoint, contract)
        checkpoint_files[str(step)] = checkpoint_tools.checkpoint_files(checkpoint)
    if not (root / "final/model.safetensors").is_file():
        raise ValueError("Final inference export missing")
    with safe_open(root / "final/model.safetensors", framework="pt", device="cpu") as final:
        with safe_open(
            root / "checkpoint-3/model.safetensors", framework="pt", device="cpu"
        ) as sealed:
            if set(final.keys()) != set(sealed.keys()):
                raise ValueError("Final and scheduled model topology differs")
            for name in final.keys():
                value = final.get_tensor(name)
                if (
                    not torch.equal(value, sealed.get_tensor(name))
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError("Final model differs from the final scheduled checkpoint")
    return {
        "completion": identity(root / "completed.json"),
        "resolved_config": identity(root / "run_config.json"),
        "checkpoint_files": checkpoint_files,
        "final_equals_scheduled_checkpoint": True,
    }


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
            "CUDA_VISIBLE_DEVICES": "4,5,6,7",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
            "EMBED_OPTIM_MAX_STEPS": "3",
            "EMBED_OPTIM_STOP_AFTER_STEP": "-1",
        }.items()
    ):
        raise ValueError("Require the explicit second four-rank offline diagnostic pool")
    if (
        work.parent != Path("/tmp")
        or not work.name.startswith("dense-prepared-entry.")
        or not work.is_dir()
        or work.is_symlink()
    ):
        raise ValueError("Require a new prepared-entry mktemp namespace")
    manifest = check_sources(args.source_manifest, args.source_sha256, root)
    handoff_path = Path("/tmp/dense-gpu-handoff.4MuYjC/receipt/result.json")
    handoff = gpu.require_handoff(handoff_path)
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group("nccl", timeout=timedelta(seconds=600), device_id=device)
    report = {
        "scope": "engineering_prepared_dense_actual_run_training_entrypoint",
        "scientific_completion": False,
        "production_deployed": False,
        "audit_execution_complete": False,
        "entrypoint_acceptance_passed": False,
        "source_manifest": identity(args.source_manifest),
        "handoff": handoff,
        "records": [],
        "boundary": "Actual unpatched prepared run_training including dataset loading/column selection, model/loss/collator construction, distributed output setup, optimizer, callbacks, checkpoint contracts, native restore, final save and completed.json. Three algorithms each run a three-step synthetic epoch and continue from step 2 in a separate output namespace. Default DDP communication; no bitwise trajectory equivalence claim, CLI/frozen-protocol bypass, formal data run, independent restart or scientific result. W&B disabled; short fixture only.",
    }
    try:
        if rank == 0 and any(work.iterdir()):
            raise ValueError("Output must be empty")
        selection = select_download(args.audit, args.audit_sha256, "padded-normuon-3e-4", 3126)
        before = verify_download(selection, args.download_root.absolute())
        report["source_checkpoint"] = before
        dataset_path = work / "diagnostic-dataset"
        if rank == 0:
            Dataset.from_list(gpu.previous.synthetic_rows(288)).save_to_disk(str(dataset_path))
        dist.barrier()
        configs = {
            c.run_id: c for c in load_matrix(root / "configs/dense_correctness_candidate.yaml")
        }
        for run_id in RUNS:
            baseline = replace(
                configs[run_id],
                model_name=before["checkpoint_root"],
                model_revision=None,
                dataset_path=str(dataset_path),
                output_root=str(work / "baseline"),
                checkpoint_fractions=(1 / 3, 2 / 3, 1.0),
            )
            continuation = replace(baseline, output_root=str(work / "continuation"))
            source_files = None
            for config, step in ((baseline, 0), (continuation, 2)):
                os.environ["WANDB_RUN_ID"] = f"{work.name}-{run_id}-from-{step}"
                checkpoint = baseline.output_dir / "checkpoint-2" if step else None
                result = train.run_training(config, str(checkpoint) if checkpoint else None)
                if result != config.output_dir / "final":
                    raise ValueError("Unexpected actual final output path")
                dist.barrier()
                numerical.require_resume_receipt(
                    config.output_dir / "checkpoint-3", numerical.receipt(config.optimizer, 4)
                )
                if rank == 0:
                    completion = check_completion(config, step)
                    if step == 0:
                        source_files = checkpoint_tools.checkpoint_files(
                            baseline.output_dir / "checkpoint-2"
                        )
                    elif (
                        checkpoint_tools.checkpoint_files(baseline.output_dir / "checkpoint-2")
                        != source_files
                    ):
                        raise ValueError("Source diagnostic checkpoint mutated on resume")
                    report["records"].append(
                        {"run_id": run_id, "resume_step": step, "completion": completion}
                    )
                    print(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "resume_step": step,
                                "actual_entrypoint_completed": True,
                            }
                        ),
                        flush=True,
                    )
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
        report["entrypoint_acceptance_passed"] = len(report["records"]) == 6
    except BaseException as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if rank == 0:
            report.update(
                observed_at_utc=datetime.now(UTC).isoformat(),
                source_bindings=[
                    identity(Path(inspect.getsourcefile(obj)).resolve())
                    for obj in (
                        train,
                        numerical,
                        gpu,
                        gpu.previous,
                        checkpoint_tools,
                        check_sources,
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
    if rank == 0 and not report["entrypoint_acceptance_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
