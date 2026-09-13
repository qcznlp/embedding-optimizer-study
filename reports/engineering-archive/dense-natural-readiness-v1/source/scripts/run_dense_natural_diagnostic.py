"""Unpatched four-rank entrypoint on authentic natural rows and untrained base."""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset

from embed_optim import dense_run_contract as contract
from embed_optim import train
from embed_optim.config import load_matrix
from scripts.audit_dense_identity_entrypoint import check_completion
from scripts.audit_prepared_dense_gpu import RUNS, check_sources


def write_new(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    args = parser.parse_args()
    work, root = args.workdir.absolute(), args.candidate_root.resolve()
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
        raise ValueError("Require the declared offline four-rank bounded diagnostic")
    if (
        work.parent != Path("/tmp")
        or not work.name.startswith("dense-natural-readiness.")
        or work.is_symlink()
    ):
        raise ValueError("Require a separate natural-data diagnostic namespace")
    source = check_sources(args.source_manifest, args.source_sha256, root)
    if contract.file_identity(args.input_receipt)["sha256"] != args.input_sha256:
        raise ValueError("Original natural-data/base receipt differs")
    inputs = contract.load_json(args.input_receipt)
    selection = contract.load_json(work / "selection.json")
    data = Dataset.load_from_disk(str(work / "dataset"))
    if len(data) != 288 or contract.digest(list(data)) != selection["selected_rows_sha256"]:
        raise ValueError("Natural-data prefix changed before worker preparation")
    matrix = {c.run_id: c for c in load_matrix(root / "configs/dense_correctness_candidate.yaml")}
    configs = [
        replace(
            matrix[key],
            run_id=f"diagnostic-natural-{matrix[key].optimizer.name}",
            dataset_path=str(work / "dataset"),
            output_root=str(work / "runs"),
            checkpoint_fractions=(1 / 3, 2 / 3, 1.0),
        )
        for key in RUNS
    ]
    expected = {}
    for config in configs:
        value, _ = contract.prepare(config, train._training_argument_values(config), 4)
        contract.require_equal(value["model"], inputs["common_identity"]["model"])
        contract.require_equal(value["source"], inputs["common_identity"]["source"])
        if (config.model_name, config.model_revision) != (
            inputs["initial_model"]["repo"],
            inputs["initial_model"]["revision"],
        ):
            raise ValueError("Diagnostic must initialize the original immutable untrained base")
        expected[config.optimizer.name] = value
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group("nccl", timeout=timedelta(seconds=600), device_id=device)
    report = {
        "scope": "engineering_natural_data_actual_run_training",
        "scientific_completion": False,
        "production_deployed": False,
        "natural_entrypoint_passed": False,
        "records": [],
    }
    try:
        if rank == 0:
            write_new(
                work / "prepared.json",
                {
                    "expected_runs": expected,
                    "original_model_identity": inputs["common_identity"]["model"],
                    "selection": selection,
                    "scientific_completion": False,
                },
            )
        dist.barrier()
        for config in configs:
            os.environ["WANDB_RUN_ID"] = f"{work.name}-{config.run_id}"
            final = train.run_training(config)
            if final != config.output_dir / "final":
                raise ValueError("Actual training returned an unexpected final path")
            dist.barrier()
            if rank == 0:
                actual = contract.read_identity(config.output_dir)
                contract.require_equal(actual, expected[config.optimizer.name])
                checked = check_completion(config, 0)
                report["records"].append(
                    {
                        "algorithm": config.optimizer.name,
                        "identity_sha256": contract.digest(actual),
                        "completion": checked,
                    }
                )
                print(
                    json.dumps(
                        {
                            "algorithm": config.optimizer.name,
                            "actual_entrypoint_completed": True,
                            "natural_rows": 288,
                        }
                    ),
                    flush=True,
                )
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
        if check_sources(args.source_manifest, args.source_sha256, root) != source:
            raise ValueError("Prepared source changed during natural-data execution")
        report["natural_entrypoint_passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if rank == 0:
            report.update(
                observed_at_utc=datetime.now(UTC).isoformat(),
                gpu_name=torch.cuda.get_device_name(device),
                world_size=4,
            )
            write_new(work / "worker-result.json", report)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
