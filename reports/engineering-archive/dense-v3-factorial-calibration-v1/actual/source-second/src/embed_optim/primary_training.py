"""Exact-identity entrypoint for the reviewed revised primary matrix.

The checked-in proposal supports --inspect only until released and committed.
There is no uncommitted/draft/short-horizon execution bypass.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from contextlib import nullcontext
from pathlib import Path

from .primary_contract import PrimaryContract, require_same


def execute(contract, run_id, experiment_root, resume=None):
    contract = contract.require_execution()
    if os.environ.get("WORLD_SIZE") != "4":
        raise ValueError("Formal primary execution requires exactly four ranks")
    if (
        int(os.environ.get("EMBED_OPTIM_MAX_STEPS", "-1")) != -1
        or int(os.environ.get("EMBED_OPTIM_STOP_AFTER_STEP", "-1")) != -1
    ):
        raise ValueError(
            "Diagnostic horizon/stop overrides are forbidden in the primary entrypoint"
        )
    # Imports that could initialize the training stack follow release/source admission.
    from . import dense_run_contract as run_contract
    from . import train
    from .config import RunConfig
    from .gpu_lease import acquire_gpu_lease, parse_gpu_tokens
    from .runtime import verify_runtime_spec

    if (
        Path(inspect.getsourcefile(train)).resolve()
        != contract.training_root / "src/embed_optim/train.py"
    ):
        raise ValueError("Actual imported training source differs from the bound runtime tree")
    expected = contract.expected_identity(run_id)
    verify_runtime_spec(contract.repository / "configs/formal_runtime.json")
    tokens = parse_gpu_tokens(os.environ.get("CUDA_VISIBLE_DEVICES", ""), expected_count=4)
    if list(tokens) not in contract.payload["gpu_pools"]:
        raise ValueError("Primary launch must use one declared disjoint four-GPU pool")
    config = RunConfig.from_dict(
        {
            **expected["recipe"],
            "dataset_path": str(Path(experiment_root) / contract.payload["data_path"]),
            "output_root": str(Path(experiment_root) / contract.payload["output_root"]),
            "wandb_project": contract.payload["wandb"]["project"],
            "wandb_entity": contract.payload["wandb"]["entity"],
        }
    )
    actual, _ = run_contract.prepare(config, train._training_argument_values(config), 4, resume)
    require_same(actual, expected)
    if resume:
        step = json.loads((Path(resume) / "trainer_state.json").read_text())["global_step"]
        contract.checkpoint(resume, run_id, step)
    lease = (
        acquire_gpu_lease(
            tokens,
            lock_dir=contract.payload["gpu_lease_root"],
            timeout_seconds=60,
            purpose=f"primary-v2-training:{run_id}",
        )
        if os.environ.get("RANK") == "0"
        else nullcontext()
    )
    with lease:
        return train.run_training(config, str(resume) if resume else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    contract = PrimaryContract.load(
        args.protocol, args.repository, args.training_root, require_released=args.execute
    )
    expected = contract.expected_identity(args.run_id)
    if args.inspect:
        print(
            json.dumps(
                {
                    "scope": contract.payload["scope"],
                    "status": contract.payload["status"],
                    "protocol_sha256": contract.sha256,
                    "run_id": args.run_id,
                    "recipe": expected["recipe"],
                    "execution": expected["execution"],
                    "checkpoint_steps": contract.payload["checkpoint_steps"],
                    "scientific_completion": False,
                    "training_executed": False,
                },
                indent=2,
            )
        )
        return
    try:
        execute(contract, args.run_id, args.experiment_root, args.resume_from_checkpoint)
    finally:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
