"""CPU-only serialization continuation audit, not a distributed-training resume test."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import get_linear_schedule_with_warmup

from embed_optim.config import load_matrix
from embed_optim.incremental_checkpoint_backup import (
    local_checkpoint_inventory,
    stat_signature,
    validate_sealed_checkpoint,
)
from embed_optim.optimizers import build_optimizer


def assert_exact(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        if (
            not isinstance(right, torch.Tensor)
            or left.dtype != right.dtype
            or left.shape != right.shape
            or not torch.equal(left, right)
        ):
            raise ValueError(f"Non-identical tensor: {path}")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError(f"Mapping keys differ: {path}")
        for key in left:
            assert_exact(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError(f"Sequence differs: {path}")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            assert_exact(a, b, f"{path}[{index}]")
    elif type(left) is not type(right) or left != right:
        raise ValueError(f"Scalar differs: {path}")


def _stack(model, optimizer_config, total_steps, warmup_steps):
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("The diagnostic requires CPU float32 parameters")
    optimizer, partition = build_optimizer(model, optimizer_config)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return optimizer, scheduler, partition


def _check_lr(optimizer, scheduler, total_steps, warmup_steps):
    step = scheduler.last_epoch
    multiplier = (
        step / max(1, warmup_steps)
        if step < warmup_steps
        else max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    )
    expected = [base * multiplier for base in scheduler.base_lrs]
    actual = [group["lr"] for group in optimizer.param_groups]
    if actual != expected or scheduler.get_last_lr() != expected:
        raise ValueError("Restored learning rate disagrees with the declared linear schedule")
    return {"step": step, "learning_rates": actual}


def _synthetic_update(model, optimizer, scheduler, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    for parameter in model.parameters():
        parameter.grad = (
            torch.randn(parameter.shape, generator=generator, dtype=torch.float32) * 1e-3
        )
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    changed = sum(not torch.equal(before[name], p) for name, p in model.named_parameters())
    if not changed or any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Synthetic continuation has no update or non-finite parameters")
    for state in optimizer.state.values():
        if any(isinstance(v, torch.Tensor) and not torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Synthetic continuation has non-finite optimizer state")
    return changed


def exercise_resume(
    model_factory,
    optimizer_config,
    saved_optimizer,
    saved_scheduler,
    total_steps,
    warmup_steps,
    seed=20260905,
):
    model = model_factory()
    optimizer, scheduler, partition = _stack(model, optimizer_config, total_steps, warmup_steps)
    # Match Trainer's ordering: construct both, then restore their saved states.
    optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
    scheduler.load_state_dict(copy.deepcopy(saved_scheduler))
    assert_exact(optimizer.state_dict(), saved_optimizer, "initial.optimizer")
    assert_exact(scheduler.state_dict(), saved_scheduler, "initial.scheduler")
    ids = [i for group in saved_optimizer["param_groups"] for i in group["params"]]
    if len(ids) != len(set(ids)) or set(ids) != set(saved_optimizer["state"]):
        raise ValueError("The saved optimizer does not cover every unique parameter")
    rates = [_check_lr(optimizer, scheduler, total_steps, warmup_steps)]
    changed_first = _synthetic_update(model, optimizer, scheduler, seed)
    rates.append(_check_lr(optimizer, scheduler, total_steps, warmup_steps))

    buffer = io.BytesIO()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        buffer,
    )
    serialized_bytes = buffer.tell()
    buffer.seek(0)
    restored = torch.load(buffer, map_location="cpu", weights_only=True)
    second_model = model_factory()
    second_optimizer, second_scheduler, second_partition = _stack(
        second_model, optimizer_config, total_steps, warmup_steps
    )
    second_model.load_state_dict(restored["model"], strict=True)
    second_optimizer.load_state_dict(restored["optimizer"])
    second_scheduler.load_state_dict(restored["scheduler"])
    assert_exact(model.state_dict(), second_model.state_dict(), "reserialized.model")
    assert_exact(optimizer.state_dict(), second_optimizer.state_dict(), "reserialized.optimizer")
    assert_exact(scheduler.state_dict(), second_scheduler.state_dict(), "reserialized.scheduler")
    assert_exact(partition, second_partition, "partition")
    del restored
    buffer.close()

    changed_second = _synthetic_update(model, optimizer, scheduler, seed + 1)
    resumed_changed = _synthetic_update(second_model, second_optimizer, second_scheduler, seed + 1)
    assert_exact(model.state_dict(), second_model.state_dict(), "continued.model")
    assert_exact(optimizer.state_dict(), second_optimizer.state_dict(), "continued.optimizer")
    assert_exact(scheduler.state_dict(), second_scheduler.state_dict(), "continued.scheduler")
    if changed_second != resumed_changed:
        raise ValueError("Different parameters updated after serialization")
    rates.append(_check_lr(optimizer, scheduler, total_steps, warmup_steps))
    names = {id(p): n for n, p in model.named_parameters()}
    return {
        "complete_for_cpu_diagnostic": True,
        "optimizer": optimizer_config.name,
        "partition": partition,
        "parameter_count": len(names),
        "optimizer_state_count": len(optimizer.state),
        "serialized_bytes": serialized_bytes,
        "gradient_seeds": [seed, seed + 1],
        "gradient_distribution": "CPU float32 standard normal scaled by 1e-3; diagnostic, not training data",
        "parameter_updates_per_step": [changed_first, changed_second],
        "learning_rate_trace": rates,
        "parameter_order_by_group": [
            [names[id(p)] for p in g["params"]] for g in optimizer.param_groups
        ],
        "initial_saved_states_exact": True,
        "intermediate_roundtrip_exact": True,
        "second_update_weights_and_states_exact": True,
        "distributed_training_resumed": False,
        "forward_or_backward_executed": False,
        "training_data_order_or_rank_rng_tested": False,
    }


def identity(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--step", type=int, default=2345)
    parser.add_argument(
        "--run-ids",
        nargs="+",
        default=["padded-adamw-3e-5", "padded-muon-3e-4", "padded-normuon-3e-4"],
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("Set CUDA_VISIBLE_DEVICES='' and HF_HUB_OFFLINE=1")
    if args.output.exists():
        raise ValueError("Refusing to overwrite a diagnostic receipt")
    root = args.repository.resolve()
    for function, relative in [
        (build_optimizer, "src/embed_optim/optimizers.py"),
        (load_matrix, "src/embed_optim/config.py"),
        (stat_signature, "src/embed_optim/incremental_checkpoint_backup.py"),
    ]:
        if Path(inspect.getsourcefile(function)).resolve() != root / relative:
            raise ValueError("Set PYTHONPATH to the audited repository's absolute src directory")
    os.chdir(root)
    from sentence_transformers import SentenceTransformer

    matrix_path = root / "configs/dense_no_packing_retrain.yaml"
    configs = {config.run_id: config for config in load_matrix(matrix_path)}
    if len(args.run_ids) != len(set(args.run_ids)) or set(args.run_ids) - set(configs):
        raise ValueError("Duplicate or unknown primary run")
    execution_path = root / "configs/dense_no_packing_execution_protocol.json"
    execution = json.loads(execution_path.read_text())
    total = execution["training"]["expected_optimizer_steps"]
    if (
        args.step not in execution["training"]["expected_checkpoint_steps"]
        or args.step + 2 >= total
    ):
        raise ValueError("Choose a scheduled nonterminal checkpoint for this two-update diagnostic")
    results = []
    for run_id in args.run_ids:
        config = configs[run_id]
        checkpoint = validate_sealed_checkpoint(config, args.step)
        before = stat_signature(checkpoint)
        inventory = local_checkpoint_inventory(checkpoint)
        saved_optimizer = torch.load(
            checkpoint / "optimizer.pt", map_location="cpu", weights_only=True
        )
        saved_scheduler = torch.load(
            checkpoint / "scheduler.pt", map_location="cpu", weights_only=True
        )
        if saved_scheduler["last_epoch"] != args.step:
            raise ValueError("Scheduler and checkpoint step differ")

        def factory():
            return SentenceTransformer(
                str(checkpoint),
                device="cpu",
                local_files_only=True,
                model_kwargs={"dtype": torch.float32, "attn_implementation": "sdpa"},
            )

        result = exercise_resume(
            factory,
            config.optimizer,
            saved_optimizer,
            saved_scheduler,
            total,
            math.ceil(config.warmup_ratio * total),
        )
        if stat_signature(checkpoint) != before:
            raise ValueError("Sealed source checkpoint changed during the diagnostic")
        results.append(
            {
                "run_id": run_id,
                "checkpoint_step": args.step,
                "checkpoint_root": str(checkpoint),
                "checkpoint_inventory": inventory,
                **result,
            }
        )
        print(
            json.dumps({"run_id": run_id, "exact": True, "parameters": result["parameter_count"]}),
            flush=True,
        )
    report = {
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "scope": "real_primary_checkpoint_cpu_optimizer_serialization_continuation",
        "implementation": identity(Path(__file__).resolve()),
        "optimizer_implementation": identity(root / "src/embed_optim/optimizers.py"),
        "configuration_loader": identity(root / "src/embed_optim/config.py"),
        "checkpoint_validation_helper": identity(
            root / "src/embed_optim/incremental_checkpoint_backup.py"
        ),
        "scheduler_implementation": identity(
            Path(inspect.getsourcefile(get_linear_schedule_with_warmup))
        ),
        "matrix": identity(matrix_path),
        "execution_protocol": identity(execution_path),
        "torch_version": torch.__version__,
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("transformers", "sentence-transformers", "safetensors")
        },
        "threads": torch.get_num_threads(),
        "records": results,
        "source_checkpoints_modified": False,
        "hf_mutations": False,
        "scientific_completion": False,
        "boundary": "Three selected real primary checkpoints by default, CPU optimizer/scheduler serialization only. No GPU, training forward/backward, rank RNG, data-loader continuation, all-checkpoint resume claim or optimizer-quality inference. Parameter order follows the current frozen model/optimizer builder; original training did not store an independent per-slot name list.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
