"""Isolated four-rank CPU save/resume verification; never a study launcher."""

from __future__ import annotations

import argparse
import copy
import functools
import gc
import hashlib
import inspect
import json
import os
import pickle
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset

from embed_optim import train
from embed_optim.callbacks import FractionalCheckpointCallback
from embed_optim.config import load_matrix
from embed_optim.losses import ExplicitDenseInfoNCELoss
from scripts import audit_dense_ddp_accumulation as previous
from scripts import audit_dense_trainer_normalization as normalization
from scripts.dense_trainer_normalization_candidate import SingleNormalizationTrainerCandidate


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rng_hashes():
    return {
        "torch_cpu": tensor_hash(torch.get_rng_state()),
        "python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
    }


def assert_exact_tree(left, right, name="state"):
    """Check payload values recursively, not pickle archive metadata or object IDs."""
    if type(left) is not type(right):
        raise AssertionError(f"{name}: different types")
    if isinstance(left, torch.Tensor):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise AssertionError(f"{name}: tensor values, shape or dtype differ")
        if left.is_floating_point() and not torch.isfinite(left).all():
            raise AssertionError(f"{name}: non-finite tensor")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"{name}: different keys")
        for key in left:
            assert_exact_tree(left[key], right[key], f"{name}/{key}")
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError(f"{name}: different lengths")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            assert_exact_tree(a, b, f"{name}/{index}")
    elif left != right:
        raise AssertionError(f"{name}: different scalar values")


def canonical_cpu_load(original_load, checkpoint, records, file, *args, **kwargs):
    """Optional CPU-only transport adapter, not an optimizer/RNG repair."""
    location = kwargs.get("map_location")
    if isinstance(location, torch.device) and location.type == "cpu" and location.index is not None:
        expected = checkpoint.resolve() / "optimizer.pt"
        if Path(file).resolve() != expected:
            raise ValueError(
                "CPU transport override is limited to this synthetic optimizer payload"
            )
        records.append({"path": str(expected), "from": str(location), "to": "cpu"})
        kwargs["map_location"] = "cpu"
    return original_load(file, *args, **kwargs)


def train_segment(config, fixture, output, trainer_kind, dropout, resume=None, canonical_cpu=False):
    model = previous.load_fixture(fixture)
    # An explicitly adversarial fixture, not a change to the formal zero-dropout model.
    dropouts = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    if not dropouts:
        raise ValueError("RNG test requires observable dropout modules")
    for module in dropouts:
        module.p = dropout
    arguments, overrides = previous.cpu_arguments(config, output)
    arguments.max_steps = overrides["max_steps"] = 3
    trainer_class = {
        "live": train.OptimizerTrainer,
        "candidate": SingleNormalizationTrainerCandidate,
    }[trainer_kind]
    trainer = trainer_class(
        model=model,
        args=arguments,
        train_dataset=Dataset.from_list(previous.synthetic_rows(normalization.ROW_COUNT)),
        loss=ExplicitDenseInfoNCELoss(model, temperature=config.resolved_temperature),
        data_collator=previous.RecordingCollator(model),
        optimizer_config=config.optimizer,
        callbacks=[FractionalCheckpointCallback((1 / 3, 2 / 3, 1.0), output)],
    )
    if arguments.world_size != 4 or arguments.gradient_accumulation_steps != 4:
        raise ValueError("Resume audit requires the actual 4 x 8 x 4 batch schedule")
    if arguments.ignore_data_skip or arguments.save_only_model:
        raise ValueError("Require ordinary full-state resumption with data skipping")
    trace, gradients = [], []
    original_step = trainer.training_step

    @functools.wraps(original_step)
    def observe_step(wrapped_model, inputs, num_items_in_batch=None):
        record = {
            "global_step": trainer.state.global_step,
            "row_ids": inputs.pop("_audit_row_ids").tolist(),
            "current_accumulation": trainer.current_gradient_accumulation_steps,
            "rng_before": rng_hashes(),
        }
        result = original_step(wrapped_model, inputs, num_items_in_batch)
        record.update(loss=float(result), rng_after=rng_hashes())
        trace.append(record)
        return result

    trainer.training_step = observe_step
    original_clip = trainer.accelerator.clip_grad_norm_

    def observe_clip(parameters, max_norm, norm_type=2):
        parameters = list(parameters)
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError("Missing or non-finite gradient")
        gradients.append(
            {
                "global_step": trainer.state.global_step,
                "raw": [tensor_hash(p.grad) for p in parameters],
                "learning_rates": [g["lr"] for g in trainer.optimizer.param_groups],
            }
        )
        result = original_clip(parameters, max_norm, norm_type)
        gradients[-1]["clipped"] = [tensor_hash(p.grad) for p in parameters]
        gradients[-1]["raw_norm"] = float(result)
        return result

    trainer.accelerator.clip_grad_norm_ = observe_clip
    transport = []
    if resume and canonical_cpu:
        original_restore = trainer._load_optimizer_and_scheduler
        original_load = torch.load

        def restore(checkpoint):
            with patch.object(
                torch,
                "load",
                functools.partial(canonical_cpu_load, original_load, resume, transport),
            ):
                return original_restore(checkpoint)

        trainer._load_optimizer_and_scheduler = restore
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    if resume and canonical_cpu and len(transport) != 1:
        raise ValueError("Expected exactly one indexed-CPU optimizer transport override")
    if trainer.state.global_step != 3 or trainer.state.epoch != 1.0:
        raise ValueError("Did not reach the exact epoch tail")
    normalization.check_schedule(trainer, 3)
    snapshot = {
        "weights": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
        "scheduler": copy.deepcopy(trainer.lr_scheduler.state_dict()),
        "rng": rng_hashes(),
        "trace": trace,
        "gradients": gradients,
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
    }
    weights = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    rank_weights = [torch.empty_like(weights) for _ in range(4)]
    dist.all_gather(rank_weights, weights)
    if not all(torch.equal(weights, other) for other in rank_weights):
        raise ValueError("Final model differs across ranks")
    dist.barrier()
    # Every rank must be able to read all saved model/optimizer/scheduler/RNG payloads.
    checkpoint = output / "checkpoint-3"
    required = ["model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json"]
    required += [f"rng_state_{r}.pth" for r in range(4)]
    if any(not (checkpoint / name).is_file() for name in required):
        raise ValueError("Missing actual Trainer checkpoint payload")
    del trainer, model
    gc.collect()
    return snapshot, {
        "argument_overrides": overrides,
        "dropout_modules": len(dropouts),
        "cpu_transport_adapter": transport,
    }


def verify_replay(reference, resumed, checkpoint_step):
    expected = copy.deepcopy(reference)
    expected["trace"] = [x for x in reference["trace"] if x["global_step"] >= checkpoint_step]
    expected["gradients"] = [
        x for x in reference["gradients"] if x["global_step"] >= checkpoint_step
    ]
    if not expected["trace"] or not expected["gradients"]:
        raise AssertionError("A replay must include at least one real update")
    assert_exact_tree(expected, resumed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--canonical-cpu-map-location", action="store_true")
    args = parser.parse_args()
    root, work = args.repository.resolve(), args.workdir.resolve()
    required = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WORLD_SIZE": "4",
    }
    if any(os.environ.get(k) != v for k, v in required.items()):
        raise ValueError("Require four offline CPU-only torchrun ranks")
    if (
        work.parent != Path("/tmp")
        or not work.name.startswith("dense-trainer-resume.")
        or not work.is_dir()
    ):
        raise ValueError("Require a fresh /tmp/dense-trainer-resume.XXXXXX mktemp directory")
    if Path(inspect.getsourcefile(train)).resolve() != root / "src/embed_optim/train.py":
        raise ValueError("PYTHONPATH selects a different study checkout")
    torch.set_num_threads(1)
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    report = {
        "scope": "engineering_actual_trainer_four_rank_cpu_save_resume",
        "complete_for_cpu_fixture": False,
        "scientific_completion": False,
        "runtime_deployed": False,
        "canonical_cpu_map_location": args.canonical_cpu_map_location,
        "records": [],
        "boundary": "Tiny two-layer ModernBERT; synthetic 288 rows; FP32/SDPA/Gloo; same process group with fresh Trainers, not independent host/process restart; workers=0; non-reentrant gradient checkpointing. Exact saved model, optimizer, scheduler, row order, per-rank CPU/Python/NumPy RNG, losses and raw/clipped gradients. Actual fractional checkpoint callback at steps 1/2/3; resume from 1 and 2 to the 32-query tail. Does not certify GPU/BF16/FA2, maximum length, primary checkpoints or normalization correctness of the live Trainer.",
    }
    try:
        if rank == 0:
            if any(work.iterdir()):
                raise ValueError("Diagnostic directory must be empty")
            normalization.create_fixture(work / "tiny-model")
        dist.barrier()
        configs = {c.run_id: c for c in load_matrix(root / "configs/dense_no_packing_retrain.yaml")}
        for kind in ("live", "candidate"):
            for dropout in (0.0, 0.1):
                for run_id in previous.RUN_IDS:
                    case = work / f"{kind}-dropout-{dropout}-{run_id}"
                    reference, metadata = train_segment(
                        configs[run_id], work / "tiny-model", case / "uninterrupted", kind, dropout
                    )
                    if len(reference["trace"]) != 9 or len(reference["gradients"]) != 3:
                        raise ValueError("Incomplete baseline microbatch/update schedule")
                    gathered_ids = [None] * 4
                    dist.all_gather_object(
                        gathered_ids, [i for x in reference["trace"] for i in x["row_ids"]]
                    )
                    ids = [i for rank_ids in gathered_ids for i in rank_ids]
                    if len(ids) != 288 or set(ids) != set(range(288)):
                        raise ValueError("Missing/repeated baseline query identities")
                    checks = []
                    for step in (1, 2):
                        checkpoint = case / "uninterrupted" / f"checkpoint-{step}"
                        resumed, resume_metadata = train_segment(
                            configs[run_id],
                            work / "tiny-model",
                            case / f"resume-{step}",
                            kind,
                            dropout,
                            checkpoint,
                            args.canonical_cpu_map_location,
                        )
                        verify_replay(reference, resumed, step)
                        ranks = [None] * 4
                        dist.all_gather_object(
                            ranks,
                            {
                                "rank": rank,
                                "trace": resumed["trace"],
                                "gradients": resumed["gradients"],
                                "rng": resumed["rng"],
                            },
                        )
                        checks.append(
                            {
                                "resume_from_step": step,
                                "exact_replay_on_all_ranks": True,
                                "cpu_transport_adapter": resume_metadata["cpu_transport_adapter"],
                                "ranks": ranks,
                            }
                        )
                    if rank == 0:
                        report["records"].append(
                            {
                                "trainer": kind,
                                "dropout": dropout,
                                "run_id": run_id,
                                **metadata,
                                "checks": checks,
                            }
                        )
                        print(
                            f"Exact CPU Trainer resume verified: {kind}, dropout={dropout}, {run_id}",
                            flush=True,
                        )
                    dist.barrier()
        report["complete_for_cpu_fixture"] = True
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if rank == 0:
            sources = [
                Path(__file__),
                Path(inspect.getsourcefile(previous)),
                Path(inspect.getsourcefile(normalization)),
                Path(inspect.getsourcefile(SingleNormalizationTrainerCandidate)),
            ]
            sources += [
                root / f"src/embed_optim/{name}.py"
                for name in ("train", "callbacks", "losses", "collators", "optimizers", "config")
            ]
            report.update(
                observed_at_utc=datetime.now(UTC).isoformat(),
                source_bindings=[previous.identity(p.resolve()) for p in sources],
            )
            (work / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
