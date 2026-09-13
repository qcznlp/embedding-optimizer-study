"""Actual four-GPU DenseOn Trainer normalization audit; never a primary run.

Independent reference: same authenticated weights and per-rank microbatches,
ordinary autograd of mean InfoNCE / actual microbatch count, then an explicit
four-rank average. It never calls Trainer.training_step or Accelerator.backward.
The objective arithmetic deliberately matches production; its separate float64
formula oracle is covered by the earlier CPU loss audit, not claimed here.
"""

from __future__ import annotations

import argparse
import functools
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
import torch.nn.functional as F
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from embed_optim import train
from embed_optim.collators import DenseGroupCollator
from embed_optim.config import load_matrix
from embed_optim.corrected_input_execution import require_independently_padded_dense
from embed_optim.losses import ExplicitDenseInfoNCELoss
from embed_optim.optimizers import parameter_partition
from scripts import audit_dense_ddp_accumulation as previous
from scripts import audit_dense_trainer_normalization as normalization
from scripts.audit_checkpoint_download import select_download, verify_download
from scripts.dense_trainer_normalization_candidate import (
    SingleNormalizationTrainerCandidate,
    verify_stack,
)
from scripts.pause_dense_evaluation_for_audit import (
    CHAIN,
    LEDGER,
    LEDGER_SHA,
    digest,
)
from scripts.pause_dense_evaluation_for_audit import (
    inspect as inspect_process,
)


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def batch_hash(batch):
    return {
        k: tensor_hash(v)
        for k, v in sorted(batch.items())
        if isinstance(v, torch.Tensor) and k != "_audit_row_ids"
    }


def require_handoff(path):
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "paused_for_correctness_audit" or not receipt.get(
        "all_preserved_bytes_unchanged"
    ):
        raise ValueError("Require a completed, evidence-preserving project handoff")
    for i, (pid, start, entrypoint) in enumerate(CHAIN):
        item = inspect_process(pid, start, entrypoint, CHAIN[i - 1][0] if i else 1)
        if item["state"] != "T":
            raise ValueError("Project dispatcher is not stopped; refuse GPU overlap")
    if digest(LEDGER) != LEDGER_SHA:
        raise ValueError("Main ledger changed after handoff")
    return {"path": str(path), "sha256": digest(path)}


def gpu_arguments(config, output):
    with patch.object(train, "SentenceTransformerTrainingArguments", side_effect=lambda **kw: kw):
        declared = train._training_arguments(config)
    if (declared["per_device_train_batch_size"], declared["gradient_accumulation_steps"]) != (8, 4):
        raise ValueError("Require unchanged formal four-rank schedule")
    overrides = {
        "output_dir": str(output),
        "max_steps": 3,
        "report_to": [],
        "logging_strategy": "no",
        "disable_tqdm": True,
        "dataloader_num_workers": 0,
        "dataloader_persistent_workers": False,
        "dataloader_prefetch_factor": None,
        "ddp_timeout": 600,
    }
    args = SentenceTransformerTrainingArguments(**{**declared, **overrides})
    if args.device.type != "cuda" or not args.bf16 or not args.tf32 or args.fp16:
        raise ValueError("Require production BF16/TF32 CUDA argument path")
    return args, overrides


def load_model(checkpoint, device):
    model = SentenceTransformer(
        str(checkpoint),
        device=str(device),
        local_files_only=True,
        model_kwargs={"dtype": torch.float32, "attn_implementation": "flash_attention_2"},
    )
    require_independently_padded_dense(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = list(model.parameters())
    if (
        len(parameters) != 134
        or model.get_sentence_embedding_dimension() != 768
        or model.max_seq_length != 8192
        or len(parameter_partition(model)["hidden"]) != 88
        or model[0].auto_model.config._attn_implementation != "flash_attention_2"
        or not model[0].auto_model.is_gradient_checkpointing
        or any(p.device != device or p.dtype != torch.float32 for p in parameters)
        or any(m.p != 0 for m in model.modules() if isinstance(m, torch.nn.Dropout))
    ):
        raise ValueError("Full-model GPU execution contract mismatch")
    return model


def manual_loss(model, features, temperature):
    if len(features) != 9:
        raise ValueError("Require query and eight own documents")
    embeddings = []
    for feature in features:
        # Match Accelerator's model-local autocast + FP32 output conversion, but
        # do not use its backward or wrapping/preparation/accumulation machinery.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embedding = model(feature)["sentence_embedding"]
        embeddings.append(F.normalize(embedding.float(), p=2, dim=-1))
    scores = (
        torch.einsum("bh,bnh->bn", embeddings[0], torch.stack(embeddings[1:], dim=1)) / temperature
    )
    if scores.shape != (8, 8) or scores.dtype != torch.float32:
        raise ValueError("Wrong reference own-candidate score shape/dtype")
    return F.cross_entropy(scores, torch.zeros(8, dtype=torch.long, device=scores.device))


def measure_gradients(actual, expected, multiplier=1.0):
    if not actual or len(actual) != len(expected):
        raise ValueError("Require all matching parameters")
    a2 = b2 = dot = residual2 = 0.0
    records = []
    for (name, p), (other, q) in zip(actual, expected, strict=True):
        if name != other or p.grad is None or q.grad is None:
            raise ValueError("Missing or reordered gradient")
        a, b = p.grad.detach() * multiplier, q.grad.detach()
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Invalid complete-gradient input")
        d = a - b
        aa, bb = a.double(), b.double()
        a2 += float(aa.square().sum())
        b2 += float(bb.square().sum())
        dot += float((aa * bb).sum())
        residual2 += float(d.double().square().sum())
        records.append(
            {
                "name": name,
                "elements": a.numel(),
                "max_absolute_error": float(d.abs().max()),
                "matches_prior_tolerance": torch.allclose(
                    a,
                    b,
                    atol=previous.TOLERANCES["gradient_atol"],
                    rtol=previous.TOLERANCES["gradient_rtol"],
                ),
            }
        )
    if min(a2, b2) <= 0:
        raise ValueError("Degenerate gradient cannot test normalization")
    return {
        "all_parameters_match_prior_tolerance": all(r["matches_prior_tolerance"] for r in records),
        "actual_norm": a2**0.5,
        "reference_norm": b2**0.5,
        "norm_ratio": (a2 / b2) ** 0.5,
        "least_squares_scale": dot / b2,
        "cosine": dot / (a2 * b2) ** 0.5,
        "relative_l2_error": (residual2 / b2) ** 0.5,
        "diagnostic_multiplier": multiplier,
        "parameters": records,
    }


def identical_rank_tensors(named):
    local = [(name, tensor_hash(value)) for name, value in named]
    gathered = [None] * 4
    dist.all_gather_object(gathered, local)
    return all(x == local for x in gathered)


def run_case(config, checkpoint, output, kind, rank, device, profile=False):
    model, reference = load_model(checkpoint, device), load_model(checkpoint, device)
    arguments, overrides = gpu_arguments(config, output)
    cls = {"live": train.OptimizerTrainer, "candidate": SingleNormalizationTrainerCandidate}[kind]
    rows = previous.synthetic_rows(normalization.ROW_COUNT)
    trainer = cls(
        model=model,
        args=arguments,
        train_dataset=Dataset.from_list(rows),
        loss=ExplicitDenseInfoNCELoss(model, temperature=config.resolved_temperature),
        data_collator=previous.RecordingCollator(model),
        optimizer_config=config.optimizer,
    )
    if arguments.world_size != 4 or trainer.accelerator.gradient_accumulation_steps != (
        1 if kind == "candidate" else 4
    ):
        raise ValueError("Unexpected actual accumulation policy")
    pending, seen, records, divisors, dtype_observations = [], set(), [], [], []
    linear = next(m for m in model[0].auto_model.modules() if isinstance(m, torch.nn.Linear))

    def observe_dtype(module, inputs, result):
        if len(dtype_observations) < 4:
            dtype_observations.append(
                {
                    "autocast_cuda": torch.is_autocast_enabled("cuda"),
                    "output_dtype": str(result.dtype),
                    "output_device": str(result.device),
                }
            )

    hook = linear.register_forward_hook(observe_dtype)
    original_step, original_clip = trainer.training_step, trainer.accelerator.clip_grad_norm_
    kernel_names = []

    @functools.wraps(original_step)
    def observe_step(wrapped, inputs, num_items_in_batch=None):
        if num_items_in_batch is not None:
            raise ValueError("Require actual no-label mean-loss path")
        ids = inputs.pop("_audit_row_ids").tolist()
        pending.append({"ids": ids, "tokens": batch_hash(inputs)})
        divisors.append(trainer.current_gradient_accumulation_steps)
        if profile and rank == 0 and trainer.state.global_step == 0 and len(pending) == 1:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                result = original_step(wrapped, inputs, num_items_in_batch)
            kernel_names.extend(
                sorted({x.key for x in prof.key_averages() if "flash" in x.key.lower()})
            )
            return result
        return original_step(wrapped, inputs, num_items_in_batch)

    def check_and_clip(parameters, max_norm, norm_type=2):
        parameters = list(parameters)
        step = trainer.state.global_step
        rank_batches, rank_divisors = [None] * 4, [None] * 4
        dist.all_gather_object(rank_batches, [x["ids"] for x in pending])
        dist.all_gather_object(rank_divisors, divisors)
        ids = normalization.validate_ids(rank_batches, seen, step)
        count = normalization.ACCUMULATION_COUNTS[step]
        if rank_divisors != [[count] * count] * 4:
            raise ValueError("Wrong Trainer full/tail divisor")
        schedule = normalization.check_schedule(trainer, step)
        reference.load_state_dict(model.state_dict())
        reference.train()
        reference.zero_grad(set_to_none=True)
        lengths = []
        for item in pending:
            batch = DenseGroupCollator(reference.preprocess)([rows[i] for i in item["ids"]])
            if batch_hash(batch) != item["tokens"]:
                raise ValueError("Independent reference token inputs differ from actual Trainer")
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            features, labels = SentenceTransformerTrainer.collect_features(None, batch)
            if labels is not None:
                raise ValueError("Unexpected labels")
            lengths.extend(int(x) for f in features for x in f["attention_mask"].sum(-1).tolist())
            (manual_loss(reference, features, config.resolved_temperature) / count).backward()
        for parameter in reference.parameters():
            if parameter.grad is None:
                raise ValueError("Reference misses a parameter gradient")
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(4)
        raw = measure_gradients(list(model.named_parameters()), list(reference.named_parameters()))
        compensated = (
            measure_gradients(
                list(model.named_parameters()), list(reference.named_parameters()), 4.0
            )
            if kind == "live"
            else None
        )
        equal_raw = identical_rank_tensors([(n, p.grad) for n, p in model.named_parameters()])
        if not equal_raw:
            raise ValueError("DDP raw gradients differ across ranks")
        reference_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), max_norm, norm_type)
        actual_norm = original_clip(parameters, max_norm, norm_type)
        clipped = measure_gradients(
            list(model.named_parameters()), list(reference.named_parameters())
        )
        record = {
            "step_before_update": step,
            "global_queries": len(ids),
            "global_ids": ids,
            "rank_microbatch_ids": rank_batches,
            "trainer_divisors": rank_divisors,
            "accelerator_divisor": trainer.accelerator.gradient_accumulation_steps,
            "raw": raw,
            "compensated_raw_diagnostic": compensated,
            "clipped": clipped,
            "actual_clip_norm": float(actual_norm),
            "reference_clip_norm": float(reference_norm),
            "raw_gradients_rank_identical": equal_raw,
            "scheduler": schedule,
            "actual_reference_token_inputs_identical": True,
            "local_token_length_min": min(lengths),
            "local_token_length_max": max(lengths),
        }
        records.append(record)
        seen.update(ids)
        pending.clear()
        divisors.clear()
        if rank == 0:
            with (output / f"gradient-step-{step}.json").open("x") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
            print(
                json.dumps(
                    {
                        "trainer": kind,
                        "run": config.run_id,
                        "step": step,
                        "global_queries": len(ids),
                        "raw_ratio": raw["norm_ratio"],
                        "raw_pass": raw["all_parameters_match_prior_tolerance"],
                        "clipped_pass": clipped["all_parameters_match_prior_tolerance"],
                    }
                ),
                flush=True,
            )
        reference.zero_grad(set_to_none=True)
        return actual_norm

    trainer.training_step = observe_step
    trainer.accelerator.clip_grad_norm_ = check_and_clip
    torch.cuda.reset_peak_memory_stats(device)
    trainer.train()
    if (
        trainer.state.global_step != 3
        or trainer.state.epoch != 1
        or seen != set(range(288))
        or pending
    ):
        raise ValueError("Incorrect full/tail epoch completion")
    if not dtype_observations or any(
        not x["autocast_cuda"] or x["output_dtype"] != "torch.bfloat16" for x in dtype_observations
    ):
        raise ValueError("Actual Trainer forward did not exercise BF16 CUDA")
    if not identical_rank_tensors([(n, p.detach()) for n, p in model.named_parameters()]):
        raise ValueError("Final model weights differ across ranks")
    result = {
        "trainer": kind,
        "run_id": config.run_id,
        "records": records,
        "all_raw_and_clipped_checks_pass": all(
            x["raw"]["all_parameters_match_prior_tolerance"]
            and x["clipped"]["all_parameters_match_prior_tolerance"]
            for x in records
        ),
        "dtype_observations": dtype_observations,
        "flash_profiler_operator_names": kernel_names,
        "final_scheduler": normalization.check_schedule(trainer, 3),
        "final_weights_rank_identical": True,
        "argument_overrides": overrides,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "normalization_policy": getattr(trainer, "normalization_policy", None),
    }
    hook.remove()
    trainer.training_step = original_step
    trainer.accelerator.clip_grad_norm_ = original_clip
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--trainer", choices=("live", "candidate"), required=True)
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
        raise ValueError("Require four owned CUDA ranks and offline/disabled external logging")
    root, work = args.repository.resolve(), args.workdir.absolute()
    if (
        work.parent != Path("/tmp")
        or not work.name.startswith("dense-gpu-normalization.")
        or not work.is_dir()
    ):
        raise ValueError("Require a fresh mktemp /tmp/dense-gpu-normalization.XXXXXX directory")
    if Path(inspect.getsourcefile(train)).resolve() != root / "src/embed_optim/train.py":
        raise ValueError("Wrong audited source checkout")
    handoff = require_handoff(args.handoff)
    verify_stack()
    torch.set_num_threads(2)
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=600), device_id=device)
    report = {
        "scope": "engineering_full_denseon_four_gpu_bf16_fa2_normalization",
        "scientific_completion": False,
        "runtime_deployed": False,
        "audit_execution_complete": False,
        "trainer": args.trainer,
        "handoff": handoff,
        "records": [],
        "tolerances": previous.TOLERANCES,
        "boundary": "Three fresh diagnostic optimizers, one authenticated trained DenseOn state, 288 distinct short synthetic queries each, actual four-GPU Trainer/FA2/BF16, groups 128/128/32, workers=0. Independent manual same-microbatch autograd/all-reduce normalization reference with matched objective arithmetic and model-local autocast. No historical optimizer continuation, maximum-length execution, data-loader worker replay, scientific/retrieval outcome, or production deployment.",
    }
    try:
        if rank == 0 and any(work.iterdir()):
            raise ValueError("Output directory must be empty")
        selection = select_download(args.audit, args.audit_sha256, "padded-normuon-3e-4", 3126)
        before = verify_download(selection, args.download_root.absolute())
        report["source_checkpoint"] = before
        dist.barrier()
        configs = {c.run_id: c for c in load_matrix(root / "configs/dense_no_packing_retrain.yaml")}
        for index, run_id in enumerate(previous.RUN_IDS):
            result = run_case(
                configs[run_id],
                Path(before["checkpoint_root"]),
                work / run_id,
                args.trainer,
                rank,
                device,
                profile=index == 0,
            )
            if rank == 0:
                report["records"].append(result)
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
        if verify_download(selection, args.download_root.absolute()) != before:
            raise ValueError("Authenticated source checkpoint changed")
        require_handoff(args.handoff)
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
            objects = [
                previous,
                normalization,
                SingleNormalizationTrainerCandidate,
                select_download,
                train,
                require_independently_padded_dense,
                SentenceTransformer,
            ]
            paths = [
                Path(__file__).resolve(),
                *[Path(inspect.getsourcefile(o)).resolve() for o in objects],
                *[
                    root / f"src/embed_optim/{x}.py"
                    for x in ("losses", "collators", "optimizers", "config")
                ],
                root / "configs/dense_no_packing_retrain.yaml",
            ]
            report.update(
                observed_at_utc=datetime.now(UTC).isoformat(),
                source_bindings=[previous.identity(p) for p in dict.fromkeys(paths)],
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
                device_name=torch.cuda.get_device_name(device),
                world_size=4,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32,
            )
            with (work / "audit.json").open("x") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")
        dist.destroy_process_group()
    if rank == 0 and not report["normalization_acceptance_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
