"""Four-rank CPU/Gloo Trainer audit on synthetic data, never a primary training run."""

from __future__ import annotations

import argparse
import functools
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
from accelerate import Accelerator
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    models,
)
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordLevel
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast, Trainer

from embed_optim import train
from embed_optim.collators import DenseGroupCollator
from embed_optim.config import load_matrix
from embed_optim.losses import ExplicitDenseInfoNCELoss
from scripts.audit_dense_loss_contract import reference_scores

# Fixed before execution. Check raw gradients before clipping so a factor-of-four
# accumulation error cannot be hidden by clipping both vectors to the same norm.
TOLERANCES = {"gradient_atol": 5e-6, "gradient_rtol": 5e-4}
RUN_IDS = ("padded-adamw-3e-5", "padded-muon-3e-4", "padded-normuon-3e-4")


def synthetic_rows(count=256):
    return [
        {
            "example_id": i,
            "query": f"question item{i} topic{i % 17}",
            "positive": f"answer item{i} topic{i % 17} relevant",
            **{
                f"negative_{j}": f"answer item{(i + j + 1) % count} topic{(i + j + 3) % 17} other"
                for j in range(7)
            },
            "length": 40 + i % 13,
        }
        for i in range(count)
    ]


def create_model_fixture(directory: Path):
    directory.mkdir(exist_ok=False)
    vocabulary = ["[PAD]", "[UNK]", "query", "document", ":", "question", "answer"]
    vocabulary += ["relevant", "other", *[f"item{i}" for i in range(256)]]
    vocabulary += [f"topic{i}" for i in range(17)]
    tokenizer = Tokenizer(WordLevel({w: i for i, w in enumerate(vocabulary)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", model_max_length=64
    ).save_pretrained(directory)
    config = ModernBertConfig(
        vocab_size=len(vocabulary),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        local_attention=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
        cls_token_id=1,
        sep_token_id=1,
        attention_dropout=0.0,
        embedding_dropout=0.0,
        mlp_dropout=0.0,
    )
    torch.manual_seed(20260905)
    ModernBertModel(config).save_pretrained(directory)


def load_fixture(directory: Path):
    encoder = models.Transformer(
        str(directory),
        max_seq_length=64,
        model_args={
            "dtype": torch.float32,
            "attn_implementation": "sdpa",
            "local_files_only": True,
        },
        tokenizer_args={"local_files_only": True},
    )
    encoder.can_flatten_inputs = False
    model = SentenceTransformer(
        modules=[encoder, models.Pooling(32, pooling_mode="mean")], device="cpu"
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Only CPU FP32 fixture parameters are allowed")
    if any(m.p != 0 for m in model.modules() if isinstance(m, torch.nn.Dropout)):
        raise ValueError("The deterministic diagnostic requires zero dropout")
    return model


def compare_gradients(actual, expected):
    if len(actual) != len(expected) or not actual:
        raise ValueError("Require the same nonempty parameter list")
    errors = []
    for (name, left), (other_name, right) in zip(actual, expected, strict=True):
        if name != other_name or left is None or right is None:
            raise ValueError("Missing, reordered or unused parameter gradient")
        if (
            left.shape != right.shape
            or not torch.isfinite(left).all()
            or not torch.isfinite(right).all()
        ):
            raise ValueError(f"Invalid gradient: {name}")
        torch.testing.assert_close(
            left,
            right,
            atol=TOLERANCES["gradient_atol"],
            rtol=TOLERANCES["gradient_rtol"],
            msg=lambda m: f"{name}: {m}",
        )
        errors.append(
            {
                "name": name,
                "elements": left.numel(),
                "max_absolute_error": float((left - right).abs().max()),
            }
        )
    return errors


def gradient_scale_diagnostic(actual, expected):
    left = torch.cat([g.detach().double().reshape(-1) for _, g in actual])
    right = torch.cat([g.detach().double().reshape(-1) for _, g in expected])
    scale = torch.dot(left, right) / torch.dot(right, right)
    return {
        "actual_norm": float(left.norm()),
        "reference_norm": float(right.norm()),
        "actual_to_reference_norm_ratio": float(left.norm() / right.norm()),
        "least_squares_scale": float(scale),
        "cosine": float(torch.nn.functional.cosine_similarity(left, right, dim=0)),
        "max_residual_after_fitted_scale": float((left - scale * right).abs().max()),
    }


def validate_global_ids(rank_batches, previously_seen):
    if len(rank_batches) != 4 or any(len(batches) != 4 for batches in rank_batches):
        raise ValueError("Expected four actual microbatches on each of four ranks")
    if any(len(batch) != 8 for batches in rank_batches for batch in batches):
        raise ValueError("Expected eight queries per actual microbatch")
    ids = [i for batches in rank_batches for batch in batches for i in batch]
    if len(set(ids)) != 128 or set(ids) & previously_seen:
        raise ValueError("Repeated or overlapping global query identities")
    if any(type(i) is not int or not 0 <= i < 256 for i in ids):
        raise ValueError("Unknown synthetic query identity")
    return ids


def cpu_arguments(config, output):
    # Capture the unchanged production declaration, then explicitly mark every
    # CPU/short-fixture override. No production argument object is mutated.
    with patch.object(train, "SentenceTransformerTrainingArguments", side_effect=lambda **kw: kw):
        declared = train._training_arguments(config)
    if (declared["per_device_train_batch_size"], declared["gradient_accumulation_steps"]) != (8, 4):
        raise ValueError("The formal four-rank schedule changed")
    overrides = {
        "output_dir": str(output),
        "use_cpu": True,
        "bf16": False,
        "tf32": False,
        "ddp_backend": "gloo",
        "ddp_timeout": 90,
        "max_steps": 2,
        "report_to": [],
        "logging_strategy": "no",
        "disable_tqdm": True,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "dataloader_persistent_workers": False,
        "dataloader_prefetch_factor": None,
    }
    return SentenceTransformerTrainingArguments(**{**declared, **overrides}), overrides


class RecordingCollator:
    def __init__(self, model):
        self.inner = DenseGroupCollator(model.preprocess)
        self.valid_label_columns = self.inner.valid_label_columns

    def __call__(self, rows):
        batch = self.inner(rows)
        batch["_audit_row_ids"] = torch.tensor([r["example_id"] for r in rows])
        return batch


def run_case(config, fixture, output, rows, rank, diagnostic_trainer_backward_only=False):
    model = load_fixture(fixture)
    reference = load_fixture(fixture) if rank == 0 else None
    arguments, overrides = cpu_arguments(config, output)
    trainer = train.OptimizerTrainer(
        model=model,
        args=arguments,
        train_dataset=Dataset.from_list(rows),
        loss=ExplicitDenseInfoNCELoss(model, temperature=config.resolved_temperature),
        data_collator=RecordingCollator(model),
        optimizer_config=config.optimizer,
    )
    backward_divisor = trainer.accelerator.gradient_accumulation_steps
    if diagnostic_trainer_backward_only:
        original_backward = trainer.accelerator.backward

        def trainer_backward_only(loss, **kwargs):
            # DIAGNOSTIC INTERVENTION ONLY. Compensate this owned accelerator's
            # second division without modifying installed or training source.
            return original_backward(loss * backward_divisor, **kwargs)

        trainer.accelerator.backward = trainer_backward_only
    pending, seen, records = [], set(), []
    original_step = trainer.training_step

    @functools.wraps(original_step)
    def observe_step(wrapped_model, inputs, num_items_in_batch=None):
        pending.append(inputs.pop("_audit_row_ids").tolist())
        return original_step(wrapped_model, inputs, num_items_in_batch)

    trainer.training_step = observe_step
    original_clip = trainer.accelerator.clip_grad_norm_

    def check_and_clip(parameters, max_norm, norm_type=2):
        parameters = list(parameters)
        actual = [
            (name, p.grad.detach().clone() if p.grad is not None else None)
            for name, p in model.named_parameters()
        ]
        if any(g is None for _, g in actual):
            raise ValueError("The tiny encoder has an unused trainable parameter")
        flat = torch.cat([g.reshape(-1) for _, g in actual])
        replicas = [torch.empty_like(flat) for _ in range(4)]
        dist.all_gather(replicas, flat)
        ids_by_rank = [None] * 4
        dist.all_gather_object(ids_by_rank, pending)
        response = [None]
        expected_clipped = None
        if rank == 0:
            try:
                if not all(torch.equal(flat, other) for other in replicas):
                    raise ValueError("Raw gradients differ across DDP ranks")
                ids = validate_global_ids(ids_by_rank, seen)
                seen.update(ids)
                reference.load_state_dict(model.state_dict())
                reference.train()
                reference.zero_grad(set_to_none=True)
                batch = DenseGroupCollator(reference.preprocess)([rows[i] for i in ids])
                features, labels = SentenceTransformerTrainer.collect_features(None, batch)
                if labels is not None or len(features) != 9:
                    raise ValueError("Unexpected explicit candidate columns")
                embeddings = [reference(f)["sentence_embedding"] for f in features]
                scores, losses = reference_scores(embeddings, config.resolved_temperature)
                if scores.shape != (128, 8):
                    raise ValueError("Reference denominator is not eight own documents")
                losses.mean().backward()
                expected = [(n, p.grad) for n, p in reference.named_parameters()]
                diagnostic = {
                    "scope": "diagnostic_gradient_scale_observation",
                    "scientific_completion": False,
                    "run_id": config.run_id,
                    "global_step_before_update": trainer.state.global_step,
                    "cpu_only_backward_compensation": diagnostic_trainer_backward_only,
                    "trainer_accumulation_divisor": trainer.current_gradient_accumulation_steps,
                    "accelerator_backward_divisor": backward_divisor,
                    "model_accepts_loss_kwargs": trainer.model_accepts_loss_kwargs,
                    "scale_diagnostic": gradient_scale_diagnostic(actual, expected),
                    "global_example_ids": ids,
                    "rank_microbatch_ids": ids_by_rank,
                }
                with (output / f"gradient-before-step-{trainer.state.global_step}.json").open(
                    "x"
                ) as handle:
                    handle.write(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n")
                raw_errors = compare_gradients(actual, expected)
                reference_norm = torch.nn.utils.clip_grad_norm_(
                    reference.parameters(), max_norm, norm_type
                )
                expected_clipped = [
                    (n, p.grad.detach().clone()) for n, p in reference.named_parameters()
                ]
                response[0] = {
                    "passed": True,
                    "global_step_before_update": trainer.state.global_step,
                    "global_example_ids": ids,
                    "rank_microbatch_ids": ids_by_rank,
                    "raw_gradient_errors": raw_errors,
                    "reference_raw_gradient_norm": float(reference_norm),
                    "identical_raw_gradients_across_four_ranks": True,
                    "candidates_per_query": 8,
                }
            except Exception as exc:
                response[0] = {"passed": False, "error_type": type(exc).__name__, "error": str(exc)}
        dist.broadcast_object_list(response, src=0)
        if not response[0]["passed"]:
            raise RuntimeError(json.dumps(response[0]))
        actual_norm = original_clip(parameters, max_norm, norm_type)
        clipped_response = [None]
        if rank == 0:
            try:
                clipped = [(n, p.grad) for n, p in model.named_parameters()]
                response[0]["clipped_gradient_errors"] = compare_gradients(
                    clipped, expected_clipped
                )
                response[0]["actual_raw_gradient_norm"] = float(actual_norm)
                torch.testing.assert_close(
                    torch.as_tensor(actual_norm),
                    torch.tensor(response[0]["reference_raw_gradient_norm"]),
                    atol=5e-6,
                    rtol=5e-4,
                )
                records.append(response[0])
                clipped_response[0] = {"passed": True}
            except Exception as exc:
                clipped_response[0] = {
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        dist.broadcast_object_list(clipped_response, src=0)
        if not clipped_response[0]["passed"]:
            raise RuntimeError(json.dumps(clipped_response[0]))
        pending.clear()
        return actual_norm

    trainer.accelerator.clip_grad_norm_ = check_and_clip
    trainer.train()
    if trainer.state.global_step != 2 or pending:
        raise ValueError("Unexpected diagnostic optimizer-step count or unfinished accumulation")
    weights = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    replicas = [torch.empty_like(weights) for _ in range(4)]
    dist.all_gather(replicas, weights)
    if not torch.isfinite(weights).all() or not all(torch.equal(weights, p) for p in replicas):
        raise ValueError("Final fixture weights are non-finite or differ across ranks")
    if rank == 0 and (len(records) != 2 or seen != set(range(256))):
        raise ValueError("Not all 256 distinct synthetic rows were checked")
    return {
        "run_id": config.run_id,
        "optimizer": config.optimizer.as_dict()
        if hasattr(config.optimizer, "as_dict")
        else config.as_dict()["optimizer"],
        "cpu_fixture_argument_overrides": overrides,
        "optimizer_partition": trainer.optimizer_partition_summary,
        "world_size": arguments.world_size,
        "gradient_accumulation_steps": arguments.gradient_accumulation_steps,
        "accelerator_backward_divisor": trainer.accelerator.gradient_accumulation_steps,
        "cpu_only_backward_compensation": diagnostic_trainer_backward_only,
        "model_accepts_loss_kwargs": trainer.model_accepts_loss_kwargs,
        "gradient_checkpointing": model[0].auto_model.is_gradient_checkpointing,
        "identical_final_weights_across_ranks": True,
        "optimizer_steps": 2,
        "checks": records,
    }


def identity(path):
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-trainer-backward-only",
        action="store_true",
        help="CPU diagnostic intervention only: compensate Accelerator's second division.",
    )
    args = parser.parse_args()
    root, work = args.repository.resolve(), args.workdir.resolve()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
        or os.environ.get("WORLD_SIZE") != "4"
    ):
        raise ValueError("Require four torchrun ranks, hidden GPUs and offline model loading")
    if (
        work.parent != Path("/tmp")
        or not work.name.startswith("dense-ddp-audit.")
        or not work.is_dir()
    ):
        raise ValueError("Use a fresh mktemp -d /tmp/dense-ddp-audit.XXXXXX work directory")
    if Path(inspect.getsourcefile(train)).resolve() != root / "src/embed_optim/train.py":
        raise ValueError("Set PYTHONPATH to the audited checkout's absolute src")
    torch.set_num_threads(1)
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    report = {
        "scope": "engineering_four_rank_cpu_dense_trainer_accumulation",
        "scientific_completion": False,
        "complete_for_cpu_fixture": False,
        "runtime_deployed": False,
        "cpu_only_backward_compensation": args.diagnostic_trainer_backward_only,
        "tolerances": TOLERANCES,
        "boundary": "Actual four-rank CPU/Gloo OptimizerTrainer, production collator/loss/optimizer builder, microbatch 8 and four accumulation steps. Tiny randomly initialized two-layer ModernBERT, short synthetic texts, FP32/SDPA and zero dropout with non-reentrant gradient checkpointing. Checks raw and clipped gradients against an independent float64 row-wise 128-query/eight-candidate reference and rank consistency over two steps per optimizer. Does not certify full DenseOn, GPU/NCCL, BF16/FlashAttention, 8192-token execution, rank-RNG resume, the 500K data loader or any optimizer-quality/scientific result.",
        "records": [],
    }
    try:
        if rank == 0:
            if any(work.iterdir()):
                raise ValueError("Diagnostic work directory is not empty")
            create_model_fixture(work / "tiny-model")
        dist.barrier()
        configs = {c.run_id: c for c in load_matrix(root / "configs/dense_no_packing_retrain.yaml")}
        for run_id in RUN_IDS:
            record = run_case(
                configs[run_id],
                work / "tiny-model",
                work / run_id,
                synthetic_rows(),
                rank,
                args.diagnostic_trainer_backward_only,
            )
            if rank == 0:
                report["records"].append(record)
                print(f"Verified CPU DDP accumulation: {run_id}", flush=True)
            dist.barrier()
        report["complete_for_cpu_fixture"] = True
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if rank == 0:
            sources = [
                Path(__file__).resolve(),
                root / "src/embed_optim/train.py",
                root / "src/embed_optim/losses.py",
                root / "src/embed_optim/collators.py",
                root / "src/embed_optim/optimizers.py",
                root / "src/embed_optim/config.py",
                root / "configs/dense_no_packing_retrain.yaml",
                Path(inspect.getsourcefile(reference_scores)),
                *[
                    Path(inspect.getsourcefile(obj))
                    for obj in (Trainer, SentenceTransformerTrainer, Accelerator, ModernBertModel)
                ],
            ]
            report.update(
                observed_at_utc=datetime.now(UTC).isoformat(),
                source_bindings=[identity(p) for p in sources],
                versions={
                    p: importlib.metadata.version(p)
                    for p in (
                        "torch",
                        "transformers",
                        "sentence-transformers",
                        "accelerate",
                        "datasets",
                    )
                },
            )
            with (work / "audit.json").open("x") as handle:
                handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
