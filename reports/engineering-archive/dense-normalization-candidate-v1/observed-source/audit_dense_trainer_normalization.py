"""Actual four-rank CPU test of a prospective repair, including an epoch tail."""

from __future__ import annotations

import argparse
import functools
import importlib.metadata
import inspect
import json
import math
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainer
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordLevel
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

from embed_optim import train
from embed_optim.collators import DenseGroupCollator
from embed_optim.config import load_matrix
from embed_optim.losses import ExplicitDenseInfoNCELoss
from scripts import audit_dense_ddp_accumulation as previous
from scripts.audit_dense_loss_contract import reference_scores
from scripts.dense_trainer_normalization_candidate import SingleNormalizationTrainerCandidate

ROW_COUNT = 288
ACCUMULATION_COUNTS = (4, 4, 1)
WORLD_SIZE = 4
MICRO_BATCH_SIZE = 8


def validate_ids(rank_batches, seen, step):
    if type(step) is not int or not 0 <= step < len(ACCUMULATION_COUNTS):
        raise ValueError("Unexpected optimizer-step index")
    count = ACCUMULATION_COUNTS[step]
    if len(rank_batches) != WORLD_SIZE or any(len(b) != count for b in rank_batches):
        raise ValueError("Incorrect complete/tail accumulation count")
    if any(len(b) != MICRO_BATCH_SIZE for batches in rank_batches for b in batches):
        raise ValueError("Incorrect microbatch size")
    ids = [i for batches in rank_batches for batch in batches for i in batch]
    if any(type(i) is not int or not 0 <= i < ROW_COUNT for i in ids):
        raise ValueError("Unknown synthetic row identity")
    if len(set(ids)) != WORLD_SIZE * MICRO_BATCH_SIZE * count or set(ids) & seen:
        raise ValueError("Repeated or overlapping synthetic row identities")
    return ids


def create_fixture(directory):
    """Cover every synthetic item, including all 32 tail identities, in vocabulary."""
    directory.mkdir(exist_ok=False)
    words = ["[PAD]", "[UNK]", "query", "document", ":", "question", "answer", "relevant", "other"]
    words += [f"item{i}" for i in range(ROW_COUNT)] + [f"topic{i}" for i in range(17)]
    tokenizer = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", model_max_length=64
    ).save_pretrained(directory)
    config = ModernBertConfig(
        vocab_size=len(words),
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


def check_schedule(trainer, step):
    """Verify actual scheduler progress, not just the declared accumulation value."""
    state = trainer.lr_scheduler.state_dict()
    if state["last_epoch"] != step:
        raise ValueError("Scheduler advanced at the wrong cadence")
    warmup = trainer.args.get_warmup_steps(3)
    factor = step / max(1, warmup) if step < warmup else (3 - step) / max(1, 3 - warmup)
    groups = []
    for group in trainer.optimizer.param_groups:
        if not math.isclose(
            group["lr"], group["initial_lr"] * factor, abs_tol=1e-15, rel_tol=1e-12
        ):
            raise ValueError("Learning rate differs from the declared linear schedule")
        groups.append(
            {"algorithm": group["algorithm"], "lr": group["lr"], "initial_lr": group["initial_lr"]}
        )
    return {"last_epoch": state["last_epoch"], "factor": factor, "groups": groups}


def run_case(config, fixture, output, rows, rank, constant_tail_control=False):
    model = previous.load_fixture(fixture)
    reference = previous.load_fixture(fixture) if rank == 0 else None
    arguments, overrides = previous.cpu_arguments(config, output)
    arguments.max_steps = overrides["max_steps"] = 3
    trainer = SingleNormalizationTrainerCandidate(
        model=model,
        args=arguments,
        train_dataset=Dataset.from_list(rows),
        loss=ExplicitDenseInfoNCELoss(model, temperature=config.resolved_temperature),
        data_collator=previous.RecordingCollator(model),
        optimizer_config=config.optimizer,
    )
    if arguments.world_size != WORLD_SIZE or arguments.gradient_accumulation_steps != 4:
        raise ValueError("Candidate altered the formal batch schedule")
    pending, seen, records, observed_divisors = [], set(), [], []
    original_step = trainer.training_step

    @functools.wraps(original_step)
    def observe_step(wrapped_model, inputs, num_items_in_batch=None):
        pending.append(inputs.pop("_audit_row_ids").tolist())
        if num_items_in_batch is not None:
            raise ValueError("This audit requires the production no-label mean-loss path")
        observed_divisors.append(trainer.current_gradient_accumulation_steps)
        if constant_tail_control:
            # Deliberately wrong, diagnostic-instance-only negative control.
            trainer.current_gradient_accumulation_steps = arguments.gradient_accumulation_steps
        return original_step(wrapped_model, inputs, num_items_in_batch)

    trainer.training_step = observe_step
    original_clip = trainer.accelerator.clip_grad_norm_

    def check_and_clip(parameters, max_norm, norm_type=2):
        parameters = list(parameters)
        actual = [
            (n, p.grad.detach().clone() if p.grad is not None else None)
            for n, p in model.named_parameters()
        ]
        if any(g is None for _, g in actual):
            raise ValueError("Unused parameter in the complete-gradient audit")
        flat = torch.cat([g.reshape(-1) for _, g in actual])
        replicas = [torch.empty_like(flat) for _ in range(WORLD_SIZE)]
        dist.all_gather(replicas, flat)
        rank_batches, rank_divisors = [None] * WORLD_SIZE, [None] * WORLD_SIZE
        dist.all_gather_object(rank_batches, pending)
        dist.all_gather_object(rank_divisors, observed_divisors)
        response, expected_clipped = [None], None
        step = trainer.state.global_step
        if rank == 0:
            try:
                if not all(torch.equal(flat, x) for x in replicas):
                    raise ValueError("Raw gradients differ across ranks")
                ids = validate_ids(rank_batches, seen, step)
                expected_count = ACCUMULATION_COUNTS[step]
                if rank_divisors != [[expected_count] * expected_count] * WORLD_SIZE:
                    raise ValueError("Trainer did not observe the actual accumulation count")
                schedule = check_schedule(trainer, step)
                reference.load_state_dict(model.state_dict())
                reference.train()
                reference.zero_grad(set_to_none=True)
                batch = DenseGroupCollator(reference.preprocess)([rows[i] for i in ids])
                features, labels = SentenceTransformerTrainer.collect_features(None, batch)
                if labels is not None or len(features) != 9:
                    raise ValueError("Wrong own-candidate feature topology")
                embeddings = [reference(f)["sentence_embedding"] for f in features]
                scores, losses = reference_scores(embeddings, config.resolved_temperature)
                if scores.shape != (len(ids), 8):
                    raise ValueError("Reference must contain eight own candidates per query")
                losses.mean().backward()
                expected = [(n, p.grad) for n, p in reference.named_parameters()]
                record = {
                    "global_step_before_update": step,
                    "global_query_count": len(ids),
                    "global_example_ids": ids,
                    "rank_microbatch_ids": rank_batches,
                    "observed_trainer_divisors": rank_divisors,
                    "used_trainer_divisor": trainer.current_gradient_accumulation_steps,
                    "accelerator_divisor": trainer.accelerator.gradient_accumulation_steps,
                    "scale_diagnostic": previous.gradient_scale_diagnostic(actual, expected),
                    "scheduler": schedule,
                    "identical_raw_gradients_across_ranks": True,
                }
                # Preserve a failed numerical comparison rather than losing its diagnostic.
                with (output / f"gradient-before-step-{step}.json").open("x") as handle:
                    handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
                record["raw_gradient_errors"] = previous.compare_gradients(actual, expected)
                reference_norm = torch.nn.utils.clip_grad_norm_(
                    reference.parameters(), max_norm, norm_type
                )
                expected_clipped = [
                    (n, p.grad.detach().clone()) for n, p in reference.named_parameters()
                ]
                record["reference_raw_gradient_norm"] = float(reference_norm)
                response[0] = {"passed": True, "record": record}
            except Exception as exc:
                response[0] = {"passed": False, "type": type(exc).__name__, "message": str(exc)}
        dist.broadcast_object_list(response, src=0)
        if not response[0]["passed"]:
            raise RuntimeError(json.dumps(response[0]))
        actual_norm = original_clip(parameters, max_norm, norm_type)
        clipped_response = [None]
        if rank == 0:
            try:
                record = response[0]["record"]
                record["clipped_gradient_errors"] = previous.compare_gradients(
                    [(n, p.grad) for n, p in model.named_parameters()], expected_clipped
                )
                record["actual_raw_gradient_norm"] = float(actual_norm)
                torch.testing.assert_close(
                    torch.as_tensor(actual_norm),
                    torch.tensor(record["reference_raw_gradient_norm"]),
                    atol=previous.TOLERANCES["gradient_atol"],
                    rtol=previous.TOLERANCES["gradient_rtol"],
                )
                records.append(record)
                seen.update(record["global_example_ids"])
                clipped_response[0] = {"passed": True}
            except Exception as exc:
                clipped_response[0] = {
                    "passed": False,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
        dist.broadcast_object_list(clipped_response, src=0)
        if not clipped_response[0]["passed"]:
            raise RuntimeError(json.dumps(clipped_response[0]))
        pending.clear()
        observed_divisors.clear()
        return actual_norm

    trainer.accelerator.clip_grad_norm_ = check_and_clip
    trainer.train()
    if trainer.state.global_step != 3 or pending or trainer.state.epoch != 1:
        raise ValueError("Expected exactly one epoch and three optimizer steps")
    final_schedule = check_schedule(trainer, 3)
    weights = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    replicas = [torch.empty_like(weights) for _ in range(WORLD_SIZE)]
    dist.all_gather(replicas, weights)
    if not torch.isfinite(weights).all() or not all(torch.equal(weights, x) for x in replicas):
        raise ValueError("Final weights differ across ranks or are non-finite")
    if rank == 0 and (len(records) != 3 or seen != set(range(ROW_COUNT))):
        raise ValueError("Incomplete full/tail coverage or row replay")
    return {
        "run_id": config.run_id,
        "checks": records,
        "optimizer_steps": 3,
        "cpu_fixture_argument_overrides": overrides,
        "optimizer_partition": trainer.optimizer_partition_summary,
        "normalization_policy": trainer.normalization_policy,
        "normalization_source_bindings": trainer.normalization_source_bindings,
        "global_batch_sizes": [128, 128, 32],
        "consumed_unique_rows": len(seen),
        "identical_final_weights_across_ranks": True,
        "final_scheduler": final_schedule,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--negative-control-constant-tail", action="store_true")
    args = parser.parse_args()
    root, work = args.repository.resolve(), args.workdir.resolve()
    required_env = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WORLD_SIZE": "4",
    }
    if any(os.environ.get(k) != v for k, v in required_env.items()):
        raise ValueError("Require four CPU-only torchrun ranks and offline model loading")
    if (
        work.parent != Path("/tmp")
        or not work.name.startswith("dense-normalization-audit.")
        or not work.is_dir()
    ):
        raise ValueError("Require a fresh mktemp /tmp/dense-normalization-audit.XXXXXX directory")
    if Path(inspect.getsourcefile(train)).resolve() != root / "src/embed_optim/train.py":
        raise ValueError("PYTHONPATH does not select the declared audited source")
    torch.set_num_threads(1)
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    report = {
        "scope": "engineering_four_rank_cpu_normalization_candidate_with_epoch_tail",
        "scientific_completion": False,
        "runtime_deployed": False,
        "complete_for_cpu_fixture": False,
        "negative_control_constant_tail": args.negative_control_constant_tail,
        "tolerances": previous.TOLERANCES,
        "records": [],
        "boundary": "Prospective single-normalization subclass of the unchanged study Trainer. Four CPU/Gloo ranks, tiny two-layer random ModernBERT, FP32/SDPA, zero dropout, non-reentrant gradient checkpointing, three optimizer steps on 288 distinct synthetic queries: 128 + 128 + 32. Independent raw/clipped full-batch gradients and actual LR cadence. Not GPU/full-model/8192-token/resume validation or a scientific result; no production code or library modification.",
    }
    try:
        if rank == 0:
            if any(work.iterdir()):
                raise ValueError("Diagnostic work directory is not empty")
            create_fixture(work / "tiny-model")
        dist.barrier()
        configs = {c.run_id: c for c in load_matrix(root / "configs/dense_no_packing_retrain.yaml")}
        for run_id in previous.RUN_IDS:
            result = run_case(
                configs[run_id],
                work / "tiny-model",
                work / run_id,
                previous.synthetic_rows(ROW_COUNT),
                rank,
                args.negative_control_constant_tail,
            )
            if rank == 0:
                report["records"].append(result)
                print(f"Verified complete/tail CPU candidate: {run_id}", flush=True)
            dist.barrier()
        report["complete_for_cpu_fixture"] = True
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if rank == 0:
            sources = [
                Path(__file__).resolve(),
                Path(inspect.getsourcefile(SingleNormalizationTrainerCandidate)),
                Path(inspect.getsourcefile(previous)),
                Path(inspect.getsourcefile(reference_scores)),
                *[
                    root / f"src/embed_optim/{name}.py"
                    for name in ("train", "losses", "collators", "optimizers", "config")
                ],
                root / "configs/dense_no_packing_retrain.yaml",
            ]
            report.update(
                observed_at_utc=datetime.now(UTC).isoformat(),
                source_bindings=[previous.identity(p) for p in sources],
                versions={
                    p: importlib.metadata.version(p)
                    for p in (
                        "torch",
                        "transformers",
                        "accelerate",
                        "sentence-transformers",
                        "datasets",
                    )
                },
            )
            with (work / "audit.json").open("x") as handle:
                handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
