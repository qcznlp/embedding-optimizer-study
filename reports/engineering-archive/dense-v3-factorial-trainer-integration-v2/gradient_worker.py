"""Four-rank CPU toy ST/InfoNCE integration. Never a DenseOn/formal run."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Dense, Dropout, LayerNorm, Pooling, WordEmbeddings
from sentence_transformers.models.tokenizer import WhitespaceTokenizer
from transformers import TrainerCallback

from embed_optim.config import OptimizerConfig
from embed_optim.factorial_v3_checkpoint import canonical, file_digest
from embed_optim.factorial_v3_trainer import FactorialTrainer, FactorialTrainingArguments
from embed_optim.losses import ExplicitDenseInfoNCELoss


class Collator:
    valid_label_columns = []

    def __init__(self, model):
        self.model = model

    def __call__(self, rows):
        result = {}
        for column in ("query", "positive", *(f"negative_{i}" for i in range(7))):
            features = self.model.tokenize([row[column] for row in rows])
            result.update({f"{column}_{k}": v for k, v in features.items()})
        result["query_row_id"] = torch.tensor([row["row_id"] for row in rows])
        return result


class SaveAndStop(TrainerCallback):
    def __init__(self, stop):
        self.stop = stop

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in (79, 157, 235, 313, 391):
            control.should_save = True
        if self.stop and state.global_step == self.stop:
            control.should_training_stop = True
        return control


class GlobalMeanGradientReference(TrainerCallback):
    """Independent FP64 sum/global-N oracle, outside Trainer/loss normalization."""

    def __init__(self, encoder, objective):
        self.encoder = encoder
        self.reference = copy.deepcopy(encoder).double()
        self.active = False
        self.features = []
        self.observations = []
        objective.register_forward_pre_hook(self.capture)

    def capture(self, module, arguments):
        if self.active:
            self.features.append([
                {key: value.detach().clone() for key, value in feature.items()}
                for feature in arguments[0]
            ])

    def on_step_begin(self, args, state, control, **kwargs):
        self.active = state.global_step in (0, 390)
        if self.active:
            self.reference.load_state_dict(self.encoder.state_dict())
            self.reference.zero_grad(set_to_none=True)
            self.features = []

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if not self.active:
            return
        if args.max_grad_norm != 0 or len(self.features) != 4:
            raise ValueError('Require four actual unclipped diagnostic micro-batches')
        total = 128 if state.global_step == 0 else 80
        rows = sum(len(features[0]['row_id']) for features in self.features)
        if rows * 4 != total:
            raise ValueError('Actual gradient-probe global query count differs')
        for features in self.features:
            representations = []
            for feature in features:
                vector = self.reference(feature)['sentence_embedding']
                representations.append(vector / torch.linalg.vector_norm(
                    vector, dim=-1, keepdim=True).clamp_min(1e-12))
            scores = torch.stack([
                (representations[0] * document).sum(dim=-1) / 0.02
                for document in representations[1:]
            ], dim=1)
            # No production loss call, cross_entropy, Trainer backward or
            # Accelerate scaling. SUM across ranks yields the global group mean.
            ((torch.logsumexp(scores, dim=1) - scores[:, 0]).sum() / total).backward()
        errors, quarter_rejected, fourfold_rejected = {}, False, False
        reference_parameters = dict(self.reference.named_parameters())
        for name, parameter in self.encoder.named_parameters():
            expected = reference_parameters[name].grad
            if parameter.grad is None or expected is None:
                raise ValueError('Missing actual or independent gradient')
            dist.all_reduce(expected, op=dist.ReduceOp.SUM)
            actual = parameter.grad.detach().double()
            torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-4)
            quarter_rejected |= not torch.allclose(actual / 4, expected, atol=5e-6, rtol=5e-4)
            fourfold_rejected |= not torch.allclose(actual * 4, expected, atol=5e-6, rtol=5e-4)
            errors[name] = float((actual - expected).abs().max())
        if not quarter_rejected or not fourfold_rejected:
            raise ValueError('The independent oracle is insensitive to normalization errors')
        self.observations.append({
            'update': state.global_step + 1, 'global_groups': total,
            'actual_micro_batches_per_rank': len(self.features),
            'absolute_tolerance': 5e-6, 'relative_tolerance': 5e-4,
            'all_parameter_max_absolute_errors': errors,
            'quarter_gradient_rejected': bool(quarter_rejected),
            'fourfold_gradient_rejected': bool(fourfold_rejected),
        })
        self.active = False


def model():
    from collections import OrderedDict

    torch.manual_seed(82719)
    vocabulary = [str(i) for i in range(32)]
    tokenizer = WhitespaceTokenizer(vocab=vocabulary, stop_words=[], do_lower_case=False)
    weights = torch.randn(len(tokenizer.get_vocab()), 3) * 0.3
    return SentenceTransformer(
        modules=OrderedDict(
            embeddings=WordEmbeddings(tokenizer, weights, update_embeddings=True),
            pool=Pooling(3, pooling_mode="mean"),
            layers=Dense(3, 3, bias=False),
            projection=Dense(3, 3, bias=False),
            norm=LayerNorm(3),
            dropout=Dropout(0.0),
        ),
        device="cpu",
    )


def data():
    values = {"row_id": list(range(50000))}
    for column, name in enumerate(("query", "positive", *(f"negative_{i}" for i in range(7)))):
        values[name] = [
            f"{(row * 7 + column * 3) % 32} {(row + column * 5) % 32}" for row in range(50000)
        ]
    return Dataset.from_dict(values)


def run(args):
    if args.mode != 'train' or args.stop or args.resume_binding:
        raise ValueError('The unclipped FP64 oracle requires a fresh full CPU diagnostic')
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("WORLD_SIZE") != "4":
        raise ValueError("This is an explicit four-rank CPU diagnostic only")
    root = args.output.resolve()
    if root.exists():
        raise ValueError("Use a fresh output namespace for every diagnostic attempt")
    training_args = FactorialTrainingArguments(
        output_dir=str(root),
        use_cpu=True,
        ddp_backend="gloo",
        report_to=[],
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        disable_tqdm=True,
        save_strategy="no",
        logging_strategy="no",
        seed=args.seed,
        max_grad_norm=0.0,
        bf16=False,
        fp16=False,
        tf32=False,
    )
    dist.barrier()
    encoder = model()
    shapes = {name: list(p.shape) for name, p in encoder.named_parameters()}
    objective = ExplicitDenseInfoNCELoss(encoder, temperature=0.02)
    consumed = []
    objective.register_forward_pre_hook(
        lambda module, arguments: consumed.append(arguments[0][0]["row_id"].detach().cpu().tolist())
    )
    gradient_reference = GlobalMeanGradientReference(encoder, objective)
    binding = json.loads(args.resume_binding.read_text()) if args.resume_binding else None
    trainer = FactorialTrainer(
        model=encoder,
        args=training_args,
        train_dataset=data(),
        loss=objective,
        data_collator=Collator(encoder),
        callbacks=[SaveAndStop(args.stop), gradient_reference],
        optimizer_config=OptimizerConfig(name=args.optimizer, lr=3e-4),
        diagnostic_shapes=shapes,
        resume_binding=binding,
    )
    if args.mode == "loader":
        loader = trainer.get_train_dataloader()
        loader.set_epoch(0)
        consumed = [batch["query_row_id"].tolist() for batch in loader]
        step = 0
    else:
        trainer.train(resume_from_checkpoint=binding["path"] if binding else None)
        step = trainer.state.global_step
    record = {
        'independent_gradient_reference': gradient_reference.observations,
        'gradient_diagnostic_boundary': 'CPU toy; dropout disabled; clipping disabled; not the stochastic resume fixture',
        "scope": "cpu-toy-factorial-trainer-diagnostic",
        "scientific_admission": False,
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "optimizer": args.optimizer,
        "mode": args.mode,
        "completed_steps": step,
        "seed": args.seed,
        "loader": trainer.loader_observation,
        "normalization": trainer.normalization_policy,
        "shapes": shapes,
        "consumed_groups": sum(map(len, consumed)),
        "consumed_batches": consumed,
        "resume_binding": binding,
        "source": {"worker_sha256": file_digest(__file__)},
        "component_identity": trainer.component_identity(),
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"rank-{dist.get_rank()}.json").open("x") as stream:
        stream.write(canonical(record) + "\n")
    if args.mode != "loader":
        torch.save(
            {
                "model": trainer.model.state_dict(),
                "optimizer": trainer._raw_optimizer().state_dict(),
                "scheduler": trainer.lr_scheduler.state_dict(),
            },
            root / f"rank-{dist.get_rank()}.pt",
        )
    if dist.get_rank() == 0:
        with (root / "saved-component-bindings.json").open("x") as stream:
            stream.write(canonical(trainer.saved_component_bindings) + "\n")
        print(
            json.dumps(
                {
                    k: record[k]
                    for k in (
                        "scope",
                        "mode",
                        "completed_steps",
                        "consumed_groups",
                        "normalization",
                    )
                }
            ),
            flush=True,
        )
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("loader", "train"), required=True)
    parser.add_argument("--optimizer", choices=("hybrid_adamw", "muon"), default="hybrid_adamw")
    parser.add_argument("--seed", type=int, choices=(314159, 271828, 161803), default=314159)
    parser.add_argument("--stop", type=int, choices=(0, 79, 157, 235, 313), default=0)
    parser.add_argument("--resume-binding", type=Path)
    args = parser.parse_args()
    try:
        run(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
