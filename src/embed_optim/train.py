from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from transformers import set_seed

from .callbacks import (
    FractionalCheckpointCallback,
    PyLateCheckpointCompatibilityCallback,
    WandbExperimentConfigCallback,
    sanitize_pylate_checkpoint,
)
from .collators import TEXT_COLUMNS, DenseGroupCollator, LateGroupCollator
from .config import RunConfig, load_matrix, save_resolved_config
from .losses import ExplicitDenseInfoNCELoss, ExplicitLateInfoNCELoss
from .optimizers import build_optimizer
from .pylate_compat import configure_pylate_compatibility


class OptimizerTrainer(SentenceTransformerTrainer):
    def __init__(self, *args, optimizer_config, **kwargs) -> None:
        self.embedding_optimizer_config = optimizer_config
        self.optimizer_partition_summary = None
        super().__init__(*args, **kwargs)

    def create_optimizer(self, model=None):
        if self.optimizer is None:
            target = self.model if model is None else model
            self.optimizer, self.optimizer_partition_summary = build_optimizer(
                target, self.embedding_optimizer_config
            )
            if self.is_world_process_zero():
                print(
                    "Optimizer parameter partition: "
                    + json.dumps(self.optimizer_partition_summary, sort_keys=True),
                    flush=True,
                )
        return self.optimizer


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _model_kwargs(config: RunConfig) -> dict:
    return {
        "dtype": torch.float32,
        "attn_implementation": "flash_attention_2" if config.flash_attention else "sdpa",
    }


def _load_model_and_loss(config: RunConfig):
    if config.model_family == "dense":
        model = SentenceTransformer(
            config.model_name,
            revision=config.model_revision,
            model_kwargs=_model_kwargs(config),
        )
        model.max_seq_length = config.max_length
        loss = ExplicitDenseInfoNCELoss(model, temperature=config.resolved_temperature)
        collator = DenseGroupCollator(model.preprocess)
    elif config.model_family == "late":
        models = configure_pylate_compatibility()
        model = models.ColBERT(
            config.model_name,
            revision=config.model_revision,
            query_length=config.max_length,
            document_length=config.max_length,
            do_query_expansion=False,
            trust_remote_code=True,
            model_kwargs=_model_kwargs(config),
        )
        # PyLate 1.6's save method still reads the pre-ST-5 private field. The
        # public replacement contains the same base metadata; PyLate appends its
        # query/document settings during save.
        if not hasattr(model, "_model_config"):
            model._model_config = model._get_model_config()
        loss = ExplicitLateInfoNCELoss(model, temperature=config.resolved_temperature)
        collator = LateGroupCollator(model)
    else:
        raise ValueError(f"Unsupported model family {config.model_family!r}")
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, loss, collator


def _training_arguments(config: RunConfig) -> SentenceTransformerTrainingArguments:
    world_size = _world_size()
    micro_global = config.micro_batch_size * world_size
    if config.global_batch_size % micro_global:
        raise ValueError(
            f"global_batch_size={config.global_batch_size} must be divisible by "
            f"micro_batch_size*world_size={micro_global}"
        )
    accumulation = config.global_batch_size // micro_global
    return SentenceTransformerTrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        max_steps=int(os.environ.get("EMBED_OPTIM_MAX_STEPS", "-1")),
        per_device_train_batch_size=config.micro_batch_size,
        gradient_accumulation_steps=accumulation,
        learning_rate=config.optimizer.lr,
        lr_scheduler_type="linear",
        warmup_steps=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        report_to=["wandb"],
        run_name=f"{config.model_family}-{config.run_id}",
        project=config.wandb_project,
        bf16=True,
        tf32=True,
        fp16=False,
        seed=config.seed,
        data_seed=config.seed,
        full_determinism=False,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=config.dataloader_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=config.dataloader_workers > 0,
        dataloader_prefetch_factor=4 if config.dataloader_workers > 0 else None,
        train_sampling_strategy="group_by_length",
        length_column_name="length",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )


def run_training(config: RunConfig, resume_from_checkpoint: str | None = None) -> Path:
    set_seed(config.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["WANDB_PROJECT"] = config.wandb_project
    if config.wandb_entity:
        os.environ["WANDB_ENTITY"] = config.wandb_entity
    os.environ.setdefault(
        "WANDB_RUN_ID", f"{config.model_family}-{config.run_id}-seed{config.seed}"
    )
    os.environ.setdefault("WANDB_RESUME", "allow")
    os.environ.setdefault("WANDB_RUN_GROUP", config.model_family)
    os.environ.setdefault(
        "WANDB_TAGS",
        f"{config.model_family},{config.optimizer.name},seed-{config.seed}",
    )
    output_dir = config.output_dir.resolve()
    if int(os.environ.get("RANK", "0")) == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_resolved_config(config, output_dir / "run_config.json")

    dataset_root = Path(config.dataset_path)
    dataset_path = dataset_root
    if (dataset_path / "dataset").is_dir():
        dataset_path = dataset_path / "dataset"
    dataset = Dataset.load_from_disk(str(dataset_path))
    required = [*TEXT_COLUMNS, "length"]
    missing = [column for column in required if column not in dataset.column_names]
    if missing:
        raise ValueError(f"Dataset {dataset_path} is missing columns: {missing}")
    dataset = dataset.select_columns(required)
    if int(os.environ.get("RANK", "0")) == 0 and (dataset_root / "manifest.json").is_file():
        shutil.copy2(dataset_root / "manifest.json", output_dir / "dataset_manifest.json")

    model, loss, collator = _load_model_and_loss(config)
    callback = FractionalCheckpointCallback(config.checkpoint_fractions, output_dir)
    callbacks = [callback, WandbExperimentConfigCallback(config.as_dict())]
    if config.model_family == "late":
        callbacks.append(PyLateCheckpointCompatibilityCallback())
    trainer = OptimizerTrainer(
        model=model,
        args=_training_arguments(config),
        train_dataset=dataset,
        loss=loss,
        data_collator=collator,
        callbacks=callbacks,
        optimizer_config=config.optimizer,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    if trainer.is_world_process_zero():
        if config.model_family == "late":
            sanitize_pylate_checkpoint(final_dir)
        trainer.state.save_to_json(str(output_dir / "trainer_state_final.json"))
        (output_dir / "completed.json").write_text(
            json.dumps(
                {
                    "run_id": config.run_id,
                    "model_family": config.model_family,
                    "global_step": trainer.state.global_step,
                    "checkpoints": sorted(callback.requested),
                    "optimizer_partition": trainer.optimizer_partition_summary,
                    "dataset_rows": len(dataset),
                    "dataset_fingerprint": dataset._fingerprint,
                    "versions": {
                        package: importlib.metadata.version(package)
                        for package in (
                            "torch",
                            "transformers",
                            "sentence-transformers",
                            "pylate",
                            "late-interaction-kernels",
                        )
                    },
                },
                indent=2,
            )
            + "\n"
        )
    return final_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one optimizer/model configuration")
    parser.add_argument("--matrix", default="configs/experiment.yaml")
    parser.add_argument("--model-family", choices=["dense", "late"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-steps", type=int, help="Smoke-test override")
    parser.add_argument("--global-batch-size", type=int, help="Smoke-test override")
    parser.add_argument("--micro-batch-size", type=int, help="Smoke-test override")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    matches = [
        config
        for config in load_matrix(args.matrix)
        if config.model_family == args.model_family and config.run_id == args.run_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one matching run, found {len(matches)}")
    config = matches[0]
    overrides = {}
    if args.global_batch_size:
        overrides["global_batch_size"] = args.global_batch_size
    if args.micro_batch_size:
        overrides["micro_batch_size"] = args.micro_batch_size
    if args.max_steps:
        # TrainingArguments is the only max-step owner; a tiny temporary dataset keeps
        # smoke tests on the exact same code path.
        overrides["epochs"] = 1.0
    if overrides:
        import dataclasses

        config = dataclasses.replace(config, **overrides)
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    if args.max_steps:
        os.environ["EMBED_OPTIM_MAX_STEPS"] = str(args.max_steps)
    run_training(config, resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
