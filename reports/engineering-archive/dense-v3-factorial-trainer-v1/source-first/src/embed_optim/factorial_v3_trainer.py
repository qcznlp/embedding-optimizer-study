"""Actual ST/Accelerate numerical Trainer for the fixed 50K reset continuation.

This internal component executes real dataloaders, training and checkpoint hooks.
It is not a formal entrypoint: source-state/calibration/data/release and whole-run
admission still belong to the separately required outer v3 experiment path.
The distinct local checkpoint receipt explicitly cannot admit scientific results.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from accelerate.data_loader import BatchSamplerShard, DataLoaderShard, SeedableRandomSampler
from accelerate.optimizer import AcceleratedOptimizer
from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

from . import factorial_v3_batches as batches
from . import factorial_v3_checkpoint as checkpoints
from . import factorial_v3_optimizer as optim
from .losses import ExplicitDenseInfoNCELoss

SCOPE = "dense-v3-factorial-trainer-component-v1"


@dataclass
class FactorialTrainingArguments(SentenceTransformerTrainingArguments):
    """Declare variable-sized, non-dropping batches after ST's DDP default rewrite."""

    batch_sampler: Any = field(default=batches.batch_sampler_factory, repr=False)
    factorial_batching_policy: str = batches.SCOPE
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 1.0
    seed: int = 314159
    warmup_steps: int | float = 0.1
    dataloader_drop_last: bool = False
    remove_unused_columns: bool = False
    ddp_find_unused_parameters: bool = False

    def __post_init__(self):
        require_requested_arguments(self)
        if self.data_seed is None:
            self.data_seed = self.seed
        super().__post_init__()
        # Only this named factorial argument class restores the explicitly
        # requested False. The primary class and upstream source are unchanged.
        self.dataloader_drop_last = False
        require_requested_arguments(self)
        config = self.accelerator_config
        if (
            config.split_batches is not False
            or config.dispatch_batches not in (None, False)
            or config.use_seedable_sampler is not True
            or config.use_stateful_dataloader is not False
        ):
            raise ValueError("Unsupported factorial Accelerator dataloader configuration")

    def to_dict(self):
        result = super().to_dict()
        # Upstream deliberately removes callables. Retain the declared policy,
        # while the live function and its source are checked independently.
        result["factorial_batching_policy"] = self.factorial_batching_policy
        return result


def require_requested_arguments(args):
    exact = {
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "dataloader_drop_last": False,
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "ignore_data_skip": False,
        "auto_find_batch_size": False,
        "save_only_model": False,
        "push_to_hub": False,
        "load_best_model_at_end": False,
        "max_steps": -1,
        "factorial_batching_policy": batches.SCOPE,
    }
    for key, expected in exact.items():
        value = getattr(args, key)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"Factorial training argument differs: {key}")
    if (
        args.batch_sampler is not batches.batch_sampler_factory
        or type(args.seed) is not int
        or args.seed not in batches.ORDER_SEEDS
        or (
            args.data_seed is not None
            and (type(args.data_seed) is not int or args.data_seed != args.seed)
        )
        or type(args.num_train_epochs) not in (int, float)
        or args.num_train_epochs != 1
        or type(args.warmup_steps) is not float
        or args.warmup_steps != 0.1
        or args.warmup_ratio is not None
        or args.lr_scheduler_type != "linear"
        or args.lr_scheduler_kwargs
        or args.eval_strategy != "no"
        or args.eval_on_start
        or args.fp16
        or args.deepspeed
        or args.fsdp
        or args.torch_compile
        or args.prompts
        or args.router_mapping
        or args.learning_rate_mapping
    ):
        raise ValueError("Factorial schedule, order, sampler or execution mode differs")


def inspect_loader(loader, args):
    require_requested_arguments(args)
    if args.world_size != 4 or not 0 <= args.process_index < 4:
        raise ValueError("Factorial Trainer requires exactly four initialized ranks")
    if type(loader) is not DataLoaderShard or len(loader) != 1564 or len(loader.dataset) != 50000:
        raise ValueError("Wrong actual factorial dataloader type or cardinality")
    shard = loader.batch_sampler
    if (
        type(shard) is not BatchSamplerShard
        or shard.num_processes != 4
        or shard.process_index != args.process_index
        or shard.split_batches is not False
        or shard.even_batches is not False
        or shard.drop_last is not False
        or shard.batch_size is not None
        or type(shard.batch_sampler) is not batches.FactorialBatchSampler
    ):
        raise ValueError("Actual four-rank factorial shard differs from the declared policy")
    sampler = shard.batch_sampler.sampler
    if type(sampler) is not SeedableRandomSampler or sampler.initial_seed != args.seed:
        raise ValueError("Actual seeded sampler does not retain the branch order seed")
    return {
        "world_size": 4,
        "rank": args.process_index,
        "groups": 50000,
        "local_micro_batches": 1564,
        "optimizer_steps": 391,
        "tail_global_groups": 80,
        "policy": batches.SCOPE,
    }


class FactorialTrainer(SentenceTransformerTrainer):
    """Inherited pinned training_step/compute_loss, with real routed save/resume.

    diagnostic_shapes is restricted to CPU fixtures. Default model topology is
    the complete DenseOn topology and must retain its deterministic backward
    policy. This class alone does not authorize any formal experiment.
    """

    def __init__(self, *, optimizer_config, resume_binding=None, diagnostic_shapes=None, **kwargs):
        from . import dense_numerical_contract as numerical

        args, model, loss, data = (
            kwargs.get(k) for k in ("args", "model", "loss", "train_dataset")
        )
        if type(args) is not FactorialTrainingArguments:
            raise ValueError("Require the explicit factorial TrainingArguments class")
        require_requested_arguments(args)
        if args.world_size != 4:
            raise ValueError("Factorial Trainer requires four initialized ranks")
        if (
            type(loss) is not ExplicitDenseInfoNCELoss
            or loss.model is not model
            or loss.temperature != 0.02
        ):
            raise ValueError("Require the actual shared-model explicit Dense mean InfoNCE loss")
        if type(data) is not Dataset or len(data) != 50000:
            raise ValueError("Require the complete fixed-size factorial Dataset")
        if any(
            kwargs.get(k) is not None
            for k in ("model_init", "eval_dataset", "evaluator", "optimizer_cls_and_kwargs")
        ) or kwargs.get("optimizers", (None, None)) != (None, None):
            raise ValueError(
                "Custom initialization/evaluation/optimizer path is outside this Trainer"
            )
        if diagnostic_shapes is not None and not args.use_cpu:
            raise ValueError("Non-DenseOn topology is restricted to explicit CPU diagnostics")
        self._diagnostic_shapes = copy.deepcopy(diagnostic_shapes)
        self._config = copy.deepcopy(optimizer_config)
        optim.configuration(self._config)
        self._layout = optim.named_layout(
            model, optim.denseon_shapes() if diagnostic_shapes is None else diagnostic_shapes
        )
        if diagnostic_shapes is None:
            numerical.require_backward(model)
            if model.max_seq_length != 8192 or model[0].can_flatten_inputs is not False:
                raise ValueError("DenseOn factorial model execution differs")
        self._resume_binding = copy.deepcopy(resume_binding)
        self._resume_step = None
        self.saved_component_bindings = []
        self._numerical = numerical
        super().__init__(**kwargs)
        if self.compute_loss_func is not None:
            raise ValueError("Custom compute_loss_func is outside the factorial mean-loss policy")
        self._identity_at_creation = self.component_identity()

    def create_accelerator_and_postprocess(self):
        self._numerical.verify_stack(type(self))
        super().create_accelerator_and_postprocess()
        self.normalization_policy = self._numerical.assign_normalization_owner(self)

    def component_identity(self):
        require_requested_arguments(self.args)
        return {
            "scope": SCOPE,
            "scientific_admission": False,
            "cpu_diagnostic": self._diagnostic_shapes is not None,
            "optimizer": optim.configuration(self._config),
            "layout": self._layout,
            "dataset_fingerprint": self.train_dataset._fingerprint,
            "dataset_columns": self.train_dataset.column_names,
            "dataset_rows": len(self.train_dataset),
            "seed": self.args.seed,
            "data_seed": self.args.data_seed,
            "world_size": self.args.world_size,
            "batching": self.args.factorial_batching_policy,
            "normalization": self.normalization_policy,
            "precision": {k: getattr(self.args, k) for k in ("bf16", "fp16", "tf32", "use_cpu")},
            "max_grad_norm": self.args.max_grad_norm,
            "gradient_checkpointing": self.args.gradient_checkpointing,
            "sources": {
                name: checkpoints.file_digest(Path(__file__).parent / name)
                for name in (
                    "factorial_v3_trainer.py",
                    "factorial_v3_batches.py",
                    "factorial_v3_optimizer.py",
                    "factorial_v3_checkpoint.py",
                    "config.py",
                    "optimizers.py",
                    "losses.py",
                    "dense_numerical_contract.py",
                )
            },
        }

    def _require_identity(self):
        current = self.component_identity()
        if current != self._identity_at_creation:
            raise ValueError("Factorial component source/data/execution changed after construction")
        if self.accelerator.gradient_accumulation_steps != 1:
            raise ValueError("Factorial Accelerator normalization owner changed")
        return current

    def get_train_dataloader(self):
        self._require_identity()
        loader = super().get_train_dataloader()
        self.loader_observation = inspect_loader(loader, self.args)
        return loader

    def create_optimizer(self, model=None):
        if self.optimizer is None:
            if model is not None and model is not self.model:
                raise ValueError("Unexpected factorial optimizer model wrapper")
            self.optimizer = optim.FactorialOptimizer(
                self.model, self._config, expected_shapes=self._diagnostic_shapes
            )
        return self.optimizer

    def _raw_optimizer(self):
        value = self.optimizer
        if type(value) is AcceleratedOptimizer:
            value = value.optimizer
        if type(value) is not optim.FactorialOptimizer:
            raise ValueError("Actual Trainer does not own the declared factorial optimizer")
        return value

    def create_scheduler(self, num_training_steps, optimizer=None):
        if (
            type(num_training_steps) is not int
            or num_training_steps != 391
            or optimizer is not None
        ):
            raise ValueError("Actual factorial scheduler must retain the 391-update horizon")
        raw = self._raw_optimizer()
        if self.lr_scheduler is None:
            self.lr_scheduler = optim.create_scheduler(raw)
            self._created_lr_scheduler = True
        optim.require_scheduler_function(self.lr_scheduler, raw, 391)
        return self.lr_scheduler

    def _bound_checkpoint(self, checkpoint):
        if (
            self._resume_binding is None
            or Path(checkpoint).resolve() != Path(self._resume_binding["path"]).resolve()
        ):
            raise ValueError("Resume requires the exact externally supplied component checkpoint")
        payload = checkpoints.read(self._resume_binding, self._require_identity())
        self._resume_step = payload["step"]
        return payload

    def _load_from_checkpoint(self, resume_from_checkpoint):
        self._bound_checkpoint(resume_from_checkpoint)
        return super()._load_from_checkpoint(resume_from_checkpoint)

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None:
            if self._resume_binding is not None:
                raise ValueError("An externally bound resume cannot silently start fresh")
            return
        self._bound_checkpoint(checkpoint)
        saved_optimizer = torch.load(
            Path(checkpoint) / "optimizer.pt", map_location="cpu", weights_only=True
        )
        saved_scheduler = torch.load(
            Path(checkpoint) / "scheduler.pt", map_location="cpu", weights_only=True
        )
        optim.restore_training_state(
            self._raw_optimizer(),
            self.lr_scheduler,
            saved_optimizer,
            saved_scheduler,
            step=self._resume_step,
        )

    def _save_checkpoint(self, model, trial):
        identity = self._require_identity()
        raw, step = self._raw_optimizer(), self.state.global_step
        if raw.completed_steps != step:
            raise ValueError("Actual optimizer updates differ from Trainer step")
        optim.inspect_scheduler(
            self.lr_scheduler.state_dict(), raw.state_dict(), self._config, step=step
        )
        output = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{step}"
        # Every rank checks before any rank is allowed to write.
        if output.exists() or output.is_symlink():
            raise ValueError("Factorial checkpoint may not overwrite existing evidence")
        torch.distributed.barrier()
        super()._save_checkpoint(model, trial)
        torch.distributed.barrier()
        if self.args.should_save:
            self.saved_component_bindings.append(checkpoints.seal(output, identity, step))
        torch.distributed.barrier()
