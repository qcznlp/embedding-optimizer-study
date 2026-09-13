"""Explicit, pinned-stack contract for newly corrected Dense training.

Legacy configuration reading is retained separately; a legacy checkpoint is not
a valid continuation input for this numerical policy. No communication hook or
global library implementation is installed by this module.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

from accelerate import Accelerator, DistributedType
from sentence_transformers import SentenceTransformerTrainer
from transformers import Trainer

from .config import DENSE_NUMERICAL_POLICY, MUON_NS_IMPLEMENTATION, NORMUON_NS_IMPLEMENTATION

VERSIONS = {"transformers": "5.3.0", "accelerate": "1.13.0", "sentence-transformers": "5.7.0"}
SOURCE_HASHES = {
    "trainer": "060eda6fcd587e79caeca7c3f246ea7bab1af410479c9776584470b8f4ac48a8",
    "accelerator": "85fe1ad1f0061598313f8566868e9f53434ac9c1d5e24089c3d03da772d40f23",
    "sentence_loss": "3ffc94ef7c3acea812ecbbed5b2ea968abf5e8bf48c881929e22e3e36b3bdd59",
}
RECEIPT_NAME = "dense_numerical_contract.json"


def verify_stack(trainer_type):
    for package, expected in VERSIONS.items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"Unvalidated numerical stack: {package}")
    objects = {
        "trainer": Trainer,
        "accelerator": Accelerator,
        "sentence_loss": SentenceTransformerTrainer.compute_loss,
    }
    for name, obj in objects.items():
        path = Path(inspect.getsourcefile(obj))
        if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_HASHES[name]:
            raise ValueError(f"Unvalidated numerical source: {name}")
    if (
        trainer_type.training_step is not Trainer.training_step
        or trainer_type.compute_loss is not SentenceTransformerTrainer.compute_loss
    ):
        raise ValueError("Require the audited inherited training-step and mean-loss dispatch")


def assign_normalization_owner(trainer):
    accelerator = trainer.accelerator
    if (
        trainer.is_deepspeed_enabled
        or trainer.is_fsdp_enabled
        or accelerator.distributed_type
        not in (DistributedType.NO, DistributedType.MULTI_CPU, DistributedType.MULTI_GPU)
    ):
        raise ValueError("Corrected normalization is limited to ordinary single-device/DDP")
    scheduled = trainer.args.gradient_accumulation_steps
    if type(scheduled) is not int or scheduled < 1:
        raise ValueError("Require a positive integral Trainer accumulation schedule")
    if accelerator.gradient_accumulation_steps != scheduled:
        raise ValueError("Unexpected initial Accelerator divisor")
    accelerator.gradient_accumulation_steps = 1
    if accelerator.gradient_accumulation_steps != 1:
        raise ValueError("Accelerator did not retain its single-normalization divisor")
    return {
        "owner": "Trainer.training_step/current_gradient_accumulation_steps",
        "trainer_scheduled_accumulation": scheduled,
        "accelerator_divisor_before": scheduled,
        "accelerator_divisor_after": 1,
    }


def configure_backward(model):
    config = model[0].auto_model.config
    attention = [
        m for m in model[0].auto_model.modules() if type(m).__name__ == "ModernBertAttention"
    ]
    if len(attention) != 22 or type(config.deterministic_flash_attn) is not bool:
        raise ValueError("Corrected policy requires the audited DenseOn attention topology")
    if any(type(m.deterministic_flash_attn) is not bool for m in attention):
        raise ValueError("Attention backward policy is not observable")
    config.deterministic_flash_attn = True
    for module in attention:
        module.deterministic_flash_attn = True
    require_backward(model)


def require_backward(model):
    config = model[0].auto_model.config
    attention = [
        m for m in model[0].auto_model.modules() if type(m).__name__ == "ModernBertAttention"
    ]
    if len(attention) != 22 or config.deterministic_flash_attn is not True:
        raise ValueError("Corrected policy requires deterministic attention backward")
    if any(m.deterministic_flash_attn is not True for m in attention):
        raise ValueError("An attention instance does not retain deterministic backward")


def require_optimizer_choice(config):
    expected = {
        "adamw": None,
        "muon": MUON_NS_IMPLEMENTATION,
        "normuon": NORMUON_NS_IMPLEMENTATION,
    }
    if config.name not in expected or config.resolved_ns_implementation != expected[config.name]:
        raise ValueError("Optimizer implementation does not satisfy the corrected primary policy")


def receipt(optimizer_config, accumulation):
    require_optimizer_choice(optimizer_config)
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError("Invalid accumulation in numerical receipt")
    return {
        "schema_version": 1,
        "numerical_policy": DENSE_NUMERICAL_POLICY,
        "optimizer": optimizer_config.name,
        "optimizer_configuration": dataclasses.asdict(optimizer_config),
        "ns_implementation": optimizer_config.resolved_ns_implementation,
        "normalization_owner": "Trainer.training_step/current_gradient_accumulation_steps",
        "trainer_scheduled_accumulation": accumulation,
        "accelerator_divisor": 1,
        "attention_backward_deterministic": True,
        "upstream_versions": VERSIONS,
        "upstream_source_hashes": SOURCE_HASHES,
    }


def require_resume_receipt(checkpoint, expected):
    path = Path(checkpoint)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Require an ordinary corrected checkpoint directory")
    source = path / RECEIPT_NAME
    if source.is_symlink() or not source.is_file():
        raise ValueError(
            "Checkpoint lacks the corrected numerical contract; do not relabel legacy state"
        )
    if json.loads(source.read_text()) != expected:
        raise ValueError("Checkpoint numerical contract differs from this run")


def require_optimizer_state(optimizer, config):
    require_optimizer_choice(config)
    for group in optimizer.param_groups:
        if group.get("dense_numerical_policy") != DENSE_NUMERICAL_POLICY:
            raise ValueError("Restored optimizer group lacks the corrected numerical policy")
        if group["algorithm"] in {"muon", "normuon"} and group.get("ns_implementation") != (
            config.resolved_ns_implementation
        ):
            raise ValueError("Restored optimizer operator implementation differs")


def require_training_request(config, resume_from_checkpoint):
    if config.numerical_policy != DENSE_NUMERICAL_POLICY or config.model_family != "dense":
        raise ValueError("New Dense training requires an explicit corrected numerical policy")
    if config.dense_can_flatten_inputs:
        raise ValueError("Corrected Dense training requires the existing independent-input policy")
    require_optimizer_choice(config.optimizer)
    output = config.output_dir
    if output.is_symlink():
        raise ValueError("Do not train into a symlinked output namespace")
    if not resume_from_checkpoint and output.exists() and any(output.iterdir()):
        raise ValueError("Fresh corrected training must not overwrite existing run artifacts")
