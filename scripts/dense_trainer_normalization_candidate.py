"""Undeployed, pinned-stack repair candidate; never imported by the live launcher.

The HF Trainer owns accumulation scheduling AND mean-loss normalization. Accelerate
still owns distributed backward, but must not divide that already-normalized loss
again. This candidate changes neither the loss, optimizer nor Trainer batch size.
Use one active Trainer per process: Accelerate's gradient state is process-shared.
GPU, resume and scientific-replication acceptance remain separate requirements.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from pathlib import Path

from accelerate import Accelerator, DistributedType
from sentence_transformers import SentenceTransformerTrainer
from transformers import Trainer

from embed_optim.losses import ExplicitDenseInfoNCELoss
from embed_optim.train import OptimizerTrainer

VERSIONS = {"transformers": "5.3.0", "accelerate": "1.13.0", "sentence-transformers": "5.7.0"}
SOURCE_HASHES = {
    "trainer": "060eda6fcd587e79caeca7c3f246ea7bab1af410479c9776584470b8f4ac48a8",
    "accelerator": "85fe1ad1f0061598313f8566868e9f53434ac9c1d5e24089c3d03da772d40f23",
    "sentence_loss": "3ffc94ef7c3acea812ecbbed5b2ea968abf5e8bf48c881929e22e3e36b3bdd59",
    "study_trainer": "e52cfcb5857aa64d4fb826c1f0b12eabe547506de25241a93b88ef1969d1720d",
}


def verify_stack():
    """Refuse to carry this version-specific correction into an unknown stack."""
    for package, expected in VERSIONS.items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"Unvalidated normalization stack: {package}")
    objects = {
        "trainer": Trainer,
        "accelerator": Accelerator,
        "sentence_loss": SentenceTransformerTrainer.compute_loss,
        "study_trainer": OptimizerTrainer,
    }
    bindings = []
    for name, obj in objects.items():
        path = Path(inspect.getsourcefile(obj)).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != SOURCE_HASHES[name]:
            raise ValueError(f"Unvalidated normalization source: {name}")
        bindings.append({"name": name, "path": str(path), "sha256": digest})
    if OptimizerTrainer.training_step is not Trainer.training_step:
        raise ValueError("The candidate requires the audited inherited training_step")
    return bindings


def assign_normalization_owner(trainer):
    """Keep the Trainer schedule; remove only Accelerate's second divisor."""
    accelerator = trainer.accelerator
    if (
        trainer.is_deepspeed_enabled
        or trainer.is_fsdp_enabled
        or accelerator.distributed_type
        not in (DistributedType.NO, DistributedType.MULTI_CPU, DistributedType.MULTI_GPU)
    ):
        raise ValueError("Only the ordinary single-device/DDP path is in candidate scope")
    scheduled = trainer.args.gradient_accumulation_steps
    if type(scheduled) is not int or scheduled < 1:
        raise ValueError("Require a positive integral Trainer accumulation schedule")
    if accelerator.gradient_accumulation_steps != scheduled:
        raise ValueError("Unexpected pre-correction Accelerator accumulation divisor")
    accelerator.gradient_accumulation_steps = 1
    if accelerator.gradient_accumulation_steps != 1:
        raise ValueError("Accelerator did not retain the single-normalization policy")
    return {
        "owner": "Trainer.training_step/current_gradient_accumulation_steps",
        "trainer_scheduled_accumulation": scheduled,
        "accelerator_divisor_before": scheduled,
        "accelerator_divisor_after": 1,
        "runtime_deployed": False,
    }


class SingleNormalizationTrainerCandidate(OptimizerTrainer):
    """Explicitly selected candidate, not a monkeypatch of any installed class."""

    def __init__(self, *args, **kwargs):
        if type(kwargs.get("loss")) is not ExplicitDenseInfoNCELoss:
            raise ValueError("Candidate requires the study's explicit Dense mean loss")
        super().__init__(*args, **kwargs)
        if self.compute_loss_func is not None:
            raise ValueError("Custom compute_loss_func is outside the audited candidate")

    def create_accelerator_and_postprocess(self):
        self.normalization_source_bindings = verify_stack()
        super().create_accelerator_and_postprocess()
        self.normalization_policy = assign_normalization_owner(self)
