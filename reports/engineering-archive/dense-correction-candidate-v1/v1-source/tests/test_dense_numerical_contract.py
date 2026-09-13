from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from accelerate import DistributedType

from embed_optim import dense_numerical_contract as numerical
from embed_optim import train
from embed_optim.config import (
    DENSE_NUMERICAL_POLICY,
    MUON_NS_IMPLEMENTATION,
    NORMUON_NS_IMPLEMENTATION,
    OptimizerConfig,
    RunConfig,
    load_matrix,
    source_wandb_run_id,
)


def configuration(tmp_path, name="adamw"):
    source = load_matrix(Path(__file__).parents[1] / "configs/dense_no_packing_retrain.yaml")[0]
    optimizer = OptimizerConfig(
        name=name,
        lr=3e-5 if name == "adamw" else 3e-4,
        ns_implementation=NORMUON_NS_IMPLEMENTATION if name == "normuon" else None,
    )
    return replace(
        source,
        optimizer=optimizer,
        output_root=str(tmp_path),
        numerical_policy=DENSE_NUMERICAL_POLICY,
    )


def fake_trainer(steps=4):
    return SimpleNamespace(
        args=SimpleNamespace(gradient_accumulation_steps=steps),
        accelerator=SimpleNamespace(
            gradient_accumulation_steps=steps, distributed_type=DistributedType.MULTI_CPU
        ),
        is_deepspeed_enabled=False,
        is_fsdp_enabled=False,
    )


@pytest.mark.parametrize("steps", [1, 2, 4, 8])
def test_single_normalization_keeps_trainer_schedule(steps):
    trainer = fake_trainer(steps)
    receipt = numerical.assign_normalization_owner(trainer)
    assert trainer.args.gradient_accumulation_steps == steps
    assert trainer.accelerator.gradient_accumulation_steps == 1
    assert receipt["accelerator_divisor_before"] == steps


@pytest.mark.parametrize("failure", ["deepspeed", "fsdp", "xla", "divisor", "zero", "float"])
def test_unsupported_normalization_fails_before_mutation(failure):
    trainer = fake_trainer()
    if failure in ("deepspeed", "fsdp"):
        setattr(trainer, f"is_{failure}_enabled", True)
    elif failure == "xla":
        trainer.accelerator.distributed_type = DistributedType.XLA
    elif failure == "divisor":
        trainer.accelerator.gradient_accumulation_steps = 8
    else:
        trainer.args.gradient_accumulation_steps = 0 if failure == "zero" else 4.0
    before = trainer.accelerator.gradient_accumulation_steps
    with pytest.raises(ValueError):
        numerical.assign_normalization_owner(trainer)
    assert trainer.accelerator.gradient_accumulation_steps == before


def test_actual_pinned_stack_and_inherited_dispatch():
    numerical.verify_stack(train.OptimizerTrainer)


def test_restore_preserves_sentence_transformers_call_signature(monkeypatch):
    trainer = object.__new__(train.OptimizerTrainer)
    trainer.dense_numerical_policy = "legacy-v1"
    observed = []
    monkeypatch.setattr(
        train.SentenceTransformerTrainer,
        "_load_from_checkpoint",
        lambda self, path: observed.append(path),
    )
    trainer._load_from_checkpoint("example-checkpoint")
    assert observed == ["example-checkpoint"]


def test_unknown_stack_is_not_auto_accepted(monkeypatch):
    monkeypatch.setattr(numerical.importlib.metadata, "version", lambda _: "future")
    with pytest.raises(ValueError, match="Unvalidated"):
        numerical.verify_stack(train.OptimizerTrainer)


@pytest.mark.parametrize("name", ["adamw", "muon", "normuon"])
def test_new_configuration_roundtrip_and_wandb_namespace(tmp_path, name):
    config = configuration(tmp_path, name)
    assert RunConfig.from_dict(config.as_dict()).as_dict() == config.as_dict()
    assert source_wandb_run_id(config).startswith("study-v5-")
    legacy = replace(config, numerical_policy="legacy-v1")
    assert source_wandb_run_id(legacy) != source_wandb_run_id(config)
    assert "numerical_policy" not in legacy.as_dict()


@pytest.mark.parametrize("name", ["adamw", "muon", "normuon"])
def test_legacy_checkpoint_is_rejected_before_loading(tmp_path, name):
    config = configuration(tmp_path, name)
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="lacks the corrected"):
        numerical.require_resume_receipt(checkpoint, numerical.receipt(config.optimizer, 4))


def test_checkpoint_policy_and_accumulation_must_match(tmp_path):
    import json

    config = configuration(tmp_path)
    expected = numerical.receipt(config.optimizer, 4)
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    path = checkpoint / numerical.RECEIPT_NAME
    path.write_text(json.dumps(expected))
    numerical.require_resume_receipt(checkpoint, expected)
    with pytest.raises(ValueError):
        numerical.require_resume_receipt(checkpoint, numerical.receipt(config.optimizer, 8))
    path.write_text(json.dumps({**expected, "accelerator_divisor": 4}))
    with pytest.raises(ValueError):
        numerical.require_resume_receipt(checkpoint, expected)


def test_legacy_training_request_fails_before_output_or_wandb(tmp_path, monkeypatch):
    config = replace(configuration(tmp_path), numerical_policy="legacy-v1")
    monkeypatch.setattr(train, "set_seed", lambda _: pytest.fail("Must reject before setup"))
    with pytest.raises(ValueError, match="explicit corrected"):
        train.run_training(config)
    assert not config.output_dir.exists()


def test_existing_fresh_run_cannot_be_overwritten(tmp_path):
    config = configuration(tmp_path)
    config.output_dir.mkdir(parents=True)
    marker = config.output_dir / "model.safetensors"
    marker.write_bytes(b"existing evidence")
    with pytest.raises(ValueError, match="overwrite"):
        numerical.require_training_request(config, None)
    assert marker.read_bytes() == b"existing evidence"


def test_new_normuon_requires_additive_implementation(tmp_path):
    config = configuration(tmp_path, "normuon")
    numerical.require_training_request(config, None)
    legacy = replace(config, optimizer=replace(config.optimizer, ns_implementation=None))
    with pytest.raises(ValueError, match="Optimizer implementation"):
        numerical.require_training_request(legacy, None)


def test_muon_cannot_silently_take_normuon_epsilon_policy():
    with pytest.raises(ValueError):
        OptimizerConfig.from_dict(
            {"name": "muon", "lr": 1e-3, "ns_implementation": NORMUON_NS_IMPLEMENTATION}
        )
    legacy = OptimizerConfig.from_dict(
        {"name": "muon", "lr": 1e-3, "ns_implementation": MUON_NS_IMPLEMENTATION}
    )
    assert legacy.resolved_ns_implementation == MUON_NS_IMPLEMENTATION


def test_restored_group_cannot_claim_old_operator_as_new(tmp_path):
    config = configuration(tmp_path, "normuon")
    group = {
        "algorithm": "normuon",
        "dense_numerical_policy": DENSE_NUMERICAL_POLICY,
        "ns_implementation": NORMUON_NS_IMPLEMENTATION,
    }
    optimizer = SimpleNamespace(param_groups=[group])
    numerical.require_optimizer_state(optimizer, config.optimizer)
    group["ns_implementation"] = MUON_NS_IMPLEMENTATION
    with pytest.raises(ValueError):
        numerical.require_optimizer_state(optimizer, config.optimizer)


def fake_model(count=22):
    attention_type = type("ModernBertAttention", (), {})
    attention = [attention_type() for _ in range(count)]
    for module in attention:
        module.deterministic_flash_attn = False
    config = SimpleNamespace(deterministic_flash_attn=False)
    return [
        SimpleNamespace(auto_model=SimpleNamespace(config=config, modules=lambda: attention))
    ], attention


def test_backward_policy_is_applied_to_every_owned_attention():
    model, attention = fake_model()
    numerical.configure_backward(model)
    numerical.require_backward(model)
    attention[-1].deterministic_flash_attn = False
    with pytest.raises(ValueError):
        numerical.require_backward(model)


def test_wrong_attention_topology_is_rejected():
    model, _ = fake_model(21)
    with pytest.raises(ValueError):
        numerical.configure_backward(model)
