"""Factory wiring tests. Calibrated rates below are synthetic, not GPU admission."""

import dataclasses
from pathlib import Path

import pytest
import torch
from test_factorial_v3_run_contract import metadata_identity

from embed_optim import factorial_v3_factory as factory
from embed_optim import factorial_v3_run_contract as contract
from embed_optim.factorial_v3_trainer import FactorialTrainingArguments


def recipe(identity, tmp_path):
    return factory.recipe(
        identity,
        tmp_path / "branch",
        tmp_path / "output",
        project="factory-test",
        entity="local-test-only",
    )


@pytest.mark.parametrize("state", ("adamw_state", "muon_state"))
@pytest.mark.parametrize("operator", ("adamw", "muon"))
@pytest.mark.parametrize("seed", (314159, 271828, 161803))
def test_all_twelve_fixed_recipe_cells(tmp_path, state, operator, seed):
    identity = metadata_identity(state, operator, seed)
    config, values = recipe(identity, tmp_path)
    assert config.model_name == identity["source"]["checkpoint"]
    assert config.model_revision is None
    assert config.optimizer.name == ("hybrid_adamw" if operator == "adamw" else "muon")
    assert config.optimizer.lr == identity["optimizer"]["lr"]
    assert config.optimizer.aux_lr == 3e-6
    assert config.checkpoint_fractions == (0.2, 0.4, 0.6, 0.8, 1.0)
    assert config.max_length == 8192 and config.resolved_temperature == 0.02
    assert config.flash_attention is True and config.dense_can_flatten_inputs is False
    assert values["seed"] == values["data_seed"] == seed
    assert values["save_strategy"] == "no" and values["dataloader_drop_last"] is False
    assert values["train_sampling_strategy"] == "random" and values["max_steps"] == -1
    assert values["output_dir"] == str(config.output_dir)
    assert values["run_name"] == identity["run_id"]
    assert values["warmup_steps"] == 0.1 and "save_safetensors" not in values
    # Real CPU dataclass construction, then a *static saved-GPU metadata*
    # simulation. No GPU Trainer/device or BF16 computation is created here.
    simulated = dict(values, use_cpu=True, bf16=False, tf32=False, report_to=[])
    args = FactorialTrainingArguments(**simulated)
    args.use_cpu, args.bf16, args.tf32 = False, True, True
    args.dataloader_pin_memory = True
    contract.require_arguments(args, identity)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("key,value", (("project", ""), ("entity", None), ("entity", False)))
def test_missing_logging_identity_is_refused(tmp_path, key, value):
    arguments = {"project": "p", "entity": "e", key: value}
    with pytest.raises(ValueError, match="logging"):
        factory.recipe(metadata_identity(), tmp_path / "branch", tmp_path / "out", **arguments)


@pytest.mark.parametrize("which", ("branch", "output"))
def test_relative_paths_are_refused(tmp_path, which):
    with pytest.raises(ValueError, match="absolute"):
        factory.recipe(
            metadata_identity(),
            Path("branch") if which == "branch" else tmp_path,
            Path("output") if which == "output" else tmp_path,
            project="p",
            entity="e",
        )


def test_production_factory_refuses_cpu_before_reading_or_initializing(monkeypatch, tmp_path):
    identity = metadata_identity()
    config, _ = recipe(identity, tmp_path)
    monkeypatch.setattr(contract, "require_run_dependencies", lambda *a: pytest.fail("early IO"))
    with pytest.raises(ValueError, match="already admitted four-rank NCCL"):
        factory.load_model(identity, config)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "key,value",
    (
        ("max_length", 512),
        ("flash_attention", False),
        ("gradient_checkpointing", False),
        ("dense_can_flatten_inputs", True),
        ("model_revision", "main"),
        ("temperature", 0.05),
    ),
)
def test_loader_refuses_recipe_overrides_before_model_or_checkpoint_read(
    monkeypatch, tmp_path, key, value
):
    identity = metadata_identity()
    config, _ = recipe(identity, tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(contract, "require_run_dependencies", lambda *a: {})
    monkeypatch.setattr(
        contract, "require_source_checkpoint", lambda *a: pytest.fail("early model IO")
    )
    with pytest.raises(ValueError):
        factory.load_model(
            identity, dataclasses.replace(config, **{key: value}), diagnostic_cpu=True
        )


def test_no_primary_argument_builder_or_numerical_override():
    import inspect

    source = inspect.getsource(factory.recipe)
    assert "primary_train._training_argument" not in source
    assert factory.RunBoundFactorialTrainer is not factory.primary_train.OptimizerTrainer
    assert "primary_train._load_model_and_loss(effective)" in inspect.getsource(factory.load_model)


def test_internal_constructor_requires_an_existing_worker():
    with pytest.raises(ValueError, match="already admitted four-rank NCCL"):
        factory.prepare_trainer(None, None, None, None, None, None, None, project="p", entity="e")
    assert not torch.cuda.is_initialized()


def mocked_constructor(monkeypatch, tmp_path):
    """Explicitly mocked orchestration only; no real process group/model/Trainer."""
    from types import SimpleNamespace

    import transformers

    identity = metadata_identity()
    data, model, loss = object(), object(), object()
    observed = []
    monkeypatch.setattr(factory, "require_gpu_context", lambda: None)
    monkeypatch.setattr(factory, "collective_phase", lambda call: call())
    monkeypatch.setattr(contract, "prepare_run", lambda *a: (identity, data))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda dest, obj: dest.__setitem__(slice(None), [obj] * 4),
    )
    monkeypatch.setattr(
        factory,
        "FactorialTrainingArguments",
        lambda **kw: SimpleNamespace(**kw, world_size=4, local_process_index=0),
    )
    monkeypatch.setattr(contract, "require_arguments", lambda *a: observed.append("arguments"))
    monkeypatch.setattr(transformers, "set_seed", lambda seed: observed.append(("seed", seed)))
    monkeypatch.setattr(factory, "load_model", lambda *a: (model, loss, {"mocked_loading": True}))

    def construct(**kwargs):
        observed.append(kwargs)
        return "mocked-Trainer-not-a-run"

    monkeypatch.setattr(factory, "RunBoundFactorialTrainer", construct)
    for key, value in {
        "WANDB_PROJECT": "p",
        "WANDB_ENTITY": "e",
        "WANDB_RUN_ID": identity["run_id"],
        "WANDB_RESUME": "never",
    }.items():
        monkeypatch.setenv(key, value)
    parameters = (
        SimpleNamespace(data_store=tmp_path / "data"),
        {},
        "adamw_state",
        "adamw",
        314159,
        Path(factory.__file__).resolve().parents[2],
        tmp_path / "out",
    )
    return identity, parameters, observed, (data, model, loss)


def test_mocked_end_to_end_constructor_wiring(monkeypatch, tmp_path):
    identity, parameters, observed, (data, model, loss) = mocked_constructor(monkeypatch, tmp_path)
    trainer, receipt = factory.prepare_trainer(*parameters, project="p", entity="e")
    assert trainer == "mocked-Trainer-not-a-run"
    assert observed[:2] == ["arguments", ("seed", 314159)]
    call = observed[-1]
    assert call["model"] is model and call["loss"] is loss and call["train_dataset"] is data
    assert call["run_identity"] == identity
    assert call["optimizer_config"].name == "hybrid_adamw"
    assert set(call) == {
        "run_identity",
        "optimizer_config",
        "model",
        "args",
        "train_dataset",
        "loss",
    }
    assert receipt["scientific_admission"] is False and receipt["training_executed"] is False
    assert not (tmp_path / "out").exists() and not torch.cuda.is_initialized()


@pytest.mark.parametrize("problem", ("existing_output", "wrong_logging", "rank_disagreement"))
def test_mocked_constructor_refuses_before_arguments_or_loading(monkeypatch, tmp_path, problem):
    identity, parameters, observed, _ = mocked_constructor(monkeypatch, tmp_path)
    if problem == "existing_output":
        (tmp_path / "out" / "dense" / identity["run_id"]).mkdir(parents=True)
    elif problem == "wrong_logging":
        monkeypatch.setenv("WANDB_ENTITY", "wrong-entity")
    else:
        monkeypatch.setattr(
            torch.distributed,
            "all_gather_object",
            lambda dest, obj: dest.__setitem__(slice(None), [obj, obj, "different", obj]),
        )
    with pytest.raises(ValueError):
        factory.prepare_trainer(*parameters, project="p", entity="e")
    assert observed == [] and not torch.cuda.is_initialized()
