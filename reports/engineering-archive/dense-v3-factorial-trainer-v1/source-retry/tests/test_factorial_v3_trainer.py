"""Component argument/actual prepared-loader and checkpoint integrity checks."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate.data_loader import prepare_data_loader, skip_first_batches
from datasets import Dataset
from torch.utils.data import DataLoader, RandomSampler

from embed_optim import factorial_v3_batches as batches
from embed_optim import factorial_v3_checkpoint as checkpoints
from embed_optim import factorial_v3_trainer as f


def args(tmp_path, **kwargs):
    values = dict(
        output_dir=str(tmp_path / "unused"),
        use_cpu=True,
        report_to=[],
        dataloader_pin_memory=False,
        disable_tqdm=True,
    )
    values.update(kwargs)
    return f.FactorialTrainingArguments(**values)


def test_explicit_arguments_retain_non_dropping_policy_and_serialization(tmp_path):
    actual = args(tmp_path)
    assert actual.dataloader_drop_last is False
    assert actual.batch_sampler is batches.batch_sampler_factory
    assert actual.data_seed == actual.seed == 314159
    assert actual.to_dict()["factorial_batching_policy"] == batches.SCOPE
    assert "batch_sampler" not in actual.to_dict()
    json.dumps(actual.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "change",
    [
        {"batch_sampler": "batch_sampler"},
        {"dataloader_drop_last": True},
        {"per_device_train_batch_size": 4},
        {"gradient_accumulation_steps": 2},
        {"seed": 42},
        {"seed": True},
        {"data_seed": 161803},
        {"num_train_epochs": 2.0},
        {"max_steps": 3},
        {"warmup_steps": 40},
        {"warmup_ratio": 0.1},
        {"lr_scheduler_type": "cosine"},
        {"remove_unused_columns": True},
        {"ignore_data_skip": True},
        {"auto_find_batch_size": True},
        {"save_only_model": True},
        {"push_to_hub": True},
        {"fp16": True},
        {"factorial_batching_policy": "legacy"},
        {"prompts": "prefix"},
        {"accelerator_config": {"split_batches": True}},
        {"accelerator_config": {"use_seedable_sampler": False}},
        {"accelerator_config": {"dispatch_batches": True}},
        {"accelerator_config": {"use_stateful_dataloader": True}},
    ],
)
def test_wrong_execution_arguments_are_not_silently_repaired(tmp_path, change):
    with pytest.raises(ValueError):
        args(tmp_path, **change)


@pytest.mark.parametrize("seed", batches.ORDER_SEEDS)
def test_actual_prepared_dataloader_and_skipped_resume_cover_the_exact_permutation(tmp_path, seed):
    actual = args(tmp_path, seed=seed)
    data = Dataset.from_dict({"row_id": list(range(50000))})
    ranks = []
    for rank in range(4):

        def prepare():
            raw = DataLoader(
                data,
                batch_sampler=batches.batch_sampler_factory(
                    data, 8, False, generator=torch.Generator().manual_seed(seed)
                ),
            )
            return prepare_data_loader(
                raw,
                num_processes=4,
                process_index=rank,
                split_batches=False,
                even_batches=False,
                use_seedable_sampler=True,
                data_seed=seed,
                rng_types=[],
            )

        loader = prepare()
        observation_args = SimpleNamespace(**vars(actual))
        observation_args.world_size = 4
        observation_args.process_index = rank
        assert f.inspect_loader(loader, observation_args)["optimizer_steps"] == 391
        loader.set_epoch(0)
        observed = [batch["row_id"].tolist() for batch in loader]
        assert len(observed) == 1564
        assert sum(map(len, observed)) == 12500
        assert list(map(len, observed[-4:])) == [5] * 4
        ranks.append(observed)
        restored = prepare()
        restored.set_epoch(0)
        restored = skip_first_batches(restored, 235 * 4)
        assert [b["row_id"].tolist() for b in restored] == observed[235 * 4 :]
    logical_order = [i for micro in range(1564) for rank in ranks for i in rank[micro]]
    expected = list(RandomSampler(range(50000), generator=torch.Generator().manual_seed(seed)))
    assert logical_order == expected


def checkpoint(tmp_path):
    directory = tmp_path / "checkpoint-79"
    directory.mkdir()
    for name in (
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
        "modules.json",
        "model.safetensors",
        *(f"rng_state_{rank}.pth" for rank in range(4)),
    ):
        (directory / name).write_bytes(b"explicit inventory-only fixture, not loadable model state")
    (directory / "trainer_state.json").write_text(
        json.dumps(
            {"global_step": 79, "max_steps": 391, "num_train_epochs": 1, "train_batch_size": 8}
        )
    )
    identity = {"scope": "inventory-unit-fixture", "scientific_admission": False}
    binding = checkpoints.seal(directory, identity, 79)
    return directory, identity, binding


def test_distinct_component_seal_requires_external_binding(tmp_path):
    directory, identity, binding = checkpoint(tmp_path)
    assert checkpoints.read(binding, identity)["scientific_admission"] is False
    assert checkpoints.file_digest(directory / checkpoints.NAME) == binding["sha256"]
    with pytest.raises(FileExistsError):
        checkpoints.seal(directory, identity, 79)


@pytest.mark.parametrize(
    "change", ["model", "rng", "extra", "symlink", "identity", "external_digest", "step", "horizon"]
)
def test_component_inventory_refuses_changed_or_missing_payload(tmp_path, change):
    directory, identity, binding = checkpoint(tmp_path)
    if change == "model":
        (directory / "model.safetensors").write_bytes(b"different")
    elif change == "rng":
        (directory / "rng_state_3.pth").rename(directory / "renamed_rng")
    elif change == "extra":
        (directory / "undeclared").write_bytes(b"extra")
    elif change == "symlink":
        (directory / "unexpected-link").symlink_to(directory / "model.safetensors")
    elif change == "identity":
        identity = {**identity, "scope": "different"}
    elif change == "external_digest":
        binding = {**binding, "sha256": "0" * 64}
    else:
        state = json.loads((directory / "trainer_state.json").read_text())
        state["global_step" if change == "step" else "max_steps"] = 80
        (directory / "trainer_state.json").write_text(json.dumps(state))
        payload = json.loads((directory / checkpoints.NAME).read_text())
        payload["files"] = checkpoints.inventory(directory)
        (directory / checkpoints.NAME).write_text(checkpoints.canonical(payload))
        binding = {**binding, "sha256": checkpoints.file_digest(directory / checkpoints.NAME)}
    with pytest.raises(ValueError):
        checkpoints.read(binding, identity)


def test_new_component_imports_intended_development_sources():
    root = Path(__file__).resolve().parents[1]
    for module in (f, batches, checkpoints):
        assert Path(module.__file__).resolve().parent == root / "src/embed_optim"


def test_sampler_callable_mutation_is_rejected_after_initialization(tmp_path):
    actual = args(tmp_path)
    mutated = copy.copy(actual)
    mutated.batch_sampler = lambda *a, **k: None
    with pytest.raises(ValueError):
        f.require_requested_arguments(mutated)
