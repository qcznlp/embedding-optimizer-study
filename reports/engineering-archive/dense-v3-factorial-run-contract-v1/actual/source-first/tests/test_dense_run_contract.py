import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset

from embed_optim import dense_numerical_contract as numerical
from embed_optim import dense_run_contract as contract
from embed_optim import train
from embed_optim.collators import TEXT_COLUMNS
from embed_optim.config import load_matrix


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("EMBED_OPTIM_MAX_STEPS", "3")
    base = tmp_path / "initial-model"
    base.mkdir()
    (base / "config.json").write_text('{"diagnostic_fixture":true}')
    (base / "model.safetensors").write_bytes(b"no deserialization in this unit fixture")
    data = tmp_path / "dataset"
    row = {**{column: "fixed text" for column in TEXT_COLUMNS}, "length": 10}
    Dataset.from_list([row] * 288).save_to_disk(str(data))
    configs = load_matrix(Path(__file__).parents[1] / "configs/dense_correctness_candidate.yaml")
    config = replace(
        configs[0],
        model_name=str(base),
        model_revision=None,
        dataset_path=str(data),
        output_root=str(tmp_path / "output"),
    )
    identity, dataset = contract.prepare(config, train._training_argument_values(config), 4)
    return config, identity, dataset


def checkpoint(tmp_path, identity, step=2):
    root = tmp_path / f"checkpoint-{step}"
    root.mkdir()
    for name in (
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
        *(f"rng_state_{rank}.pth" for rank in range(4)),
    ):
        (root / name).write_bytes(b"deliberately inert unit-test payload")
    (root / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    (root / numerical.RECEIPT_NAME).write_text(json.dumps(identity["numerical_contract"]))
    contract.seal_checkpoint(root, identity, step)
    return root


def forbidden(*args, **kwargs):
    pytest.fail("Admission reached a side effect or model/pickle loading")


def test_actual_admission_and_content_identical_relocation(fixture, tmp_path, monkeypatch):
    config, expected, dataset = fixture
    sealed = checkpoint(tmp_path, expected)
    data, base = tmp_path / "relocated-data", tmp_path / "relocated-model"
    shutil.copytree(config.dataset_path, data)
    shutil.copytree(config.model_name, base)
    moved = replace(
        config,
        dataset_path=str(data),
        model_name=str(base),
        output_root=str(tmp_path / "new-output"),
        wandb_project="new-project",
    )
    actual, observed = contract.prepare(moved, train._training_argument_values(moved), 4, sealed)
    assert actual == expected and len(observed) == len(dataset) == 288

    def admitted(*args):
        raise RuntimeError("admitted before device setup")

    monkeypatch.setattr(train, "_training_arguments", admitted)
    monkeypatch.setattr(train, "set_seed", forbidden)
    monkeypatch.setattr(train, "_load_model_and_loss", forbidden)
    with pytest.raises(RuntimeError, match="admitted before device"):
        train.run_training(moved, str(sealed))
    assert not moved.output_dir.exists()


@pytest.mark.parametrize(
    "change",
    [
        "seed",
        "epochs",
        "max_length",
        "temperature",
        "max_grad_norm",
        "warmup_ratio",
        "checkpoint_fractions",
        "global_and_micro",
        "dataloader_workers",
        "model_revision",
        "max_steps",
        "optimizer_lr",
        "data_content",
        "model_content",
        "source",
        "environment",
    ],
)
def test_changed_identity_fails_in_actual_entry_before_setup(
    fixture, tmp_path, monkeypatch, change
):
    config, expected, _ = fixture
    sealed = checkpoint(tmp_path, expected)
    values = {
        "seed": 123,
        "epochs": 2.0,
        "max_length": 512,
        "temperature": 0.03,
        "max_grad_norm": 0.5,
        "warmup_ratio": 0.2,
        "checkpoint_fractions": (0.5, 1.0),
        "dataloader_workers": 4,
        "model_revision": "a" * 40,
    }
    if change in values:
        config = replace(config, **{change: values[change]})
    elif change == "global_and_micro":
        config = replace(config, global_batch_size=64, micro_batch_size=4)
    elif change == "max_steps":
        monkeypatch.setenv("EMBED_OPTIM_MAX_STEPS", "4")
    elif change == "optimizer_lr":
        config = replace(config, optimizer=replace(config.optimizer, lr=0.123))
    elif change == "data_content":
        data = tmp_path / "changed-data"
        rows = Dataset.load_from_disk(config.dataset_path).to_list()
        rows[0]["query"] = "other text"
        Dataset.from_list(rows).save_to_disk(str(data))
        config = replace(config, dataset_path=str(data))
    elif change == "model_content":
        (Path(config.model_name) / "model.safetensors").write_bytes(b"different initial weights")
    elif change == "source":
        source = contract.source_identity()
        source["local_files"][0]["sha256"] = "f" * 64
        monkeypatch.setattr(contract, "source_identity", lambda: source)
    elif change == "environment":
        monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    monkeypatch.setattr(train, "_training_arguments", forbidden)
    monkeypatch.setattr(train, "set_seed", forbidden)
    monkeypatch.setattr(train, "_load_model_and_loss", forbidden)
    before = contract.inventory(sealed)
    with pytest.raises(ValueError):
        train.run_training(config, str(sealed))
    assert contract.inventory(sealed) == before
    assert not config.output_dir.exists()


@pytest.mark.parametrize(
    "payload",
    [
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state_3.pth",
        "training_args.bin",
        "trainer_state.json",
    ],
)
def test_payload_replacement_rejected_before_parent_restore(
    fixture, tmp_path, monkeypatch, payload
):
    config, expected, _ = fixture
    sealed = checkpoint(tmp_path, expected)
    (sealed / payload).write_bytes(b"replaced payload")
    monkeypatch.setattr(train, "_training_arguments", forbidden)
    monkeypatch.setattr(train, "set_seed", forbidden)
    with pytest.raises(ValueError):
        train.run_training(config, str(sealed))
    trainer = object.__new__(train.OptimizerTrainer)
    trainer.dense_numerical_policy = config.numerical_policy
    trainer.embedding_optimizer_config = config.optimizer
    trainer.dense_run_identity = expected
    trainer.args = SimpleNamespace(**expected["execution"]["training_arguments"], world_size=4)
    monkeypatch.setattr(train.SentenceTransformerTrainer, "_load_from_checkpoint", forbidden)
    monkeypatch.setattr(
        train.SentenceTransformerTrainer, "_load_optimizer_and_scheduler", forbidden
    )
    with pytest.raises(ValueError):
        trainer._load_from_checkpoint(str(sealed))
    with pytest.raises(ValueError):
        trainer._load_optimizer_and_scheduler(str(sealed))


@pytest.mark.parametrize("missing", [contract.RECEIPT_NAME, contract.SEAL_NAME, "rng_state_0.pth"])
def test_missing_full_receipt_or_rank_state_rejected(fixture, tmp_path, missing):
    config, expected, _ = fixture
    sealed = checkpoint(tmp_path, expected)
    # Preserve the exact generated payload by moving it, not by deleting evidence.
    (sealed / missing).rename(tmp_path / ("removed-" + missing))
    with pytest.raises(ValueError):
        contract.prepare(config, train._training_argument_values(config), 4, sealed)


def test_incomplete_rank_state_cannot_be_sealed(fixture, tmp_path):
    _, expected, _ = fixture
    root = tmp_path / "checkpoint-1"
    root.mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        contract.seal_checkpoint(root, expected, 1)
    assert not (root / contract.SEAL_NAME).exists()


def test_checkpoint_without_admission_cannot_be_saved(monkeypatch):
    trainer = object.__new__(train.OptimizerTrainer)
    trainer.dense_numerical_policy = train.DENSE_NUMERICAL_POLICY
    trainer.dense_run_identity = None
    monkeypatch.setattr(train.SentenceTransformerTrainer, "_save_checkpoint", forbidden)
    with pytest.raises(ValueError, match="complete run admission"):
        trainer._save_checkpoint(None, None)


def test_existing_contract_is_preserved_and_different_identity_rejected(fixture, tmp_path):
    _, expected, _ = fixture
    contract.write_identity(tmp_path, expected)
    before = (tmp_path / contract.RECEIPT_NAME).read_bytes()
    contract.write_identity(tmp_path, expected)
    assert (tmp_path / contract.RECEIPT_NAME).read_bytes() == before
    changed = json.loads(contract.canonical(expected))
    changed["recipe"]["seed"] += 1
    with pytest.raises(ValueError):
        contract.write_identity(tmp_path, changed)
    assert (tmp_path / contract.RECEIPT_NAME).read_bytes() == before


@pytest.mark.parametrize("mode", ["unknown", "final", "later"])
def test_existing_output_is_not_silently_adopted(fixture, tmp_path, mode):
    config, expected, _ = fixture
    sealed = checkpoint(tmp_path, expected)
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "marker").write_text("preserve")
    if mode != "unknown":
        contract.write_identity(config.output_dir, expected)
        (config.output_dir / ("final" if mode == "final" else "checkpoint-3")).mkdir()
    with pytest.raises(ValueError):
        contract.require_output(config.output_dir, expected, sealed)
    assert (config.output_dir / "marker").read_text() == "preserve"


def test_in_place_incomplete_run_keeps_original_metadata(fixture, tmp_path):
    config, expected, _ = fixture
    source = checkpoint(tmp_path, expected)
    config.output_dir.mkdir(parents=True)
    contract.write_identity(config.output_dir, expected)
    shutil.copytree(source, config.output_dir / source.name)
    contract.require_output(config.output_dir, expected, source)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_ambiguous_or_nonfinite_json_is_rejected(tmp_path, text):
    path = tmp_path / "contract.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        contract.load_json(path)


def test_identity_json_distinguishes_boolean_and_integer():
    with pytest.raises(ValueError):
        contract.require_equal({"value": True}, {"value": 1})


def test_contract_and_payload_symlinks_are_rejected(fixture, tmp_path):
    _, expected, _ = fixture
    sealed = checkpoint(tmp_path, expected)
    payload = sealed / "optimizer.pt"
    retained = tmp_path / "retained-optimizer.pt"
    payload.rename(retained)
    payload.symlink_to(retained)
    with pytest.raises(ValueError):
        contract.require_checkpoint(sealed, expected)


def test_directory_addition_during_hash_is_rejected(tmp_path, monkeypatch):
    (tmp_path / "data").write_text("one")
    original = contract.file_identity

    def changed(path, **kwargs):
        result = original(path, **kwargs)
        (tmp_path / "new-file").write_text("two")
        return result

    monkeypatch.setattr(contract, "file_identity", changed)
    with pytest.raises(ValueError, match="Directory changed"):
        contract.inventory(tmp_path)


def test_actual_arguments_cannot_mutate_bound_runtime(fixture):
    _, identity, _ = fixture
    values = dict(identity["execution"]["training_arguments"])
    contract.require_arguments(SimpleNamespace(**values, world_size=4), identity["execution"])
    values["bf16"] = False
    with pytest.raises(ValueError):
        contract.require_arguments(SimpleNamespace(**values, world_size=4), identity["execution"])
