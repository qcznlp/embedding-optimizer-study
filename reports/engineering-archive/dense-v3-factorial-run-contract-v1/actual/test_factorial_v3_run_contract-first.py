"""Source-bound run component tests, NOT genuine calibration/GPU admission.

Recorded genuine input metadata is used to test identity comparisons. Calibrated
rates and any constructed completion files below are explicitly synthetic. The
actual complete data/tensor read is a separate, CPU-only archived command.
"""

import copy
import dataclasses
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset
from safetensors.torch import save_file
from transformers import Trainer, TrainerState

from embed_optim import factorial_v3_bound_trainer as bound
from embed_optim import factorial_v3_run_contract as contract
from embed_optim.factorial_v3_trainer import FactorialTrainer, FactorialTrainingArguments
from embed_optim.primary_contract import canonical, digest, file_identity, read_json
from embed_optim.runtime import verify_runtime_spec

REPOSITORY = Path(__file__).resolve().parents[1]
PARENT = (
    REPOSITORY
    / "reports/engineering-archive/dense-v3-factorial-calibration-v1/actual/actual-inputs-third.json"
)


def recorded_inputs():
    assert file_identity(PARENT)["sha256"] == contract.CALIBRATION_SOURCE_SHA
    return copy.deepcopy(read_json(PARENT)["inputs"])


def metadata_identity(state="adamw_state", operator="adamw", seed=314159):
    """Identity wiring with a synthetic rate, not the public preparation path."""
    inputs = recorded_inputs()
    sources = contract.source_identity(REPOSITORY, Path(contract.__file__).resolve().parents[2])
    cell = {
        "state": state,
        "operator": operator,
        "seed": seed,
        "hidden_lr": 1e-4 if operator == "adamw" else 1e-3,
        "auxiliary_lr": 3e-6,
        "optimizer_state": "fresh_zero_state_after_calibration",
        "source": inputs["sources"][state],
        "branch_data": inputs["branch"],
        "steps": 391,
        "checkpoint_steps": list(contract.STEPS),
        "execution_authorized": False,
        "calibration_sha256": digest({"synthetic_calibration_metadata": state}),
    }
    data = {
        "rows": 50000,
        "columns": list(contract.TEXT_COLUMNS),
        "dataset_fingerprint": "f" * 16,
        "parent_dataset_fingerprint": "e" * 16,
        "view_operations": [
            {"operation": "select_columns", "columns": list(contract.TEXT_COLUMNS)}
        ],
        "accepted_all_field_values_sha256": contract.BRANCH_FIELDS_SHA,
        "selected_sample_ids_sha256": contract.BRANCH_IDS_SHA,
        "selected_text_values_sha256": "a" * 64,
    }
    return contract.make_identity(inputs, cell, data, sources)


def component(identity):
    return {
        "scope": "dense-v3-factorial-trainer-component-v1",
        "scientific_admission": False,
        "cpu_diagnostic": False,
        "bound_factorial_run": copy.deepcopy(identity),
        "optimizer": identity["optimizer"],
        "layout": identity["source"]["named_layout"],
        "dataset_fingerprint": identity["data"]["dataset_fingerprint"],
        "dataset_columns": list(contract.TEXT_COLUMNS),
        "dataset_rows": 50000,
        "seed": identity["seed"],
        "data_seed": identity["seed"],
        "world_size": 4,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "batching": contract.batches.SCOPE,
        "precision": {"bf16": True, "fp16": False, "tf32": True, "use_cpu": False},
        "normalization": {
            "owner": "Trainer.training_step/current_gradient_accumulation_steps",
            "trainer_scheduled_accumulation": 4,
            "accelerator_divisor_before": 4,
            "accelerator_divisor_after": 1,
        },
        "sources": {
            n: identity["sources"][f"src/embed_optim/{n}"]["sha256"]
            for n in contract.COMPONENT_SOURCES
        },
        "bound_run_source": identity["sources"]["src/embed_optim/factorial_v3_bound_trainer.py"],
        "runtime": verify_runtime_spec(identity["runtime_spec"]["path"]),
    }


def metadata_arguments(tmp_path, identity):
    """Construct on CPU, then simulate saved GPU flags for static-reader tests only."""
    args = FactorialTrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        tf32=False,
        report_to="none",
        run_name=identity["run_id"],
        seed=identity["seed"],
        learning_rate=identity["optimizer"]["lr"],
        weight_decay=0.01,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=4,
    )
    args.use_cpu, args.bf16, args.tf32 = False, True, True
    return args


@pytest.mark.parametrize("state", contract.STATES)
@pytest.mark.parametrize("operator", ["adamw", "muon"])
@pytest.mark.parametrize("seed", contract.batches.ORDER_SEEDS)
def test_all_twelve_identity_wirings_do_not_authorize_execution(state, operator, seed):
    identity = metadata_identity(state, operator, seed)
    assert contract.require_identity(identity) == identity
    assert identity["source"]["run_id"] == contract.STATES[state]
    assert identity["scientific_admission"] is False
    assert identity["execution_authorized"] is False
    assert identity["optimizer"]["adamw_lr"] == 3e-6
    contract.require_component(component(identity), identity)


@pytest.mark.parametrize(
    "path,value",
    [
        (("execution_authorized",), True),
        (("scientific_admission",), True),
        (("schema_version",), True),
        (("seed",), 42),
        (("operator",), "normuon"),
        (("source", "run_id"), "padded-adamw-3e-5"),
        (("source", "checkpoint_step"), 3907),
        (("source", "immutable_remote_commit"), "0" * 40),
        (("source", "native_checkpoint", "run_identity_sha256"), "0" * 64),
        (("data", "rows"), 49984),
        (("data", "selected_sample_ids_sha256"), "0" * 64),
        (("data", "accepted_all_field_values_sha256"), "0" * 64),
        (("data", "view_operations"), []),
        (("data", "dataset_fingerprint"), "not-a-fingerprint"),
        (("training", "last_global_batch_size"), 64),
        (("training", "in_batch_negatives"), True),
        (("training", "max_length"), 512),
        (("training", "steps"), 390),
        (("training", "bf16"), False),
        (("calibration_sha256",), "unbound"),
        (("optimizer", "adamw_lr"), 1e-4),
        (("optimizer", "lr"), 0.0),
        (("runtime_spec", "sha256"), "0" * 64),
        (("sources", "src/embed_optim/optimizers.py", "sha256"), "0" * 64),
    ],
)
def test_changed_identity_fields_refused(path, value):
    identity = metadata_identity()
    target = identity
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        contract.require_identity(identity)


def test_current_assembly_and_runtime_are_read_without_gpu_initialization():
    before = torch.cuda.is_initialized()
    runtime = contract.require_run_dependencies(metadata_identity())
    assert runtime["packages"]["torch"] == "2.9.1"
    assert torch.cuda.is_initialized() is before is False


def test_current_run_module_drift_is_not_hidden_by_parent_bindings():
    identity = metadata_identity()
    identity["sources"]["src/embed_optim/factorial_v3_bound_trainer.py"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        contract.require_run_dependencies(identity)


@pytest.mark.parametrize(
    "path,value",
    [
        (("cpu_diagnostic",), True),
        (("scope",), "legacy"),
        (("world_size",), 1),
        (("data_seed",), 42),
        (("dataset_rows",), 49984),
        (("precision", "bf16"), False),
        (("normalization", "accelerator_divisor_after"), 4),
        (("bound_factorial_run", "calibration_sha256"), "0" * 64),
        (("runtime", "packages", "transformers"), "unverified"),
        (("sources", "optimizers.py"), "0" * 64),
    ],
)
def test_unbound_or_changed_component_cannot_be_a_run(path, value):
    identity = metadata_identity()
    observed = component(identity)
    target = observed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        contract.require_component(observed, identity)


@pytest.mark.parametrize(
    "name,value",
    [
        ("use_cpu", True),
        ("bf16", False),
        ("data_seed", 42),
        ("seed", 42),
        ("learning_rate", 0.004),
        ("run_name", "another-run"),
        ("weight_decay", 0.0),
        ("gradient_checkpointing_kwargs", {"use_reentrant": True}),
        ("dataloader_num_workers", 0),
        ("dataloader_persistent_workers", False),
        ("dataloader_prefetch_factor", 2),
        ("dataloader_pin_memory", False),
        ("dataloader_drop_last", True),
        ("gradient_accumulation_steps", 1),
        ("train_sampling_strategy", "group_by_length"),
        ("save_total_limit", 1),
    ],
)
def test_changed_serialized_arguments_refused(tmp_path, name, value):
    identity = metadata_identity()
    args = metadata_arguments(tmp_path, identity)
    contract.require_arguments(args, identity)
    setattr(args, name, value)
    with pytest.raises(ValueError):
        contract.require_arguments(args, identity)


def test_saved_argument_round_trip_is_not_a_gpu_run(tmp_path):
    identity = metadata_identity()
    args = metadata_arguments(tmp_path, identity)
    path = tmp_path / "training_args.bin"
    torch.save(args, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    contract.require_arguments(restored, identity)
    assert torch.cuda.is_initialized() is False


def test_actual_text_hash_catches_spoofed_dataset_fingerprint():
    identity = metadata_identity()
    data = Dataset.from_dict({key: [f"synthetic {key}"] * 50000 for key in contract.TEXT_COLUMNS})
    observed = hashlib.sha256()
    for row in data:
        observed.update(canonical(row) + b"\n")
    identity["data"]["dataset_fingerprint"] = data._fingerprint
    identity["data"]["selected_text_values_sha256"] = observed.hexdigest()
    contract.require_dataset(data, identity)
    bad = Dataset.from_dict({key: [f"changed {key}"] * 50000 for key in contract.TEXT_COLUMNS})
    bad._fingerprint = data._fingerprint
    with pytest.raises(ValueError, match="text values"):
        contract.require_dataset(bad, identity)


def test_bound_trainer_does_not_override_numerical_training_or_save_hooks():
    assert bound.RunBoundFactorialTrainer.training_step is Trainer.training_step
    for name in (
        "compute_loss",
        "create_optimizer",
        "create_scheduler",
        "get_train_dataloader",
        "_save_checkpoint",
        "_load_from_checkpoint",
        "_load_optimizer_and_scheduler",
    ):
        assert getattr(bound.RunBoundFactorialTrainer, name) is getattr(FactorialTrainer, name)


def test_no_process_group_cannot_enter_finalization():
    with pytest.raises(ValueError, match="four-rank"):
        bound.collective_phase(lambda: None)


def rng_fixture():
    return {
        "python": random.Random(53).getstate(),
        "numpy": np.random.RandomState(54).get_state(),
        "cpu": torch.Generator().manual_seed(55).get_state(),
        "cuda": [torch.zeros(16, dtype=torch.uint8) for _ in range(4)],
    }


def test_rng_structure_reader_preserves_all_global_cpu_rng_states():
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    contract.inspect_rank_rng(rng_fixture())
    after = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert before[0] == after[0]
    assert before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "key,value",
    [
        ("cpu", torch.zeros(1)),
        ("cpu", torch.zeros(1, dtype=torch.uint8)),
        ("cuda", []),
        ("cuda", [torch.zeros(2)]),
        ("python", (1,)),
        ("numpy", ()),
    ],
)
def test_malformed_rng_refused(key, value):
    rng = rng_fixture()
    rng[key] = value
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        contract.inspect_rank_rng(rng)


@pytest.mark.parametrize(
    "key,value",
    [
        ("global_step", 390),
        ("max_steps", 392),
        ("num_train_epochs", 2),
        ("train_batch_size", 32),
        ("epoch", 0.999),
    ],
)
def test_incomplete_final_training_state_refused(key, value):
    state = dataclasses.asdict(
        TrainerState(
            global_step=391, max_steps=391, num_train_epochs=1, train_batch_size=8, epoch=1.0
        )
    )
    contract.require_final_state(state)
    state[key] = value
    with pytest.raises(ValueError):
        contract.require_final_state(state)


def test_tiny_weights_cannot_pass_the_default_full_denseon_reader(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"tiny": torch.zeros(2, 2)}, str(path))
    with pytest.raises(ValueError, match="134"):
        contract.inspect_model_weights(path)


def test_legacy_component_cannot_reach_tensor_deserialization(tmp_path, monkeypatch):
    identity = metadata_identity()
    observed = component(identity)
    observed["cpu_diagnostic"] = True

    def forbidden(*args, **kwargs):
        raise AssertionError("Refused component must not deserialize tensors")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="CPU/legacy/unbound"):
        contract.inspect_checkpoint({"path": str(tmp_path), "sha256": "0" * 64}, identity, observed)


def test_missing_genuine_calibrations_cannot_prepare_a_formal_run(tmp_path):
    from embed_optim.factorial_v3_inputs import EVIDENCE, Locations

    locations = Locations(
        REPOSITORY,
        Path("/root/embedding-optimizer-primary-v3"),
        Path("/root/embedding-optimizer-v3-experiment"),
        Path("/root/embedding-optimizer-study/data"),
        REPOSITORY / EVIDENCE,
    )
    # Refuse before any expensive real data/native admission: this unit controls
    # the missing-output boundary, not a second genuine input verification.
    with pytest.raises(ValueError, match="Both actual calibration"):
        contract.prepare_run(
            locations,
            {},
            "adamw_state",
            "adamw",
            314159,
            Path(contract.__file__).resolve().parents[2],
        )
