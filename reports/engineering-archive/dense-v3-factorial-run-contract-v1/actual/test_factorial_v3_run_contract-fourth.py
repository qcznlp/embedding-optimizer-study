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
from embed_optim import factorial_v3_checkpoint as checkpoints
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
    # ST intentionally disabled pinning for the CPU construction above. Restore
    # only the simulated saved-GPU metadata; no GPU Trainer is instantiated.
    args.dataloader_pin_memory = True
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
    assert identity["optimizer"]["aux_lr"] == 3e-6
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
        (("optimizer", "aux_lr"), 1e-4),
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
    assert runtime["packages"] == read_json(recorded_inputs()["runtime_spec"]["path"])["packages"]
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


def write_new_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def checkpoint_fixture(root, identity):
    """Five real content seals over tiny SYNTHETIC, non-DenseOn payload files."""
    root.mkdir(parents=True, exist_ok=True)
    observed = component(identity)
    args = metadata_arguments(root, identity)
    stages = []
    for step in contract.STEPS:
        path = root / f"checkpoint-{step}"
        path.mkdir()
        for name in contract.INFERENCE_FILES:
            target = path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f"synthetic structural fixture only: {name}\n".encode())
        (path / "README.md").write_text("Synthetic component fixture. Not a trained retriever.\n")
        torch.save({}, path / "optimizer.pt")
        torch.save({}, path / "scheduler.pt")
        torch.save(args, path / "training_args.bin")
        for rank in range(4):
            torch.save(rng_fixture(), path / f"rng_state_{rank}.pth")
        state = dataclasses.asdict(
            TrainerState(
                global_step=step,
                max_steps=391,
                num_train_epochs=1,
                train_batch_size=8,
                epoch=step / 391,
            )
        )
        write_new_json(path / "trainer_state.json", state)
        binding = checkpoints.seal(path, observed, step)
        stages.append({"step": step, "directory": path.name, "sha256": binding["sha256"]})
    return observed, stages


def completion_fixture(root, identity):
    import shutil

    observed, stages = checkpoint_fixture(root, identity)
    final = root / "final"
    final.mkdir()
    for name in (*contract.INFERENCE_FILES, "README.md", "training_args.bin"):
        (final / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "checkpoint-391" / name, final / name)
    write_new_json(root / contract.RUN_NAME, identity)
    write_new_json(
        root / "trainer_state_final.json", read_json(root / "checkpoint-391/trainer_state.json")
    )
    inventory = checkpoints.inventory(final)
    value = {
        "scope": contract.SCOPE,
        "status": "training_complete",
        "scientific_admission": False,
        "run_identity": identity,
        "component_identity": observed,
        "optimizer_steps": 391,
        "dataset_rows": 50000,
        "checkpoints": stages,
        "final_inference_files": [r for r in inventory if r["path"] in contract.INFERENCE_FILES],
        "final_files": inventory,
        "trainer_state_final": file_identity(root / "trainer_state_final.json"),
    }
    write_new_json(root / contract.RUN_COMPLETE, value)
    return value, file_identity(root / contract.RUN_COMPLETE)


def simulated_tensor_reader(binding, identity, expected_component):
    """Only the deep tensor stage is simulated; native seals still read in full."""
    native = checkpoints.read(binding, expected_component)
    return {"step": native["step"], "simulated_tensor_read": True}


def test_whole_run_container_wiring_with_explicitly_simulated_tensor_reads(tmp_path, monkeypatch):
    identity = metadata_identity()
    _, binding = completion_fixture(tmp_path, identity)
    monkeypatch.setattr(contract, "inspect_checkpoint", simulated_tensor_reader)
    result = contract.inspect_complete_run(tmp_path, binding, identity)
    assert [r["step"] for r in result["checkpoints"]] == list(contract.STEPS)
    assert all(r["simulated_tensor_read"] for r in result["checkpoints"])
    assert result["scientific_admission"] is False


def test_default_deep_reader_rejects_the_same_synthetic_container(tmp_path):
    identity = metadata_identity()
    _, binding = completion_fixture(tmp_path, identity)
    with pytest.raises(ValueError):
        contract.inspect_complete_run(tmp_path, binding, identity)


@pytest.mark.parametrize(
    "change",
    [
        "short_horizon",
        "missing_stage",
        "reordered_stages",
        "different_run",
        "scientific_promotion",
        "missing_final_file",
        "changed_final_epoch",
        "extra_final_payload",
    ],
)
def test_rehashed_semantic_completion_changes_are_refused(tmp_path, monkeypatch, change):
    identity = metadata_identity()
    value, _ = completion_fixture(tmp_path, identity)
    if change == "short_horizon":
        value["optimizer_steps"] = 390
    elif change == "missing_stage":
        value["checkpoints"] = value["checkpoints"][:-1]
    elif change == "reordered_stages":
        value["checkpoints"][0], value["checkpoints"][1] = (
            value["checkpoints"][1],
            value["checkpoints"][0],
        )
    elif change == "different_run":
        value["run_identity"]["calibration_sha256"] = "0" * 64
    elif change == "scientific_promotion":
        value["scientific_admission"] = True
    elif change == "missing_final_file":
        value["final_inference_files"] = value["final_inference_files"][:-1]
    elif change == "changed_final_epoch":
        state = read_json(tmp_path / "trainer_state_final.json")
        state["epoch"] = 0.99
        (tmp_path / "trainer_state_final.json").write_text(json.dumps(state))
        value["trainer_state_final"] = file_identity(tmp_path / "trainer_state_final.json")
    elif change == "extra_final_payload":
        (tmp_path / "final/extra.json").write_text("{}\n")
        value["final_files"] = checkpoints.inventory(tmp_path / "final")
    (tmp_path / contract.RUN_COMPLETE).write_text(json.dumps(value))
    binding = file_identity(tmp_path / contract.RUN_COMPLETE)
    monkeypatch.setattr(contract, "inspect_checkpoint", simulated_tensor_reader)
    with pytest.raises(ValueError):
        contract.inspect_complete_run(tmp_path, binding, identity)


@pytest.mark.parametrize(
    "change", ["nonintegral_counter_type", "hidden_component_file", "wrong_last_checkpoint_epoch"]
)
def test_completion_container_additional_counterexamples(tmp_path, monkeypatch, change):
    identity = metadata_identity()
    value, _ = completion_fixture(tmp_path, identity)
    if change == "nonintegral_counter_type":
        value["optimizer_steps"] = 391.0
    elif change == "hidden_component_file":
        (tmp_path / "final" / checkpoints.NAME).write_text("synthetic extra file\n")
    elif change == "wrong_last_checkpoint_epoch":
        checkpoint = tmp_path / "checkpoint-391"
        state = read_json(checkpoint / "trainer_state.json")
        state["epoch"] = 0.99
        (checkpoint / "trainer_state.json").write_text(json.dumps(state))
        seal = read_json(checkpoint / checkpoints.NAME)
        seal["files"] = checkpoints.inventory(checkpoint)
        (checkpoint / checkpoints.NAME).write_text(json.dumps(seal))
        value["checkpoints"][-1]["sha256"] = checkpoints.file_digest(checkpoint / checkpoints.NAME)
    (tmp_path / contract.RUN_COMPLETE).write_text(json.dumps(value))
    monkeypatch.setattr(contract, "inspect_checkpoint", simulated_tensor_reader)
    with pytest.raises(ValueError):
        contract.inspect_complete_run(
            tmp_path, file_identity(tmp_path / contract.RUN_COMPLETE), identity
        )


def test_public_complete_run_reader_cannot_skip_missing_calibrations(tmp_path):
    with pytest.raises(ValueError, match="Both actual calibration"):
        contract.read_complete_run(
            None,
            {},
            state="adamw_state",
            operator="adamw",
            seed=314159,
            source_root=tmp_path,
            root=tmp_path,
            binding={"bytes": 0, "sha256": "0" * 64},
        )


def test_public_reader_joins_supplied_cell_with_explicitly_simulated_preparation(
    tmp_path, monkeypatch
):
    identity = metadata_identity()
    _, binding = completion_fixture(tmp_path, identity)
    calls = []

    def simulated_prepare(*args):
        calls.append(args)
        return identity, None

    monkeypatch.setattr(contract, "prepare_run", simulated_prepare)
    monkeypatch.setattr(contract, "inspect_checkpoint", simulated_tensor_reader)
    marker = {"explicitly_simulated_calibrations": True}
    result = contract.read_complete_run(
        "fixture-locations",
        marker,
        state="adamw_state",
        operator="adamw",
        seed=314159,
        source_root=tmp_path,
        root=tmp_path,
        binding=binding,
    )
    assert calls == [("fixture-locations", marker, "adamw_state", "adamw", 314159, tmp_path)]
    assert len(result["artifacts"]["checkpoints"]) == 5
    assert result["scientific_admission"] is result["execution_authorized"] is False
