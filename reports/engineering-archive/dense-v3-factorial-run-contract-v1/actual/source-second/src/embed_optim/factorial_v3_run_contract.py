"""Run identity and complete saved-state reading for the v3 crossed continuation.

Preparation and numerical artifact acceptance are separate from GPU admission,
controller authority and scientific publication. No old runtime lock is changed.
"""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

from . import factorial_v3_batches as batches
from . import factorial_v3_calibration as calibration
from . import factorial_v3_checkpoint as checkpoints
from . import factorial_v3_optimizer as optim
from .config import OptimizerConfig
from .factorial_v3_inputs import AUDITS, STATES, load_inputs
from .primary_contract import (
    canonical,
    digest,
    file_identity,
    inspect_sealed_checkpoint,
    read_json,
    require_same,
    verify_file,
)

SCOPE = "dense-v3-factorial-source-bound-run-v1"
STEPS = (79, 157, 235, 313, 391)
TEXT_COLUMNS = ("query", "positive", *(f"negative_{i}" for i in range(7)))
RUN_NAME = "factorial_run_identity.json"
RUN_COMPLETE = "factorial_run_complete.json"
CALIBRATION_SOURCE_SHA = "3640dc3a0deaa142d7e09eabfcc8450ab786db1b7302e04f470518557a63176f"
PARENT_SOURCE_FILES_SHA = "247848d22aed020f3b50a2788ffd6a72a78d96730b2e52bcdb033450c7436380"
BRANCH_FIELDS_SHA = "c729ac2ffa5cdbdb59bfd19f3d18caacf7c3ebb35e4cf67280564ef4bbfe6f12"
BRANCH_IDS_SHA = "a05f02c97a299abe43f43ee5452339cb381c62a43acabb89bba1696de27892a4"
RUNTIME_SHA = "b90de16b3796bb8e9d0badcedec688da55a037d46244a0d2a7f5838795cfe77a"
NEW_SOURCES = ("factorial_v3_run_contract.py", "factorial_v3_bound_trainer.py")
SOURCE_RECORD_SHA = {
    "adamw_state": "81ed3424b4bb2c2c76e1308064e3079197bbace46e3f4007d7c69089293dd0f5",
    "muon_state": "9182083a79c0beebbfb484bd6ac360b619964ea7abe093ab7e92b78e0058fb28",
}
COMPONENT_SOURCES = (
    "factorial_v3_trainer.py",
    "factorial_v3_batches.py",
    "factorial_v3_optimizer.py",
    "factorial_v3_checkpoint.py",
    "config.py",
    "optimizers.py",
    "losses.py",
    "dense_numerical_contract.py",
)
INFERENCE_FILES = (
    "model.safetensors",
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "1_Pooling/config.json",
)


def source_identity(repository, source_root):
    """Keep all 66 accepted files intact; bind the two new integration modules."""
    path = (
        Path(repository)
        / "reports/engineering-archive/dense-v3-factorial-calibration-v1/actual/actual-inputs-third.json"
    )
    if file_identity(path)["sha256"] != CALIBRATION_SOURCE_SHA:
        raise ValueError("Accepted calibration source parent differs")
    accepted = read_json(path)
    files = copy.deepcopy(accepted["source_files"])
    for relative, expected in files.items():
        verify_file(Path(source_root) / relative, expected)
    if len(files) != 66 or digest(files) != PARENT_SOURCE_FILES_SHA:
        raise ValueError("Accepted parent does not contain the exact 66-file closure")
    for name in NEW_SOURCES:
        relative = f"src/embed_optim/{name}"
        files[relative] = file_identity(Path(source_root) / relative)
    if file_identity(__file__) != files["src/embed_optim/factorial_v3_run_contract.py"]:
        raise ValueError("Run contract imported from a different source assembly")
    calibration.require_sources()
    return files


def branch_dataset(inputs):
    """One explicit nine-column view of the unchanged complete branch Dataset."""
    from datasets import Dataset

    if (
        inputs.get("accepted_audit_digests") != AUDITS
        or inputs["branch"]["rows"] != 50000
        or inputs["branch"]["all_field_values_sha256"] != BRANCH_FIELDS_SHA
        or inputs["branch"]["selected_sample_ids_sha256"] != BRANCH_IDS_SHA
    ):
        raise ValueError("Require the complete genuine fixed-branch input")
    root = Path(inputs["branch"]["path"])
    dataset = Dataset.load_from_disk(str(root / "dataset"))
    if len(dataset) != 50000 or dataset.column_names != inputs["branch"]["columns"]:
        raise ValueError("Actual branch Dataset differs from the admitted complete view")
    fields, texts = hashlib.sha256(), hashlib.sha256()
    for row in dataset:
        fields.update(canonical(row) + b"\n")
        texts.update(canonical({k: row[k] for k in TEXT_COLUMNS}) + b"\n")
    if fields.hexdigest() != inputs["branch"]["all_field_values_sha256"]:
        raise ValueError("Actual branch field values differ from the accepted intact groups")
    selected = dataset.select_columns(list(TEXT_COLUMNS))
    return selected, {
        "rows": len(selected),
        "columns": selected.column_names,
        "dataset_fingerprint": selected._fingerprint,
        "parent_dataset_fingerprint": dataset._fingerprint,
        "view_operations": [{"operation": "select_columns", "columns": list(TEXT_COLUMNS)}],
        "accepted_all_field_values_sha256": inputs["branch"]["all_field_values_sha256"],
        "selected_sample_ids_sha256": inputs["branch"]["selected_sample_ids_sha256"],
        "selected_text_values_sha256": texts.hexdigest(),
    }


def make_identity(inputs, cell, data, sources):
    """Pure constructor; its outer caller must authenticate real calibration files."""
    state, operator, seed = (cell[k] for k in ("state", "operator", "seed"))
    if (
        state not in STATES
        or operator not in ("adamw", "muon")
        or type(seed) is not int
        or seed not in batches.ORDER_SEEDS
    ):
        raise ValueError("Run cell is outside the fixed crossed design")
    require_same(cell["source"], inputs["sources"][state])
    require_same(cell["branch_data"], inputs["branch"])
    for key, value in {
        "auxiliary_lr": 3e-6,
        "optimizer_state": "fresh_zero_state_after_calibration",
        "steps": 391,
        "checkpoint_steps": list(STEPS),
        "execution_authorized": False,
    }.items():
        require_same(cell[key], value)
    config = OptimizerConfig(
        name="hybrid_adamw" if operator == "adamw" else "muon", lr=cell["hidden_lr"]
    )
    identity = {
        "schema_version": 1,
        "scope": SCOPE,
        "scientific_admission": False,
        "execution_authorized": False,
        "run_id": f"factorial-v3-{state}-{operator}-seed{seed}",
        "state": state,
        "operator": operator,
        "seed": seed,
        "source": copy.deepcopy(inputs["sources"][state]),
        "input_sha256": digest(inputs),
        "calibration_sha256": cell["calibration_sha256"],
        "data": copy.deepcopy(data),
        "optimizer": optim.configuration(config),
        "sources": copy.deepcopy(sources),
        "runtime_spec": copy.deepcopy(inputs["runtime_spec"]),
        "training": {
            "world_size": 4,
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "global_batch_size": 128,
            "last_global_batch_size": 80,
            "epochs": 1.0,
            "steps": 391,
            "checkpoint_steps": list(STEPS),
            "max_length": 8192,
            "temperature": 0.02,
            "hard_negatives": 7,
            "in_batch_negatives": False,
            "max_grad_norm": 1.0,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "bf16": True,
            "fp16": False,
            "tf32": True,
            "use_cpu": False,
            "dataloader_workers": 8,
            "dataloader_pin_memory": True,
            "dataloader_persistent_workers": True,
            "dataloader_prefetch_factor": 4,
            "batching_policy": batches.SCOPE,
            "optimizer_initialization": "fresh_zero_state_after_calibration",
        },
    }
    require_identity(identity)
    return identity


def require_identity(identity):
    if not isinstance(identity, dict) or set(identity) != {
        "schema_version",
        "scope",
        "scientific_admission",
        "execution_authorized",
        "run_id",
        "state",
        "operator",
        "seed",
        "source",
        "input_sha256",
        "calibration_sha256",
        "data",
        "optimizer",
        "sources",
        "runtime_spec",
        "training",
    }:
        raise ValueError("Incomplete or extended factorial run identity")
    if (
        identity.get("scope") != SCOPE
        or type(identity.get("schema_version")) is not int
        or identity["schema_version"] != 1
        or identity.get("scientific_admission") is not False
        or identity.get("execution_authorized") is not False
    ):
        raise ValueError("Not a prepared v3 factorial run identity")
    state, operator, seed = (identity[k] for k in ("state", "operator", "seed"))
    if (
        state not in STATES
        or operator not in ("adamw", "muon")
        or type(seed) is not int
        or seed not in batches.ORDER_SEEDS
    ):
        raise ValueError("Wrong fixed factorial cell")
    if (
        identity["source"]["run_id"] != STATES[state]
        or identity["source"]["checkpoint_step"] != 2345
        or identity["run_id"] != f"factorial-v3-{state}-{operator}-seed{seed}"
        or identity["data"]["rows"] != 50000
        or identity["data"]["columns"] != list(TEXT_COLUMNS)
    ):
        raise ValueError("Wrong source, branch view or run name")
    if (
        digest({k: v for k, v in identity["source"].items() if k != "checkpoint"})
        != SOURCE_RECORD_SHA[state]
    ):
        raise ValueError("Source provenance is not one of the two accepted genuine states")
    expected = OptimizerConfig(
        name="hybrid_adamw" if operator == "adamw" else "muon", lr=identity["optimizer"]["lr"]
    )
    require_same(identity["optimizer"], optim.configuration(expected))
    train = identity["training"]
    required = {
        "world_size": 4,
        "micro_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "global_batch_size": 128,
        "last_global_batch_size": 80,
        "epochs": 1.0,
        "steps": 391,
        "checkpoint_steps": list(STEPS),
        "max_length": 8192,
        "temperature": 0.02,
        "hard_negatives": 7,
        "in_batch_negatives": False,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "use_cpu": False,
        "dataloader_workers": 8,
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": True,
        "dataloader_prefetch_factor": 4,
        "batching_policy": batches.SCOPE,
        "optimizer_initialization": "fresh_zero_state_after_calibration",
    }
    require_same(train, required)
    data = identity["data"]
    if set(data) != {
        "rows",
        "columns",
        "dataset_fingerprint",
        "parent_dataset_fingerprint",
        "view_operations",
        "accepted_all_field_values_sha256",
        "selected_sample_ids_sha256",
        "selected_text_values_sha256",
    }:
        raise ValueError("Incomplete exact branch-view identity")
    require_same(
        data["view_operations"], [{"operation": "select_columns", "columns": list(TEXT_COLUMNS)}]
    )
    if (
        type(data["rows"]) is not int
        or data["accepted_all_field_values_sha256"] != BRANCH_FIELDS_SHA
        or data["selected_sample_ids_sha256"] != BRANCH_IDS_SHA
    ):
        raise ValueError("Run does not use the accepted intact branch groups")
    for key in ("dataset_fingerprint", "parent_dataset_fingerprint"):
        _require_hex(data[key], 16)
    _require_hex(data["selected_text_values_sha256"], 64)
    sources = identity["sources"]
    new = {f"src/embed_optim/{n}" for n in NEW_SOURCES}
    if (
        not isinstance(sources, dict)
        or len(sources) != 68
        or not new <= set(sources)
        or digest({k: v for k, v in sources.items() if k not in new}) != PARENT_SOURCE_FILES_SHA
    ):
        raise ValueError("Run must retain every unchanged accepted parent source")
    for name in new:
        value = sources[name]
        if (
            set(value) != {"bytes", "sha256"}
            or type(value["bytes"]) is not int
            or value["bytes"] <= 0
        ):
            raise ValueError("Invalid new run-source identity")
        _require_hex(value["sha256"], 64)
    runtime = identity["runtime_spec"]
    if (
        set(runtime) != {"path", "bytes", "sha256"}
        or type(runtime["bytes"]) is not int
        or runtime["bytes"] != 1019
        or runtime["sha256"] != RUNTIME_SHA
        or not isinstance(runtime["path"], str)
        or not Path(runtime["path"]).is_absolute()
    ):
        raise ValueError("Run changed the fixed formal runtime specification")
    for key in ("input_sha256", "calibration_sha256"):
        _require_hex(identity[key], 64)
    return copy.deepcopy(identity)


def _require_hex(value, length):
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("Missing canonical content/fingerprint binding")


def require_run_dependencies(identity):
    """Verify this process's whole assembly and runtime; no GPU is initialized."""
    from .runtime import verify_runtime_spec

    require_identity(identity)
    root = Path(__file__).resolve().parents[2]
    for relative, expected in identity["sources"].items():
        verify_file(root / relative, expected)
    spec = identity["runtime_spec"]
    verify_file(spec["path"], spec)
    return verify_runtime_spec(spec["path"])


def require_dataset(dataset, identity):
    """Check actual text values once at construction, not just a mutable fingerprint."""
    from datasets import Dataset

    require_identity(identity)
    expected = identity["data"]
    if (
        type(dataset) is not Dataset
        or len(dataset) != 50000
        or dataset.column_names != expected["columns"]
        or dataset._fingerprint != expected["dataset_fingerprint"]
    ):
        raise ValueError("Trainer received a different actual branch Dataset view")
    observed = hashlib.sha256()
    for row in dataset:
        observed.update(canonical(row) + b"\n")
    if observed.hexdigest() != expected["selected_text_values_sha256"]:
        raise ValueError("Actual Trainer text values differ from the prepared branch")


def require_arguments(args, identity):
    """Validate static saved/live fields without accessing device initialization."""
    from .factorial_v3_trainer import FactorialTrainingArguments, require_requested_arguments

    if type(args) is not FactorialTrainingArguments:
        raise ValueError("Run requires its exact factorial argument class")
    require_requested_arguments(args)
    for key, value in {
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "use_cpu": False,
        "seed": identity["seed"],
        "data_seed": identity["seed"],
        "run_name": identity["run_id"],
        "learning_rate": identity["optimizer"]["lr"],
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataloader_num_workers": 8,
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": True,
        "dataloader_prefetch_factor": 4,
        "save_on_each_node": False,
        "save_total_limit": None,
        "train_sampling_strategy": "random",
    }.items():
        require_same(getattr(args, key), value)


def require_source_checkpoint(identity):
    require_identity(identity)
    source = identity["source"]
    path = Path(source["checkpoint"])
    declared = read_json(path / "dense_run_contract.json")
    actual = inspect_sealed_checkpoint(path, declared, 2345)
    native = source["native_checkpoint"]
    require_same(
        actual, {k: native[k] for k in ("step", "run_identity_sha256", "checkpoint_seal", "files")}
    )
    return actual


def prepare_run(locations, calibration_directories, state, operator, seed, source_root):
    """Read both real calibration chains before preparing one of twelve run identities."""
    sources = source_identity(locations.repository, source_root)
    inputs = load_inputs(locations)
    if set(calibration_directories) != set(STATES):
        raise ValueError("Both actual calibration outputs are required")
    completed = {
        label: calibration.read_calibration(
            inputs,
            label,
            calibration_directories[label]["path"],
            calibration_directories[label]["calibration_binding"],
        )
        for label in STATES
    }
    cells = calibration.branch_cells(inputs, completed)
    matched = [
        c for c in cells if (c["state"], c["operator"], c["seed"]) == (state, operator, seed)
    ]
    if len(matched) != 1:
        raise ValueError("Require one unique declared state/operator/order-seed cell")
    dataset, view = branch_dataset(inputs)
    return make_identity(inputs, matched[0], view, sources), dataset


def require_component(component, identity):
    require_identity(identity)
    if (
        component.get("scope") != "dense-v3-factorial-trainer-component-v1"
        or component.get("scientific_admission") is not False
        or component.get("cpu_diagnostic") is not False
    ):
        raise ValueError("A CPU/legacy/unbound component is not a genuine factorial run")
    require_same(component["bound_factorial_run"], identity)
    require_same(component["optimizer"], identity["optimizer"])
    require_same(component["layout"], identity["source"]["named_layout"])
    for key in ("dataset_fingerprint", "dataset_columns", "dataset_rows"):
        source_key = {"dataset_columns": "columns", "dataset_rows": "rows"}.get(key, key)
        require_same(component[key], identity["data"][source_key])
    for key, value in {
        "seed": identity["seed"],
        "data_seed": identity["seed"],
        "world_size": 4,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "batching": batches.SCOPE,
    }.items():
        require_same(component[key], value)
    require_same(
        component["precision"], {"bf16": True, "fp16": False, "tf32": True, "use_cpu": False}
    )
    require_same(
        component["normalization"],
        {
            "owner": "Trainer.training_step/current_gradient_accumulation_steps",
            "trainer_scheduled_accumulation": 4,
            "accelerator_divisor_before": 4,
            "accelerator_divisor_after": 1,
        },
    )
    if set(component["sources"]) != set(COMPONENT_SOURCES):
        raise ValueError("Saved component does not bind every original numerical source")
    for name, value in component["sources"].items():
        if value != identity["sources"][f"src/embed_optim/{name}"]["sha256"]:
            raise ValueError("Saved component numerical source differs from this run")
    require_same(
        component["bound_run_source"],
        identity["sources"]["src/embed_optim/factorial_v3_bound_trainer.py"],
    )
    from .runtime import load_runtime_spec, runtime_problems

    verify_file(identity["runtime_spec"]["path"], identity["runtime_spec"])
    if runtime_problems(load_runtime_spec(identity["runtime_spec"]["path"]), component["runtime"]):
        raise ValueError("Saved run component does not have the fixed package/runtime versions")


def inspect_model_weights(path):
    """Read every actual DenseOn tensor, without claiming a run or GPU execution."""
    import torch
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as store:
        shapes = {n.removeprefix("0.model."): shape for n, shape in optim.denseon_shapes().items()}
        if set(store.keys()) != set(shapes):
            raise ValueError("Checkpoint does not have all 134 DenseOn tensors")
        for name in store.keys():
            tensor = store.get_tensor(name)
            if (
                list(tensor.shape) != shapes[name]
                or tensor.dtype != torch.float32
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError("Invalid saved DenseOn weight shape/dtype/value")
    return {
        "tensors": 134,
        "parameters": sum(math.prod(s) for s in shapes.values()),
        "dtype": "torch.float32",
    }


def inspect_rank_rng(rng):
    """Validate in temporary CPU RNG objects only; do not change global RNG/device state."""
    import random

    import numpy as np
    import torch

    if not isinstance(rng, dict) or not {"python", "numpy", "cpu", "cuda"} <= set(rng):
        raise ValueError("Missing genuine GPU rank-local RNG state")
    if (
        not torch.is_tensor(rng["cpu"])
        or rng["cpu"].dtype != torch.uint8
        or rng["cpu"].ndim != 1
        or not rng["cpu"].numel()
    ):
        raise ValueError("Invalid CPU RNG state")
    random.Random().setstate(rng["python"])
    np.random.RandomState().set_state(rng["numpy"])
    torch.Generator(device="cpu").set_state(rng["cpu"])
    cuda = rng["cuda"] if isinstance(rng["cuda"], list) else [rng["cuda"]]
    if not cuda or any(
        not torch.is_tensor(t) or t.dtype != torch.uint8 or t.ndim != 1 or not t.numel()
        for t in cuda
    ):
        raise ValueError("Invalid CUDA RNG state")


def require_final_state(state):
    for key, expected in {
        "global_step": 391,
        "max_steps": 391,
        "num_train_epochs": 1,
        "train_batch_size": 8,
    }.items():
        require_same(state[key], expected)
    if type(state.get("epoch")) not in (int, float) or state["epoch"] != 1.0:
        raise ValueError("Final Trainer state has not consumed its complete epoch")


def inspect_checkpoint(binding, identity, expected_component):
    """Read a complete native save and every named model/moment tensor on CPU."""
    import torch

    require_component(expected_component, identity)
    payload = checkpoints.read(binding, expected_component)
    step = payload["step"]
    if step not in STEPS:
        raise ValueError("Checkpoint is not a declared fifth of the fixed branch")
    path = Path(binding["path"])
    config = OptimizerConfig(**identity["optimizer"])
    saved_optimizer = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
    scheduler = torch.load(path / "scheduler.pt", map_location="cpu", weights_only=True)
    state = optim.inspect_state(saved_optimizer, expected_component["layout"], config, step=step)
    optim.inspect_scheduler(scheduler, saved_optimizer, config, step=step)
    inspect_model_weights(path / "model.safetensors")
    args = torch.load(path / "training_args.bin", map_location="cpu", weights_only=False)
    require_arguments(args, identity)
    # Authentication precedes decoding rank-local RNG state. It is never installed
    # into the reader's RNGs or used to initialize a device.
    for rank in range(4):
        rng = torch.load(path / f"rng_state_{rank}.pth", map_location="cpu", weights_only=False)
        inspect_rank_rng(rng)
    checkpoints.read(binding, expected_component)
    return {
        "step": step,
        "component_binding": copy.deepcopy(binding),
        "optimizer": state,
        "model_tensors": 134,
        "rank_rng_states": 4,
        "run_identity_sha256": digest(identity),
    }


def inspect_complete_run(root, binding, identity):
    """All five declared stages and the exact final inference weights are required."""
    require_identity(identity)
    require_run_dependencies(identity)
    root = Path(root)
    verify_file(root / RUN_COMPLETE, binding)
    completed = read_json(root / RUN_COMPLETE)
    require_same(read_json(root / RUN_NAME), identity)
    require_same(completed["run_identity"], identity)
    if (
        completed.get("scope") != SCOPE
        or completed.get("status") != "training_complete"
        or completed.get("scientific_admission") is not False
        or completed.get("optimizer_steps") != 391
        or completed.get("dataset_rows") != 50000
    ):
        raise ValueError("Incomplete or mis-scoped factorial run")
    component = completed["component_identity"]
    require_component(component, identity)
    saved = completed["checkpoints"]
    if len(saved) != 5 or sorted(p.name for p in root.glob("checkpoint-*")) != sorted(
        f"checkpoint-{s}" for s in STEPS
    ):
        raise ValueError("Run must retain exactly its five declared checkpoints")
    observations = []
    for step, item in zip(STEPS, saved, strict=True):
        if item["step"] != step or item["directory"] != f"checkpoint-{step}":
            raise ValueError("Checkpoint receipt order/stage differs")
        native = {"path": str(root / item["directory"]), "sha256": item["sha256"]}
        observations.append(inspect_checkpoint(native, identity, component))
    last = {r["path"]: r for r in checkpoints.inventory(root / "checkpoint-391")}
    final_files = completed["final_inference_files"]
    if not final_files or len({r["path"] for r in final_files}) != len(final_files):
        raise ValueError("Missing or duplicate final inference inventory")
    from .primary_contract import relative_path

    for row in final_files:
        relative_path(row["path"])
        require_same(row, last[row["path"]])
        verify_file(root / "final" / row["path"], row)
    if {r["path"] for r in final_files} != set(INFERENCE_FILES):
        raise ValueError("Incomplete final inference package")
    final_inventory = checkpoints.inventory(root / "final")
    require_same(final_inventory, completed["final_files"])
    if {r["path"] for r in final_inventory} != set(INFERENCE_FILES) | {
        "README.md",
        "training_args.bin",
    }:
        raise ValueError("Unexpected or missing final package payload")
    import torch

    require_arguments(
        torch.load(root / "final/training_args.bin", map_location="cpu", weights_only=False),
        identity,
    )
    verify_file(root / "trainer_state_final.json", completed["trainer_state_final"])
    require_final_state(read_json(root / "trainer_state_final.json"))
    require_same(
        read_json(root / "trainer_state_final.json")["global_step"], observations[-1]["step"]
    )
    verify_file(root / RUN_COMPLETE, binding)
    return {
        "scope": SCOPE,
        "whole_run_artifacts_verified": True,
        "scientific_admission": False,
        "run_identity_sha256": digest(identity),
        "checkpoints": observations,
        "final_inference_files": copy.deepcopy(final_files),
        "optimizer_steps": 391,
        "dataset_rows": 50000,
    }
