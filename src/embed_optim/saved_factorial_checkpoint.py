"""Read authenticated saved factorial checkpoints using explicit local paths.

This is historical artifact inspection, not fresh-run source admission, training,
resume equivalence or scientific completion. The saved identities are preserved.
Only the runtime specification's *read location* is provided locally; its pinned
bytes and the original saved runtime/identity comparisons remain mandatory.
Checkpoint receipts must be authenticated by a trusted external digest before
any pickle-backed training arguments or RNG state may be decoded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from . import factorial_v3_batches as batches
from . import factorial_v3_checkpoint as checkpoints
from . import factorial_v3_optimizer as optim
from .config import OptimizerConfig
from .factorial_v3_run_contract import (
    CHECKPOINT_POLICY,
    COLLATOR_POLICY,
    COMPONENT_SOURCES,
    STEPS,
    _require_hex,
    inspect_model_weights,
    inspect_rank_rng,
    require_arguments,
    require_final_state,
    require_identity,
)
from .primary_contract import digest, file_identity, read_json, require_same, verify_file
from .runtime import verify_runtime_spec

SCOPE = "dense-v3-authenticated-saved-factorial-checkpoint-read-v1"


def _ordinary_path(value):
    path = Path(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or any(part.is_symlink() for part in (path, *path.parents))
    ):
        raise ValueError("Require an explicit absolute local path without symlinks")
    return path


def _strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate saved receipt key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Nonfinite saved receipt JSON")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def authenticated_component(checkpoint, expected_sha256, runtime_spec):
    """Authenticate the complete envelope and local runtime before tensor decoding."""
    path, runtime_spec = _ordinary_path(checkpoint), _ordinary_path(runtime_spec)
    _require_hex(expected_sha256, 64)
    manifest = path / checkpoints.NAME
    before = file_identity(manifest)
    if before["sha256"] != expected_sha256:
        raise ValueError("External checkpoint receipt digest differs")
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Checkpoint receipt changed during reading")
    payload = _strict_json(raw)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "scope",
        "scientific_admission",
        "identity",
        "step",
        "files",
    }:
        raise ValueError("Incomplete saved checkpoint envelope")
    component = payload["identity"]
    if not isinstance(component, dict) or not isinstance(
        component.get("bound_factorial_run"), dict
    ):
        raise ValueError("Missing original bound factorial run")
    identity = component["bound_factorial_run"]
    require_component(component, identity, runtime_spec_path=runtime_spec)
    binding = {"path": str(path), "sha256": expected_sha256}
    checked = checkpoints.read(binding, component)
    if checked != payload or file_identity(manifest) != before:
        raise ValueError("Saved checkpoint changed during authentication")
    return binding, identity, component, runtime_spec


def read_checkpoint(checkpoint, expected_sha256, runtime_spec):
    """Decode only a trusted complete save; never load an old producer path."""
    binding, identity, component, local_runtime = authenticated_component(
        checkpoint, expected_sha256, runtime_spec
    )
    reader_runtime = verify_runtime_spec(local_runtime)
    native = inspect_checkpoint(binding, identity, component, runtime_spec_path=local_runtime)
    runtime_binding = verify_file(local_runtime, identity["runtime_spec"])
    # Native returns the original run digest. Do not substitute a relocated identity.
    return {
        "scope": SCOPE,
        "native": native,
        "runtime_spec": {"path": str(local_runtime), **runtime_binding},
        "reader_runtime": reader_runtime,
        "original_run_identity_sha256": digest(identity),
        "original_component_identity_sha256": digest(component),
        "saved_identities_unchanged": True,
        "checkpoint_tensor_integrity_verified": True,
        "source_training_reexecuted": False,
        "fresh_run_source_admission": False,
        "gpu_resume_equivalence": False,
        "scientific_completion": False,
    }


def require_component(component, identity, *, runtime_spec_path):
    require_identity(identity)
    if (
        component.get("scope") != "dense-v3-factorial-trainer-component-v1"
        or component.get("scientific_admission") is not False
        or component.get("cpu_diagnostic") is not False
    ):
        raise ValueError("A CPU/legacy/unbound component is not a genuine factorial run")
    require_same(component["bound_factorial_run"], identity)
    require_same(component["input_collator"], COLLATOR_POLICY)
    require_same(component["checkpoint_callback"], CHECKPOINT_POLICY)
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

    verify_file(runtime_spec_path, identity["runtime_spec"])
    if runtime_problems(load_runtime_spec(runtime_spec_path), component["runtime"]):
        raise ValueError("Saved run component does not have the fixed package/runtime versions")


def inspect_checkpoint(binding, identity, expected_component, *, runtime_spec_path):
    """Read a complete native save and every named model/moment tensor on CPU."""
    import torch

    require_component(expected_component, identity, runtime_spec_path=runtime_spec_path)
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
    if step == STEPS[-1]:
        require_final_state(read_json(path / "trainer_state.json"))
    checkpoints.read(binding, expected_component)
    return {
        "step": step,
        "component_binding": copy.deepcopy(binding),
        "optimizer": state,
        "model_tensors": 134,
        "rank_rng_states": 4,
        "run_identity_sha256": digest(identity),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-receipt-sha256",
        required=True,
        help="Trusted external digest, not a hash supplied by an untrusted checkpoint.",
    )
    parser.add_argument("--runtime-spec", type=Path, required=True)
    args = parser.parse_args(argv)
    result = read_checkpoint(args.checkpoint, args.expected_receipt_sha256, args.runtime_spec)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
