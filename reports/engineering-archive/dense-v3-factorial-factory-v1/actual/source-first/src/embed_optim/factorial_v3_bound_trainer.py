"""Bind genuine source/calibration/data identity inside every native Trainer save.

This is an internal GPU component, not a launcher or permission to take a device.
The inherited training step, loss, batching, optimizer and save/resume math stay
unchanged. Actual four-GPU admission remains a separate required experiment.
"""

from __future__ import annotations

import copy
from pathlib import Path

from . import factorial_v3_run_contract as contract
from .callbacks import FractionalCheckpointCallback
from .collators import DenseGroupCollator
from .factorial_v3_calibration import _verify_loaded_weights
from .factorial_v3_trainer import FactorialTrainer
from .primary_contract import file_identity


def input_handlers(model, output_dir):
    """Use the unchanged primary prompt collator and exact five-fraction callback."""
    return DenseGroupCollator(model.preprocess), FractionalCheckpointCallback(
        tuple(contract.CHECKPOINT_POLICY["fractions"]), output_dir
    )


def require_collator(collator, model):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.data_collator import (
        SentenceTransformerDataCollator,
    )

    if (
        type(collator) is not DenseGroupCollator
        or type(collator.inner) is not SentenceTransformerDataCollator
    ):
        raise ValueError("Run changed the declared explicit prompt collator")
    actual = collator.inner.preprocess_fn
    if (
        getattr(actual, "__self__", None) is not model
        or getattr(actual, "__func__", None) is not SentenceTransformer.preprocess
    ):
        raise ValueError("Collator does not preprocess through the same unmodified model")
    if collator.inner.prompts != contract.COLLATOR_POLICY["prompts"]:
        raise ValueError("Run changed query/document prompt routing")


def collective_phase(action):
    """Propagate ordinary per-rank admission/IO failures before the next phase.

    This is not recovery from a killed rank, failed device or broken collective.
    Such failures still require the separately owned launcher's failure handling.
    """
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() != 4:
        raise ValueError("Finalization requires the existing four-rank process group")
    value, error = None, None
    try:
        value = action()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [None] * 4
    dist.all_gather_object(errors, error)
    if any(e is not None for e in errors):
        raise ValueError(f"Factorial finalization phase failed; preserve partial outputs: {errors}")
    return value


class RunBoundFactorialTrainer(FactorialTrainer):
    def __init__(self, *, run_identity, **kwargs):
        self._bound_run = contract.require_identity(run_identity)
        args, model, data = (kwargs.get(k) for k in ("args", "model", "train_dataset"))
        if kwargs.get("diagnostic_shapes") is not None or args.use_cpu:
            raise ValueError("Genuine run binding cannot admit a CPU/toy diagnostic")
        if kwargs.get("data_collator") is not None or kwargs.get("callbacks"):
            raise ValueError("The bound run owns its prompt collator and checkpoint callback")
        self._bound_runtime = contract.require_run_dependencies(self._bound_run)
        contract.require_arguments(args, self._bound_run)
        contract.require_dataset(data, self._bound_run)
        # A fresh constructor loads the original source even when train() will
        # subsequently restore an externally bound same-run checkpoint.
        contract.require_source_checkpoint(self._bound_run)
        _verify_loaded_weights(model, self._bound_run["source"]["checkpoint"])
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError("Run constructor inherited calibration gradients")
        if model[0].model.config._attn_implementation != "flash_attention_2":
            raise ValueError("Run did not load its declared GPU attention backend")
        collator, self._fractional_callback = input_handlers(model, args.output_dir)
        require_collator(collator, model)
        kwargs["data_collator"] = collator
        kwargs["callbacks"] = [self._fractional_callback]
        super().__init__(**kwargs)
        contract.require_component(self.component_identity(), self._bound_run)

    def component_identity(self):
        contract.require_arguments(self.args, self._bound_run)
        require_collator(self.data_collator, self.model)
        callback = self._fractional_callback
        if (
            callback not in self.callback_handler.callbacks
            or type(callback) is not FractionalCheckpointCallback
            or list(callback.fractions) != contract.CHECKPOINT_POLICY["fractions"]
            or callback.output_dir != Path(self.args.output_dir)
            or callback.targets not in ((), contract.STEPS)
        ):
            raise ValueError("Run changed its five-checkpoint callback policy")
        value = super().component_identity()
        value["bound_factorial_run"] = copy.deepcopy(self._bound_run)
        value["bound_run_source"] = file_identity(Path(__file__))
        value["runtime"] = copy.deepcopy(self._bound_runtime)
        value["input_collator"] = copy.deepcopy(contract.COLLATOR_POLICY)
        value["checkpoint_callback"] = copy.deepcopy(contract.CHECKPOINT_POLICY)
        return value

    def save_run_completion(self):
        """Save the final package/receipt only after the actual fixed horizon.

        Call on all four ranks. This writes new artifacts only and leaves the
        five native component seals untouched. The independent whole-run reader
        still has to inspect the resulting content-bound completion receipt.
        """
        import dataclasses
        import json

        import torch.distributed as dist

        from . import factorial_v3_checkpoint as checkpoints
        from .primary_contract import verify_file

        root = Path(self.args.output_dir)
        final = root / "final"

        def preflight():
            identity = self._require_identity()
            contract.require_component(identity, self._bound_run)
            contract.require_final_state(dataclasses.asdict(self.state))
            if self._raw_optimizer().completed_steps != 391:
                raise ValueError("Cannot finalize a partial factorial run")
            for path in (
                final,
                root / contract.RUN_NAME,
                root / contract.RUN_COMPLETE,
                root / "trainer_state_final.json",
            ):
                if path.exists() or path.is_symlink():
                    raise ValueError("Do not overwrite an earlier final export or receipt")
            if sorted(p.name for p in root.glob("checkpoint-*")) != sorted(
                f"checkpoint-{s}" for s in contract.STEPS
            ):
                raise ValueError("Cannot finalize without exactly five declared checkpoints")
            return identity

        identity = collective_phase(preflight)
        collective_phase(lambda: self.save_model(str(final)))

        def write_completion():
            if not self.is_world_process_zero():
                return None

            def save_new(path, payload):
                with path.open("x") as stream:
                    stream.write(
                        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
                    )

            saved = []
            for step in contract.STEPS:
                directory = root / f"checkpoint-{step}"
                native = {
                    "path": str(directory.resolve()),
                    "sha256": checkpoints.file_digest(directory / checkpoints.NAME),
                }
                checkpoints.read(native, identity)
                saved.append(
                    {"step": step, "directory": directory.name, "sha256": native["sha256"]}
                )
            final_state = root / "trainer_state_final.json"
            save_new(final_state, dataclasses.asdict(self.state))
            inventory = {r["path"]: r for r in checkpoints.inventory(root / "checkpoint-391")}
            final_files = []
            for name in contract.INFERENCE_FILES:
                verify_file(final / name, inventory[name])
                final_files.append(inventory[name])
            final_inventory = contract.final_inventory(final)
            save_new(root / contract.RUN_NAME, self._bound_run)
            receipt = {
                "scope": contract.SCOPE,
                "status": "training_complete",
                "scientific_admission": False,
                "run_identity": self._bound_run,
                "component_identity": identity,
                "optimizer_steps": 391,
                "dataset_rows": 50000,
                "checkpoints": saved,
                "final_inference_files": final_files,
                "final_files": final_inventory,
                "trainer_state_final": file_identity(final_state),
            }
            save_new(root / contract.RUN_COMPLETE, receipt)
            return file_identity(root / contract.RUN_COMPLETE)

        binding = collective_phase(write_completion)
        observed = [binding]
        dist.broadcast_object_list(observed, src=0)
        dist.barrier()
        return observed[0]
