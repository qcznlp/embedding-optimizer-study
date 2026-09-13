"""Bind genuine source/calibration/data identity inside every native Trainer save.

This is an internal GPU component, not a launcher or permission to take a device.
The inherited training step, loss, batching, optimizer and save/resume math stay
unchanged. Actual four-GPU admission remains a separate required experiment.
"""

from __future__ import annotations

import copy
from pathlib import Path

from . import factorial_v3_run_contract as contract
from .factorial_v3_calibration import _verify_loaded_weights
from .factorial_v3_trainer import FactorialTrainer
from .primary_contract import file_identity, require_same


class RunBoundFactorialTrainer(FactorialTrainer):
    def __init__(self, *, run_identity, **kwargs):
        self._bound_run = contract.require_identity(run_identity)
        args, model, data = (kwargs.get(k) for k in ("args", "model", "train_dataset"))
        if kwargs.get("diagnostic_shapes") is not None or args.use_cpu:
            raise ValueError("Genuine run binding cannot admit a CPU/toy diagnostic")
        if data._fingerprint != self._bound_run["data"][
            "dataset_fingerprint"
        ] or data.column_names != list(contract.TEXT_COLUMNS):
            raise ValueError("Trainer received a different actual branch Dataset view")
        for key, value in {
            "bf16": True,
            "fp16": False,
            "tf32": True,
            "seed": self._bound_run["seed"],
            "max_grad_norm": 1.0,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "dataloader_num_workers": 8,
            "dataloader_pin_memory": True,
            "dataloader_persistent_workers": True,
            "dataloader_prefetch_factor": 4,
        }.items():
            require_same(getattr(args, key), value)
        # A fresh constructor loads the original source even when train() will
        # subsequently restore an externally bound same-run checkpoint.
        contract.require_source_checkpoint(self._bound_run)
        _verify_loaded_weights(model, self._bound_run["source"]["checkpoint"])
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError("Run constructor inherited calibration gradients")
        if model[0].model.config._attn_implementation != "flash_attention_2":
            raise ValueError("Run did not load its declared GPU attention backend")
        super().__init__(**kwargs)
        contract.require_component(self.component_identity(), self._bound_run)

    def component_identity(self):
        value = super().component_identity()
        value["bound_factorial_run"] = copy.deepcopy(self._bound_run)
        value["bound_run_source"] = file_identity(Path(__file__))
        return value

    def save_run_completion(self):
        """Save the final package/receipt only after the actual fixed horizon.

        Call on all four ranks. This writes new artifacts only and leaves the
        five native component seals untouched. The independent whole-run reader
        still has to inspect the resulting content-bound completion receipt.
        """
        import json
        import torch.distributed as dist
        from . import factorial_v3_checkpoint as checkpoints

        identity = self._require_identity()
        contract.require_component(identity, self._bound_run)
        if self.state.global_step != 391 or self._raw_optimizer().completed_steps != 391:
            raise ValueError("Cannot finalize a partial factorial run")
        root = Path(self.args.output_dir)
        final = root / "final"
        if final.exists() or final.is_symlink():
            raise ValueError("Do not overwrite an earlier final export")
        dist.barrier()
        self.save_model(str(final))
        dist.barrier()
        binding = None
        if self.is_world_process_zero():

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
            if final_state.exists():
                raise ValueError("Do not overwrite a previous final Trainer state")
            self.state.save_to_json(str(final_state))
            inventory = {r["path"]: r for r in checkpoints.inventory(root / "checkpoint-391")}
            final_files = []
            for name in contract.INFERENCE_FILES:
                from .primary_contract import verify_file

                verify_file(final / name, inventory[name])
                final_files.append(inventory[name])
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
                "trainer_state_final": file_identity(final_state),
            }
            save_new(root / contract.RUN_COMPLETE, receipt)
            binding = file_identity(root / contract.RUN_COMPLETE)
        observed = [binding]
        dist.broadcast_object_list(observed, src=0)
        dist.barrier()
        return observed[0]
