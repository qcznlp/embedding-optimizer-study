from __future__ import annotations

import json
import math
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


def sanitize_pylate_checkpoint(path: str | Path) -> None:
    """Remove ST 5 fields that PyLate 1.6's Dense loader cannot accept."""

    path = Path(path)
    for config_path in path.glob("*_Dense/config.json"):
        config = json.loads(config_path.read_text())
        changed = False
        for key in ("module_input_name", "module_output_name"):
            changed |= config.pop(key, None) is not None
        if changed:
            config_path.write_text(json.dumps(config, indent=4) + "\n")


class FractionalCheckpointCallback(TrainerCallback):
    """Request checkpoints at exact fractions of the realized optimizer steps."""

    def __init__(self, fractions: tuple[float, ...], output_dir: str | Path) -> None:
        if not fractions or any(not 0 < fraction <= 1 for fraction in fractions):
            raise ValueError(f"Checkpoint fractions must lie in (0, 1], got {fractions}")
        self.fractions = tuple(sorted(set(fractions)))
        self.output_dir = Path(output_dir)
        self.targets: tuple[int, ...] = ()
        self.requested: set[int] = set()

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        del kwargs
        self.targets = tuple(
            sorted(
                set(
                    min(state.max_steps, math.ceil(state.max_steps * fraction))
                    for fraction in self.fractions
                )
            )
        )
        self.requested.update(target for target in self.targets if target <= state.global_step)
        if args is None or args.process_index == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "checkpoint_schedule.json").write_text(
                json.dumps(
                    {
                        "max_steps": state.max_steps,
                        "fractions": self.fractions,
                        "steps": self.targets,
                    },
                    indent=2,
                )
                + "\n"
            )
        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        del args, kwargs
        if state.global_step in self.targets and state.global_step not in self.requested:
            control.should_save = True
            self.requested.add(state.global_step)
        return control


class PyLateCheckpointCompatibilityCallback(TrainerCallback):
    """Make each freshly saved PyLate checkpoint directly reloadable."""

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        del kwargs
        if args.process_index == 0:
            sanitize_pylate_checkpoint(Path(args.output_dir) / f"checkpoint-{state.global_step}")
        return control


class WandbExperimentConfigCallback(TrainerCallback):
    """Attach the resolved research configuration to the Trainer-created run."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        del state, kwargs
        if args.process_index == 0:
            import wandb

            if wandb.run is not None:
                wandb.config.update(self.payload, allow_val_change=True)
        return control
