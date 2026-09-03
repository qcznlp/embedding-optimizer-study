from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
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


class StopAfterStepCallback(TrainerCallback):
    """Stop a diagnostic replay without shortening its scheduler horizon."""

    def __init__(self, target_step: int) -> None:
        if target_step <= 0:
            raise ValueError(f"Stop step must be positive, got {target_step}")
        self.target_step = target_step

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        del args, kwargs
        if state.global_step >= self.target_step:
            raise ValueError(
                f"Stop step {self.target_step} must be after resumed step {state.global_step}"
            )
        if self.target_step > state.max_steps:
            raise ValueError(
                f"Stop step {self.target_step} exceeds training horizon {state.max_steps}"
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
        if state.global_step >= self.target_step:
            control.should_save = True
            control.should_training_stop = True
        return control


class AcceptedTimingCallback(TrainerCallback):
    """Persist non-overlapping wall-time segments at durable checkpoint boundaries."""

    def __init__(self, output_dir: str | Path) -> None:
        self.path = Path(output_dir) / "accepted_timing.json"
        self.started_monotonic: float | None = None
        self.started_at_utc: str | None = None
        self.start_step: int | None = None

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _read(self) -> dict:
        if not self.path.is_file():
            return {
                "schema_version": 1,
                "segments": [],
                "total_wall_time_seconds_max_rank": 0.0,
            }
        payload = json.loads(self.path.read_text())
        if payload.get("schema_version") != 1 or not isinstance(payload.get("segments"), list):
            raise RuntimeError(f"Invalid accepted timing ledger: {self.path}")
        return payload

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        del kwargs
        current_step = int(state.global_step)
        if args.process_index == 0:
            payload = self._read()
            if payload["segments"]:
                last_step = int(payload["segments"][-1]["end_step_inclusive"])
                if last_step > current_step:
                    raise RuntimeError(
                        f"Accepted timing ledger ends at step {last_step}, "
                        f"after resumed step {current_step}"
                    )
        self.started_monotonic = time.monotonic()
        self.started_at_utc = self._utc_now()
        self.start_step = current_step
        return control

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        del kwargs
        if self.started_monotonic is None or self.started_at_utc is None or self.start_step is None:
            raise RuntimeError("Accepted timing callback did not observe train begin")
        end_step = int(state.global_step)
        if end_step <= self.start_step:
            return control
        completed_monotonic = time.monotonic()
        completed_at_utc = self._utc_now()
        wall_time_seconds = completed_monotonic - self.started_monotonic
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            duration = torch.tensor(wall_time_seconds, dtype=torch.float64, device=args.device)
            torch.distributed.all_reduce(duration, op=torch.distributed.ReduceOp.MAX)
            wall_time_seconds = float(duration.item())
        if args.process_index != 0:
            self.started_monotonic = completed_monotonic
            self.started_at_utc = completed_at_utc
            self.start_step = end_step
            return control
        payload = self._read()
        segments = payload["segments"]
        if segments:
            last_step = int(segments[-1]["end_step_inclusive"])
            if last_step != self.start_step:
                raise RuntimeError(
                    f"Accepted timing ledger ends at step {last_step}, "
                    f"but this segment starts at {self.start_step}"
                )
        segment = {
            "start_step_exclusive": self.start_step,
            "end_step_inclusive": end_step,
            "started_at_utc": self.started_at_utc,
            "checkpoint_completed_at_utc": completed_at_utc,
            "wall_time_seconds_max_rank": wall_time_seconds,
        }
        segments.append(segment)
        payload["total_wall_time_seconds_max_rank"] = sum(
            float(item["wall_time_seconds_max_rank"]) for item in segments
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(self.path)
        self.started_monotonic = completed_monotonic
        self.started_at_utc = completed_at_utc
        self.start_step = end_step
        return control


def accepted_timing_summary(path: str | Path, expected_final_step: int) -> dict:
    """Require a complete timing ledger before a run can write its completion marker."""

    path = Path(path)
    payload = json.loads(path.read_text())
    segments = payload.get("segments")
    if payload.get("schema_version") != 1 or not isinstance(segments, list) or not segments:
        raise RuntimeError(f"Invalid accepted timing ledger: {path}")
    if int(segments[-1]["end_step_inclusive"]) != expected_final_step:
        raise RuntimeError(
            f"Accepted timing ledger ends at {segments[-1].get('end_step_inclusive')}, "
            f"expected {expected_final_step}"
        )
    total = sum(float(segment["wall_time_seconds_max_rank"]) for segment in segments)
    recorded_total = float(payload["total_wall_time_seconds_max_rank"])
    if (
        not math.isfinite(total)
        or total <= 0
        or not math.isclose(recorded_total, total, rel_tol=1e-9, abs_tol=1e-6)
    ):
        raise RuntimeError(f"Accepted timing ledger total does not match its segments: {path}")
    return {
        "schema_version": 1,
        "segments": len(segments),
        "total_wall_time_seconds_max_rank": total,
    }


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
