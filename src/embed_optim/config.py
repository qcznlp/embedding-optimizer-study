from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Literal

import yaml

ModelFamily = Literal["dense", "late"]
OptimizerName = Literal["adamw", "muon", "normuon"]


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    name: OptimizerName
    lr: float
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    momentum: float = 0.95
    normuon_beta2: float = 0.95
    aux_lr: float = 3e-6
    aux_beta1: float = 0.9
    aux_beta2: float = 0.999
    aux_eps: float = 1e-8
    ns_steps: int = 5
    adjust_lr_fn: str = "original"

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "OptimizerConfig":
        return cls(**values)


@dataclasses.dataclass(frozen=True)
class RunConfig:
    run_id: str
    model_family: ModelFamily
    optimizer: OptimizerConfig
    model_name: str
    dataset_path: str
    model_revision: str | None = None
    output_root: str = "outputs"
    seed: int = 42
    epochs: float = 1.0
    global_batch_size: int = 128
    micro_batch_size: int = 8
    temperature: float | None = None
    max_length: int = 8192
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    dataloader_workers: int = 8
    gradient_checkpointing: bool = True
    flash_attention: bool = True
    wandb_project: str = "embedding-optimizer-study"
    wandb_entity: str | None = None
    checkpoint_fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)

    @property
    def resolved_temperature(self) -> float:
        if self.temperature is not None:
            return self.temperature
        return 0.02 if self.model_family == "dense" else 0.001

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.model_family / self.run_id

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "RunConfig":
        values = dict(values)
        values["optimizer"] = OptimizerConfig.from_dict(values["optimizer"])
        if "checkpoint_fractions" in values:
            values["checkpoint_fractions"] = tuple(values["checkpoint_fractions"])
        return cls(**values)


def load_matrix(path: str | Path) -> list[RunConfig]:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    common = raw.get("common", {})
    models = raw["models"]
    runs: list[RunConfig] = []
    for model_family, model_values in models.items():
        for optimizer_values in raw["optimizers"]:
            opt = dict(optimizer_values)
            run_id = opt.pop("id")
            values = {
                **common,
                **model_values,
                "run_id": run_id,
                "model_family": model_family,
                "optimizer": opt,
            }
            runs.append(RunConfig.from_dict(values))
    return runs


def save_resolved_config(config: RunConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n")
