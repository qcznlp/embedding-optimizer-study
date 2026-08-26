from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Literal

import yaml

ModelFamily = Literal["dense", "late"]
OptimizerName = Literal["adamw", "hybrid_adamw", "muon", "normuon"]
MUON_NS_IMPLEMENTATION = "unfused-bfloat16-v1"


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
        values = dict(values)
        implementation = values.pop("ns_implementation", None)
        config = cls(**values)
        if implementation is not None and (
            config.name not in {"muon", "normuon"} or implementation != MUON_NS_IMPLEMENTATION
        ):
            raise ValueError(f"Unsupported optimizer implementation {implementation!r}")
        return config


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
        values = dataclasses.asdict(self)
        if self.optimizer.name in {"muon", "normuon"}:
            values["optimizer"]["ns_implementation"] = MUON_NS_IMPLEMENTATION
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "RunConfig":
        values = dict(values)
        values["optimizer"] = OptimizerConfig.from_dict(values["optimizer"])
        if "checkpoint_fractions" in values:
            values["checkpoint_fractions"] = tuple(values["checkpoint_fractions"])
        return cls(**values)


def _resolve_matrix_path(path: str | Path, prefix: Path | None = None) -> Path:
    """Resolve a source-tree config, falling back to wheel data for bundled defaults."""

    path = Path(path)
    if path.is_file() or path.is_absolute() or path.parent != Path("configs"):
        return path
    prefix = Path(sys.prefix) if prefix is None else prefix
    installed = prefix / "share" / "embedding-optimizer-study" / "configs" / path.name
    return installed if installed.is_file() else path


def resolve_matrix_path(path: str | Path, prefix: Path | None = None) -> Path:
    return _resolve_matrix_path(path, prefix)


def load_matrix(path: str | Path) -> list[RunConfig]:
    path = resolve_matrix_path(path)
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


def matrix_runtime_spec(path: str | Path) -> Path | None:
    """Return a matrix's optional formal-runtime spec, resolved beside the matrix."""

    matrix_path = resolve_matrix_path(path)
    raw = yaml.safe_load(matrix_path.read_text())
    value = raw.get("formal_runtime")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid formal_runtime in {matrix_path}")
    runtime_path = Path(value)
    return runtime_path if runtime_path.is_absolute() else matrix_path.parent / runtime_path


def save_resolved_config(config: RunConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n")


def source_wandb_run_id(config: RunConfig) -> str:
    if config.optimizer.name in {"muon", "normuon"}:
        version = "v3"
    elif config.optimizer.name == "hybrid_adamw":
        version = "v4"
    else:
        version = "v2"
    suffix = f"-{MUON_NS_IMPLEMENTATION}" if version == "v3" else ""
    return f"study-{version}-{config.model_family}-{config.run_id}-seed{config.seed}{suffix}"
