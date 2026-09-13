from __future__ import annotations

import math
import re
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim import _functional as optim_f

from .config import MUON_NS_IMPLEMENTATION, OptimizerConfig

_NO_DECAY = re.compile(r"(?:^|\.)(?:bias|.*norm(?:\d+)?\.weight)$", re.IGNORECASE)


def parameter_partition_name(name: str, ndim: int) -> str:
    """Return the training-time optimizer partition for a named tensor.

    Keeping this rule independent of ``nn.Parameter`` lets offline checkpoint analyses reproduce
    the exact routing used during training without instantiating the model.
    """
    lowered = name.lower()
    if (
        ndim == 2
        and (lowered.startswith("layers.") or ".layers." in lowered)
        and "embedding" not in lowered
        and "classifier" not in lowered
        and "head" not in lowered
    ):
        return "hidden"
    if ndim >= 2 and _NO_DECAY.search(name) is None:
        return "aux_decay"
    return "aux_no_decay"


def _is_hidden_matrix(name: str, parameter: nn.Parameter) -> bool:
    return parameter_partition_name(name, parameter.ndim) == "hidden"


def _uses_weight_decay(name: str, parameter: nn.Parameter) -> bool:
    return parameter_partition_name(name, parameter.ndim) == "aux_decay"


def parameter_partition(model: nn.Module) -> dict[str, list[tuple[str, nn.Parameter]]]:
    result = {"hidden": [], "aux_decay": [], "aux_no_decay": []}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        result[parameter_partition_name(name, parameter.ndim)].append((name, parameter))
    return result


def partition_summary(
    partition: dict[str, list[tuple[str, nn.Parameter]]],
) -> dict[str, dict[str, int]]:
    return {
        key: {
            "tensors": len(values),
            "parameters": sum(parameter.numel() for _, parameter in values),
        }
        for key, values in partition.items()
    }


def _zeropower_via_newton_schulz(gradient: Tensor, steps: int, eps: float = 1e-7) -> Tensor:
    """Apply Muon's bfloat16 polynomial without the unstable fused ``torch.addmm`` path."""
    if gradient.ndim != 2:
        raise ValueError(f"Muon requires 2D tensors, got {tuple(gradient.shape)}")
    update = gradient.bfloat16()
    transposed = update.size(0) > update.size(1)
    if transposed:
        update = update.T
    update = update / update.norm().clamp_min(eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = update @ update.T
        update = a * update + (b * gram + c * gram @ gram) @ update
    return update.T if transposed else update


def _adjust_muon_lr(lr: float, adjust_lr_fn: str | None, shape: torch.Size) -> float:
    rows, columns = shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        ratio = math.sqrt(max(1, rows / columns))
    elif adjust_lr_fn == "match_rms_adamw":
        ratio = 0.2 * math.sqrt(max(rows, columns))
    else:
        ratio = 1.0
    return lr * ratio


def _muon_update(
    gradient: Tensor,
    momentum_buffer: Tensor,
    momentum: float,
    ns_steps: int,
) -> Tensor:
    momentum_buffer.lerp_(gradient, 1 - momentum)
    update = gradient.lerp(momentum_buffer, momentum)
    return _zeropower_via_newton_schulz(update, ns_steps)


def _normuon_update(
    gradient: Tensor,
    momentum_buffer: Tensor,
    second_moment: Tensor,
    momentum: float,
    beta2: float,
    ns_steps: int,
) -> Tensor:
    momentum_buffer.lerp_(gradient, 1 - momentum)
    update = gradient.lerp(momentum_buffer, momentum)
    update = _zeropower_via_newton_schulz(update, ns_steps).to(gradient.dtype)
    original_norm = update.norm()
    row_second_moment = torch.mean(update.square(), dim=-1, keepdim=True)
    second_moment.lerp_(row_second_moment, 1 - beta2)
    update.mul_(second_moment.sqrt().add_(1e-10).reciprocal())
    update.mul_(original_norm / update.norm().add_(1e-10))
    update.mul_(math.sqrt(max(1, gradient.size(0) / gradient.size(1))))
    return update


class EmbeddingOptimizer(Optimizer):
    """One checkpoint-friendly optimizer mixing Muon/NorMuon with fused AdamW."""

    def __init__(self, param_groups: list[dict]) -> None:
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            algorithm = group["algorithm"]
            if algorithm == "adamw":
                self._adamw_step(group)
            elif algorithm == "muon":
                self._muon_step(group)
            elif algorithm == "normuon":
                self._normuon_step(group)
            else:
                raise ValueError(f"Unknown parameter-group algorithm {algorithm!r}")
        return loss

    def _adamw_step(self, group: dict) -> None:
        params: list[Tensor] = []
        grads: list[Tensor] = []
        exp_avgs: list[Tensor] = []
        exp_avg_sqs: list[Tensor] = []
        state_steps: list[Tensor] = []
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = self.state[parameter]
            if not state:
                state["step"] = torch.zeros((), dtype=torch.float32, device=parameter.device)
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
            params.append(parameter)
            grads.append(parameter.grad)
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(state["step"])
        if not params:
            return
        fused = all(parameter.is_cuda for parameter in params)
        optim_f.adamw(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            [],
            state_steps,
            foreach=None if fused else True,
            fused=fused,
            capturable=False,
            differentiable=False,
            amsgrad=False,
            beta1=group["betas"][0],
            beta2=group["betas"][1],
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=False,
        )

    def _muon_step(self, group: dict) -> None:
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = self.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(parameter)
            update = _muon_update(
                parameter.grad,
                state["momentum_buffer"],
                momentum=group["momentum"],
                ns_steps=group["ns_steps"],
            )
            adjusted_lr = _adjust_muon_lr(group["lr"], group["adjust_lr_fn"], parameter.shape)
            parameter.mul_(1 - group["lr"] * group["weight_decay"])
            parameter.add_(update, alpha=-adjusted_lr)

    def _normuon_step(self, group: dict) -> None:
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = self.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(parameter)
                state["second_moment"] = torch.zeros_like(parameter[..., :1])
            update = _normuon_update(
                parameter.grad,
                state["momentum_buffer"],
                state["second_moment"],
                momentum=group["momentum"],
                beta2=group["beta2"],
                ns_steps=group["ns_steps"],
            )
            parameter.mul_(1 - group["lr"] * group["weight_decay"])
            parameter.add_(update, alpha=-group["lr"])


def build_optimizer(
    model: nn.Module, config: OptimizerConfig
) -> tuple[EmbeddingOptimizer, dict[str, dict[str, int]]]:
    partition = parameter_partition(model)
    summary = partition_summary(partition)
    all_parameters = sum(item["parameters"] for item in summary.values())
    if all_parameters == 0:
        raise ValueError("Model has no trainable parameters")

    if config.name == "adamw":
        decay = partition["hidden"] + partition["aux_decay"]
        groups = [
            {
                "params": [parameter for _, parameter in decay],
                "algorithm": "adamw",
                "lr": config.lr,
                "betas": (config.beta1, config.beta2),
                "eps": config.eps,
                "weight_decay": config.weight_decay,
            },
            {
                "params": [parameter for _, parameter in partition["aux_no_decay"]],
                "algorithm": "adamw",
                "lr": config.lr,
                "betas": (config.beta1, config.beta2),
                "eps": config.eps,
                "weight_decay": 0.0,
            },
        ]
    elif config.name == "hybrid_adamw":
        if not partition["hidden"]:
            raise ValueError("No transformer hidden matrices matched the hybrid AdamW partition")
        groups = [
            {
                "params": [parameter for _, parameter in partition["hidden"]],
                "algorithm": "adamw",
                "lr": config.lr,
                "betas": (config.beta1, config.beta2),
                "eps": config.eps,
                "weight_decay": config.weight_decay,
            },
            {
                "params": [parameter for _, parameter in partition["aux_decay"]],
                "algorithm": "adamw",
                "lr": config.aux_lr,
                "betas": (config.aux_beta1, config.aux_beta2),
                "eps": config.aux_eps,
                "weight_decay": config.weight_decay,
            },
            {
                "params": [parameter for _, parameter in partition["aux_no_decay"]],
                "algorithm": "adamw",
                "lr": config.aux_lr,
                "betas": (config.aux_beta1, config.aux_beta2),
                "eps": config.aux_eps,
                "weight_decay": 0.0,
            },
        ]
    else:
        if not partition["hidden"]:
            raise ValueError("No transformer hidden matrices matched the Muon partition")
        groups = [
            {
                "params": [parameter for _, parameter in partition["hidden"]],
                "algorithm": config.name,
                "lr": config.lr,
                "momentum": config.momentum,
                "beta2": config.normuon_beta2,
                "ns_steps": config.ns_steps,
                "ns_implementation": MUON_NS_IMPLEMENTATION,
                "adjust_lr_fn": config.adjust_lr_fn,
                "weight_decay": config.weight_decay,
            },
            {
                "params": [parameter for _, parameter in partition["aux_decay"]],
                "algorithm": "adamw",
                "lr": config.aux_lr,
                "betas": (config.aux_beta1, config.aux_beta2),
                "eps": config.aux_eps,
                "weight_decay": config.weight_decay,
            },
            {
                "params": [parameter for _, parameter in partition["aux_no_decay"]],
                "algorithm": "adamw",
                "lr": config.aux_lr,
                "betas": (config.aux_beta1, config.aux_beta2),
                "eps": config.aux_eps,
                "weight_decay": 0.0,
            },
        ]
    groups = [group for group in groups if group["params"]]
    return EmbeddingOptimizer(groups), summary
