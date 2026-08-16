from __future__ import annotations

import math
import re
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim import _functional as optim_f
from torch.optim._muon import muon as functional_muon

from .config import OptimizerConfig

_NO_DECAY = re.compile(r"(?:^|\.)(?:bias|.*norm(?:\d+)?\.weight)$", re.IGNORECASE)


def _is_hidden_matrix(name: str, parameter: nn.Parameter) -> bool:
    lowered = name.lower()
    return (
        parameter.ndim == 2
        and (lowered.startswith("layers.") or ".layers." in lowered)
        and "embedding" not in lowered
        and "classifier" not in lowered
        and "head" not in lowered
    )


def _uses_weight_decay(name: str, parameter: nn.Parameter) -> bool:
    return parameter.ndim >= 2 and _NO_DECAY.search(name) is None


def parameter_partition(model: nn.Module) -> dict[str, list[tuple[str, nn.Parameter]]]:
    result = {"hidden": [], "aux_decay": [], "aux_no_decay": []}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if _is_hidden_matrix(name, parameter):
            result["hidden"].append((name, parameter))
        elif _uses_weight_decay(name, parameter):
            result["aux_decay"].append((name, parameter))
        else:
            result["aux_no_decay"].append((name, parameter))
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
        params: list[Tensor] = []
        grads: list[Tensor] = []
        momentum_buffers: list[Tensor] = []
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = self.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(parameter)
            params.append(parameter)
            grads.append(parameter.grad)
            momentum_buffers.append(state["momentum_buffer"])
        functional_muon(
            params,
            grads,
            momentum_buffers,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            momentum=group["momentum"],
            nesterov=True,
            ns_coefficients=(3.4445, -4.7750, 2.0315),
            ns_steps=group["ns_steps"],
            eps=1e-7,
            adjust_lr_fn=group["adjust_lr_fn"],
            has_complex=False,
        )

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
