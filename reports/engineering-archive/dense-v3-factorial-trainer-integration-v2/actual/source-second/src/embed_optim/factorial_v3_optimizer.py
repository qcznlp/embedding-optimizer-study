"""Named reset/persistence boundary for the crossed DenseOn continuation.

This uses the existing AdamW/Muon factory and never changes primary routing or
primary numerical policy. It is a training component, not source-checkpoint,
calibration, data, runtime, whole-run, or publication admission. Its caller must
establish those identities separately before constructing a trainable model.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import json
import math

import torch

from .config import MUON_NS_IMPLEMENTATION, OptimizerConfig
from .optimizers import EmbeddingOptimizer, build_optimizer, parameter_partition_name

SCOPE = "dense-v3-factorial-named-reset-optimizer-v1"
ROLES = ("hidden", "aux_decay", "aux_no_decay")
HORIZON = 391
WARMUP_RATIO = 0.1


def denseon_shapes():
    """Pinned ST 5.7 named-parameter namespace, not safetensors key order."""
    shapes = {
        "embeddings.tok_embeddings.weight": [50368, 768],
        "embeddings.norm.weight": [768],
        "final_norm.weight": [768],
    }
    for layer in range(22):
        for suffix, shape in (
            ("attn.Wqkv.weight", [2304, 768]),
            ("attn.Wo.weight", [768, 768]),
            ("mlp.Wi.weight", [2304, 768]),
            ("mlp.Wo.weight", [768, 1152]),
            ("mlp_norm.weight", [768]),
        ):
            shapes[f"layers.{layer}.{suffix}"] = shape.copy()
        if layer:
            shapes[f"layers.{layer}.attn_norm.weight"] = [768]
    return {f"0.model.{name}": shape for name, shape in shapes.items()}


def _positive_integer(value):
    if type(value) is not int or value < 1:
        raise ValueError("Require a positive integral factorial step/horizon")
    return value


def _number(value, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Require a finite non-boolean optimizer number")
    if positive and value <= 0:
        raise ValueError("Require a positive calibrated hidden learning rate")
    return value


def configuration(config):
    """Only the two frozen operators; LR provenance is the caller's obligation."""
    if type(config) is not OptimizerConfig or config.name not in {"hybrid_adamw", "muon"}:
        raise ValueError("Factorial requires routed AdamW or Muon, not primary AdamW/NorMuon")
    _number(config.lr, positive=True)

    def serialized(value):
        result = dataclasses.asdict(value)
        implementation = result.pop("ns_implementation", None)
        if value.name == "muon":
            if implementation not in (None, MUON_NS_IMPLEMENTATION):
                raise ValueError("Factorial Muon requires the audited unfused BF16 transform")
            result["ns_implementation"] = MUON_NS_IMPLEMENTATION
        elif implementation is not None:
            raise ValueError("Routed AdamW has no Newton-Schulz implementation")
        return result

    actual = serialized(config)
    expected = serialized(OptimizerConfig(name=config.name, lr=config.lr))
    # Compare types too: True is not an admissible replacement for a numeric 1.
    if actual != expected or any(type(actual[k]) is not type(v) for k, v in expected.items()):
        raise ValueError("Factorial optimizer settings differ from the fixed design")
    if config.name == "muon" and actual.get("ns_implementation") != MUON_NS_IMPLEMENTATION:
        raise ValueError("Factorial Muon requires the audited unfused BF16 transform")
    return copy.deepcopy(actual)


def named_layout(model, expected_shapes):
    """Observe every trainable parameter; aliases/frozen/extra tensors fail closed."""
    if not isinstance(expected_shapes, dict) or not expected_shapes:
        raise ValueError("Require an explicit complete named model topology")
    named = list(model.named_parameters(remove_duplicate=False))
    if len({n for n, _ in named}) != len(named) or len({id(p) for _, p in named}) != len(named):
        raise ValueError("Aliased or duplicate factorial model parameters")
    if {n: list(p.shape) for n, p in named} != expected_shapes:
        raise ValueError("Loaded model names/shapes differ from the declared topology")
    if any(not p.requires_grad or p.dtype != torch.float32 for _, p in named):
        raise ValueError("Factorial training requires all declared parameters trainable in FP32")
    groups = []
    offset = 0
    for role in ROLES:
        members = [
            {"name": name, "shape": list(p.shape), "dtype": str(p.dtype)}
            for name, p in named
            if parameter_partition_name(name, p.ndim) == role
        ]
        if not members:
            raise ValueError("Factorial requires all three nonempty parameter partitions")
        groups.append(
            {"role": role, "params": list(range(offset, offset + len(members))), "members": members}
        )
        offset += len(members)
    return groups


def _group_settings(config):
    configuration(config)

    def adam(auxiliary, decay):
        return {
            "algorithm": "adamw",
            "betas": (config.aux_beta1, config.aux_beta2)
            if auxiliary
            else (config.beta1, config.beta2),
            "eps": config.aux_eps if auxiliary else config.eps,
            "weight_decay": config.weight_decay if decay else 0.0,
        }

    hidden = (
        adam(False, True)
        if config.name == "hybrid_adamw"
        else {
            "algorithm": "muon",
            "momentum": config.momentum,
            "beta2": config.normuon_beta2,
            "ns_steps": config.ns_steps,
            "ns_implementation": MUON_NS_IMPLEMENTATION,
            "adjust_lr_fn": config.adjust_lr_fn,
            "weight_decay": config.weight_decay,
        }
    )
    return [hidden, adam(True, True), adam(True, False)]


def _require_settings(group, settings, base_lr):
    required = {*settings, "lr", "params", "param_names"}
    if set(group) not in (required, required | {"initial_lr"}):
        raise ValueError("Unknown/missing factorial optimizer group fields")
    for key, value in settings.items():
        if group[key] != value or type(group[key]) is not type(value):
            raise ValueError(f"Factorial optimizer group setting differs: {key}")
    if _number(group["lr"]) < 0:
        raise ValueError("Negative optimizer learning rate")
    if "initial_lr" in group and group["initial_lr"] != base_lr:
        raise ValueError("Restored initial learning rate differs from calibration")


def inspect_state(payload, layout, config, *, step):
    """Inspect tensor state BEFORE load; trusted checkpoint step is supplied outside."""
    if type(step) is not int or step < 0:
        raise ValueError("Invalid factorial optimizer step")
    expected_meta = {
        "schema_version": 1,
        "scope": SCOPE,
        "layout": layout,
        "optimizer_configuration": configuration(config),
        "steps": step,
    }
    if not isinstance(payload, dict) or set(payload) != {"state", "param_groups", "factorial_v3"}:
        raise ValueError("Optimizer lacks the explicit factorial named-state contract")
    if json.dumps(payload["factorial_v3"], sort_keys=True, allow_nan=False) != json.dumps(
        expected_meta, sort_keys=True, allow_nan=False
    ):
        raise ValueError("Factorial named routing, configuration or step differs")
    if type(payload["factorial_v3"].get("steps")) is not int:
        raise ValueError("Factorial saved step must be an integer")
    groups, states = payload["param_groups"], payload["state"]
    if not isinstance(groups, list) or len(groups) != 3 or not isinstance(states, dict):
        raise ValueError("Wrong factorial optimizer group/state topology")
    settings = _group_settings(config)
    base_lrs = [config.lr, config.aux_lr, config.aux_lr]
    expected_ids = []
    for group, group_layout, required, lr in zip(groups, layout, settings, base_lrs, strict=True):
        _require_settings(group, required, lr)
        members, ids = group_layout["members"], group_layout["params"]
        if (
            group["params"] != ids
            or any(type(pid) is not int for pid in group["params"])
            or group["param_names"] != [m["name"] for m in members]
        ):
            raise ValueError("Factorial parameter IDs or names/order differ")
        expected_ids.extend(ids)
        if step == 0:
            continue
        fields = (
            {"step", "exp_avg", "exp_avg_sq"}
            if required["algorithm"] == "adamw"
            else {"momentum_buffer"}
        )
        for pid, member in zip(ids, members, strict=True):
            values = states.get(pid)
            if not isinstance(values, dict) or set(values) != fields:
                raise ValueError("Wrong or missing named optimizer moments")
            for key, tensor in values.items():
                shape = [] if key == "step" else member["shape"]
                if (
                    not torch.is_tensor(tensor)
                    or tensor.layout != torch.strided
                    or tensor.dtype != torch.float32
                    or list(tensor.shape) != shape
                    or not bool(torch.isfinite(tensor).all())
                ):
                    raise ValueError("Invalid named optimizer tensor shape/dtype/finiteness")
            if "step" in values and values["step"].item() != step:
                raise ValueError("Adam moment counter differs from the trusted checkpoint step")
            if "exp_avg_sq" in values and bool((values["exp_avg_sq"] < 0).any()):
                raise ValueError("Negative Adam second moment")
    if any(type(pid) is not int for pid in states) or (
        states if step == 0 else set(states) != set(expected_ids)
    ):
        raise ValueError("Reset state is not empty or saved state coverage differs")
    return {"groups": 3, "parameter_states": len(states), "step": step, "named_routing": True}


class FactorialOptimizer(EmbeddingOptimizer):
    """Same numerical kernels, with fresh-state and named save/load enforcement.

    Construct from newly loaded model weights. A continuation within this branch
    must call prepare_resume with the externally checked checkpoint step before
    the usual PyTorch load_state_dict. Calibration states are not an input.
    """

    def __init__(self, model, config, *, expected_shapes=None):
        self._configuration = copy.deepcopy(config)
        configuration(config)
        self._layout = named_layout(
            model, denseon_shapes() if expected_shapes is None else expected_shapes
        )
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError("Fresh factorial optimizer must not inherit calibration gradients")
        factory, self.partition_summary = build_optimizer(model, config)
        if factory.state:
            raise ValueError("Factory unexpectedly supplied prior optimizer state")
        super().__init__(factory.param_groups)
        self._named_parameters = dict(model.named_parameters())
        self._steps = 0
        self._resume_step = None
        for group, layout in zip(self.param_groups, self._layout, strict=True):
            group["param_names"] = [m["name"] for m in layout["members"]]
        self._require_live_layout()
        self.state_dict()  # Observe the empty reset, not a declaration of reset.

    @property
    def named_parameter_layout(self):
        return copy.deepcopy(self._layout)

    @property
    def completed_steps(self):
        return self._steps

    def _require_live_layout(self):
        if len(self.param_groups) != 3:
            raise ValueError("Live factorial group topology differs")
        for group, layout, settings, lr in zip(
            self.param_groups,
            self._layout,
            _group_settings(self._configuration),
            [self._configuration.lr, self._configuration.aux_lr, self._configuration.aux_lr],
            strict=True,
        ):
            _require_settings(group, settings, lr)
            names = [m["name"] for m in layout["members"]]
            if group["param_names"] != names or [id(p) for p in group["params"]] != [
                id(self._named_parameters[n]) for n in names
            ]:
                raise ValueError("Live parameter objects no longer follow the named routing")

    def step(self, closure=None):
        if closure is not None or self._resume_step is not None:
            raise ValueError("Factorial step requires ordinary gradients and completed restore")
        self._require_live_layout()
        if any(
            p.grad is None or p.grad.is_sparse or p.grad.dtype != p.dtype
            for p in self._named_parameters.values()
        ):
            raise ValueError("Factorial step requires dense gradients for every declared parameter")
        result = super().step()
        self._steps += 1
        return result

    def state_dict(self):
        self._require_live_layout()
        payload = super().state_dict()
        payload["factorial_v3"] = {
            "schema_version": 1,
            "scope": SCOPE,
            "layout": self.named_parameter_layout,
            "optimizer_configuration": configuration(self._configuration),
            "steps": self._steps,
        }
        inspect_state(payload, self._layout, self._configuration, step=self._steps)
        return payload

    def prepare_resume(self, expected_step):
        _positive_integer(expected_step)
        if self._steps or self.state or self._resume_step is not None:
            raise ValueError("Resume requires a fresh, not previously stepped/armed optimizer")
        self._resume_step = expected_step

    def load_state_dict(self, state_dict):
        if self._resume_step is None or self._steps or self.state:
            raise ValueError("Externally admitted resume step is required before state loading")
        self._require_live_layout()
        inspect_state(state_dict, self._layout, self._configuration, step=self._resume_step)
        # The standard loader maps integer IDs by position; order was checked
        # above. Never rely on shape multisets or on param_names alone to remap.
        payload = {k: state_dict[k] for k in ("state", "param_groups")}
        super().load_state_dict(payload)
        self._steps, self._resume_step = self._resume_step, None
        self.state_dict()


def inspect_scheduler(scheduler, optimizer_state, config, *, step, horizon=HORIZON):
    _positive_integer(horizon)
    if type(step) is not int or not 0 <= step <= horizon:
        raise ValueError("Factorial scheduler step outside its horizon")
    configuration(config)
    # LambdaLR.load_state_dict updates its instance dictionary. Ignoring an
    # extra key can replace its optimizer or step method after partial restore.
    if (
        not isinstance(scheduler, dict)
        or set(scheduler)
        != {
            "base_lrs",
            "last_epoch",
            "_step_count",
            "_is_initial",
            "_get_lr_called_within_step",
            "_last_lr",
            "lr_lambdas",
        }
        or scheduler["_is_initial"] is not False
        or scheduler["_get_lr_called_within_step"] is not False
    ):
        raise ValueError("Unknown/missing factorial scheduler fields or active transition flags")
    warmup = math.ceil(WARMUP_RATIO * horizon)
    factor = (
        step / max(1, warmup)
        if step < warmup
        else max(0.0, (horizon - step) / max(1, horizon - warmup))
    )
    if (
        type(scheduler.get("last_epoch")) is not int
        or type(scheduler.get("_step_count")) is not int
        or scheduler["last_epoch"] != step
        or scheduler["_step_count"] != step + 1
        or scheduler.get("lr_lambdas") != [{}, {}, {}]
    ):
        raise ValueError("Factorial scheduler counters/lambda state differ")
    initial = [config.lr, config.aux_lr, config.aux_lr]
    current = [lr * factor for lr in initial]
    for key, target in (("base_lrs", initial), ("_last_lr", current)):
        values = scheduler.get(key)
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError("Factorial scheduler group coverage differs")
        for actual, expected in zip(values, target, strict=True):
            if not math.isclose(_number(actual), expected, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError("Factorial scheduler learning rate differs")
    for group, base, lr in zip(optimizer_state["param_groups"], initial, current, strict=True):
        if group.get("initial_lr") != base or not math.isclose(
            _number(group["lr"]), lr, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError("Optimizer learning rate does not match the saved scheduler")
    return {"step": step, "horizon": horizon, "warmup_steps": warmup, "groups": 3}


def create_scheduler(optimizer, *, horizon=HORIZON):
    from transformers import get_linear_schedule_with_warmup

    _positive_integer(horizon)
    if type(optimizer) is not FactorialOptimizer or optimizer.completed_steps:
        raise ValueError("Create the declared scheduler before restoring/stepping the optimizer")
    return get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=math.ceil(WARMUP_RATIO * horizon), num_training_steps=horizon
    )


def require_scheduler_function(scheduler, optimizer, horizon):
    from transformers.optimization import _get_linear_schedule_with_warmup_lr_lambda

    _positive_integer(horizon)
    if (
        type(scheduler) is not torch.optim.lr_scheduler.LambdaLR
        or scheduler.optimizer is not optimizer
    ):
        raise ValueError("Require the actual declared optimizer/scheduler pair")
    expected = {
        "num_warmup_steps": math.ceil(WARMUP_RATIO * horizon),
        "num_training_steps": horizon,
    }
    if len(scheduler.lr_lambdas) != 3 or any(
        not isinstance(fn, functools.partial)
        or fn.func is not _get_linear_schedule_with_warmup_lr_lambda
        or fn.args
        or fn.keywords != expected
        for fn in scheduler.lr_lambdas
    ):
        raise ValueError("Actual scheduler function or declared horizon differs")


def restore_training_state(
    optimizer, scheduler, saved_optimizer, saved_scheduler, *, step, horizon=HORIZON
):
    """Validate BOTH payloads and the live pair before changing either object.

    External checkpoint seals and the model-weight load are the caller's job.
    This routine never retrofits missing names or accepts calibration state.
    """
    if type(optimizer) is not FactorialOptimizer or optimizer.completed_steps or optimizer.state:
        raise ValueError("Restore requires the fresh factorial optimizer")
    _positive_integer(step)
    require_scheduler_function(scheduler, optimizer, horizon)
    inspect_scheduler(
        scheduler.state_dict(),
        optimizer.state_dict(),
        optimizer._configuration,
        step=0,
        horizon=horizon,
    )
    inspect_state(
        saved_optimizer, optimizer.named_parameter_layout, optimizer._configuration, step=step
    )
    inspect_scheduler(
        saved_scheduler, saved_optimizer, optimizer._configuration, step=step, horizon=horizon
    )
    optimizer.prepare_resume(step)
    optimizer.load_state_dict(saved_optimizer)
    scheduler.load_state_dict(saved_scheduler)
    require_scheduler_function(scheduler, optimizer, horizon)
    return inspect_scheduler(
        scheduler.state_dict(),
        optimizer.state_dict(),
        optimizer._configuration,
        step=step,
        horizon=horizon,
    )
