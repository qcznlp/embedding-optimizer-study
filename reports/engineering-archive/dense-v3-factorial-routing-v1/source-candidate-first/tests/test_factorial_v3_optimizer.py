"""Actual CPU updates and named persistence, not primary/factorial run admission."""

import copy
import dataclasses
import inspect
import io
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from embed_optim import config as optimizer_config_module
from embed_optim import factorial_v3_optimizer as f
from embed_optim import optimizers as optimizer_module
from embed_optim.config import OptimizerConfig


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(7, 3)
        self.layers = nn.ModuleList([nn.Linear(3, 3, bias=False), nn.Linear(3, 3, bias=False)])
        self.norm = nn.LayerNorm(3)
        with torch.no_grad():
            for i, p in enumerate(self.parameters()):
                p.copy_(torch.arange(p.numel()).reshape(p.shape) / 17 + (i + 1) / 11)


SHAPES = {
    "embeddings.weight": [7, 3],
    "layers.0.weight": [3, 3],
    "layers.1.weight": [3, 3],
    "norm.weight": [3],
    "norm.bias": [3],
}


def test_actual_imports_come_from_this_source_tree_not_a_foreign_install():
    package = Path(__file__).resolve().parents[1] / "src/embed_optim"
    for module in (f, optimizer_config_module, optimizer_module):
        assert Path(inspect.getsourcefile(module)).resolve() == package / (
            module.__name__.split(".")[-1] + ".py"
        )


def pair(name, *, model=None, horizon=10):
    model = Tiny() if model is None else model
    config = OptimizerConfig(name=name, lr=0.0003)
    optimizer = f.FactorialOptimizer(model, config, expected_shapes=SHAPES)
    scheduler = f.create_scheduler(optimizer, horizon=horizon)
    return model, config, optimizer, scheduler


def gradients(model, step):
    for i, p in enumerate(model.parameters()):
        p.grad = (torch.sin(torch.arange(p.numel()).reshape(p.shape) + i + step) / 13).float()


def advance(model, optimizer, scheduler, begin, end):
    for step in range(begin, end):
        gradients(model, step)
        optimizer.step()
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def saved_pair(name):
    model, config, optimizer, scheduler = pair(name)
    advance(model, optimizer, scheduler, 0, 4)
    return (
        model,
        config,
        optimizer,
        scheduler,
        copy.deepcopy(optimizer.state_dict()),
        copy.deepcopy(scheduler.state_dict()),
    )


@pytest.mark.parametrize("name", ["hybrid_adamw", "muon"])
def test_fresh_actual_factory_observes_same_named_matrices_and_empty_state(name):
    model, config, optimizer, scheduler = pair(name)
    assert [g["param_names"] for g in optimizer.param_groups] == [
        ["layers.0.weight", "layers.1.weight"],
        ["embeddings.weight"],
        ["norm.weight", "norm.bias"],
    ]
    assert optimizer.state_dict()["state"] == {}
    assert optimizer.completed_steps == 0
    assert optimizer.partition_summary["hidden"]["tensors"] == 2
    assert f.inspect_state(
        optimizer.state_dict(), optimizer.named_parameter_layout, config, step=0
    )["named_routing"]
    f.inspect_scheduler(scheduler.state_dict(), optimizer.state_dict(), config, step=0, horizon=10)


@pytest.mark.parametrize("name", ["adamw", "normuon", "sgd"])
def test_nonfactorial_operators_are_not_admitted(name):
    with pytest.raises(ValueError, match="routed AdamW or Muon"):
        f.FactorialOptimizer(Tiny(), OptimizerConfig(name=name, lr=0.0003), expected_shapes=SHAPES)


@pytest.mark.parametrize(
    "field,value",
    [
        ("lr", 0),
        ("lr", float("nan")),
        ("lr", True),
        ("aux_lr", 1e-5),
        ("weight_decay", 0.0),
        ("beta1", 0.8),
        ("beta2", 0.9),
        ("eps", 1e-6),
        ("aux_beta1", 0.8),
        ("aux_beta2", 0.9),
        ("aux_eps", 1e-6),
        ("momentum", 0.9),
        ("ns_steps", 4),
        ("adjust_lr_fn", "match_rms_adamw"),
    ],
)
def test_frozen_optimizer_settings_are_not_silently_changed(field, value):
    config = dataclasses.replace(OptimizerConfig(name="muon", lr=0.0003), **{field: value})
    with pytest.raises(ValueError):
        f.FactorialOptimizer(Tiny(), config, expected_shapes=SHAPES)


@pytest.mark.parametrize(
    "change", ["alias", "frozen", "dtype", "names", "shape", "gradient", "missing_group"]
)
def test_fresh_model_observation_rejects_wrong_topology_or_calibration_gradient(change):
    model, shapes = Tiny(), copy.deepcopy(SHAPES)
    if change == "alias":
        model.layers[1].weight = model.layers[0].weight
    elif change == "frozen":
        model.norm.bias.requires_grad_(False)
    elif change == "dtype":
        model.double()
    elif change == "names":
        shapes["other"] = shapes.pop("layers.0.weight")
    elif change == "shape":
        shapes["layers.0.weight"] = [1, 9]
    elif change == "gradient":
        gradients(model, 0)
    else:
        del model.norm
        shapes.pop("norm.weight")
        shapes.pop("norm.bias")
    with pytest.raises(ValueError):
        f.FactorialOptimizer(model, OptimizerConfig(name="muon", lr=0.0003), expected_shapes=shapes)


@pytest.mark.parametrize("name", ["hybrid_adamw", "muon"])
def test_saved_resume_uses_actual_torch_payload_and_matches_uninterrupted_updates(name):
    model, config, optimizer, scheduler, saved_opt, saved_sched = saved_pair(name)
    buffer = io.BytesIO()
    torch.save(
        {"model": model.state_dict(), "optimizer": saved_opt, "scheduler": saved_sched}, buffer
    )
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True, map_location="cpu")
    continued = Tiny()
    continued.load_state_dict(saved["model"])
    _, _, restored, restored_scheduler = pair(name, model=continued)
    receipt = f.restore_training_state(
        restored, restored_scheduler, saved["optimizer"], saved["scheduler"], step=4, horizon=10
    )
    assert receipt["step"] == 4
    advance(model, optimizer, scheduler, 4, 10)
    advance(continued, restored, restored_scheduler, 4, 10)
    for original, replay in zip(model.parameters(), continued.parameters(), strict=True):
        torch.testing.assert_close(original, replay, rtol=0, atol=0)
    assert restored.completed_steps == optimizer.completed_steps == 10
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    f.inspect_state(restored.state_dict(), optimizer.named_parameter_layout, config, step=10)
    f.inspect_scheduler(
        restored_scheduler.state_dict(), restored.state_dict(), config, step=10, horizon=10
    )


@pytest.mark.parametrize("name", ["hybrid_adamw", "muon"])
def test_reset_from_reached_weights_does_not_inherit_eight_step_calibration_state(name):
    model, _, calibration, schedule = pair(name)
    advance(model, calibration, schedule, 0, 8)
    assert calibration.state and calibration.completed_steps == 8
    # Loading weights alone does not transfer moments, gradients or counters.
    reset_model = Tiny()
    reset_model.load_state_dict(model.state_dict())
    _, _, reset, _ = pair(name, model=reset_model)
    assert reset.state_dict()["state"] == {} and reset.completed_steps == 0
    with pytest.raises(ValueError, match="Externally admitted"):
        reset.load_state_dict(calibration.state_dict())


@pytest.mark.parametrize("name", ["hybrid_adamw", "muon"])
@pytest.mark.parametrize(
    "change",
    [
        "names",
        "ids",
        "layout",
        "missing_name",
        "missing_state",
        "extra_state",
        "nonfinite",
        "negative_second",
        "shape",
        "dtype",
        "counter",
        "recipe",
        "metadata_step",
        "schema_bool",
        "group_setting",
        "scheduler_counter",
        "scheduler_lr",
        "scheduler_horizon",
        "untrusted_step",
    ],
)
def test_invalid_restore_refuses_before_mutating_either_fresh_object(name, change):
    _, _, _, _, saved_opt, saved_sched = saved_pair(name)
    _, _, target, scheduler = pair(name)
    expected_step, horizon = 4, 10
    if change == "names":
        saved_opt["param_groups"][0]["param_names"].reverse()
    elif change == "ids":
        saved_opt["param_groups"][0]["params"].reverse()
    elif change == "layout":
        saved_opt["factorial_v3"]["layout"][0]["members"].reverse()
    elif change == "missing_name":
        saved_opt["param_groups"][0].pop("param_names")
    elif change == "missing_state":
        saved_opt["state"].pop(0)
    elif change == "extra_state":
        saved_opt["state"][99] = saved_opt["state"][0]
    elif change in {"nonfinite", "shape", "dtype"}:
        key = "exp_avg" if name == "hybrid_adamw" else "momentum_buffer"
        value = saved_opt["state"][0][key]
        if change == "nonfinite":
            value[0, 0] = float("nan")
        elif change == "shape":
            saved_opt["state"][0][key] = value.reshape(1, 9)
        else:
            saved_opt["state"][0][key] = value.double()
    elif change == "negative_second":
        saved_opt["state"][2]["exp_avg_sq"][0, 0] = -1
    elif change == "counter":
        saved_opt["state"][2]["step"] = torch.tensor(8.0)
    elif change == "recipe":
        saved_opt["factorial_v3"]["optimizer_configuration"]["aux_lr"] *= 2
    elif change == "metadata_step":
        saved_opt["factorial_v3"]["steps"] = 8
    elif change == "schema_bool":
        saved_opt["factorial_v3"]["schema_version"] = True
    elif change == "group_setting":
        saved_opt["param_groups"][1]["eps"] = 1e-4
    elif change == "scheduler_counter":
        saved_sched["_step_count"] += 1
    elif change == "scheduler_lr":
        saved_sched["_last_lr"][1] *= 2
    elif change == "scheduler_horizon":
        horizon = 20
    else:
        expected_step = 5
    pristine_opt, pristine_sched = (
        copy.deepcopy(target.state_dict()),
        copy.deepcopy(scheduler.state_dict()),
    )
    with pytest.raises(ValueError):
        f.restore_training_state(
            target, scheduler, saved_opt, saved_sched, step=expected_step, horizon=horizon
        )
    assert target.state_dict() == pristine_opt
    assert scheduler.state_dict() == pristine_sched
    assert target.completed_steps == 0 and target._resume_step is None


@pytest.mark.parametrize("name", ["hybrid_adamw", "muon"])
def test_live_equal_shape_parameter_swap_refused_before_update(name):
    model, _, optimizer, _ = pair(name)
    before = [p.detach().clone() for p in model.parameters()]
    gradients(model, 0)
    optimizer.param_groups[0]["params"].reverse()
    with pytest.raises(ValueError, match="Live parameter objects"):
        optimizer.step()
    assert not optimizer.state and optimizer.completed_steps == 0
    for p, original in zip(model.parameters(), before, strict=True):
        assert torch.equal(p, original)


@pytest.mark.parametrize("name", ["hybrid_adamw", "muon"])
def test_missing_gradient_does_not_silently_skip_a_matrix(name):
    model, _, optimizer, _ = pair(name)
    gradients(model, 0)
    model.layers[1].weight.grad = None
    with pytest.raises(ValueError, match="every declared parameter"):
        optimizer.step()
    assert not optimizer.state and optimizer.completed_steps == 0


def test_routed_adamw_matches_independent_torch_adamw_groups():
    model, _, actual, schedule = pair("hybrid_adamw")
    reference = Tiny()
    optimizer = torch.optim.AdamW(
        [
            {"params": list(reference.layers.parameters()), "lr": 0.0003, "weight_decay": 0.01},
            {"params": list(reference.embeddings.parameters()), "lr": 3e-6, "weight_decay": 0.01},
            {"params": list(reference.norm.parameters()), "lr": 3e-6, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
        fused=False,
    )
    for step in range(10):
        gradients(model, step)
        gradients(reference, step)
        for group, reference_group in zip(actual.param_groups, optimizer.param_groups, strict=True):
            reference_group["lr"] = group["lr"]
        actual.step()
        optimizer.step()
        schedule.step()
        for p, q in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(p, q, rtol=0, atol=2e-7)


def reference_muon(gradient, momentum):
    # Independent literal calculation with scalar matrix products. The declared
    # BF16 expression rounds c*A before its product with A, not c*(A@A).
    # Those mathematically equal expressions are NOT the same BF16 operator.
    def product(left, right):
        return torch.tensor(
            [
                [
                    sum(float(left[i, k]) * float(right[k, j]) for k in range(left.shape[1]))
                    for j in range(right.shape[1])
                ]
                for i in range(left.shape[0])
            ],
            dtype=torch.bfloat16,
        )

    momentum = 0.95 * momentum + 0.05 * gradient
    update = (0.05 * gradient + 0.95 * momentum).bfloat16()
    transpose = update.shape[0] > update.shape[1]
    if transpose:
        update = update.T
    update = update / torch.maximum(update.norm(), torch.tensor(1e-7, dtype=torch.bfloat16))
    for _ in range(5):
        gram = product(update, update.T)
        polynomial = -4.7750 * gram + product(2.0315 * gram, gram)
        update = 3.4445 * update + product(polynomial, update)
    return (update.T if transpose else update), momentum


@pytest.mark.parametrize("shape", [(3, 3), (2, 3), (3, 2)])
def test_muon_updates_match_independent_polynomial_and_aspect_scaling(shape):
    model = Tiny()
    model.layers[0].weight = nn.Parameter(
        torch.arange(math.prod(shape)).reshape(shape).float() / 17
    )
    shapes = {**SHAPES, "layers.0.weight": list(shape)}
    config = OptimizerConfig(name="muon", lr=0.0003)
    optimizer = f.FactorialOptimizer(model, config, expected_shapes=shapes)
    reference = model.layers[0].weight.detach().clone()
    momentum = torch.zeros_like(reference)
    for step in range(8):
        gradients(model, step)
        direction, momentum = reference_muon(model.layers[0].weight.grad, momentum)
        reference *= 1 - 0.0003 * 0.01
        reference.add_(direction, alpha=-0.0003 * math.sqrt(max(1, shape[0] / shape[1])))
        optimizer.step()
        torch.testing.assert_close(model.layers[0].weight, reference, rtol=0, atol=0)


def test_complete_denseon_default_is_not_replaced_by_the_tiny_fixture():
    shapes = f.denseon_shapes()
    assert len(shapes) == 134 and sum(math.prod(s) for s in shapes.values()) == 149014272
    assert "0.model.layers.0.attn_norm.weight" not in shapes
    assert shapes["0.model.layers.21.mlp.Wi.weight"] == [2304, 768]
    with pytest.raises(ValueError, match="declared topology"):
        f.FactorialOptimizer(Tiny(), OptimizerConfig(name="muon", lr=0.0003))
