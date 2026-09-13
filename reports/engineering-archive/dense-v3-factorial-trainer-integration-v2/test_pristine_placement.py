"""Actual empty optimizer placement must not become a checkpoint-admission bypass."""

import copy

import pytest
import torch
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from torch import nn

from embed_optim.config import OptimizerConfig
from embed_optim.factorial_v3_optimizer import FactorialOptimizer, create_scheduler


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(7, 3)
        self.layers = nn.ModuleList([nn.Linear(3, 3, bias=False), nn.Linear(3, 3, bias=False)])
        self.norm = nn.LayerNorm(3)


def fresh(name):
    model = Tiny()
    optimizer = FactorialOptimizer(
        model, OptimizerConfig(name=name, lr=3e-4),
        expected_shapes={n: list(p.shape) for n, p in model.named_parameters()},
    )
    scheduler = create_scheduler(optimizer)
    return model, optimizer, scheduler


@pytest.mark.parametrize('name', ('hybrid_adamw', 'muon'))
def test_actual_accelerate_pristine_placement_is_an_exact_no_op(name):
    accelerator = Accelerator(cpu=True)
    model, optimizer, scheduler = fresh(name)
    before = copy.deepcopy(optimizer.state_dict())
    parameters = [id(p) for g in optimizer.param_groups for p in g['params']]
    wrapped = accelerator.prepare_optimizer(optimizer, device_placement=True)
    assert type(wrapped) is AcceleratedOptimizer
    assert wrapped.optimizer is optimizer
    assert optimizer.state_dict() == before
    assert optimizer.completed_steps == 0 and not optimizer.state
    assert parameters == [id(p) for g in optimizer.param_groups for p in g['params']]
    assert scheduler.optimizer is optimizer


@pytest.mark.parametrize('name', ('hybrid_adamw', 'muon'))
@pytest.mark.parametrize('change', ('lr', 'initial_lr', 'extra', 'names', 'ids', 'moment', 'step'))
def test_empty_round_trip_cannot_change_state_or_routing(name, change):
    model, optimizer, scheduler = fresh(name)
    before = copy.deepcopy(optimizer.state_dict())
    payload = copy.deepcopy(before)
    if change == 'lr':
        payload['param_groups'][0]['lr'] = 1e-5
    elif change == 'initial_lr':
        payload['param_groups'][0]['initial_lr'] = 1e-5
    elif change == 'extra':
        payload['undeclared'] = True
    elif change == 'names':
        payload['param_groups'][0]['param_names'].reverse()
    elif change == 'ids':
        payload['param_groups'][0]['params'].reverse()
    elif change == 'moment':
        payload['state'][0] = {'momentum_buffer': torch.ones(3, 3)}
    else:
        payload['factorial_v3']['steps'] = 1
    with pytest.raises(ValueError):
        optimizer.load_state_dict(payload)
    assert optimizer.state_dict() == before
    assert optimizer.completed_steps == 0 and not optimizer.state


@pytest.mark.parametrize('name', ('hybrid_adamw', 'muon'))
def test_existing_optimizer_cannot_rewind_to_empty_via_placement(name):
    model, optimizer, scheduler = fresh(name)
    empty = copy.deepcopy(optimizer.state_dict())
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()
    assert optimizer.completed_steps == 1 and optimizer.state
    with pytest.raises(ValueError, match='Externally admitted'):
        optimizer.load_state_dict(empty)
    assert optimizer.completed_steps == 1 and optimizer.state
