import copy

import pytest
import torch
from torch import nn

from embed_optim.config import OptimizerConfig
from scripts.audit_checkpoint_optimizer_resume import _stack, _synthetic_update, exercise_resume


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(16, 8)
        self.layers = nn.ModuleList([nn.Linear(8, 8, bias=False)])
        self.norm = nn.LayerNorm(8)


def setup_checkpoint(name):
    model = TinyEncoder()
    config = OptimizerConfig(name=name, lr=1e-3, aux_lr=1e-4)
    optimizer, scheduler, _ = _stack(model, config, total_steps=100, warmup_steps=10)
    # The initial zero learning-rate step still initializes momentum/moment state.
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    for seed in [13, 14]:
        _synthetic_update(model, optimizer, scheduler, seed)
    return model, config, optimizer.state_dict(), scheduler.state_dict()


@pytest.mark.parametrize("name", ["adamw", "muon", "normuon"])
def test_real_state_schema_roundtrip_continues_exactly(name):
    model, config, optimizer, scheduler = setup_checkpoint(name)
    result = exercise_resume(lambda: copy.deepcopy(model), config, optimizer, scheduler, 100, 10)
    assert result["initial_saved_states_exact"] is True
    assert result["intermediate_roundtrip_exact"] is True
    assert result["second_update_weights_and_states_exact"] is True
    assert [r["step"] for r in result["learning_rate_trace"]] == [3, 4, 5]
    assert result["parameter_count"] == result["optimizer_state_count"] == 4
    assert result["distributed_training_resumed"] is False


def test_rejects_incorrect_saved_learning_rate():
    model, config, optimizer, scheduler = setup_checkpoint("normuon")
    optimizer["param_groups"][0]["lr"] = 0.0
    with pytest.raises(ValueError, match="learning rate"):
        exercise_resume(lambda: copy.deepcopy(model), config, optimizer, scheduler, 100, 10)


def test_rejects_missing_parameter_momentum_state():
    model, config, optimizer, scheduler = setup_checkpoint("muon")
    optimizer["state"].pop(next(iter(optimizer["state"])))
    with pytest.raises(ValueError, match="every unique parameter"):
        exercise_resume(lambda: copy.deepcopy(model), config, optimizer, scheduler, 100, 10)
