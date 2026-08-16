import copy
import math

import pytest
import torch
from torch import nn

from embed_optim.config import OptimizerConfig
from embed_optim.optimizers import (
    _normuon_update,
    build_optimizer,
    parameter_partition,
    partition_summary,
)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(16, 8)
        self.layers = nn.ModuleList([nn.Linear(8, 8, bias=True), nn.Linear(8, 8, bias=False)])
        self.norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 4, bias=False)

    def forward(self, tokens):
        hidden = self.embeddings(tokens).mean(1)
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
        return self.head(self.norm(hidden))


def _official_normuon_reference(gradient, momentum_buffer, second_moment, beta, beta2, steps):
    momentum_buffer.lerp_(gradient, 1 - beta)
    update = gradient.clone().lerp_(momentum_buffer, beta).bfloat16()
    transposed = update.size(-2) > update.size(-1)
    if transposed:
        update = update.mT
    update = update / (update.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = update @ update.mT
        update = a * update + (b * gram + c * gram @ gram) @ update
    if transposed:
        update = update.mT
    update = update.to(gradient.dtype)
    original_norm = update.norm(dim=(-2, -1), keepdim=True)
    row_second_moment = torch.mean(update * update, dim=-1, keepdim=True)
    second_moment.lerp_(row_second_moment, 1 - beta2)
    update.mul_(second_moment.sqrt().add_(1e-10).reciprocal())
    normalized_norm = update.norm(dim=(-2, -1), keepdim=True)
    update.mul_(original_norm / normalized_norm.add_(1e-10))
    update.mul_(math.sqrt(max(1, gradient.size(-2) / gradient.size(-1))))
    return update


def test_parameter_partition_keeps_embeddings_and_head_out_of_muon():
    partition = parameter_partition(TinyEncoder())
    summary = partition_summary(partition)
    assert summary["hidden"]["tensors"] == 2
    assert {name for name, _ in partition["hidden"]} == {"layers.0.weight", "layers.1.weight"}
    assert any(name == "embeddings.weight" for name, _ in partition["aux_decay"])
    assert any(name == "head.weight" for name, _ in partition["aux_decay"])


def test_normuon_update_matches_pinned_official_reference():
    torch.manual_seed(19)
    gradient = torch.randn(8, 4)
    local_momentum = torch.randn_like(gradient)
    reference_momentum = local_momentum.clone()
    local_second_moment = torch.rand(8, 1)
    reference_second_moment = local_second_moment.clone()

    local = _normuon_update(
        gradient,
        local_momentum,
        local_second_moment,
        momentum=0.95,
        beta2=0.95,
        ns_steps=5,
    )
    reference = _official_normuon_reference(
        gradient,
        reference_momentum,
        reference_second_moment,
        beta=0.95,
        beta2=0.95,
        steps=5,
    )

    torch.testing.assert_close(local, reference)
    torch.testing.assert_close(local_momentum, reference_momentum)
    torch.testing.assert_close(local_second_moment, reference_second_moment)


@pytest.mark.parametrize("name,lr", [("adamw", 1e-3), ("muon", 1e-3), ("normuon", 1e-3)])
def test_optimizer_steps_and_round_trips(name, lr):
    torch.manual_seed(7)
    model = TinyEncoder()
    optimizer, _ = build_optimizer(model, OptimizerConfig(name=name, lr=lr, aux_lr=1e-3))
    before = copy.deepcopy(model.state_dict())
    loss = model(torch.randint(0, 16, (4, 3))).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    assert any(not torch.equal(before[key], value) for key, value in model.state_dict().items())

    state = optimizer.state_dict()
    restored, _ = build_optimizer(model, OptimizerConfig(name=name, lr=lr, aux_lr=1e-3))
    restored.load_state_dict(state)
    assert len(restored.state) == len(optimizer.state)
