import copy

import pytest
import torch
from torch import nn

from embed_optim.config import OptimizerConfig
from embed_optim.optimizers import build_optimizer, parameter_partition, partition_summary


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


def test_parameter_partition_keeps_embeddings_and_head_out_of_muon():
    partition = parameter_partition(TinyEncoder())
    summary = partition_summary(partition)
    assert summary["hidden"]["tensors"] == 2
    assert {name for name, _ in partition["hidden"]} == {"layers.0.weight", "layers.1.weight"}
    assert any(name == "embeddings.weight" for name, _ in partition["aux_decay"])
    assert any(name == "head.weight" for name, _ in partition["aux_decay"])


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
