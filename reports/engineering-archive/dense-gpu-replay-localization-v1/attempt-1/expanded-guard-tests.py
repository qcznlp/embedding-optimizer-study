import copy
import unittest

import torch

from embed_optim.config import OptimizerConfig
from embed_optim.optimizers import build_optimizer
from scripts.audit_dense_gpu_replay_update import (
    difference,
    direct_step,
    exact_update,
    operator_trace,
    optimizer_layout,
)


class ReplayUpdateGuards(unittest.TestCase):
    def test_difference_retains_small_inequality(self):
        result = difference(torch.tensor([1.0]), torch.tensor([1.0 + 1e-7]))
        self.assertFalse(result["bitwise_equal"])
        self.assertEqual(result["unequal_elements"], 1)

    def test_topology_not_broadcasted(self):
        with self.assertRaises(ValueError):
            difference(torch.ones(2, 2), torch.ones(2))

    def test_nonfinite_not_accepted(self):
        with self.assertRaises(ValueError):
            difference(torch.tensor([float("nan")]), torch.ones(1))

    def test_traces_match_production_in_both_orientations(self):
        torch.manual_seed(7201)
        for algorithm in ("muon", "normuon"):
            for shape in ((8, 4), (4, 8), (4, 4)):
                with self.subTest(algorithm=algorithm, shape=shape):
                    gradient = torch.randn(shape)
                    state = {"momentum_buffer": torch.randn(shape)}
                    if algorithm == "normuon":
                        state["second_moment"] = torch.rand(shape[0], 1)
                    before = copy.deepcopy(state)
                    stages = operator_trace(
                        gradient,
                        state,
                        {"algorithm": algorithm, "momentum": 0.95, "beta2": 0.95, "ns_steps": 5},
                    )
                    self.assertIn("newton_schulz_5", stages)
                    self.assertEqual(stages["operator_update"].shape, gradient.shape)
                    for key in state:
                        self.assertTrue(torch.equal(state[key], before[key]))

    def test_trace_rejects_adam(self):
        with self.assertRaises(ValueError):
            operator_trace(torch.ones(2, 2), {}, {"algorithm": "adamw"})

    def test_direct_step_rejects_unobserved_state(self):
        with self.assertRaises(ValueError):
            direct_step(
                OptimizerConfig(name="muon", lr=0.001),
                {"weights": {"unexpected": torch.ones(2, 2)}},
                ["layers.0.weight"],
                [],
                [torch.ones(2, 2)],
            )

    def test_direct_replay_all_optimizers_preserves_entry(self):
        torch.manual_seed(4019)
        for algorithm in ("adamw", "muon", "normuon"):
            with self.subTest(algorithm=algorithm):
                model = torch.nn.Module()
                model.layers = torch.nn.ModuleList([torch.nn.Linear(4, 3, bias=False)])
                model.embedding = torch.nn.Embedding(5, 4)
                model.norm = torch.nn.LayerNorm(4)
                config = OptimizerConfig(name=algorithm, lr=0.001)
                optimizer, _ = build_optimizer(model, config)
                named = list(model.named_parameters())
                for _, p in named:
                    p.grad = torch.randn_like(p)
                optimizer.step()
                entry = {
                    "weights": copy.deepcopy(model.state_dict()),
                    "optimizer": copy.deepcopy(optimizer.state_dict()),
                }
                before = copy.deepcopy(entry)
                gradients = [torch.randn_like(p) for _, p in named]
                layout = optimizer_layout(optimizer, named)
                replayed = direct_step(config, entry, [n for n, _ in named], layout, gradients)
                for (_, p), gradient in zip(named, gradients, strict=True):
                    p.grad = gradient
                optimizer.step()
                exact_update(
                    replayed, {"weights": model.state_dict(), "optimizer": optimizer.state_dict()}
                )
                exact_update(entry, before)

    def test_direct_replay_rejects_wrong_parameter_routing(self):
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([torch.nn.Linear(4, 3, bias=False)])
        config = OptimizerConfig(name="muon", lr=0.001)
        optimizer, _ = build_optimizer(model, config)
        named = list(model.named_parameters())
        with self.assertRaises(ValueError):
            direct_step(
                config,
                {"weights": model.state_dict(), "optimizer": optimizer.state_dict()},
                [n for n, _ in named],
                [["wrong.name"]],
                [torch.ones_like(p) for _, p in named],
            )
