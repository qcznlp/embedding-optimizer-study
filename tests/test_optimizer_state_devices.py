"""CPU adapter controls; real four-GPU resume equivalence is a separate check."""

import copy

import pytest
import torch

from embed_optim.optimizer_state_devices import place_adam_step_counters


def fixture(muon=False):
    first = torch.nn.Parameter(torch.ones(2, 2))
    second = torch.nn.Parameter(torch.ones(2))
    groups = [
        {
            "params": [first],
            "param_names": ["hidden"],
            "algorithm": "muon" if muon else "adamw",
            "lr": 0.01,
        },
        {"params": [second], "param_names": ["norm"], "algorithm": "adamw", "lr": 0.001},
    ]
    optimizer = torch.optim.Optimizer(groups, defaults={})
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            optimizer.state[parameter] = (
                {"momentum_buffer": torch.ones_like(parameter)}
                if group["algorithm"] == "muon"
                else {
                    "step": torch.tensor(313.0),
                    "exp_avg": torch.ones_like(parameter),
                    "exp_avg_sq": torch.full_like(parameter, 2),
                }
            )
    return optimizer, first, second


@pytest.mark.parametrize("muon,expected", [(False, 2), (True, 1)])
def test_named_value_preserving_placement(muon, expected):
    optimizer, _, _ = fixture(muon)
    before = copy.deepcopy(optimizer.state_dict())
    pointers = {
        id(p): {key: id(value) for key, value in state.items()}
        for p, state in optimizer.state.items()
    }
    report = place_adam_step_counters(optimizer, expected_step=313)
    assert report["named_adam_counters"] == expected
    assert report["counters_moved"] == 0
    assert report["counter_values_bitwise_preserved"]
    after = optimizer.state_dict()
    assert after["param_groups"] == before["param_groups"]
    for pid, state in before["state"].items():
        for key, tensor in state.items():
            assert torch.equal(tensor, after["state"][pid][key])
    assert pointers == {
        id(p): {key: id(value) for key, value in state.items()}
        for p, state in optimizer.state.items()
    }
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("expected_step", [True, 0, -1, 313.0, 314, None])
def test_untrusted_or_different_step_rejected(expected_step):
    optimizer, _, _ = fixture()
    with pytest.raises(ValueError):
        place_adam_step_counters(optimizer, expected_step=expected_step)


@pytest.mark.parametrize(
    "bad_step",
    [
        313,
        torch.tensor([313.0]),
        torch.tensor(313.0, dtype=torch.float64),
        torch.tensor(float("nan")),
        torch.tensor(float("inf")),
    ],
)
def test_invalid_counter_rejected(bad_step):
    optimizer, first, _ = fixture()
    optimizer.state[first]["step"] = bad_step
    with pytest.raises(ValueError):
        place_adam_step_counters(optimizer, expected_step=313)


@pytest.mark.parametrize("field", ["step", "exp_avg", "exp_avg_sq"])
def test_missing_state_rejected(field):
    optimizer, first, _ = fixture()
    del optimizer.state[first][field]
    with pytest.raises(ValueError):
        place_adam_step_counters(optimizer, expected_step=313)


@pytest.mark.parametrize("field", ["exp_avg", "exp_avg_sq"])
@pytest.mark.parametrize("bad", [torch.ones(3), torch.ones(2, 2, dtype=torch.float64)])
def test_malformed_moments_rejected(field, bad):
    optimizer, first, _ = fixture()
    optimizer.state[first][field] = bad
    with pytest.raises(ValueError):
        place_adam_step_counters(optimizer, expected_step=313)


def test_duplicate_parameter_rejected():
    optimizer, first, _ = fixture()
    optimizer.param_groups[1]["params"] = [first]
    with pytest.raises(ValueError):
        place_adam_step_counters(optimizer, expected_step=313)


@pytest.mark.parametrize("labels", [None, [], ["hidden"], [True]])
def test_invalid_names_rejected(labels):
    optimizer, _, _ = fixture()
    optimizer.param_groups[1]["param_names"] = labels
    with pytest.raises(ValueError):
        place_adam_step_counters(optimizer, expected_step=313)


def test_unknown_group_rejected():
    optimizer, _, _ = fixture()
    optimizer.param_groups[1]["algorithm"] = "unknown"
    with pytest.raises(ValueError):
        place_adam_step_counters(optimizer, expected_step=313)


def test_extra_state_rejected_before_any_placement():
    optimizer, first, second = fixture()
    original = optimizer.state[first]["step"]
    optimizer.state[second]["extra"] = 1
    with pytest.raises(ValueError):
        place_adam_step_counters(optimizer, expected_step=313)
    assert optimizer.state[first]["step"] is original


def test_empty_state_is_not_initialized():
    optimizer, first, _ = fixture()
    del optimizer.state[first]
    with pytest.raises(ValueError):
        place_adam_step_counters(optimizer, expected_step=313)
    assert first not in optimizer.state


def test_non_fp32_parameter_rejected():
    optimizer, first, _ = fixture()
    first.data = first.data.double()
    with pytest.raises(ValueError):
        place_adam_step_counters(optimizer, expected_step=313)
