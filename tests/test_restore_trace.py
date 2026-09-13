import copy
from types import SimpleNamespace

import pytest
import torch

from embed_optim_restore.trace import Capture, cpu_tree, fingerprint, require_stop_boundary


def test_hooks_observe_but_never_replace_or_mutate_inputs_and_gradients():
    expected = [[{"input_ids": torch.tensor([[i, i + 1]])}] for i in range(4)]
    capture = Capture(expected, ["weight"])
    actual = copy.deepcopy(expected)
    for i in range(4):
        assert capture.loss_input(None, (actual[i], None)) is None
        grad = torch.tensor([float(i), -float(i)], dtype=torch.float32)
        before = grad.clone()
        assert capture.leaf_hook("weight")(grad) is None
        assert torch.equal(grad, before)
    capture.require_complete()
    assert fingerprint(expected) == fingerprint(actual)
    assert torch.equal(capture.local_sum["weight"], torch.tensor([6.0, -6.0]))


@pytest.mark.parametrize("kind", ["token", "shape", "dtype", "extra"])
def test_feature_mismatch_is_not_admitted(kind):
    base = [[{"input_ids": torch.tensor([[1, 2]])}] for _ in range(4)]
    capture = Capture(base, ["weight"])
    changed = copy.deepcopy(base[0])
    if kind == "token":
        changed[0]["input_ids"][0, 0] = 3
    elif kind == "shape":
        changed[0]["input_ids"] = changed[0]["input_ids"].reshape(2, 1)
    elif kind == "dtype":
        changed[0]["input_ids"] = changed[0]["input_ids"].float()
    else:
        for v in base:
            capture.loss_input(None, (v, None))
    with pytest.raises(ValueError):
        capture.loss_input(None, (changed, None))


def test_incomplete_or_extra_gradient_events_refused():
    capture = Capture([[]] * 4, ["weight"])
    with pytest.raises(ValueError):
        capture.require_complete()
    for _ in range(4):
        capture.loss_input(None, ([], None))
        capture.leaf_hook("weight")(torch.ones(1))
    capture.require_complete()
    with pytest.raises(ValueError):
        capture.leaf_hook("weight")(torch.ones(1))


@pytest.mark.parametrize("step,raw,accumulation", [(314, 313, 4), (313, 314, 4), (313, 313, 1)])
def test_wrong_stop_boundary_refused(step, raw, accumulation):
    trainer = SimpleNamespace(
        state=SimpleNamespace(global_step=step),
        _raw_optimizer=lambda: SimpleNamespace(completed_steps=raw),
        current_gradient_accumulation_steps=accumulation,
    )
    with pytest.raises(ValueError):
        require_stop_boundary(trainer)


def test_cpu_copy_is_independent_and_recursive():
    value = {"a": [torch.tensor([1.0])], "b": (None, 3)}
    copied = cpu_tree(value)
    copied["a"][0].zero_()
    assert value["a"][0].item() == 1 and copied["b"] == value["b"]
