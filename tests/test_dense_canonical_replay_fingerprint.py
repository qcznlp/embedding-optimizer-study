from collections import OrderedDict

import pytest
import torch

from scripts.audit_dense_gpu_canonical_replay import fingerprint, replay_fingerprint


def test_mapping_order_is_not_a_tensor_difference():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


def test_fingerprint_retains_container_key_and_tensor_types():
    assert fingerprint({1: 2}) != fingerprint({"1": 2})
    assert fingerprint({"a": 1}) != fingerprint(OrderedDict(a=1))
    assert fingerprint([1]) != fingerprint((1,))
    assert fingerprint(torch.ones(2)) != fingerprint(torch.ones(2, dtype=torch.float64))


def test_small_tensor_difference_not_hidden_by_hash():
    assert fingerprint(torch.tensor([1.0])) != fingerprint(torch.tensor([1.0 + 1e-7]))


def test_nonfinite_tensor_rejected():
    with pytest.raises(ValueError):
        fingerprint(torch.tensor([float("nan")]))


def test_restart_filters_only_already_consumed_steps():
    snapshot = {
        "entry_states": {1: {"w": torch.ones(1)}},
        "gradient_values": {1: {}},
        "trace": [{"global_step": 0}, {"global_step": 1}],
        "gradients": [{"global_step": 0}, {"global_step": 1}],
        "epoch": 1.0,
    }
    actual = replay_fingerprint(snapshot, 1)
    expected = {
        "trace": [{"global_step": 1}],
        "gradients": [{"global_step": 1}],
        "epoch": 1.0,
        "entry_states": snapshot["entry_states"],
        "gradient_values": snapshot["gradient_values"],
    }
    assert actual == {
        "entry": fingerprint(snapshot["entry_states"][1]),
        "replayed_state": fingerprint(expected),
    }


def test_all_future_entries_and_raw_gradient_values_are_sealed():
    snapshot = {
        "entry_states": {1: {"w": torch.ones(1)}, 2: {"w": torch.ones(1)}},
        "gradient_values": {1: {"raw": torch.ones(1)}, 2: {"raw": torch.ones(1)}},
        "trace": [{"global_step": 1}, {"global_step": 2}],
        "gradients": [{"global_step": 1}, {"global_step": 2}],
    }
    original = replay_fingerprint(snapshot, 1)
    snapshot["entry_states"][2]["w"].add_(1)
    assert original != replay_fingerprint(snapshot, 1)
    snapshot["entry_states"][2]["w"].sub_(1)
    assert original == replay_fingerprint(snapshot, 1)
    snapshot["gradient_values"][2]["raw"].add_(1)
    assert original != replay_fingerprint(snapshot, 1)


def test_empty_replay_rejected():
    with pytest.raises(ValueError):
        replay_fingerprint({"trace": [], "gradients": []}, 1)
