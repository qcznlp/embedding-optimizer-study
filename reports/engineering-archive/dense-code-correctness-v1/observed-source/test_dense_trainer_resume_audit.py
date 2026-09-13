"""Guards for the CPU diagnostic; these tests do not certify primary resumption."""

import copy
from unittest.mock import Mock

import pytest
import torch

from scripts import audit_dense_trainer_resume as audit


def test_tensor_digest_includes_dtype_and_shape():
    values = torch.arange(8, dtype=torch.float32)
    assert audit.tensor_hash(values) == audit.tensor_hash(values.clone())
    assert audit.tensor_hash(values) != audit.tensor_hash(values.reshape(2, 4))
    assert audit.tensor_hash(values) != audit.tensor_hash(values.to(torch.float64))


@pytest.mark.parametrize("kind", ["shape", "dtype", "value", "nan", "keys", "length", "type"])
def test_exact_tree_rejects_a_mismatch(kind):
    left = {"x": torch.ones(4), "items": [1, 2]}
    right = copy.deepcopy(left)
    if kind == "shape":
        right["x"] = right["x"].reshape(2, 2)
    elif kind == "dtype":
        right["x"] = right["x"].double()
    elif kind == "value":
        right["x"][0] += 1e-6
    elif kind == "nan":
        left["x"][0] = right["x"][0] = float("nan")
    elif kind == "keys":
        right["extra"] = True
    elif kind == "length":
        right["items"].append(3)
    else:
        right["items"] = tuple(right["items"])
    with pytest.raises(AssertionError):
        audit.assert_exact_tree(left, right)


def test_numerical_measurement_does_not_relabel_a_bitwise_failure():
    left = torch.ones(4)
    right = left.clone()
    right[0] += 1e-6
    [record] = audit.measure_tensor_tree(left, right)
    assert not record["bitwise_equal"]
    assert record["within_prior_gradient_tolerances"]
    assert record["max_absolute_error"] > 0
    with pytest.raises(AssertionError):
        audit.assert_exact_tree(left, right)


def test_numerical_measurement_retains_failed_prior_tolerance():
    [record] = audit.measure_tensor_tree(torch.ones(4), torch.ones(4) * 2)
    assert not record["within_prior_gradient_tolerances"]


@pytest.mark.parametrize("location", [torch.device("cpu"), "cpu", None])
def test_cpu_adapter_preserves_unindexed_loads(tmp_path, location):
    original = Mock(return_value="loaded")
    records = []
    file = tmp_path / "scheduler.pt"
    result = audit.canonical_cpu_load(
        original, tmp_path, records, file, map_location=location, weights_only=True
    )
    assert result == "loaded"
    original.assert_called_once_with(file, map_location=location, weights_only=True)
    assert records == []


def test_cpu_adapter_is_limited_to_the_owned_optimizer_payload(tmp_path):
    original = Mock(return_value="loaded")
    records = []
    file = tmp_path / "optimizer.pt"
    audit.canonical_cpu_load(
        original, tmp_path, records, file, map_location=torch.device("cpu:0"), weights_only=True
    )
    original.assert_called_once_with(file, map_location="cpu", weights_only=True)
    assert records == [{"path": str(file.resolve()), "from": "cpu:0", "to": "cpu"}]
    with pytest.raises(ValueError, match="limited"):
        audit.canonical_cpu_load(
            original, tmp_path, records, tmp_path / "other.pt", map_location=torch.device("cpu:0")
        )


def snapshot():
    return {
        "weights": {"x": torch.ones(3)},
        "optimizer": {"step": torch.tensor(3.0)},
        "scheduler": {"last_epoch": 3},
        "rng": {"cpu": "final"},
        "global_step": 3,
        "epoch": 1.0,
        "trace": [
            {"global_step": i, "row_ids": [i], "rng_before": "b", "rng_after": "a", "loss": 0.1}
            for i in range(3)
        ],
        "gradients": [
            {
                "global_step": i,
                "learning_rates": [0.001],
                "raw": ["r"],
                "clipped": ["c"],
                "raw_norm": 1.0,
            }
            for i in range(3)
        ],
        "gradient_values": {
            i: {"raw": [torch.ones(3)], "clipped": [torch.ones(3)]} for i in range(3)
        },
        "entry_states": {i: {"weights": torch.ones(3), "optimizer": {"step": i}} for i in range(3)},
    }


def resumed_snapshot(reference, step):
    resumed = copy.deepcopy(reference)
    for name in ("trace", "gradients"):
        resumed[name] = [r for r in resumed[name] if r["global_step"] >= step]
    for name in ("gradient_values", "entry_states"):
        resumed[name] = {k: v for k, v in resumed[name].items() if k >= step}
    return resumed


@pytest.mark.parametrize("step", [1, 2])
def test_replay_checks_skip_only_the_saved_prefix(step):
    reference = snapshot()
    resumed = resumed_snapshot(reference, step)
    audit.verify_replay(reference, resumed, step)
    report = audit.measure_replay(reference, resumed, step)
    assert report["bitwise_replay"]["passed"]
    assert report["exact_loaded_entry_state"]


@pytest.mark.parametrize("kind", ["entry_weights", "entry_optimizer", "row_id", "rng", "scheduler"])
def test_measurement_cannot_hide_a_state_or_data_replay_error(kind):
    reference = snapshot()
    resumed = resumed_snapshot(reference, 1)
    if kind == "entry_weights":
        resumed["entry_states"][1]["weights"][0] += 1e-6
    elif kind == "entry_optimizer":
        resumed["entry_states"][1]["optimizer"]["step"] = 2
    elif kind == "row_id":
        resumed["trace"][0]["row_ids"] = [7]
    elif kind == "rng":
        resumed["trace"][0]["rng_before"] = "wrong"
    else:
        resumed["scheduler"]["last_epoch"] = 2
    with pytest.raises(AssertionError):
        audit.measure_replay(reference, resumed, 1)
