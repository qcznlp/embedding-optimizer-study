"""CPU-only guards for the separately executed, actual four-rank diagnostic."""

from types import SimpleNamespace

import pytest
import torch

from embed_optim.config import OptimizerConfig, RunConfig
from scripts import audit_dense_ddp_accumulation as audit


def rank_batches():
    return [
        [list(range(rank * 32 + batch * 8, rank * 32 + batch * 8 + 8)) for batch in range(4)]
        for rank in range(4)
    ]


def test_exact_global_batch_has_128_distinct_query_groups():
    ids = audit.validate_global_ids(rank_batches(), set())
    assert len(ids) == 128 and set(ids) == set(range(128))


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate",
        "prior_step_overlap",
        "missing_rank",
        "missing_microbatch",
        "wrong_microbatch",
        "unknown_id",
    ],
)
def test_global_batch_rejects_sampling_or_accumulation_drift(kind):
    batches, seen = rank_batches(), set()
    if kind == "duplicate":
        batches[3][3][-1] = 0
    elif kind == "prior_step_overlap":
        seen.add(0)
    elif kind == "missing_rank":
        batches.pop()
    elif kind == "missing_microbatch":
        batches[0].pop()
    elif kind == "wrong_microbatch":
        batches[0][0].pop()
    else:
        batches[0][0][0] = 256
    with pytest.raises(ValueError):
        audit.validate_global_ids(batches, seen)


def test_raw_gradient_check_rejects_scaling_that_norm_clipping_can_hide():
    expected = torch.tensor([2.0, 3.0, -4.0])
    actual = 4 * expected
    torch.testing.assert_close(actual / actual.norm(), expected / expected.norm())
    with pytest.raises(AssertionError):
        audit.compare_gradients([("weight", actual)], [("weight", expected)])


def test_gradient_check_accepts_identical_values_and_keeps_inputs_unchanged():
    value = torch.randn(8, 4)
    before = value.clone()
    result = audit.compare_gradients([("weight", value)], [("weight", value.clone())])
    assert result == [{"name": "weight", "elements": 32, "max_absolute_error": 0.0}]
    assert torch.equal(value, before)


@pytest.mark.parametrize("scale", [0.25, 1.0])
def test_scale_observation_does_not_turn_rescaled_gradients_into_a_pass(scale):
    reference = torch.tensor([2.0, 3.0, -4.0])
    actual = reference * scale
    result = audit.gradient_scale_diagnostic([("weight", actual)], [("weight", reference)])
    assert result["least_squares_scale"] == pytest.approx(scale)
    assert result["actual_to_reference_norm_ratio"] == pytest.approx(scale)
    assert result["cosine"] == pytest.approx(1.0)
    if scale != 1.0:
        with pytest.raises(AssertionError):
            audit.compare_gradients([("weight", actual)], [("weight", reference)])


@pytest.mark.parametrize("kind", ["missing", "name", "shape", "nonfinite"])
def test_gradient_check_rejects_incomplete_or_invalid_values(kind):
    value = torch.ones(3)
    other_name = "other" if kind == "name" else "weight"
    other = None if kind == "missing" else torch.ones(4 if kind == "shape" else 3)
    if kind == "nonfinite":
        other[0] = float("nan")
    with pytest.raises(ValueError):
        audit.compare_gradients([("weight", value)], [(other_name, other)])


def test_cpu_overrides_preserve_the_actual_production_batch_and_clipping_declaration(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(
        audit, "SentenceTransformerTrainingArguments", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    config = RunConfig(
        run_id="fixture",
        model_family="dense",
        optimizer=OptimizerConfig("muon", 3e-4),
        model_name="fixture",
        dataset_path="fixture",
    )
    args, overrides = audit.cpu_arguments(config, tmp_path)
    assert (args.per_device_train_batch_size, args.gradient_accumulation_steps) == (8, 4)
    assert args.max_grad_norm == 1.0
    assert args.gradient_checkpointing and args.gradient_checkpointing_kwargs == {
        "use_reentrant": False
    }
    assert args.dataloader_drop_last and args.train_sampling_strategy == "group_by_length"
    assert args.use_cpu and not args.bf16 and not args.tf32
    assert args.report_to == [] and args.max_steps == 2
    assert "gradient_accumulation_steps" not in overrides
    assert "per_device_train_batch_size" not in overrides


def test_synthetic_dataset_is_not_real_training_data():
    rows = audit.synthetic_rows()
    assert len(rows) == 256
    assert {row["example_id"] for row in rows} == set(range(256))
    assert all(len([k for k in row if k.startswith("negative_")]) == 7 for row in rows)
    assert all("item" in row["query"] for row in rows)


def test_tiny_modernbert_fixture_loads_offline_with_real_parameter_routing(tmp_path):
    from embed_optim.optimizers import parameter_partition

    audit.create_model_fixture(tmp_path / "tiny-model")
    model = audit.load_fixture(tmp_path / "tiny-model")
    partition = parameter_partition(model)
    assert len(partition["hidden"]) == 8
    assert len(partition["aux_decay"]) == 1
    assert model[0].auto_model.is_gradient_checkpointing
    assert model[0].can_flatten_inputs is False
    assert all(p.device.type == "cpu" for p in model.parameters())
