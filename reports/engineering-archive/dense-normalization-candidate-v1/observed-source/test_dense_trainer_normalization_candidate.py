"""Repair guards; the standalone four-rank numerical audit is separate."""

from types import SimpleNamespace

import pytest
from accelerate import DistributedType

from scripts import audit_dense_trainer_normalization as audit
from scripts import dense_trainer_normalization_candidate as candidate


def mock_trainer(steps=4, accelerator_steps=4, distributed=DistributedType.MULTI_CPU):
    return SimpleNamespace(
        args=SimpleNamespace(gradient_accumulation_steps=steps),
        accelerator=SimpleNamespace(
            gradient_accumulation_steps=accelerator_steps, distributed_type=distributed
        ),
        is_deepspeed_enabled=False,
        is_fsdp_enabled=False,
    )


@pytest.mark.parametrize("steps", [1, 2, 4, 8])
def test_owner_assignment_keeps_trainer_batch_schedule(steps):
    trainer = mock_trainer(steps, steps)
    receipt = candidate.assign_normalization_owner(trainer)
    assert trainer.args.gradient_accumulation_steps == steps
    assert trainer.accelerator.gradient_accumulation_steps == 1
    assert receipt["accelerator_divisor_before"] == steps
    assert receipt["runtime_deployed"] is False


@pytest.mark.parametrize("kind", ["deepspeed", "fsdp", "xla", "wrong_divisor", "zero", "float"])
def test_owner_assignment_rejects_unsupported_runtime_before_mutation(kind):
    trainer = mock_trainer()
    if kind == "deepspeed":
        trainer.is_deepspeed_enabled = True
    elif kind == "fsdp":
        trainer.is_fsdp_enabled = True
    elif kind == "xla":
        trainer.accelerator.distributed_type = DistributedType.XLA
    elif kind == "wrong_divisor":
        trainer.accelerator.gradient_accumulation_steps = 8
    elif kind == "zero":
        trainer.args.gradient_accumulation_steps = 0
    else:
        trainer.args.gradient_accumulation_steps = 4.0
    before = trainer.accelerator.gradient_accumulation_steps
    with pytest.raises(ValueError):
        candidate.assign_normalization_owner(trainer)
    assert trainer.accelerator.gradient_accumulation_steps == before


def test_candidate_rejects_loss_replacement():
    with pytest.raises(ValueError, match="explicit Dense mean loss"):
        candidate.SingleNormalizationTrainerCandidate(loss=object())


def test_stack_guard_verifies_actual_pinned_source_bytes():
    records = candidate.verify_stack()
    assert {r["name"]: r["sha256"] for r in records} == candidate.SOURCE_HASHES


def test_stack_guard_rejects_unvalidated_package_version(monkeypatch):
    monkeypatch.setattr(candidate.importlib.metadata, "version", lambda _: "future-version")
    with pytest.raises(ValueError, match="Unvalidated normalization stack"):
        candidate.verify_stack()


def batches(step):
    count = audit.ACCUMULATION_COUNTS[step]
    start = 128 * step
    return [
        [
            list(range(start + rank * 8 * count + b * 8, start + rank * 8 * count + (b + 1) * 8))
            for b in range(count)
        ]
        for rank in range(4)
    ]


def test_complete_and_partial_groups_cover_one_epoch_without_replay():
    seen = set()
    for step, expected in enumerate((128, 128, 32)):
        ids = audit.validate_ids(batches(step), seen, step)
        assert len(ids) == expected
        seen.update(ids)
    assert seen == set(range(288))


@pytest.mark.parametrize(
    "kind",
    [
        "tail_padded",
        "missing_rank",
        "duplicate",
        "replayed",
        "unknown",
        "wrong_micro",
        "wrong_step",
    ],
)
def test_tail_ids_reject_replay_or_silent_batch_shape_changes(kind):
    values, seen, step = batches(2), set(range(256)), 2
    if kind == "tail_padded":
        values = [x * 4 for x in values]
    elif kind == "missing_rank":
        values.pop()
    elif kind == "duplicate":
        values[3][0][-1] = values[0][0][0]
    elif kind == "replayed":
        values[0][0][0] = 0
    elif kind == "unknown":
        values[0][0][0] = 288
    elif kind == "wrong_micro":
        values[0][0].pop()
    else:
        step = 3
    with pytest.raises(ValueError):
        audit.validate_ids(values, seen, step)


def test_tail_item_vocabulary_is_not_out_of_vocabulary(tmp_path):
    audit.create_fixture(tmp_path / "model")
    model = audit.previous.load_fixture(tmp_path / "model")
    tokenizer = model[0].tokenizer
    encoded = tokenizer(" ".join(f"item{i}" for i in range(288)), add_special_tokens=False)
    assert len(encoded["input_ids"]) == 288
    assert tokenizer.unk_token_id not in encoded["input_ids"]


@pytest.mark.parametrize("wrong", ["cadence", "rate"])
def test_scheduler_check_rejects_changed_update_or_lr_cadence(wrong):
    trainer = SimpleNamespace(
        args=SimpleNamespace(get_warmup_steps=lambda _: 1),
        lr_scheduler=SimpleNamespace(
            state_dict=lambda: {"last_epoch": 4 if wrong == "cadence" else 2}
        ),
        optimizer=SimpleNamespace(
            param_groups=[
                {
                    "algorithm": "adamw",
                    "initial_lr": 0.001,
                    "lr": 0.001 if wrong == "rate" else 0.0005,
                }
            ]
        ),
    )
    with pytest.raises(ValueError):
        audit.check_schedule(trainer, 2)
