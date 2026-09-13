import random
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embed_optim_restore.paired import capture_rng, repeated_backward, restore_rng


def test_cpu_rng_round_trip_preserves_python_numpy_and_torch_streams():
    state = capture_rng()
    first = (random.random(), np.random.random(), torch.rand(4))
    restore_rng(state)
    second = (random.random(), np.random.random(), torch.rand(4))
    assert first[:2] == second[:2] and torch.equal(first[2], second[2])
    assert not torch.cuda.is_initialized()


def fixture():
    events = []
    model = SimpleNamespace(zero_grad=lambda **kw: events.append(("zero", kw)))

    @contextmanager
    def no_sync(**kw):
        assert kw == {"model": model}
        events.append("enter-no-sync")
        yield
        events.append("exit-no-sync")

    def step(actual_model, batch, count):
        assert actual_model is model and count is None
        events.append(("step", batch["index"]))
        batch["index"] = -1
        return torch.tensor(1.0)

    trainer = SimpleNamespace(
        model_wrapped=model,
        model_accepts_loss_kwargs=True,
        compute_loss_func=None,
        args=SimpleNamespace(device=torch.device("cpu")),
        get_batch_samples=lambda it, n, device: (list(it), None),
        accelerator=SimpleNamespace(
            no_sync=no_sync,
            gradient_state=SimpleNamespace(
                _set_sync_gradients=lambda flag: events.append(("sync", flag))
            ),
        ),
        training_step=step,
    )
    return trainer, events


def test_second_pass_uses_native_step_and_sync_order_without_updates():
    trainer, events = fixture()
    batches = [{"index": i} for i in range(4)]
    boundary = []
    losses = repeated_backward(trainer, batches, capture_rng(), lambda t: boundary.append(t))
    assert boundary == [trainer, trainer] and losses == [1.0] * 4
    assert [e for e in events if isinstance(e, tuple) and e[0] == "step"] == [
        ("step", i) for i in range(4)
    ]
    assert [e for e in events if isinstance(e, tuple) and e[0] == "sync"] == [
        ("sync", False)
    ] * 3 + [("sync", True)]
    assert events.count("enter-no-sync") == events.count("exit-no-sync") == 3
    assert batches == [{"index": i} for i in range(4)]


@pytest.mark.parametrize("kind", ["count", "normalization"])
def test_invalid_pair_rejected_before_zeroing(kind):
    trainer, events = fixture()
    batches = [{"index": i} for i in range(4)]
    if kind == "count":
        batches.pop()
    else:
        trainer.compute_loss_func = lambda: None
    with pytest.raises(ValueError):
        repeated_backward(trainer, batches, capture_rng(), lambda t: None)
    assert events == []


def test_non_none_actual_item_count_is_refused_before_zeroing():
    trainer, events = fixture()
    trainer.get_batch_samples = lambda it, n, device: (list(it), torch.tensor(32))
    with pytest.raises(ValueError, match="None item count"):
        repeated_backward(trainer, [{"index": i} for i in range(4)], capture_rng(), lambda t: None)
    assert events == []


@pytest.mark.parametrize("accepts_loss_kwargs", [False, True])
def test_original_installed_training_step_scales_none_count_for_both_flags(accepts_loss_kwargs):
    import hashlib
    import inspect
    from pathlib import Path

    from accelerate import Accelerator
    from transformers import Trainer

    assert (
        hashlib.sha256(Path(inspect.getsourcefile(Trainer)).read_bytes()).hexdigest()
        == "060eda6fcd587e79caeca7c3f246ea7bab1af410479c9776584470b8f4ac48a8"
    )
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    owner = SimpleNamespace(
        _prepare_context_parallel_inputs=lambda m, i: (nullcontext, i),
        _prepare_inputs=lambda i: i,
        compute_loss_context_manager=nullcontext,
        compute_loss=lambda m, i, **kw: m.weight.square().sum(),
        optimizer=None,
        state=SimpleNamespace(global_step=313),
        args=SimpleNamespace(torch_empty_cache_steps=None, optim="adamw_torch", n_gpu=1),
        model_accepts_loss_kwargs=accepts_loss_kwargs,
        compute_loss_func=None,
        current_gradient_accumulation_steps=4,
        accelerator=Accelerator(cpu=True, gradient_accumulation_steps=1),
    )
    assert (
        Trainer._get_num_items_in_batch(
            owner, [{"query_input_ids": torch.ones(1)}], torch.device("cpu")
        )
        is None
    )
    result = Trainer.training_step(owner, model, {}, num_items_in_batch=None)
    assert result.item() == 1.0 and model.weight.grad.item() == 1.0
