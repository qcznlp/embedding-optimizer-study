from contextlib import contextmanager
from types import SimpleNamespace
import random
import numpy as np
import pytest
import torch
from paired import capture_rng, restore_rng, repeated_backward


def test_cpu_rng_round_trip_preserves_python_numpy_and_torch_streams():
    state = capture_rng()
    first = (random.random(), np.random.random(), torch.rand(4))
    restore_rng(state)
    second = (random.random(), np.random.random(), torch.rand(4))
    assert first[:2] == second[:2] and torch.equal(first[2], second[2])
    assert not torch.cuda.is_initialized()


def fixture():
    events = []
    model = SimpleNamespace(zero_grad=lambda **kw: events.append(('zero', kw)))
    @contextmanager
    def no_sync(**kw):
        assert kw == {'model': model}
        events.append('enter-no-sync')
        yield
        events.append('exit-no-sync')
    def step(actual_model, batch, count):
        assert actual_model is model and count is None
        events.append(('step', batch['index']))
        batch['index'] = -1
        return torch.tensor(1.)
    trainer = SimpleNamespace(model_wrapped=model, model_accepts_loss_kwargs=False,
        accelerator=SimpleNamespace(no_sync=no_sync,
            gradient_state=SimpleNamespace(_set_sync_gradients=lambda flag: events.append(('sync', flag)))),
        training_step=step)
    return trainer, events


def test_second_pass_uses_native_step_and_sync_order_without_updates():
    trainer, events = fixture()
    batches = [{'index': i} for i in range(4)]
    boundary = []
    losses = repeated_backward(trainer, batches, capture_rng(), lambda t: boundary.append(t))
    assert boundary == [trainer, trainer] and losses == [1.]*4
    assert [e for e in events if isinstance(e,tuple) and e[0]=='step'] == [('step',i) for i in range(4)]
    assert [e for e in events if isinstance(e,tuple) and e[0]=='sync'] == [('sync',False)]*3+[('sync',True)]
    assert events.count('enter-no-sync') == events.count('exit-no-sync') == 3
    assert batches == [{'index':i} for i in range(4)]


@pytest.mark.parametrize('kind', ['count', 'normalization'])
def test_invalid_pair_rejected_before_zeroing(kind):
    trainer, events = fixture()
    batches = [{'index':i} for i in range(4)]
    if kind == 'count':
        batches.pop()
    else:
        trainer.model_accepts_loss_kwargs = True
    with pytest.raises(ValueError):
        repeated_backward(trainer, batches, capture_rng(), lambda t: None)
    assert events == []
