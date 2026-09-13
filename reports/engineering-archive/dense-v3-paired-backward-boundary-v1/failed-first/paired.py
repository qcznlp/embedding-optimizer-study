"""Same-input second backward through the unchanged native training_step."""
from contextlib import nullcontext
import copy
import random
import numpy as np
import torch


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state().clone(),
            'cuda': torch.cuda.get_rng_state().clone() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        if not torch.cuda.is_initialized():
            raise ValueError('Do not initialize a new GPU while restoring diagnostic RNG')
        torch.cuda.set_rng_state(state['cuda'])


def repeated_backward(trainer, raw_batches, rng, require_boundary):
    """No optimizer, clipping, scheduler, or original numerical function replacement."""
    require_boundary(trainer)
    if len(raw_batches) != 4 or trainer.model_accepts_loss_kwargs:
        raise ValueError('Require four fixed explicit-loss batches without item-count scaling')
    trainer.model_wrapped.zero_grad(set_to_none=True)
    restore_rng(rng)
    losses = []
    for index, batch in enumerate(raw_batches):
        trainer.accelerator.gradient_state._set_sync_gradients(index == 3)
        context = nullcontext() if index == 3 else trainer.accelerator.no_sync(model=trainer.model_wrapped)
        with context:
            loss = trainer.training_step(trainer.model_wrapped, copy.deepcopy(batch), None)
        losses.append(float(loss.detach().cpu()))
    require_boundary(trainer)
    return losses
