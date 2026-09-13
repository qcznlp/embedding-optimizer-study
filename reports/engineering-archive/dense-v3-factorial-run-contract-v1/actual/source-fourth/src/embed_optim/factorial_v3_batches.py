"""Exact 50K all-group batching for the four-rank reset continuation.

The first 390 updates retain the upstream sample order and 16 batches of eight
query groups each. The final 80 groups form 16 batches of five. Sharding batches
round-robin onto four ranks gives four equally sized micro-batches per update,
so the audited mean-loss / accumulation / DDP reductions remain a group mean.

This requires an explicit factorial Trainer integration: ordinary ST DDP forces
drop_last=True, which the factory below rejects. No primary defaults are changed.
"""

from __future__ import annotations

import torch
from torch.utils.data import BatchSampler, RandomSampler

ROWS = 50000
WORLD_SIZE = 4
ACCUMULATION = 4
GLOBAL_BATCH = 128
MICRO_BATCH = 8
ORDER_SEEDS = (314159, 271828, 161803)
SCOPE = "dense-v3-factorial-complete-balanced-tail-v1"


class FactorialBatchSampler(BatchSampler):
    """Variable-size batch sampler preserving a supplied complete permutation."""

    def __init__(self, sampler):
        if len(sampler) != ROWS:
            raise ValueError("Factorial batching requires exactly 50,000 query groups")
        super().__init__(sampler, batch_size=MICRO_BATCH, drop_last=False)
        # Accelerate explicitly requires this marker plus even_batches=False
        # for genuinely variable micro-batches. Do not pretend tail batches are 8.
        self.batch_size = None

    def __len__(self):
        return 391 * WORLD_SIZE * ACCUMULATION

    def __iter__(self):
        order = list(self.sampler)
        if (
            len(order) != ROWS
            or any(type(index) is not int for index in order)
            or sorted(order) != list(range(ROWS))
        ):
            raise ValueError("Factorial sampler must yield each query group exactly once")
        for begin in range(0, ROWS, GLOBAL_BATCH):
            update = order[begin : begin + GLOBAL_BATCH]
            if len(update) not in (128, 80):
                raise ValueError("Unexpected factorial logical-batch tail")
            size = len(update) // (WORLD_SIZE * ACCUMULATION)
            for offset in range(0, len(update), size):
                yield update[offset : offset + size]


def batch_sampler_factory(
    dataset, batch_size, drop_last, valid_label_columns=None, generator=None, seed=0
):
    """ST callable hook; explicit non-dropping factorial arguments are required."""
    if type(batch_size) is not int or batch_size != MICRO_BATCH or drop_last is not False:
        raise ValueError("Require explicit factorial batch size 8 and non-dropping arguments")
    if not isinstance(generator, torch.Generator) or generator.initial_seed() not in ORDER_SEEDS:
        raise ValueError("Require one of the three predeclared branch-order generators")
    if seed not in (None, 0):
        raise ValueError("The supplied order generator, not a second seed, owns shuffling")
    return FactorialBatchSampler(RandomSampler(dataset, generator=generator))
