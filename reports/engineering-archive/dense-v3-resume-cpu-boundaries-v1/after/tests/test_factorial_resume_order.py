"""CPU index-order regression, not token/gradient or GPU-resume equivalence."""

import pytest
import torch
from accelerate.data_loader import (
    BatchSamplerShard,
    DataLoaderShard,
    SeedableRandomSampler,
    skip_first_batches,
)

from embed_optim.factorial_v3_batches import ORDER_SEEDS, FactorialBatchSampler


def local_batches(seed, rank, skipped):
    dataset = range(50000)
    sampler = SeedableRandomSampler(
        dataset, generator=torch.Generator().manual_seed(seed), data_seed=seed
    )
    shard = BatchSamplerShard(
        FactorialBatchSampler(sampler),
        num_processes=4,
        process_index=rank,
        split_batches=False,
        even_batches=False,
    )
    loader = DataLoaderShard(dataset, batch_sampler=shard, device=torch.device("cpu"))
    if skipped:
        loader = skip_first_batches(loader, skipped)
    loader.set_epoch(0)
    return [batch.tolist() for batch in loader]


@pytest.mark.parametrize("seed", ORDER_SEEDS)
@pytest.mark.parametrize("rank", range(4))
def test_resume_after_step_313_preserves_entire_remaining_rank_order(seed, rank):
    full = local_batches(seed, rank, 0)
    resumed = local_batches(seed, rank, 1252)
    assert len(full) == 1564 and len(resumed) == 312
    assert full[313 * 4 :] == resumed
    assert sum(map(len, resumed)) == 2484
    assert all(len(batch) == 8 for batch in resumed[:-4])
    assert all(len(batch) == 5 for batch in resumed[-4:])
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("seed,skipped", [(314159, 1251), (271828, 1252)])
def test_index_comparison_detects_wrong_skip_or_order_seed(seed, skipped):
    assert local_batches(314159, 0, 1252) != local_batches(seed, 0, skipped)
