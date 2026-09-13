"""Actual ST hook/Accelerate shards and exact gradient coefficients; CPU only."""

import copy
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch
from accelerate.data_loader import BatchSamplerShard
from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainer
from torch.utils.data import RandomSampler

from embed_optim import factorial_v3_batches as f
from embed_optim.losses import ExplicitDenseInfoNCELoss


def batches(seed, *, rank=None):
    data = Dataset.from_dict({"sample_id": range(f.ROWS)})
    owner = SimpleNamespace(args=SimpleNamespace(batch_sampler=f.batch_sampler_factory))
    actual = SentenceTransformerTrainer.get_batch_sampler(
        owner,
        data,
        batch_size=8,
        drop_last=False,
        valid_label_columns=[],
        generator=torch.Generator().manual_seed(seed),
    )
    if rank is None:
        return list(actual)
    shard = BatchSamplerShard(
        actual, num_processes=4, process_index=rank, split_batches=False, even_batches=False
    )
    result = list(shard)
    assert len(result) == len(shard) == 1564
    return result


@pytest.mark.parametrize("seed", f.ORDER_SEEDS)
def test_actual_hook_retains_exact_original_permutation_and_all_50000_groups(seed):
    grouped = batches(seed)
    original = list(RandomSampler(range(f.ROWS), generator=torch.Generator().manual_seed(seed)))
    assert [i for b in grouped for i in b] == original
    assert len(grouped) == 6256
    assert [len(b) for b in grouped] == [8] * 6240 + [5] * 16
    assert [i for b in grouped for i in b[:]] == [i for b in batches(seed) for i in b]


@pytest.mark.parametrize("seed", f.ORDER_SEEDS)
def test_actual_four_shards_cover_every_group_once_with_equal_micro_sizes(seed):
    ranks = [batches(seed, rank=r) for r in range(4)]
    ids = [i for rank in ranks for b in rank for i in b]
    assert sorted(ids) == list(range(50000))
    for rank in ranks:
        assert sum(map(len, rank)) == 12500
        assert [len(b) for b in rank[-4:]] == [5, 5, 5, 5]
    for update in range(391):
        micro_sizes = [len(batch) for rank in ranks for batch in rank[4 * update : 4 * update + 4]]
        assert len(set(micro_sizes)) == 1
        total = 128 if update < 390 else 80
        assert sum(micro_sizes) == total
        # mean loss, accumulation divisor 4, DDP mean over 4 ranks
        assert {Fraction(1, size * 4 * 4) for size in micro_sizes} == {Fraction(1, total)}


@pytest.mark.parametrize("step", [79, 157, 235, 313, 391])
def test_fresh_same_seed_sampler_prefix_matches_checkpoint_resume_boundary(step):
    for rank in range(4):
        original = batches(314159, rank=rank)
        restored = batches(314159, rank=rank)
        assert restored[4 * step :] == original[4 * step :]
        assert len(restored[4 * step :]) == 4 * (391 - step)


@pytest.mark.parametrize("change", ["rows", "duplicate", "missing", "bool", "out_of_range"])
def test_bad_sampler_refuses_before_yielding_first_batch(change):
    order = list(range(50000))
    if change == "rows":
        order.pop()
    elif change == "duplicate":
        order[-1] = order[0]
    elif change == "missing":

        class Misreported(list):
            def __len__(self):
                return 50000

        order = Misreported(order[:-1])
    elif change == "bool":
        order[0] = False
    else:
        order[-1] = 50000
    with pytest.raises(ValueError):
        next(iter(f.FactorialBatchSampler(order)))


@pytest.mark.parametrize("size,drop,seed", [(4, False, 314159), (8, True, 314159), (8, False, 42)])
def test_upstream_primary_defaults_do_not_silently_admit_the_new_branch(size, drop, seed):
    with pytest.raises(ValueError):
        f.batch_sampler_factory(
            range(50000), size, drop, generator=torch.Generator().manual_seed(seed)
        )


@pytest.mark.parametrize("update", [0, 390])
def test_actual_infonce_gradient_matches_complete_logical_batch(update):
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.arange(20, dtype=torch.float64).reshape(5, 4) / 19 + 0.1
            )

        def forward(self, features):
            return {"sentence_embedding": features["values"] @ self.weight}

    ranks = [batches(314159, rank=r)[4 * update : 4 * update + 4] for r in range(4)]
    ids = [index for rank in ranks for micro in rank for index in micro]
    assert len(ids) == (128 if update == 0 else 80)

    def features(selected):
        values = torch.tensor(selected, dtype=torch.float64)[:, None]
        offset = torch.arange(5, dtype=torch.float64)[None, :]
        return [{"values": torch.sin(values * 0.0123 + offset + column / 7)} for column in range(9)]

    complete = Encoder()
    ExplicitDenseInfoNCELoss(complete, temperature=0.02)(features(ids)).backward()
    rank_gradients = []
    for rank in ranks:
        local = copy.deepcopy(complete)
        local.zero_grad(set_to_none=True)
        objective = ExplicitDenseInfoNCELoss(local, temperature=0.02)
        for micro in rank:
            (objective(features(micro)) / 4).backward()
        rank_gradients.append(local.weight.grad)
    observed = torch.stack(rank_gradients).mean(dim=0)
    torch.testing.assert_close(observed, complete.weight.grad, rtol=1e-12, atol=1e-12)
