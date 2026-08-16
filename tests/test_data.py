import pytest

from embed_optim.data import _seed_for, allocate_quotas


def test_allocate_quotas_is_exact_and_deterministic():
    counts = {"a": 3, "b": 2, "c": 1}
    assert allocate_quotas(counts, 5) == {"a": 2, "b": 2, "c": 1}
    assert sum(allocate_quotas(counts, 4).values()) == 4


def test_allocate_quotas_rejects_invalid_total():
    with pytest.raises(ValueError):
        allocate_quotas({"a": 2}, 3)


def test_stable_seed_depends_on_every_component():
    assert _seed_for(42, "nq", 10) == _seed_for(42, "nq", 10)
    assert _seed_for(42, "nq", 10) != _seed_for(42, "nq", 11)
