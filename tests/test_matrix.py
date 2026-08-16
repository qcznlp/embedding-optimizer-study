from types import SimpleNamespace

from embed_optim.matrix import Pool, _pop_next


def _run(family, run_id):
    return SimpleNamespace(model_family=family, run_id=run_id)


def test_pool_prefers_its_family():
    queues = {"dense": [_run("dense", "d")], "late": [_run("late", "l")]}
    selected = _pop_next(Pool("0,1", 1, "dense"), queues, {})
    assert selected.run_id == "d"
    assert [item.run_id for item in queues["late"]] == ["l"]


def test_pool_steals_after_preferred_family_drains():
    queues = {"dense": [], "late": [_run("late", "l1"), _run("late", "l2")]}
    selected = _pop_next(Pool("0,1", 1, "dense"), queues, {})
    assert selected.run_id == "l1"


def test_pool_waits_for_its_running_preferred_job_before_stealing():
    queues = {"dense": [], "late": [_run("late", "l")]}
    running = {"a": SimpleNamespace(config=_run("dense", "active"))}
    assert _pop_next(Pool("2,3", 2, "dense"), queues, running) is None
