import json
from types import SimpleNamespace

from embed_optim.matrix import (
    Pool,
    _checkpoint_is_resumable,
    _latest_resumable_checkpoint,
    _pop_next,
    _run_is_complete,
)


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


def _write_checkpoint(root, step, *, complete=True):
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    for name in (
        "config.json",
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
        "model.safetensors",
    ):
        (checkpoint / name).write_bytes(b"state")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    for rank in range(4):
        (checkpoint / f"rng_state_{rank}.pth").write_bytes(b"rng")
    if not complete:
        (checkpoint / "optimizer.pt").write_bytes(b"")
    return checkpoint


def test_checkpoint_resume_selection_ignores_interrupted_latest_write(tmp_path):
    output = tmp_path / "dense" / "run"
    valid = _write_checkpoint(output, 10)
    interrupted = _write_checkpoint(output, 20, complete=False)
    config = SimpleNamespace(output_dir=output)

    assert _checkpoint_is_resumable(valid)
    assert not _checkpoint_is_resumable(interrupted)
    assert _latest_resumable_checkpoint(config) == valid


def test_checkpoint_resume_selection_requires_matching_state_step(tmp_path):
    checkpoint = _write_checkpoint(tmp_path, 10)
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 9}))
    assert not _checkpoint_is_resumable(checkpoint)


def test_run_completion_requires_consistent_terminal_artifacts(tmp_path):
    output = tmp_path / "dense" / "run"
    steps = [2, 4, 6, 8, 10]
    for step in steps:
        _write_checkpoint(output, step)
    final = output / "final"
    final.mkdir()
    (final / "model.safetensors").write_bytes(b"model")
    (output / "checkpoint_schedule.json").write_text(json.dumps({"steps": steps}))
    (output / "trainer_state_final.json").write_text(json.dumps({"global_step": 10}))
    completed = {
        "run_id": "run",
        "model_family": "dense",
        "global_step": 10,
        "checkpoints": steps,
    }
    (output / "completed.json").write_text(json.dumps(completed))
    config = SimpleNamespace(output_dir=output, run_id="run", model_family="dense")

    assert _run_is_complete(config)
    (output / "completed.json").write_text("{")
    assert not _run_is_complete(config)
