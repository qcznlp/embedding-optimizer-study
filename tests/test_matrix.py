import json
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from embed_optim.matrix import (
    Pool,
    _checkpoint_is_resumable,
    _latest_resumable_checkpoint,
    _pop_next,
    _run_is_complete,
    parse_args,
    run_matrix,
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


def test_two_pools_claim_distinct_jobs_from_one_family():
    queues = {
        "dense": [],
        "late": [_run("late", "l1"), _run("late", "l2")],
    }
    running = {}

    first = _pop_next(Pool("0,1", 1, "dense"), queues, running)
    running["a"] = SimpleNamespace(config=first)
    second = _pop_next(Pool("2,3", 2, "late"), queues, running)

    assert (first.run_id, second.run_id) == ("l1", "l2")
    assert queues == {"dense": [], "late": []}


def _write_checkpoint(root, step, *, complete=True):
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    for name in (
        "config.json",
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
    ):
        (checkpoint / name).write_bytes(b"state")
    (checkpoint / "model.safetensors").write_bytes(f"state-{step}".encode())
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


def test_checkpoint_resume_selection_falls_back_after_deep_audit_failure(tmp_path, monkeypatch):
    output = tmp_path / "dense" / "run"
    older = _write_checkpoint(output, 10)
    latest = _write_checkpoint(output, 20)
    (output / "checkpoint_schedule.json").write_text(json.dumps({"steps": [10, 20, 30, 40, 50]}))
    config = SimpleNamespace(output_dir=output)
    audited = []

    def deep_problems(checkpoint, expected_step, world_size, config, final_step):
        audited.append((checkpoint.name, expected_step, world_size, final_step))
        return ["corrupt optimizer"] if checkpoint == latest else []

    monkeypatch.setattr("embed_optim.aggregate._deep_checkpoint_problems", deep_problems)

    assert _latest_resumable_checkpoint(config) == older
    assert audited == [
        ("checkpoint-20", 20, 4, 50),
        ("checkpoint-10", 10, 4, 50),
    ]


def test_checkpoint_resume_selection_uses_hybrid_contract(tmp_path, monkeypatch):
    output = tmp_path / "dense" / "hybrid"
    checkpoint = _write_checkpoint(output, 10)
    (output / "checkpoint_schedule.json").write_text(json.dumps({"steps": [10, 20, 30, 40, 50]}))
    config = SimpleNamespace(
        output_dir=output,
        optimizer=SimpleNamespace(name="hybrid_adamw"),
    )
    audited = []

    def hybrid_problems(candidate, selected_config, expected_step, final_step, *, world_size):
        audited.append((candidate.name, selected_config, expected_step, final_step, world_size))
        return []

    monkeypatch.setattr(
        "embed_optim.supplemental_training_audit.hybrid_checkpoint_problems",
        hybrid_problems,
    )
    generic = MagicMock(side_effect=AssertionError("generic hybrid audit must not run"))
    monkeypatch.setattr("embed_optim.aggregate._deep_checkpoint_problems", generic)

    assert _latest_resumable_checkpoint(config) == checkpoint
    assert audited == [("checkpoint-10", config, 10, 50, 4)]
    generic.assert_not_called()


def test_checkpoint_resume_selection_rejects_unchanged_model_payload(tmp_path, monkeypatch):
    output = tmp_path / "dense" / "run"
    older = _write_checkpoint(output, 10)
    latest = _write_checkpoint(output, 20)
    (latest / "model.safetensors").write_bytes((older / "model.safetensors").read_bytes())
    (output / "checkpoint_schedule.json").write_text(json.dumps({"steps": [10, 20, 30, 40, 50]}))
    config = SimpleNamespace(output_dir=output)
    monkeypatch.setattr(
        "embed_optim.aggregate._deep_checkpoint_problems", lambda *args, **kwargs: []
    )

    assert _latest_resumable_checkpoint(config) == older


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


def test_corrected_dense_completion_requires_padded_execution_receipt(tmp_path):
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
    config = SimpleNamespace(
        output_dir=output,
        run_id="run",
        model_family="dense",
        dense_can_flatten_inputs=False,
    )

    assert not _run_is_complete(config)
    completed["input_execution"] = {
        "mode": "independently_padded",
        "sentence_transformers_can_flatten_inputs": False,
    }
    (output / "completed.json").write_text(json.dumps(completed))
    assert _run_is_complete(config)


def test_run_completion_validates_declared_accepted_timing(tmp_path):
    output = tmp_path / "dense" / "run"
    steps = [2, 4, 6, 8, 10]
    for step in steps:
        _write_checkpoint(output, step)
    final = output / "final"
    final.mkdir()
    (final / "model.safetensors").write_bytes(b"model")
    (output / "checkpoint_schedule.json").write_text(json.dumps({"steps": steps}))
    (output / "trainer_state_final.json").write_text(json.dumps({"global_step": 10}))
    timing = {
        "schema_version": 1,
        "segments": [
            {
                "start_step_exclusive": 0,
                "end_step_inclusive": 10,
                "wall_time_seconds_max_rank": 5.0,
            }
        ],
        "total_wall_time_seconds_max_rank": 5.0,
    }
    (output / "accepted_timing.json").write_text(json.dumps(timing))
    completed = {
        "run_id": "run",
        "model_family": "dense",
        "global_step": 10,
        "checkpoints": steps,
        "accepted_timing": {
            "schema_version": 1,
            "segments": 1,
            "total_wall_time_seconds_max_rank": 5.0,
        },
    }
    (output / "completed.json").write_text(json.dumps(completed))
    config = SimpleNamespace(output_dir=output, run_id="run", model_family="dense")

    assert _run_is_complete(config)
    timing["total_wall_time_seconds_max_rank"] = 4.0
    (output / "accepted_timing.json").write_text(json.dumps(timing))
    assert not _run_is_complete(config)


def test_failed_job_is_retried_before_later_family_config(monkeypatch, tmp_path):
    first = _run("dense", "d1")
    second = _run("dense", "d2")
    third = _run("dense", "d3")
    for config in (first, second, third):
        config.output_dir = tmp_path / config.run_id

    launched = []
    return_codes = iter([1, 0, 0, 0])
    completed = set()

    def fake_launch(config, *args, **kwargs):
        launched.append(config.run_id)
        return_code = next(return_codes)
        if return_code == 0:
            completed.add(config.run_id)
        process = MagicMock(spec=subprocess.Popen)
        process.poll.return_value = return_code
        return SimpleNamespace(
            config=config,
            process=process,
            log_handle=MagicMock(),
            started=0.0,
        )

    monkeypatch.setattr("embed_optim.matrix.load_matrix", lambda _: [first, second, third])
    monkeypatch.setattr("embed_optim.matrix._complete", lambda config: config.run_id in completed)
    monkeypatch.setattr("embed_optim.matrix.matrix_runtime_spec", lambda _: None)
    monkeypatch.setattr("embed_optim.matrix._launch", fake_launch)
    monkeypatch.setattr("embed_optim.matrix.time.monotonic", lambda: 60.0)
    monkeypatch.setattr("embed_optim.matrix.time.sleep", lambda _: None)
    args = SimpleNamespace(
        matrix=tmp_path / "matrix.yaml",
        families=["dense"],
        run_ids=[],
        gpus_a="0,1,2,3",
        gpus_b="4,5,6,7",
        port_a=29510,
        port_b=29520,
        log_dir=tmp_path / "logs",
        max_retries=2,
        fail_fast=False,
        dry_run=False,
    )

    assert run_matrix(args) == 0
    # Pool B may steal d2 before pool A observes d1's failure.  Once the
    # failure is known, d1 must be retried before the still-queued d3.
    assert launched == ["d1", "d2", "d1", "d3"]


def test_persistent_failure_exhausts_bounded_retry_budget(monkeypatch, tmp_path):
    config = _run("dense", "broken")
    config.output_dir = tmp_path / config.run_id
    launched = []

    def fake_launch(config, *args, **kwargs):
        launched.append(config.run_id)
        process = MagicMock(spec=subprocess.Popen)
        process.poll.return_value = 17
        return SimpleNamespace(
            config=config,
            process=process,
            log_handle=MagicMock(),
            started=0.0,
        )

    monkeypatch.setattr("embed_optim.matrix.load_matrix", lambda _: [config])
    monkeypatch.setattr("embed_optim.matrix._complete", lambda _: False)
    monkeypatch.setattr("embed_optim.matrix.matrix_runtime_spec", lambda _: None)
    monkeypatch.setattr("embed_optim.matrix._launch", fake_launch)
    monkeypatch.setattr("embed_optim.matrix.time.monotonic", lambda: 60.0)
    monkeypatch.setattr("embed_optim.matrix.time.sleep", lambda _: None)
    args = SimpleNamespace(
        matrix=tmp_path / "matrix.yaml",
        families=["dense"],
        run_ids=[],
        gpus_a="0,1,2,3",
        gpus_b="4,5,6,7",
        port_a=29510,
        port_b=29520,
        log_dir=tmp_path / "logs",
        max_retries=2,
        fail_fast=False,
        dry_run=False,
    )

    assert run_matrix(args) == 1
    assert launched == ["broken", "broken", "broken"]


def test_zero_exit_without_completion_artifacts_is_retried(monkeypatch, tmp_path):
    config = _run("late", "missing-terminal-state")
    config.output_dir = tmp_path / config.run_id
    launched = []

    def fake_launch(config, *args, **kwargs):
        launched.append(config.run_id)
        process = MagicMock(spec=subprocess.Popen)
        process.poll.return_value = 0
        return SimpleNamespace(
            config=config,
            process=process,
            log_handle=MagicMock(),
            started=0.0,
        )

    monkeypatch.setattr("embed_optim.matrix.load_matrix", lambda _: [config])
    monkeypatch.setattr("embed_optim.matrix._complete", lambda _: len(launched) >= 2)
    monkeypatch.setattr("embed_optim.matrix.matrix_runtime_spec", lambda _: None)
    monkeypatch.setattr("embed_optim.matrix._launch", fake_launch)
    monkeypatch.setattr("embed_optim.matrix.time.monotonic", lambda: 60.0)
    monkeypatch.setattr("embed_optim.matrix.time.sleep", lambda _: None)
    args = SimpleNamespace(
        matrix=tmp_path / "matrix.yaml",
        families=["late"],
        run_ids=[],
        gpus_a="0,1,2,3",
        gpus_b="4,5,6,7",
        port_a=29510,
        port_b=29520,
        log_dir=tmp_path / "logs",
        max_retries=2,
        fail_fast=False,
        dry_run=False,
    )

    assert run_matrix(args) == 0
    assert launched == ["missing-terminal-state", "missing-terminal-state"]


def test_matrix_cli_rejects_negative_retry_budget():
    with pytest.raises(SystemExit):
        parse_args(["--max-retries", "-1"])


def test_matrix_cli_defaults_dense_and_requires_explicit_late_opt_in():
    assert parse_args([]).families == ["dense"]
    assert parse_args(["--families", "late"]).families == ["late"]
