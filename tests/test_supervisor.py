import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim.supervisor import parse_args, supervise


def _args(**overrides):
    values = {
        "matrix": "configs/experiment.yaml",
        "families": ["dense", "late"],
        "run_ids": [],
        "gpus_a": "3,5,6,7",
        "gpus_b": "0,1,2,4",
        "port_a": 29520,
        "port_b": 29510,
        "log_dir": "logs/training",
        "python": "/venv/python",
        "state_file": None,
        "wait_for_pid": None,
        "wait_for_pids": [],
        "poll_seconds": 2.0,
        "restart_delay": 3.0,
        "max_launches": 0,
        "sequential_families": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _config(family, run_id):
    return SimpleNamespace(model_family=family, run_id=run_id)


def test_supervisor_adopts_process_then_runs_until_complete(monkeypatch):
    configs = [_config("late", "muon"), _config("dense", "normuon")]
    completed = set()
    monkeypatch.setattr("embed_optim.supervisor.load_matrix", lambda path: configs)
    monkeypatch.setattr(
        "embed_optim.supervisor._run_is_complete", lambda config: config.run_id in completed
    )
    pid_checks = iter((True, False))
    sleeps = []
    commands = []

    def run(command, check):
        commands.append((command, check))
        completed.update(config.run_id for config in configs)
        return SimpleNamespace(returncode=1)

    result = supervise(
        _args(wait_for_pid=1234),
        run_command=run,
        pid_exists=lambda pid: next(pid_checks),
        sleeper=sleeps.append,
    )

    assert result == 0
    assert sleeps == [2.0]
    assert len(commands) == 1
    command, check = commands[0]
    assert not check
    assert command[:3] == ["/venv/python", "-m", "embed_optim.matrix"]
    assert command[command.index("--families") + 1 : command.index("--gpus-a")] == [
        "dense",
        "late",
    ]


def test_supervisor_waits_for_every_adopted_training_pid(monkeypatch):
    config = _config("dense", "adamw")
    completed = set()
    monkeypatch.setattr("embed_optim.supervisor.load_matrix", lambda path: [config])
    monkeypatch.setattr(
        "embed_optim.supervisor._run_is_complete", lambda config: config.run_id in completed
    )
    alive = {101: iter((True, False)), 202: iter((True, True, False))}
    sleeps = []

    def pid_exists(pid):
        return next(alive[pid])

    def run(command, check):
        completed.add("adamw")
        return SimpleNamespace(returncode=0)

    result = supervise(
        _args(wait_for_pids=[101, 202]),
        run_command=run,
        pid_exists=pid_exists,
        sleeper=sleeps.append,
    )

    assert result == 0
    assert sleeps == [2.0, 2.0]


def test_supervisor_publishes_atomic_recovery_state(monkeypatch, tmp_path):
    config = _config("dense", "adamw")
    completed = set()
    monkeypatch.setattr("embed_optim.supervisor.load_matrix", lambda path: [config])
    monkeypatch.setattr(
        "embed_optim.supervisor._run_is_complete", lambda config: config.run_id in completed
    )

    def run(command, check):
        completed.add("adamw")
        return SimpleNamespace(returncode=0)

    state_file = tmp_path / "nested" / "state.json"
    result = supervise(
        _args(state_file=state_file),
        run_command=run,
        sleeper=lambda seconds: None,
    )

    state = json.loads(state_file.read_text())
    assert result == 0
    assert state["phase"] == "complete"
    assert state["remaining_runs"] == []
    assert state["launches"] == 1
    assert state["selected_runs"] == ["dense/adamw"]
    assert not state_file.with_name(".state.json.tmp").exists()


def test_supervisor_respects_launch_limit(monkeypatch):
    config = _config("late", "muon")
    monkeypatch.setattr("embed_optim.supervisor.load_matrix", lambda path: [config])
    monkeypatch.setattr("embed_optim.supervisor._run_is_complete", lambda config: False)
    launches = []
    sleeps = []

    result = supervise(
        _args(families=["late"], max_launches=1),
        run_command=lambda command, check: (
            launches.append(command) or SimpleNamespace(returncode=1)
        ),
        sleeper=sleeps.append,
    )

    assert result == 1
    assert len(launches) == 1
    assert sleeps == []


def test_supervisor_skips_matrix_when_selected_runs_are_complete(monkeypatch):
    config = _config("dense", "adamw")
    monkeypatch.setattr("embed_optim.supervisor.load_matrix", lambda path: [config])
    monkeypatch.setattr("embed_optim.supervisor._run_is_complete", lambda config: True)

    result = supervise(
        _args(families=["dense"]),
        run_command=lambda *args, **kwargs: pytest.fail("matrix should not launch"),
    )

    assert result == 0


def test_supervisor_preserves_requested_family_order(monkeypatch):
    configs = [_config("late", "muon"), _config("dense", "normuon")]
    completed = set()
    monkeypatch.setattr("embed_optim.supervisor.load_matrix", lambda path: configs)
    monkeypatch.setattr(
        "embed_optim.supervisor._run_is_complete", lambda config: config.run_id in completed
    )
    launched_families = []

    def run(command, check):
        start = command.index("--families") + 1
        end = command.index("--gpus-a")
        families = command[start:end]
        launched_families.append(families)
        completed.update(config.run_id for config in configs if config.model_family in families)
        return SimpleNamespace(returncode=0)

    result = supervise(
        _args(families=["late", "dense"], sequential_families=True),
        run_command=run,
        sleeper=lambda seconds: None,
    )

    assert result == 0
    assert launched_families == [["late"], ["dense"]]


def test_supervisor_cli_rejects_invalid_intervals():
    with pytest.raises(SystemExit):
        parse_args(["--poll-seconds", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--wait-for-pid", "-1"])
    with pytest.raises(SystemExit):
        parse_args(["--wait-for-pids", "123", "-1"])


def test_supervisor_cli_defaults_dense_and_requires_explicit_late_opt_in():
    assert parse_args([]).families == ["dense"]
    assert parse_args(["--families", "late"]).families == ["late"]


def test_corrected_control_plane_recovery_lock_binds_current_sources():
    repository = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (repository / "configs/dense_no_packing_control_plane_recovery.json").read_text()
    )
    assert protocol["status"] == "corrected_training_control_plane_recovery_lock"
    assert protocol["scientific_plan_change"] is False
    assert len(protocol["adopted_training_pids"]) == 8
    assert protocol["incident"]["fatal_training_markers"] == {
        "cuda_oom": 0,
        "traceback": 0,
        "non_finite": 0,
        "nccl_data_plane_error": 0,
    }
    for group in ("parent_bindings", "source_bindings"):
        for identity in protocol[group].values():
            path = repository / identity["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == identity["sha256"]
            if "bytes" in identity:
                assert path.stat().st_size == identity["bytes"]
