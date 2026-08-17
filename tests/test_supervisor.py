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
        "wait_for_pid": None,
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
