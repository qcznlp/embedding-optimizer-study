from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim.evaluation_supervisor import (
    _aggregate_command,
    _argv_contains_command_fragment,
    _evaluation_command,
    _wandb_command,
    parse_args,
    supervise,
)


def test_command_match_ignores_adoption_declaration_values():
    fragment = "scripts/eval/late_interaction.py"

    assert _argv_contains_command_fragment(
        ["python", "/repo/scripts/eval/late_interaction.py", "--models", "checkpoint"],
        fragment,
    )
    assert not _argv_contains_command_fragment(
        ["python", "-m", "supervisor", "--wait-for-command", fragment],
        fragment,
    )
    assert not _argv_contains_command_fragment(
        ["python", "-m", "supervisor", f"--wait-for-command={fragment}"],
        fragment,
    )
    assert not _argv_contains_command_fragment(
        [
            "bash",
            "-c",
            f"exec python -m supervisor --wait-for-command {fragment} --poll-seconds 60",
        ],
        fragment,
    )
    assert _argv_contains_command_fragment(
        [
            "bash",
            "-c",
            f"python {fragment} --models checkpoint; "
            f"python -m supervisor --wait-for-command {fragment}",
        ],
        fragment,
    )


def _args(**overrides):
    values = {
        "matrix": "configs/experiment.yaml",
        "families": ["dense", "late"],
        "scope_amendment": None,
        "gpus_a": "3,5,6,7",
        "gpus_b": "0,1,2,4",
        "late_port_a": 29610,
        "late_port": 29620,
        "results_root": "results/decontaminated-beir",
        "log_dir": "logs/evaluation",
        "output_dir": "reports",
        "python": "/system/python",
        "worker_python": "/worker/python",
        "training_poll_seconds": 2.0,
        "wait_for_pids": [],
        "wait_for_commands": [],
        "wait_poll_seconds": 4.0,
        "restart_delay": 3.0,
        "max_launches": 0,
        "skip_wandb_sync": False,
        "evaluation_only": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _config(family, run_id):
    return SimpleNamespace(model_family=family, run_id=run_id)


def test_evaluation_supervisor_waits_and_retries_until_strict_coverage(monkeypatch):
    configs = [_config("dense", "adamw"), _config("late", "muon")]
    completion_checks = iter((False, True, True, True))
    monkeypatch.setattr("embed_optim.evaluation_supervisor.load_matrix", lambda matrix: configs)
    monkeypatch.setattr(
        "embed_optim.evaluation_supervisor._run_is_complete",
        lambda config: next(completion_checks),
    )
    return_codes = iter((1, 1, 0, 0, 0, 0))
    commands = []
    sleeps = []

    def run(command, check):
        commands.append((command, check))
        return SimpleNamespace(returncode=next(return_codes))

    result = supervise(_args(), run_command=run, sleeper=sleeps.append)

    assert result == 0
    assert sleeps == [2.0, 3.0]
    assert [command[0][2] for command in commands] == [
        "embed_optim.evaluate_matrix",
        "embed_optim.aggregate",
        "embed_optim.evaluate_matrix",
        "embed_optim.aggregate",
        "embed_optim.wandb_sync",
        "embed_optim.aggregate",
    ]
    assert all(not check for _, check in commands)


def test_evaluation_supervisor_accepts_complete_coverage_after_worker_failure(monkeypatch):
    config = _config("dense", "adamw")
    monkeypatch.setattr("embed_optim.evaluation_supervisor.load_matrix", lambda matrix: [config])
    monkeypatch.setattr("embed_optim.evaluation_supervisor._run_is_complete", lambda config: True)
    return_codes = iter((1, 0, 0, 0))

    result = supervise(
        _args(),
        run_command=lambda command, check: SimpleNamespace(returncode=next(return_codes)),
        sleeper=lambda seconds: pytest.fail("complete coverage should not sleep"),
    )

    assert result == 0


def test_evaluation_supervisor_dense_scope_ignores_late_training(monkeypatch):
    configs = [_config("dense", "adamw"), _config("late", "muon")]
    checked = []
    monkeypatch.setattr("embed_optim.evaluation_supervisor.load_matrix", lambda matrix: configs)

    def complete(config):
        checked.append(config.model_family)
        return config.model_family == "dense"

    monkeypatch.setattr("embed_optim.evaluation_supervisor._run_is_complete", complete)
    commands = []

    def run(command, check):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    result = supervise(
        _args(
            families=["dense"],
            scope_amendment=Path("configs/dense_scope_amendment.json"),
            evaluation_only=True,
        ),
        run_command=run,
    )

    assert result == 0
    assert checked == ["dense"]
    assert [command[2] for command in commands] == [
        "embed_optim.evaluate_matrix",
        "embed_optim.aggregate",
    ]
    assert all("late" not in command for command in commands)


def test_evaluation_supervisor_validates_dense_scope_before_training_io(monkeypatch):
    monkeypatch.setattr(
        "embed_optim.evaluation_supervisor.load_matrix",
        lambda matrix: pytest.fail("scope must be validated before reading training state"),
    )
    with pytest.raises(ValueError, match="requires --scope-amendment"):
        supervise(_args(families=["dense"], scope_amendment=None))


def test_evaluation_supervisor_adopts_multiple_coordinators_before_recovery(monkeypatch):
    config = _config("dense", "adamw")
    monkeypatch.setattr("embed_optim.evaluation_supervisor.load_matrix", lambda matrix: [config])
    monkeypatch.setattr("embed_optim.evaluation_supervisor._run_is_complete", lambda config: True)
    states = {101: iter((True, False, False)), 202: iter((True, True, False))}
    sleeps = []
    commands = []

    def pid_exists(pid):
        return next(states[pid])

    def run(command, check):
        commands.append(command[2])
        return SimpleNamespace(returncode=0)

    result = supervise(
        _args(wait_for_pids=[101, 202], evaluation_only=True),
        run_command=run,
        pid_exists=pid_exists,
        matching_command_pids=lambda fragment: [],
        sleeper=sleeps.append,
    )

    assert result == 0
    assert sleeps == [4.0, 4.0]
    assert commands == ["embed_optim.evaluate_matrix", "embed_optim.aggregate"]


def test_evaluation_only_skips_wandb_and_keeps_strict_report(monkeypatch):
    config = _config("late", "muon")
    monkeypatch.setattr("embed_optim.evaluation_supervisor.load_matrix", lambda matrix: [config])
    monkeypatch.setattr("embed_optim.evaluation_supervisor._run_is_complete", lambda config: True)
    commands = []

    def run(command, check):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    result = supervise(_args(evaluation_only=True), run_command=run)

    assert result == 0
    assert [command[2] for command in commands] == [
        "embed_optim.evaluate_matrix",
        "embed_optim.aggregate",
    ]
    assert commands[-1][-1] == "--strict"


def test_evaluation_supervisor_adopts_orphan_workers_by_command(monkeypatch):
    config = _config("late", "muon")
    monkeypatch.setattr("embed_optim.evaluation_supervisor.load_matrix", lambda matrix: [config])
    monkeypatch.setattr("embed_optim.evaluation_supervisor._run_is_complete", lambda config: True)
    matches = iter(([301, 302], [302], []))
    sleeps = []

    result = supervise(
        _args(wait_for_commands=["scripts/eval/late_interaction.py"], evaluation_only=True),
        run_command=lambda command, check: SimpleNamespace(returncode=0),
        matching_command_pids=lambda fragment: next(matches),
        sleeper=sleeps.append,
    )

    assert result == 0
    assert sleeps == [4.0, 4.0]


def test_evaluation_supervisor_retries_finalization_without_relaunching_evaluation(
    monkeypatch,
):
    config = _config("dense", "adamw")
    monkeypatch.setattr("embed_optim.evaluation_supervisor.load_matrix", lambda matrix: [config])
    monkeypatch.setattr("embed_optim.evaluation_supervisor._run_is_complete", lambda config: True)
    return_codes = iter((0, 0, 1, 0, 0))
    commands = []
    sleeps = []

    def run(command, check):
        commands.append(command[2])
        return SimpleNamespace(returncode=next(return_codes))

    result = supervise(_args(), run_command=run, sleeper=sleeps.append)

    assert result == 0
    assert commands == [
        "embed_optim.evaluate_matrix",
        "embed_optim.aggregate",
        "embed_optim.wandb_sync",
        "embed_optim.wandb_sync",
        "embed_optim.aggregate",
    ]
    assert sleeps == [3.0]


def test_evaluation_supervisor_respects_launch_limit(monkeypatch):
    config = _config("late", "muon")
    monkeypatch.setattr("embed_optim.evaluation_supervisor.load_matrix", lambda matrix: [config])
    monkeypatch.setattr("embed_optim.evaluation_supervisor._run_is_complete", lambda config: True)
    commands = []

    result = supervise(
        _args(max_launches=1),
        run_command=lambda command, check: (
            commands.append(command) or SimpleNamespace(returncode=1)
        ),
        sleeper=lambda seconds: pytest.fail("launch limit should not sleep"),
    )

    assert result == 1
    assert len(commands) == 2


def test_evaluation_supervisor_commands_pin_worker_and_strict_audit():
    evaluation = _evaluation_command(_args())
    aggregate = _aggregate_command(_args())
    wandb = _wandb_command(_args())

    assert evaluation[:3] == ["/system/python", "-m", "embed_optim.evaluate_matrix"]
    assert evaluation[evaluation.index("--worker-python") + 1] == "/worker/python"
    assert evaluation[evaluation.index("--families") + 1 : evaluation.index("--gpus-a")] == [
        "dense",
        "late",
    ]
    assert aggregate[:3] == ["/system/python", "-m", "embed_optim.aggregate"]
    assert aggregate[-1] == "--strict"
    assert wandb[:3] == ["/system/python", "-m", "embed_optim.wandb_sync"]
    assert wandb[-3:] == ["--families", "dense", "late"]

    dense = _args(
        families=["dense"],
        scope_amendment=Path("configs/dense_scope_amendment.json"),
    )
    dense_evaluation = _evaluation_command(dense)
    assert dense_evaluation[
        dense_evaluation.index("--families") + 1 : dense_evaluation.index("--gpus-a")
    ] == ["dense"]
    assert dense_evaluation[-2:] == [
        "--scope-amendment",
        "configs/dense_scope_amendment.json",
    ]
    dense_aggregate = _aggregate_command(dense)
    assert dense_aggregate[
        dense_aggregate.index("--families") + 1 : dense_aggregate.index("--scope-amendment")
    ] == ["dense"]
    assert dense_aggregate[dense_aggregate.index("--scope-amendment") + 1] == (
        "configs/dense_scope_amendment.json"
    )
    dense_wandb = _wandb_command(dense)
    assert dense_wandb[-4:] == [
        "--families",
        "dense",
        "--scope-amendment",
        "configs/dense_scope_amendment.json",
    ]


def test_evaluation_supervisor_cli_rejects_invalid_intervals():
    with pytest.raises(SystemExit):
        parse_args(["--training-poll-seconds", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--restart-delay", "-1"])
    with pytest.raises(SystemExit):
        parse_args(["--wait-poll-seconds", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--wait-for-pid", "-1"])
    with pytest.raises(SystemExit):
        parse_args(["--wait-for-command", ""])


def test_evaluation_supervisor_cli_defaults_dense_and_requires_explicit_late_opt_in():
    assert parse_args([]).families == ["dense"]
    assert parse_args(["--families", "dense", "late"]).families == ["dense", "late"]
