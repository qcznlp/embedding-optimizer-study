from types import SimpleNamespace

import pytest

from embed_optim.evaluation_supervisor import (
    _aggregate_command,
    _evaluation_command,
    _wandb_command,
    parse_args,
    supervise,
)


def _args(**overrides):
    values = {
        "matrix": "configs/experiment.yaml",
        "gpus_a": "3,5,6,7",
        "gpus_b": "0,1,2,4",
        "late_port_a": 29610,
        "late_port": 29620,
        "results_root": "results/decontaminated-beir",
        "log_dir": "logs/evaluation",
        "output_dir": "reports",
        "blog": "docs/blog.md",
        "python": "/system/python",
        "worker_python": "/worker/python",
        "training_poll_seconds": 2.0,
        "restart_delay": 3.0,
        "max_launches": 0,
        "skip_wandb_sync": False,
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
    final_aggregate = _aggregate_command(_args(), render_blog=True)
    wandb = _wandb_command(_args())

    assert evaluation[:3] == ["/system/python", "-m", "embed_optim.evaluate_matrix"]
    assert evaluation[evaluation.index("--worker-python") + 1] == "/worker/python"
    assert evaluation[evaluation.index("--families") + 1 : evaluation.index("--gpus-a")] == [
        "dense",
        "late",
    ]
    assert aggregate[:3] == ["/system/python", "-m", "embed_optim.aggregate"]
    assert aggregate[-2:] == ["--strict", "--no-render-blog"]
    assert final_aggregate[-1] == "--strict"
    assert "--no-render-blog" not in final_aggregate
    assert wandb[:3] == ["/system/python", "-m", "embed_optim.wandb_sync"]


def test_evaluation_supervisor_cli_rejects_invalid_intervals():
    with pytest.raises(SystemExit):
        parse_args(["--training-poll-seconds", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--restart-delay", "-1"])
