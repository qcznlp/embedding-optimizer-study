from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from embed_optim.post_eval_pipeline import (
    TransientProgressAuditError,
    _strict_progress,
    parse_args,
    pipeline_steps,
    supervise_post_eval,
)


def _progress(
    path: Path, *, complete: bool = True, unexpected: int = 0, error: str | None = None
) -> None:
    valid = 1680 if complete else 120
    path.parent.mkdir(parents=True, exist_ok=True)
    if error is not None:
        path.write_text(json.dumps({"schema_version": 1, "complete": False, "error": error}) + "\n")
        return
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": complete,
                "error": None,
                "expected_units": 1680,
                "valid_units": valid,
                "missing_units": 1680 - valid,
                "unexpected_units": unexpected,
            }
        )
        + "\n"
    )


def _args(tmp_path: Path, *extra: str):
    return parse_args(
        [
            "--progress",
            str(tmp_path / "progress.json"),
            "--workdir",
            str(Path.cwd()),
            "--log-dir",
            str(tmp_path / "logs"),
            "--settle-seconds",
            "0",
            "--skip-wandb-sync",
            "--skip-validation",
            *extra,
        ]
    )


def test_pipeline_dry_run_covers_all_post_evaluation_gates(tmp_path: Path, capsys):
    args = _args(tmp_path, "--dry-run")
    steps = pipeline_steps(args)

    assert len(steps) == 49
    assert steps[0].name == "strict-evaluation-audit"
    assert steps[-1].name == "paper-draft-build"
    assert [step.name for step in steps].index("mechanism-bridge") < [
        step.name for step in steps
    ].index("mechanism-blog-render")
    assert [step.name for step in steps].index("common-state-matrix") < [
        step.name for step in steps
    ].index("training-dense-representation-matrix")
    assert [step.name for step in steps].index("weight-space-reaudit") < [
        step.name for step in steps
    ].index("training-dynamics-summary")
    assert [step.name for step in steps].index("training-dynamics-summary") < [
        step.name for step in steps
    ].index("training-dynamics-plot")
    assert [step.name for step in steps].index("recipe-validation-summary") < [
        step.name for step in steps
    ].index("confirmatory-data-preparation")
    assert [step.name for step in steps].index("confirmatory-matrix-generation") < [
        step.name for step in steps
    ].index("common-state-matrix")
    assert [step.name for step in steps].index("common-state-summary") < [
        step.name for step in steps
    ].index("short-branch-matrix-generation")
    assert [step.name for step in steps].index("functional-intervention-summary") < [
        step.name for step in steps
    ].index("training-dense-representation-matrix")
    assert [step.name for step in steps].index("hybrid-adamw-training") < [
        step.name for step in steps
    ].index("hybrid-adamw-evaluation")
    assert [step.name for step in steps].index("confirmatory-matrix-generation") < [
        step.name for step in steps
    ].index("confirmatory-training-seed-314159")
    assert [step.name for step in steps].index("confirmatory-training-seed-161803") < [
        step.name for step in steps
    ].index("confirmatory-evaluation")
    assert [step.name for step in steps].index("short-branch-matrix-generation") < [
        step.name for step in steps
    ].index("short-branch-training-seed-314159")
    assert [step.name for step in steps].index("short-branch-training-seed-161803") < [
        step.name for step in steps
    ].index("short-branch-evaluation")
    assert [step.name for step in steps].index("short-branch-evaluation-audit") < [
        step.name for step in steps
    ].index("short-branch-summary")
    assert [step.name for step in steps].index("paper-evidence-audit") < [
        step.name for step in steps
    ].index("paper-draft-build")
    late = next(step for step in steps if step.name == "late-token-dynamics-plot")
    assert late.command[1] == "-c"
    assert supervise_post_eval(args) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert [item["name"] for item in rendered] == [step.name for step in steps]


def test_pipeline_wait_gate_and_ledger_are_complete(tmp_path: Path):
    args = _args(tmp_path, "--wait-pids", "12345")
    _progress(Path(args.progress))
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        kwargs["stdout"].write("fixture command output\n")
        return subprocess.CompletedProcess(command, 0)

    assert supervise_post_eval(args, run_command=run, pid_exists=lambda pid: False) == 0
    assert len(commands) == 49
    ledger = json.loads((Path(args.log_dir) / "pipeline-ledger.json").read_text())
    assert ledger["complete"] is True
    assert ledger["wait_pids"] == [12345]
    assert len(ledger["steps"]) == 49
    assert all(step["complete"] for step in ledger["steps"])
    assert all(len(step["attempts"]) == 1 for step in ledger["steps"])
    assert len(list(Path(args.log_dir).glob("*.log"))) == 49


def test_pipeline_retries_then_records_failed_step(tmp_path: Path):
    args = _args(tmp_path, "--step-retries", "1")
    _progress(Path(args.progress))
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write("failed\n")
        return subprocess.CompletedProcess(command, 17)

    assert supervise_post_eval(args, run_command=fail, sleeper=lambda seconds: None) == 1
    assert len(calls) == 2
    ledger = json.loads((Path(args.log_dir) / "pipeline-ledger.json").read_text())
    assert ledger["complete"] is False
    assert ledger["failed_step"] == "strict-evaluation-audit"
    assert len(ledger["steps"][0]["attempts"]) == 2


def test_strict_progress_rejects_unexpected_results(tmp_path: Path):
    progress = tmp_path / "progress.json"
    _progress(progress, unexpected=1)
    with pytest.raises(ValueError, match="Invalid strict evaluation progress"):
        _strict_progress(progress)


def test_pipeline_waits_through_transient_audit_error_while_evaluator_is_live(
    tmp_path: Path, capsys
):
    args = _args(tmp_path, "--wait-pids", "12345")
    progress = Path(args.progress)
    _progress(progress, error="source temporarily unavailable")
    pid_checks = 0

    def pid_exists(pid):
        nonlocal pid_checks
        assert pid == 12345
        pid_checks += 1
        return pid_checks == 1

    def recover(seconds):
        assert seconds >= 0
        _progress(progress)

    def run(command, **kwargs):
        kwargs["stdout"].write("fixture command output\n")
        return subprocess.CompletedProcess(command, 0)

    assert supervise_post_eval(args, run_command=run, sleeper=recover, pid_exists=pid_exists) == 0
    assert "temporarily unavailable" in capsys.readouterr().out


def test_strict_progress_exposes_transient_watcher_error(tmp_path: Path):
    progress = tmp_path / "progress.json"
    _progress(progress, error="audit failed")
    with pytest.raises(TransientProgressAuditError, match="audit failed"):
        _strict_progress(progress)
