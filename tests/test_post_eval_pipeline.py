from __future__ import annotations

import importlib
import json
import shutil
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

    assert len(steps) == 56
    assert steps[0].name == "strict-evaluation-audit"
    assert steps[-1].name == "paper-final-strict-audit"
    assert steps[-1].command[-1] == "--strict"
    assert [step.name for step in steps].index("strict-blog-render") < [
        step.name for step in steps
    ].index("retrieval-dynamics-summary")
    assert [step.name for step in steps].index("training-dynamics-summary") < [
        step.name for step in steps
    ].index("retrieval-dynamics-summary")
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
    assert [step.name for step in steps].index("confirmatory-data-preparation") < [
        step.name for step in steps
    ].index("confirmatory-data-source-audit")
    confirmatory_source_audit = next(
        step for step in steps if step.name == "confirmatory-data-source-audit"
    )
    assert confirmatory_source_audit.command[-2:] == ("--audit-only", "--verify-source")
    assert [step.name for step in steps].index("confirmatory-data-source-audit") < [
        step.name for step in steps
    ].index("confirmatory-matrix-generation")
    assert [step.name for step in steps].index("confirmatory-matrix-generation") < [
        step.name for step in steps
    ].index("common-state-matrix")
    assert [step.name for step in steps].index("common-state-summary") < [
        step.name for step in steps
    ].index("basis-sensitivity-analysis")
    assert [step.name for step in steps].index("basis-sensitivity-analysis") < [
        step.name for step in steps
    ].index("basis-sensitivity-audit")
    basis_audit = next(step for step in steps if step.name == "basis-sensitivity-audit")
    assert basis_audit.command[-2:] == ("--audit-only", "--verify-inputs")
    assert [step.name for step in steps].index("basis-sensitivity-audit") < [
        step.name for step in steps
    ].index("short-branch-matrix-generation")
    assert [step.name for step in steps].index("functional-intervention-summary") < [
        step.name for step in steps
    ].index("training-dense-representation-matrix")
    assert [step.name for step in steps].index("hybrid-adamw-training") < [
        step.name for step in steps
    ].index("hybrid-adamw-evaluation")
    hybrid_evaluation = next(step for step in steps if step.name == "hybrid-adamw-evaluation")
    assert hybrid_evaluation.command[2] == "embed_optim.hybrid_evaluation"
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
    short_training_audit = next(
        step for step in steps if step.name == "short-branch-training-audit"
    )
    assert short_training_audit.command[2:] == (
        "embed_optim.short_branch_evaluation",
        "--training-audit-only",
    )
    assert [step.name for step in steps].index("short-branch-training-audit") < [
        step.name for step in steps
    ].index("short-branch-evaluation")
    assert [step.name for step in steps].index("short-branch-evaluation-audit") < [
        step.name for step in steps
    ].index("short-branch-summary")
    assert [step.name for step in steps].index("short-branch-summary") < [
        step.name for step in steps
    ].index("outcome-blog-render")
    assert [step.name for step in steps].index("confirmatory-summary") < [
        step.name for step in steps
    ].index("outcome-blog-render")
    assert [step.name for step in steps].index("outcome-blog-render") < [
        step.name for step in steps
    ].index("paper-results-render")
    assert [step.name for step in steps].index("paper-results-render") < [
        step.name for step in steps
    ].index("paper-evidence-audit")
    assert [step.name for step in steps].index("paper-evidence-audit") < [
        step.name for step in steps
    ].index("paper-draft-build")
    assert [step.name for step in steps].index("paper-draft-build") < [
        step.name for step in steps
    ].index("paper-final-strict-audit")
    late = next(step for step in steps if step.name == "late-token-dynamics-plot")
    assert late.command[1] == "-c"
    assert supervise_post_eval(args) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert [item["name"] for item in rendered] == [step.name for step in steps]


def test_full_pipeline_commands_have_importable_cli_contracts(tmp_path: Path):
    args = _args(tmp_path)
    args.skip_wandb_sync = False
    args.skip_validation = False
    steps = pipeline_steps(args)

    assert len(steps) == 62
    for step in steps:
        command = list(step.command)
        if len(command) >= 3 and command[1] == "-m":
            module = importlib.import_module(command[2])
            parser = getattr(module, "parse_args", None)
            if parser is not None:
                parser(command[3:])
        elif len(command) >= 3 and command[1] == "-c":
            compile(command[2], f"<pipeline:{step.name}>", "exec")
        else:
            assert shutil.which(command[0]) is not None, step


def test_distribution_build_uses_uv_instead_of_shadowable_python_module(tmp_path: Path):
    args = _args(tmp_path)
    args.skip_validation = False

    distribution = next(step for step in pipeline_steps(args) if step.name == "distribution-build")

    assert distribution.command == ("uv", "build")
    steps = pipeline_steps(args)
    assert [step.name for step in steps].index("distribution-build") < [
        step.name for step in steps
    ].index("distribution-audit")
    assert [step.name for step in steps].index("distribution-audit") < [
        step.name for step in steps
    ].index("paper-final-strict-audit")


def test_all_training_steps_have_bounded_worker_retries(tmp_path: Path):
    args = _args(tmp_path)
    args.worker_retries = 3

    training_steps = [
        step
        for step in pipeline_steps(args)
        if step.name == "hybrid-adamw-training"
        or step.name.startswith("confirmatory-training-seed-")
        or step.name.startswith("short-branch-training-seed-")
    ]

    assert len(training_steps) == 7
    assert all(
        step.command[step.command.index("--max-retries") + 1] == "3" for step in training_steps
    )


def test_pipeline_wait_gate_and_ledger_are_complete(tmp_path: Path):
    args = _args(
        tmp_path,
        "--wait-pids",
        "12345",
        "--wait-for-command",
        "scripts/eval/dense_parallel.py",
    )
    _progress(Path(args.progress))
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        kwargs["stdout"].write("fixture command output\n")
        return subprocess.CompletedProcess(command, 0)

    assert (
        supervise_post_eval(
            args,
            run_command=run,
            pid_exists=lambda pid: False,
            matching_command_pids=lambda fragment: [],
        )
        == 0
    )
    assert len(commands) == 56
    ledger = json.loads((Path(args.log_dir) / "pipeline-ledger.json").read_text())
    assert ledger["complete"] is True
    assert ledger["wait_pids"] == [12345]
    assert ledger["wait_for_commands"] == ["scripts/eval/dense_parallel.py"]
    assert len(ledger["steps"]) == 56
    assert all(step["complete"] for step in ledger["steps"])
    assert all(len(step["attempts"]) == 1 for step in ledger["steps"])
    assert len(list(Path(args.log_dir).glob("*.log"))) == 56


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


def test_pipeline_resume_skips_only_matching_completed_prefix(tmp_path: Path):
    args = _args(tmp_path, "--step-retries", "0")
    _progress(Path(args.progress))
    initial_calls = []

    def fail_second(command, **kwargs):
        initial_calls.append(command)
        kwargs["stdout"].write("initial attempt\n")
        return subprocess.CompletedProcess(command, 0 if len(initial_calls) == 1 else 17)

    assert supervise_post_eval(args, run_command=fail_second) == 1
    failed = json.loads((Path(args.log_dir) / "pipeline-ledger.json").read_text())
    assert failed["steps"][0]["complete"] is True
    assert failed["failed_step"] == "strict-blog-render"

    args.resume = True
    resumed_calls = []

    def succeed(command, **kwargs):
        resumed_calls.append(command)
        kwargs["stdout"].write("resumed attempt\n")
        return subprocess.CompletedProcess(command, 0)

    assert supervise_post_eval(args, run_command=succeed) == 0
    assert len(resumed_calls) == 55
    assert resumed_calls[0][2] == "embed_optim.aggregate"
    ledger = json.loads((Path(args.log_dir) / "pipeline-ledger.json").read_text())
    assert ledger["complete"] is True
    assert ledger["resume_count"] == 1
    assert len(ledger["steps"]) == 56
    assert ledger["steps"][0] == failed["steps"][0]
    assert ledger["resume_history"][0]["completed_prefix"] == 1
    assert ledger["resume_history"][0]["failed_step"] == "strict-blog-render"
    archive = Path(ledger["resume_history"][0]["source"])
    assert json.loads(archive.read_text()) == failed
    assert len(list(Path(args.log_dir).glob("*.resume-1.attempt-1.log"))) == 55


def test_pipeline_resume_reexecutes_after_completed_command_drift(tmp_path: Path):
    args = _args(tmp_path)
    _progress(Path(args.progress))

    def succeed(command, **kwargs):
        kwargs["stdout"].write("fixture command output\n")
        return subprocess.CompletedProcess(command, 0)

    assert supervise_post_eval(args, run_command=succeed) == 0
    ledger_path = Path(args.log_dir) / "pipeline-ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["complete"] = False
    ledger["steps"][0]["command"][-1] = "changed-command"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    args.resume = True
    calls = []

    def rerun(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write("rerun\n")
        return subprocess.CompletedProcess(command, 0)

    assert supervise_post_eval(args, run_command=rerun) == 0
    assert len(calls) == 56


def test_pipeline_resume_requires_an_existing_valid_ledger(tmp_path: Path):
    args = _args(tmp_path, "--resume")
    _progress(Path(args.progress))

    with pytest.raises(ValueError, match="Cannot resume invalid pipeline ledger"):
        supervise_post_eval(args)


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


def test_pipeline_waits_for_replacement_evaluator_commands_after_complete_coverage(
    tmp_path: Path, capsys
):
    fragment = "scripts/eval/late_interaction.py"
    args = _args(tmp_path, "--wait-for-command", fragment)
    _progress(Path(args.progress))
    matches = iter(([301, 302], [302], []))
    sleeps = []

    def run(command, **kwargs):
        kwargs["stdout"].write("fixture command output\n")
        return subprocess.CompletedProcess(command, 0)

    assert (
        supervise_post_eval(
            args,
            run_command=run,
            sleeper=sleeps.append,
            matching_command_pids=lambda observed: next(matches),
        )
        == 0
    )
    assert sleeps == [args.poll_seconds, args.poll_seconds]
    assert "command_matches" in capsys.readouterr().out


def test_strict_progress_exposes_transient_watcher_error(tmp_path: Path):
    progress = tmp_path / "progress.json"
    _progress(progress, error="audit failed")
    with pytest.raises(TransientProgressAuditError, match="audit failed"):
        _strict_progress(progress)


def test_post_eval_cli_rejects_empty_command_fragment():
    with pytest.raises(SystemExit):
        parse_args(["--wait-for-command", ""])
