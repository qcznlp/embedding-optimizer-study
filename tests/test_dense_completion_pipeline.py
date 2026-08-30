from argparse import Namespace
from pathlib import Path

import pytest

from embed_optim.dense_completion_pipeline import (
    PipelineStep,
    _matching_completed_prefix,
    parse_args,
    pipeline_steps,
)


def _args():
    return Namespace(
        workdir=Path.cwd(),
        scope_amendment=Path("configs/dense_scope_amendment.json"),
        python="/usr/bin/python3",
        gpus="0,1,2,3,4,5,6,7",
        gpus_b="4,5,6,7",
        worker_retries=2,
        include_validation=False,
    )


def test_dense_pipeline_never_schedules_late_family():
    steps = pipeline_steps(_args())
    commands = [token for step in steps for token in step.command]

    assert "late" not in commands
    assert [step.name for step in steps[:7]] == [
        "hybrid-training-audit",
        "confirmatory-training-audit-seed-314159",
        "confirmatory-training-audit-seed-271828",
        "confirmatory-training-audit-seed-161803",
        "short-branch-training-audit-seed-314159",
        "short-branch-training-audit-seed-271828",
        "short-branch-training-audit-seed-161803",
    ]
    assert all(
        any(token.endswith("/configs/dense_scope_amendment.json") for token in step.command)
        for step in steps[7:]
    )


def test_dense_pipeline_resume_prefix_requires_same_command():
    steps = [PipelineStep("one", ("python", "one")), PipelineStep("two", ("python", "two"))]
    previous = {
        "steps": [
            {"name": "one", "command": ["python", "one"], "complete": True},
            {"name": "two", "command": ["python", "changed"], "complete": True},
        ]
    }

    assert _matching_completed_prefix(previous, steps) == 1


def test_dense_pipeline_cli_rejects_partial_gpu_set():
    with pytest.raises(SystemExit):
        parse_args(["--gpus", "0,1,2,3"])
