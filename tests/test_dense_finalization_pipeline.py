from __future__ import annotations

import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim.dense_completion_pipeline import CORE_STEP_NAMES, _step_contract
from embed_optim.dense_completion_pipeline import PipelineStep as CompletionStep
from embed_optim.dense_finalization_pipeline import (
    PipelineStep,
    ProcessIdentity,
    _matching_completed_prefix,
    _read_completion_ledger,
    _wait_for_completion,
    parse_args,
    pipeline_steps,
    run_pipeline,
)

SCOPE = {
    "path": "/scope.json",
    "sha256": "a" * 64,
    "status": "user_directed_post_hoc_scope_amendment",
    "amended_at_utc": "2026-08-31T00:00:00Z",
    "claim_boundary": "Dense only",
}
TRAINING_INPUTS = {
    "training_plan": {
        "path": "/frozen-dense-plan.json",
        "bytes": 123,
        "sha256": "b" * 64,
    },
    "training_ledgers": [
        {"pool": "a", "path": "/pool-a.json", "bytes": 456, "sha256": "c" * 64},
        {"pool": "b", "path": "/pool-b.json", "bytes": 789, "sha256": "d" * 64},
    ],
}


@pytest.fixture(autouse=True)
def _current_training_inputs(monkeypatch):
    monkeypatch.setattr(
        "embed_optim.dense_finalization_pipeline._validate_training_inputs",
        lambda **_kwargs: TRAINING_INPUTS,
    )
    source = Path("src/embed_optim/dense_completion_pipeline.py").resolve()
    monkeypatch.setattr(
        "embed_optim.dense_finalization_pipeline._repository_contract_sources",
        lambda _repository: (source,),
    )
    monkeypatch.setattr(
        "embed_optim.dense_finalization_pipeline.completion_pipeline_steps",
        lambda _args: [CompletionStep(name, ("python", name)) for name in CORE_STEP_NAMES],
    )


def _step_args(tmp_path: Path, *, include_wandb: bool = False) -> Namespace:
    return Namespace(
        workdir=tmp_path,
        scope_amendment=Path("configs/dense_scope_amendment.json"),
        python="/usr/bin/python3",
        include_wandb=include_wandb,
    )


def _valid_completion(repository: Path) -> dict[str, object]:
    training_inputs = json.loads(json.dumps(TRAINING_INPUTS))
    steps = [CompletionStep(name, ("python", name)) for name in CORE_STEP_NAMES]
    contract = _step_contract(steps)
    pipeline_arguments = {
        "workdir": str(repository.resolve()),
        "scope_amendment": SCOPE["path"],
        "python": "python",
        "gpus": "0,1,2,3,4,5,6,7",
        "gpus_b": "4,5,6,7",
        "worker_retries": 2,
        "include_validation": False,
    }
    binding = {
        "scope_amendment": SCOPE,
        "training_plan": training_inputs["training_plan"],
        "training_ledgers": training_inputs["training_ledgers"],
        "step_contract_sha256": contract["sha256"],
        "pipeline_arguments": pipeline_arguments,
    }
    return {
        "schema_version": 1,
        "complete": True,
        "families": ["dense"],
        "scope_amendment": SCOPE,
        "training_plan": training_inputs["training_plan"],
        "training_ledgers": training_inputs["training_ledgers"],
        "step_contract": contract,
        "pipeline_arguments": pipeline_arguments,
        "input_binding": binding,
        "steps": [
            {
                "index": index,
                "name": step.name,
                "command": list(step.command),
                "input_binding": binding,
                "complete": True,
            }
            for index, step in enumerate(steps, start=1)
        ],
    }


def _run_args(tmp_path: Path, completion: Path, *, resume: bool = False) -> Namespace:
    return Namespace(
        workdir=tmp_path,
        scope_amendment=Path("configs/dense_scope_amendment.json"),
        completion_ledger=completion,
        log_dir=Path("logs/finalization"),
        python=sys.executable,
        include_wandb=False,
        wait_pid=None,
        wait_command_fragment="embed_optim.dense_completion_pipeline",
        poll_seconds=0.001,
        step_retries=0,
        retry_delay=0.0,
        resume=resume,
    )


def _write_completion(path: Path, payload: dict[str, object] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or _valid_completion(path.parent)), encoding="utf-8")


def test_steps_expand_to_dense_only_release_contract(tmp_path: Path):
    steps = pipeline_steps(_step_args(tmp_path))

    assert [step.name for step in steps] == [
        "discovery-report",
        "retrieval-dynamics",
        "mechanism-report",
        "outcome-report",
        "paper-results",
        "paper-audit-strict",
        "tests",
        "ruff-check",
        "ruff-format-check",
        "paper-build",
        "distribution-build",
        "distribution-audit",
    ]
    for step in steps[:6]:
        assert "--families" in step.command
        family_index = step.command.index("--families")
        assert step.command[family_index + 1] == "dense"
        assert "--scope-amendment" in step.command
    assert "--strict" in steps[0].command
    assert "--strict" in steps[5].command
    assert steps[6].command[-2:] == ("pytest", "-q")
    assert steps[7].command[-4:] == ("check", "src", "tests", "scripts/eval")
    assert steps[8].command[-5:] == (
        "format",
        "--check",
        "src",
        "tests",
        "scripts/eval",
    )
    assert steps[9].command == ("make", "-C", "paper", "clean", "all")
    assert steps[10].command == ("uv", "build")
    assert all("wandb_sync" not in step.command for step in steps)


def test_wandb_is_an_explicit_opt_in_and_dense_only(tmp_path: Path):
    steps = pipeline_steps(_step_args(tmp_path, include_wandb=True))

    assert steps[-1].name == "wandb-sync-dense"
    assert steps[-1].command[2] == "embed_optim.wandb_sync"
    assert steps[-1].command[-4:-2] == ("--families", "dense")
    assert steps[-1].command[-2] == "--scope-amendment"
    assert steps[-1].command[-1].endswith("/configs/dense_scope_amendment.json")
    assert "late" not in steps[-1].command


def test_wait_tracks_process_start_identity_until_exit(monkeypatch, tmp_path: Path):
    command = f"{sys.executable} -m embed_optim.dense_completion_pipeline"
    identities = iter(
        [
            ProcessIdentity(123, 456, command),
            ProcessIdentity(123, 456, command),
            None,
        ]
    )
    sleeps: list[float] = []
    monkeypatch.setattr(
        "embed_optim.dense_finalization_pipeline._read_process_identity",
        lambda _pid: next(identities),
    )
    monkeypatch.setattr("embed_optim.dense_finalization_pipeline.time.sleep", sleeps.append)
    args = _run_args(tmp_path, tmp_path / "completion.json")
    args.wait_pid = 123

    _wait_for_completion(args)

    assert sleeps == [args.poll_seconds]


def test_wait_rejects_wrong_or_reused_pid(monkeypatch, tmp_path: Path):
    args = _run_args(tmp_path, tmp_path / "completion.json")
    args.wait_pid = 123
    monkeypatch.setattr(
        "embed_optim.dense_finalization_pipeline._read_process_identity",
        lambda _pid: ProcessIdentity(123, 456, "unrelated-command"),
    )
    with pytest.raises(RuntimeError, match="not the requested Dense completion"):
        _wait_for_completion(args)

    expected = f"{sys.executable} -m embed_optim.dense_completion_pipeline"
    identities = iter([ProcessIdentity(123, 456, expected), ProcessIdentity(123, 789, expected)])
    monkeypatch.setattr(
        "embed_optim.dense_finalization_pipeline._read_process_identity",
        lambda _pid: next(identities),
    )
    with pytest.raises(RuntimeError, match="was reused"):
        _wait_for_completion(args)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(complete=False),
        lambda payload: payload.update(families=["dense", "late"]),
        lambda payload: payload.update(scope_amendment={"sha256": "wrong"}),
        lambda payload: payload.update(steps=[{"name": "pending", "complete": False}]),
        lambda payload: payload.update(failed_step="spectral-transplant-summary"),
        lambda payload: payload["training_plan"].update(sha256="0" * 64),
        lambda payload: payload["step_contract"].update(sha256="0" * 64),
        lambda payload: payload["steps"][0].update(input_binding={}),
        lambda payload: payload.pop("input_binding"),
    ],
)
def test_completion_ledger_is_a_strict_dense_scope_gate(tmp_path: Path, mutation):
    path = tmp_path / "completion.json"
    payload = _valid_completion(tmp_path)
    mutation(payload)
    _write_completion(path, payload)

    with pytest.raises(RuntimeError, match="Dense completion ledger"):
        _read_completion_ledger(path, expected_scope=SCOPE, repository=tmp_path)


def test_completion_ledger_source_hashes_exact_bytes(tmp_path: Path):
    path = tmp_path / "completion.json"
    _write_completion(path)

    payload, source = _read_completion_ledger(
        path,
        expected_scope=SCOPE,
        repository=tmp_path,
    )

    raw = path.read_bytes()
    assert payload["complete"] is True
    assert source == {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def test_completion_ledger_rejects_internally_consistent_stale_implementation_contract(
    tmp_path: Path,
):
    path = tmp_path / "completion.json"
    payload = _valid_completion(tmp_path)
    contract = payload["step_contract"]
    contract["implementation_sources"][0]["sha256"] = "0" * 64
    body = {
        "steps": contract["steps"],
        "implementation_sources": contract["implementation_sources"],
    }
    contract["sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload["input_binding"]["step_contract_sha256"] = contract["sha256"]
    for step in payload["steps"]:
        step["input_binding"]["step_contract_sha256"] = contract["sha256"]
    _write_completion(path, payload)

    with pytest.raises(RuntimeError, match="step contract differs from current inputs"):
        _read_completion_ledger(path, expected_scope=SCOPE, repository=tmp_path)


def test_completion_ledger_rejects_self_signed_noncanonical_commands(tmp_path: Path):
    path = tmp_path / "completion.json"
    payload = _valid_completion(tmp_path)
    payload["steps"][0]["command"] = ["python", "-c", "pass"]
    self_signed_steps = [
        CompletionStep(step["name"], tuple(step["command"])) for step in payload["steps"]
    ]
    contract = _step_contract(self_signed_steps)
    payload["step_contract"] = contract
    payload["input_binding"]["step_contract_sha256"] = contract["sha256"]
    for step in payload["steps"]:
        step["input_binding"]["step_contract_sha256"] = contract["sha256"]
    _write_completion(path, payload)

    with pytest.raises(RuntimeError, match="step contract differs from current inputs"):
        _read_completion_ledger(path, expected_scope=SCOPE, repository=tmp_path)


def test_pipeline_writes_atomic_complete_ledger_and_hashed_logs(monkeypatch, tmp_path: Path):
    completion = tmp_path / "completion.json"
    _write_completion(completion)
    args = _run_args(tmp_path, completion)
    steps = [
        PipelineStep("one", (sys.executable, "-c", "print('one')")),
        PipelineStep("two", (sys.executable, "-c", "print('two')")),
    ]
    monkeypatch.setattr(
        "embed_optim.dense_finalization_pipeline.resolve_scope",
        lambda *_args, **_kwargs: (("dense",), SCOPE),
    )
    monkeypatch.setattr(
        "embed_optim.dense_finalization_pipeline.pipeline_steps", lambda _args: steps
    )

    assert run_pipeline(args) == 0

    ledger_path = tmp_path / args.log_dir / "pipeline-ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["complete"] is True
    assert ledger["families"] == ["dense"]
    assert ledger["scope_amendment"] == SCOPE
    assert (
        ledger["completion_ledger"]["sha256"] == hashlib.sha256(completion.read_bytes()).hexdigest()
    )
    assert ledger["step_contract"]["sha256"] == ledger["input_binding"]["step_contract_sha256"]
    assert [record["name"] for record in ledger["steps"]] == ["one", "two"]
    for record in ledger["steps"]:
        assert record["input_binding"] == ledger["input_binding"]
        attempt = record["attempts"][-1]
        log_path = Path(attempt["log"]["path"])
        assert attempt["return_code"] == 0
        assert attempt["log"]["bytes"] == log_path.stat().st_size
        assert attempt["log"]["sha256"] == hashlib.sha256(log_path.read_bytes()).hexdigest()


def test_pipeline_retries_a_failed_step_and_records_both_attempts(monkeypatch, tmp_path: Path):
    completion = tmp_path / "completion.json"
    _write_completion(completion)
    args = _run_args(tmp_path, completion)
    args.step_retries = 1
    step = PipelineStep("retry", ("synthetic-command",))
    return_codes = iter((9, 0))

    def fake_run(*_args, stdout, **_kwargs):
        return_code = next(return_codes)
        stdout.write(f"return code {return_code}\n")
        return SimpleNamespace(returncode=return_code)

    monkeypatch.setattr(
        "embed_optim.dense_finalization_pipeline.resolve_scope",
        lambda *_args, **_kwargs: (("dense",), SCOPE),
    )
    monkeypatch.setattr(
        "embed_optim.dense_finalization_pipeline.pipeline_steps", lambda _args: [step]
    )
    monkeypatch.setattr("embed_optim.dense_finalization_pipeline.subprocess.run", fake_run)

    assert run_pipeline(args) == 0

    ledger = json.loads(
        (tmp_path / args.log_dir / "pipeline-ledger.json").read_text(encoding="utf-8")
    )
    attempts = ledger["steps"][0]["attempts"]
    assert [attempt["return_code"] for attempt in attempts] == [9, 0]
    assert all(len(attempt["log"]["sha256"]) == 64 for attempt in attempts)


def test_resume_reruns_prefix_when_step_contract_changes(monkeypatch, tmp_path: Path):
    completion = tmp_path / "completion.json"
    _write_completion(completion)
    args = _run_args(tmp_path, completion)
    first = PipelineStep("one", (sys.executable, "-c", "print('one')"))
    failing = PipelineStep("two", (sys.executable, "-c", "raise SystemExit(7)"))
    current_steps = [first, failing]
    monkeypatch.setattr(
        "embed_optim.dense_finalization_pipeline.resolve_scope",
        lambda *_args, **_kwargs: (("dense",), SCOPE),
    )
    monkeypatch.setattr(
        "embed_optim.dense_finalization_pipeline.pipeline_steps", lambda _args: current_steps
    )

    assert run_pipeline(args) == 1
    ledger_path = tmp_path / args.log_dir / "pipeline-ledger.json"
    failed = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert failed["failed_step"] == "two"
    first_record = failed["steps"][0]

    current_steps[1] = PipelineStep("two", (sys.executable, "-c", "print('recovered')"))
    args.resume = True
    assert run_pipeline(args) == 0

    resumed = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert resumed["complete"] is True
    assert "failed_step" not in resumed
    assert resumed["steps"][0] != first_record
    assert resumed["steps"][0]["input_binding"] == resumed["input_binding"]
    assert resumed["steps"][1]["command"] == list(current_steps[1].command)


def test_complete_resume_revalidates_completion_and_reruns_when_it_changes(
    monkeypatch, tmp_path: Path
):
    completion = tmp_path / "completion.json"
    _write_completion(completion)
    args = _run_args(tmp_path, completion)
    step = PipelineStep("one", (sys.executable, "-c", "print('one')"))
    reads = 0
    original_read = _read_completion_ledger

    def counted_read(*read_args, **read_kwargs):
        nonlocal reads
        reads += 1
        return original_read(*read_args, **read_kwargs)

    monkeypatch.setattr(
        "embed_optim.dense_finalization_pipeline.resolve_scope",
        lambda *_args, **_kwargs: (("dense",), SCOPE),
    )
    monkeypatch.setattr(
        "embed_optim.dense_finalization_pipeline.pipeline_steps", lambda _args: [step]
    )
    monkeypatch.setattr(
        "embed_optim.dense_finalization_pipeline._read_completion_ledger", counted_read
    )

    assert run_pipeline(args) == 0
    ledger_path = tmp_path / args.log_dir / "pipeline-ledger.json"
    first = json.loads(ledger_path.read_text(encoding="utf-8"))
    args.resume = True
    before_complete_resume = reads
    assert run_pipeline(args) == 0
    assert reads > before_complete_resume
    rerun = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert rerun["steps"][0]["finished_at"] != first["steps"][0]["finished_at"]

    completion_payload = json.loads(completion.read_text(encoding="utf-8"))
    completion_payload["finished_at"] = "new-provenance"
    _write_completion(completion, completion_payload)
    assert run_pipeline(args) == 0
    changed = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert changed["completion_ledger"] != first["completion_ledger"]
    assert changed["steps"][0]["finished_at"] != first["steps"][0]["finished_at"]


def test_resume_prefix_requires_command_identity():
    steps = [
        PipelineStep("one", ("python", "one")),
        PipelineStep("two", ("python", "two")),
    ]
    previous = {
        "steps": [
            {"name": "one", "command": ["python", "one"], "complete": True},
            {"name": "two", "command": ["python", "changed"], "complete": True},
        ]
    }

    assert _matching_completed_prefix(previous, steps) == 1


def test_resume_prefix_requires_current_completion_and_step_contract_binding():
    steps = [PipelineStep("one", ("python", "one"))]
    binding = {
        "scope_amendment": SCOPE,
        "completion_ledger": {"path": "/completion", "bytes": 1, "sha256": "e" * 64},
        "step_contract_sha256": "f" * 64,
    }
    previous = {
        "steps": [
            {
                "name": "one",
                "command": ["python", "one"],
                "input_binding": binding,
                "complete": True,
            }
        ]
    }

    assert _matching_completed_prefix(previous, steps, binding) == 1
    changed = {
        **binding,
        "completion_ledger": {**binding["completion_ledger"], "sha256": "0" * 64},
    }
    assert _matching_completed_prefix(previous, steps, changed) == 0


def test_cli_rejects_invalid_wait_and_retry_values():
    with pytest.raises(SystemExit):
        parse_args(["--wait-pid", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--poll-seconds", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--step-retries", "-1"])
