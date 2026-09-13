"""Exercise real archived controller loops with only temporary leases and fake commands."""

import ast
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reports/engineering-archive/main-resume-v1"


def load_controller(side):
    name = f"embed_optim._main_resume_{side}"
    spec = importlib.util.spec_from_file_location(
        name, ARCHIVE / side / "corrected_completion_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def exercise(root, monkeypatch, side="after", mutate=None, orphan=False):
    module = load_controller(side)
    args = module.parse_args(
        [
            "--workdir",
            str(root),
            "--matrix",
            str(root / "matrix.yaml"),
            "--resume",
            "--retry-delay",
            "0",
            "--max-attempts",
            "1",
        ]
    )
    steps = [
        module.PipelineStep("stage-one", ("stage-one",)),
        module.PipelineStep("stage-two", ("stage-two",)),
    ]
    contract = {
        "schema_version": 1,
        "sha256": "a" * 64,
        "steps": [
            {"index": i, "name": s.name, "command": list(s.command)} for i, s in enumerate(steps, 1)
        ],
    }
    directory = root / args.log_dir
    directory.mkdir(parents=True)
    ledger = module._new_ledger(contract)
    ledger["backups"] = {
        "fixture-run": {
            "run_id": "fixture-run",
            "complete": True,
            "attempts": [{"attempt": 1, "return_code": 0}],
        }
    }
    ledger["contract_migrations"] = [{"scope": "prior-fixture-transition"}]
    originals = {}
    for i, step in enumerate(steps, 1):
        path = directory / f"{i:02d}-{step.name}.attempt-1.log"
        path.write_text(f"original {step.name} evidence\n")
        originals[str(path)] = path.read_bytes()
        ledger["steps"].append(
            {
                "index": i,
                "name": step.name,
                "command": list(step.command),
                "complete": i == 1,
                "attempts": [
                    {
                        "attempt": 1,
                        "return_code": 0 if i == 1 else 1,
                        "log": module._file_identity(path, root),
                    }
                ],
            }
        )
    ledger["status"] = "retry_wait"
    ledger["failed_step"] = "02-stage-two"
    if orphan:
        path = directory / "02-stage-two.attempt-2.log"
        path.write_text("orphan evidence written before a controller crash\n")
        originals[str(path)] = path.read_bytes()
    if mutate:
        mutate(ledger)
    initial = deepcopy(ledger)
    ledger_path = directory / "pipeline-ledger.json"
    ledger_path.write_text(json.dumps(ledger))
    initial_bytes = ledger_path.read_bytes()
    monkeypatch.setattr(module, "resolve_matrix_path", lambda value: Path(value))
    monkeypatch.setattr(
        module, "_selected_configs", lambda _: [SimpleNamespace(run_id="fixture-run")]
    )
    monkeypatch.setattr(module, "_run_is_complete", lambda _: True)
    monkeypatch.setattr(module, "pipeline_steps", lambda *a: steps)
    monkeypatch.setattr(module, "_contract", lambda *a: contract)
    monkeypatch.setattr(module, "_backup_command", lambda *a, **kw: ("audit-backup",))
    calls = []

    def command(values, **kwargs):
        calls.append(list(values))
        kwargs["stdout"].write(f"new {values[0]} output\n")
        return SimpleNamespace(returncode=0)

    error = None
    try:
        code = module.run_pipeline(
            args, run_command=command, sleeper=lambda _: pytest.fail("Unexpected wait")
        )
    except RuntimeError as exception:
        error, code = str(exception), None
    return {
        "exit_code": code,
        "error": error,
        "calls": calls,
        "initial": initial,
        "ledger": json.loads(ledger_path.read_bytes()),
        "ledger_bytes_unchanged": ledger_path.read_bytes() == initial_bytes,
        "original_logs_unchanged": {p: Path(p).read_bytes() == raw for p, raw in originals.items()},
    }


def test_archived_live_loop_reproduces_repeated_step_and_overwritten_logs(tmp_path, monkeypatch):
    result = exercise(tmp_path, monkeypatch, side="before")
    assert result["exit_code"] == 0
    assert result["calls"] == [["audit-backup"], ["stage-one"], ["stage-two"]]
    assert len(result["ledger"]["steps"][1]["attempts"]) == 1
    assert not any(result["original_logs_unchanged"].values())


def test_resume_preserves_finished_step_attempts_logs_and_prior_history(tmp_path, monkeypatch):
    result = exercise(tmp_path, monkeypatch)
    assert result["exit_code"] == 0
    assert result["calls"] == [["audit-backup"], ["stage-two"]]
    assert result["ledger"]["steps"][0] == result["initial"]["steps"][0]
    assert (
        result["ledger"]["steps"][1]["attempts"][0] == result["initial"]["steps"][1]["attempts"][0]
    )
    assert result["ledger"]["steps"][1]["attempts"][-1]["attempt"] == 2
    assert all(result["original_logs_unchanged"].values())
    assert result["ledger"]["contract_migrations"] == result["initial"]["contract_migrations"]
    assert result["ledger"]["backups"]["fixture-run"]["complete"] is True


def test_resume_retains_orphan_attempt_log_without_overwriting_it(tmp_path, monkeypatch):
    result = exercise(tmp_path, monkeypatch, orphan=True)
    assert result["exit_code"] == 0
    assert result["ledger"]["steps"][1]["attempts"][-1]["attempt"] == 3
    assert all(result["original_logs_unchanged"].values())
    assert result["ledger"]["steps"][1]["preserved_orphan_attempt_logs"][0]["attempt"] == 2


def test_candidate_preserves_all_17_commands_arguments_and_unrelated_source():
    before, after = load_controller("before"), load_controller("after")
    args = before.parse_args(["--workdir", str(ROOT), "--python", "/usr/bin/python3"])
    old = before.pipeline_steps(args, ROOT)
    new = after.pipeline_steps(args, ROOT)
    assert len(old) == len(new) == 17
    assert [(s.name, s.command) for s in old] == [(s.name, s.command) for s in new]
    allowed = {
        "_load_ledger",
        "_validate_step_records",
        "_run_attempt",
        "_run_until_success",
        "run_pipeline",
    }

    def retained(side):
        tree = ast.parse((ARCHIVE / side / "corrected_completion_pipeline.py").read_text())
        return [
            ast.dump(n, include_attributes=False)
            for n in tree.body
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) or n.name not in allowed
        ]

    assert retained("before") == retained("after")


def test_resume_rejects_modified_completed_attempt_log(tmp_path, monkeypatch):
    result = exercise(tmp_path, monkeypatch)
    module = load_controller("after")
    log = result["ledger"]["steps"][0]["attempts"][0]["log"]
    (tmp_path / log["path"]).write_text("modified old output")
    steps = [
        module.PipelineStep("stage-one", ("stage-one",)),
        module.PipelineStep("stage-two", ("stage-two",)),
    ]
    with pytest.raises(RuntimeError, match="log identity changed"):
        module._validate_step_records(result["ledger"], steps, tmp_path)


def test_attempt_creation_cannot_overwrite_an_existing_log(tmp_path):
    module = load_controller("after")
    path = tmp_path / "existing.log"
    path.write_text("retained evidence")
    with pytest.raises(FileExistsError):
        module._run_attempt(
            ("never-run",),
            repository=tmp_path,
            log_path=path,
            lease_fd=-1,
            run_command=lambda *a, **kw: pytest.fail("Unexpected command"),
        )
    assert path.read_text() == "retained evidence"


@pytest.mark.parametrize(
    "mutation",
    ["reordered", "changed_command", "extra_step", "false_success", "completed_after_failed"],
)
def test_undeclared_step_history_is_rejected_before_work_or_ledger_write(
    tmp_path, monkeypatch, mutation
):
    def change(ledger):
        if mutation == "reordered":
            ledger["steps"].reverse()
        elif mutation == "changed_command":
            ledger["steps"][0]["command"] = ["unreviewed"]
        elif mutation == "extra_step":
            ledger["steps"].append(deepcopy(ledger["steps"][-1]))
        elif mutation == "false_success":
            ledger["steps"][0]["attempts"][-1]["return_code"] = 1
        else:
            ledger["steps"][0]["complete"] = False
            ledger["steps"][1]["complete"] = True
            ledger["steps"][1]["attempts"][-1]["return_code"] = 0

    result = exercise(tmp_path, monkeypatch, mutate=change)
    assert result["error"]
    assert result["calls"] == []
    assert result["ledger_bytes_unchanged"] is True
    assert all(result["original_logs_unchanged"].values())
