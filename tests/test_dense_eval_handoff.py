from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim import dense_eval_handoff
from embed_optim.dense_eval_handoff import ConditionNotReady, HandoffDecision
from embed_optim.family_training_queue import QueueJob


def _handoff_fixture(tmp_path, monkeypatch, *, active_pool="b"):
    repository = tmp_path
    configs = repository / "configs"
    configs.mkdir()
    (repository / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    plan = configs / "dense_training_queue.json"
    plan.write_text('{"frozen":true}\n')
    scope = configs / "dense_scope_amendment.json"
    scope.write_text('{"dense":true}\n')
    jobs_by_pool = {"a": [], "b": []}
    complete: dict[str, bool] = {}
    for pool in ("a", "b"):
        for index in range(9):
            matrix = configs / f"{pool}-seed-{index}.yaml"
            matrix.write_text("runs: []\n")
            config = SimpleNamespace(
                model_family="dense",
                run_id=f"{pool}-run-{index}",
                output_dir=repository / "outputs" / pool / f"run-{index}",
            )
            job = QueueJob(
                "short-branch" if index >= 5 else "confirmatory",
                matrix,
                config,
            )
            jobs_by_pool[pool].append(job)
            complete[job.identity] = True
    remaining = jobs_by_pool[active_pool][-1]
    complete[remaining.identity] = False
    payload = {"scope_amendment": {"path": "configs/dense_scope_amendment.json"}}
    monkeypatch.setattr(
        dense_eval_handoff,
        "load_queue_plan",
        lambda _path: (plan.resolve(), payload, jobs_by_pool),
    )
    monkeypatch.setattr(
        dense_eval_handoff,
        "resolve_scope",
        lambda families, amendment: (("dense",), {"path": str(amendment)}),
    )

    ledgers = {}
    gpus = {"a": "0,1,2,3", "b": "4,5,6,7"}
    for pool in ("a", "b"):
        ledger = repository / "logs" / f"queue-{pool}.json"
        ledger.parent.mkdir(exist_ok=True)
        records = []
        for index, job in enumerate(jobs_by_pool[pool], start=1):
            if not complete[job.identity]:
                break
            records.append(
                {
                    "index": index,
                    "identity": job.identity,
                    "matrix": str(job.matrix.resolve()),
                    "output_dir": str(job.config.output_dir.resolve()),
                    "attempts": [{"return_code": 0}],
                    "complete": True,
                }
            )
        ledger.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "complete": pool != active_pool,
                    "plan": {
                        "path": str(plan.resolve()),
                        "sha256": dense_eval_handoff._sha256(plan),
                    },
                    "pool": pool,
                    "family": "dense",
                    "gpus": gpus[pool],
                    "jobs": records,
                }
            )
        )
        ledgers[pool] = ledger
    queue_pids = {"a": 111, "b": 222}
    child_pid = 333
    active_pid = queue_pids[active_pool]
    idle_pid = queue_pids["b" if active_pool == "a" else "a"]
    argv = {
        active_pid: [
            "/python",
            "-m",
            "embed_optim.family_training_queue",
            "--plan",
            str(plan.resolve()),
            "--pool",
            active_pool,
            "--gpus",
            gpus[active_pool],
            "--ledger",
            str(ledgers[active_pool].resolve()),
        ],
        child_pid: [
            "/python",
            "-m",
            "embed_optim.matrix",
            "--matrix",
            str(remaining.matrix.resolve()),
            "--families",
            "dense",
            "--run-ids",
            remaining.config.run_id,
            "--gpus-a",
            gpus[active_pool],
            "--gpus-b",
            gpus[active_pool],
        ],
    }
    args = Namespace(
        plan=plan.resolve(),
        scope_amendment=scope.resolve(),
        ledger_a=ledgers["a"],
        ledger_b=ledgers["b"],
        queue_pid_a=queue_pids["a"],
        queue_pid_b=queue_pids["b"],
        gpus_a=gpus["a"],
        gpus_b=gpus["b"],
    )
    return SimpleNamespace(
        args=args,
        complete=complete,
        jobs=jobs_by_pool,
        remaining=remaining,
        ledgers=ledgers,
        argv=argv,
        active_pid=active_pid,
        idle_pid=idle_pid,
        child_pid=child_pid,
        plan=plan,
        scope=scope,
    )


def test_handoff_requires_17_durable_statuses_and_exact_active_identity(tmp_path, monkeypatch):
    fixture = _handoff_fixture(tmp_path, monkeypatch)
    reads = []

    def read_argv(pid):
        reads.append(pid)
        return fixture.argv.get(pid)

    decision = dense_eval_handoff.inspect_handoff_condition(
        fixture.args,
        run_is_complete=lambda config: fixture.complete[
            next(
                job.identity
                for jobs in fixture.jobs.values()
                for job in jobs
                if job.config is config
            )
        ],
        read_argv=read_argv,
        read_children=lambda pid: [fixture.child_pid],
    )

    assert decision.idle_pool == "a"
    assert decision.active_pool == "b"
    assert decision.gpu_tokens == "0,1,2,3"
    assert decision.remaining_identity == fixture.remaining.identity
    assert len(decision.job_statuses) == 18
    assert sum(item["artifact_complete"] for item in decision.job_statuses) == 17
    assert set(reads) == {fixture.idle_pid, fixture.active_pid, fixture.child_pid}


def test_handoff_fails_closed_on_a_second_unfinished_job(tmp_path, monkeypatch):
    fixture = _handoff_fixture(tmp_path, monkeypatch)
    fixture.complete[fixture.jobs["a"][-1].identity] = False

    with pytest.raises(ConditionNotReady, match="durably reconciled|unfinished jobs"):
        dense_eval_handoff.inspect_handoff_condition(
            fixture.args,
            run_is_complete=lambda config: fixture.complete[
                next(
                    job.identity
                    for jobs in fixture.jobs.values()
                    for job in jobs
                    if job.config is config
                )
            ],
            read_argv=lambda pid: fixture.argv.get(pid),
            read_children=lambda pid: [fixture.child_pid],
        )


def test_handoff_rejects_a_mismatched_active_run(tmp_path, monkeypatch):
    fixture = _handoff_fixture(tmp_path, monkeypatch)
    child = fixture.argv[fixture.child_pid]
    child[child.index("--run-ids") + 1] = "different-run"

    with pytest.raises(RuntimeError, match="remaining Dense run"):
        dense_eval_handoff.inspect_handoff_condition(
            fixture.args,
            run_is_complete=lambda config: config is not fixture.remaining.config,
            read_argv=lambda pid: fixture.argv.get(pid),
            read_children=lambda pid: [fixture.child_pid],
        )


def test_handoff_rejects_a_live_idle_queue_pid(tmp_path, monkeypatch):
    fixture = _handoff_fixture(tmp_path, monkeypatch)
    fixture.argv[fixture.idle_pid] = ["/python", "-m", "embed_optim.family_training_queue"]

    with pytest.raises(ConditionNotReady, match="still alive"):
        dense_eval_handoff.inspect_handoff_condition(
            fixture.args,
            run_is_complete=lambda config: config is not fixture.remaining.config,
            read_argv=lambda pid: fixture.argv.get(pid),
            read_children=lambda pid: [fixture.child_pid],
        )


def test_restore_decision_revalidates_the_canonical_17_of_18_boundary(tmp_path, monkeypatch):
    fixture = _handoff_fixture(tmp_path, monkeypatch)
    decision = dense_eval_handoff.inspect_handoff_condition(
        fixture.args,
        run_is_complete=lambda config: config is not fixture.remaining.config,
        read_argv=lambda pid: fixture.argv.get(pid),
        read_children=lambda pid: [fixture.child_pid],
    )
    payload = dense_eval_handoff._decision_payload(decision)

    restored = dense_eval_handoff._restore_decision(fixture.args, payload)

    assert restored.idle_pool == decision.idle_pool
    assert restored.active_pool == decision.active_pool
    assert restored.remaining_identity == fixture.remaining.identity

    payload["job_statuses"][0]["ledger_complete"] = False
    with pytest.raises(RuntimeError, match="17/18 boundary"):
        dense_eval_handoff._restore_decision(fixture.args, payload)


def test_early_evaluation_command_is_explicit_dense_only(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    monkeypatch.setattr(
        dense_eval_handoff,
        "load_confirmatory_protocol",
        lambda _path: (protocol.resolve(), {"training": {"matrix_output_dir": str(tmp_path)}}),
    )
    args = Namespace(
        protocol=protocol,
        matrix_dir=tmp_path,
        python="/python",
        experiment_matrix=tmp_path / "experiment.yaml",
        validation_spec=tmp_path / "validation.json",
        results_root=tmp_path / "results",
        scope_amendment=tmp_path / "scope.json",
        evaluation_log_dir=tmp_path / "logs",
        worker_python="/worker/python",
        gpu_lock_dir=tmp_path / "locks",
        gpu_lock_timeout_seconds=123.0,
        receipt=tmp_path / "receipt.json",
    )
    decision = HandoffDecision("a", "b", "0,1,2,3", "remaining", (), {}, {})

    command = dense_eval_handoff._evaluation_command(args, decision)

    assert command[:3] == ["/python", "-m", "embed_optim.confirmatory_evaluation"]
    assert command[command.index("--families") + 1] == "dense"
    assert command[command.index("--gpus-a") + 1] == "0,1,2,3"
    assert command[command.index("--gpus-b") + 1] == "0,1,2,3"
    task_start = command.index("--tasks") + 1
    task_end = command.index("--scope-amendment")
    assert tuple(command[task_start:task_end]) == dense_eval_handoff.EARLY_PARTIAL_TASKS
    assert len(command[task_start:task_end]) * 3 * 3 == 72
    assert "late" not in command


def test_early_receipt_is_independent_from_final_receipt(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    args = dense_eval_handoff.parse_args(
        [
            "--ledger-a",
            "a.json",
            "--ledger-b",
            "b.json",
            "--queue-pid-a",
            "111",
            "--queue-pid-b",
            "222",
            "--workdir",
            str(tmp_path),
        ]
    )
    assert args.receipt == Path("reports/confirmatory/early-partial-evaluation-receipt.json")
    assert args.receipt.name != "evaluation-receipt.json"

    with pytest.raises(SystemExit):
        dense_eval_handoff.parse_args(
            [
                "--ledger-a",
                "a.json",
                "--ledger-b",
                "b.json",
                "--queue-pid-a",
                "111",
                "--queue-pid-b",
                "222",
                "--workdir",
                str(tmp_path),
                "--receipt",
                "reports/confirmatory/evaluation-receipt.json",
            ]
        )


def test_condition_timeout_is_durably_recorded(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    scope = tmp_path / "scope.json"
    scope.write_text("{}")
    monkeypatch.setattr(
        dense_eval_handoff, "load_queue_plan", lambda path: (plan.resolve(), {}, {})
    )
    monkeypatch.setattr(dense_eval_handoff, "resolve_scope", lambda *a, **k: (("dense",), {}))
    monkeypatch.setattr(dense_eval_handoff, "_base_identity", lambda *a, **k: {"fixture": 1})
    monkeypatch.setattr(
        dense_eval_handoff,
        "inspect_handoff_condition",
        lambda args: (_ for _ in ()).throw(ConditionNotReady("not yet")),
    )
    ledger = tmp_path / "handoff.json"
    args = Namespace(
        plan=plan,
        scope_amendment=scope,
        handoff_ledger=ledger,
        supervisor_lock=tmp_path / "handoff.lock",
        resume=False,
        gpus_a="0,1,2,3",
        gpus_b="4,5,6,7",
        queue_pid_a=111,
        queue_pid_b=222,
        condition_timeout_seconds=0.001,
        poll_seconds=0.001,
    )

    assert dense_eval_handoff.run_handoff(args) == 2
    payload = json.loads(ledger.read_text())
    assert payload["status"] == "condition-timeout"
    assert payload["error"] == "not yet"


def test_result_receipt_must_equal_fresh_audit(tmp_path, monkeypatch):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"complete": True, "stale": True}))
    fresh = {
        "complete": True,
        "tasks": list(dense_eval_handoff.EARLY_PARTIAL_TASKS),
        "valid_units": 72,
        "expected_units": 72,
    }
    monkeypatch.setattr(dense_eval_handoff, "audit_confirmatory_evaluations", lambda *a, **k: fresh)
    args = Namespace(
        protocol=tmp_path / "protocol.json",
        experiment_matrix=tmp_path / "experiment.yaml",
        validation_spec=tmp_path / "validation.json",
        matrix_dir=tmp_path,
        results_root=tmp_path / "results",
        scope_amendment=tmp_path / "scope.json",
        receipt=receipt,
        workdir=tmp_path,
    )

    with pytest.raises(RuntimeError, match="exactly match"):
        dense_eval_handoff._result_receipt(args)


def test_run_cleanup_on_wait_exception(tmp_path, monkeypatch):
    cleaned = []

    class FakeProcess:
        pid = 12345

        def wait(self, timeout=None):
            raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProcess())
    monkeypatch.setattr(
        dense_eval_handoff,
        "_terminate_and_reap_process_group",
        lambda process, **kwargs: cleaned.append(process.pid),
    )
    args = Namespace(
        process_log=tmp_path / "process.log",
        workdir=tmp_path,
        evaluation_timeout_seconds=10,
        termination_grace_seconds=0.01,
    )

    with pytest.raises(KeyboardInterrupt):
        dense_eval_handoff._run_with_timeout(["ignored"], args)
    assert cleaned == [12345]


def test_nonzero_evaluator_cleans_residual_process_group(tmp_path):
    child_file = tmp_path / "child.pid"
    code = (
        "import pathlib,subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"pathlib.Path({str(child_file)!r}).write_text(str(p.pid)); sys.exit(1)"
    )
    args = Namespace(
        process_log=tmp_path / "process.log",
        workdir=tmp_path,
        evaluation_timeout_seconds=10,
        termination_grace_seconds=0.05,
    )

    assert dense_eval_handoff._run_with_timeout([sys.executable, "-c", code], args) == 1
    child_pid = int(child_file.read_text())
    deadline = time.monotonic() + 2
    while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    if Path(f"/proc/{child_pid}").exists():
        # A killed orphan can briefly remain as a zombie, but it must not be running.
        stat = Path(f"/proc/{child_pid}/stat").read_text()
        assert stat[stat.rfind(")") + 2 :].split()[0] == "Z"


def test_resume_pid_reuse_fails_closed(tmp_path, monkeypatch):
    frozen = {"pid": 2222, "pgid": 2222, "start_time_ticks": 1, "argv": ["old"], "cwd": "/x"}
    monkeypatch.setattr(
        dense_eval_handoff,
        "_process_identity",
        lambda pid: {**frozen, "start_time_ticks": 2},
    )
    killed = []
    monkeypatch.setattr(os, "killpg", lambda *values: killed.append(values))

    with pytest.raises(RuntimeError, match="reused"):
        dense_eval_handoff._cleanup_resumed_attempt(
            {"process": frozen}, Namespace(termination_grace_seconds=0.01)
        )
    assert killed == []


def test_resume_after_condition_reaches_zero_does_not_reinspect(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    scope = tmp_path / "scope.json"
    scope.write_text("{}")
    ledger_path = tmp_path / "handoff.json"
    base = {"fixture": 1}
    ledger_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_identity": base,
                "status": "error",
                "complete": False,
                "handoff_condition": {"frozen": True},
                "attempts": [{"command": ["evaluate"]}],
            }
        )
    )
    decision = HandoffDecision("a", "b", "0,1,2,3", "remaining", (), {}, {})
    monkeypatch.setattr(dense_eval_handoff, "load_queue_plan", lambda path: (plan, {}, {}))
    monkeypatch.setattr(dense_eval_handoff, "resolve_scope", lambda *a, **k: (("dense",), {}))
    monkeypatch.setattr(dense_eval_handoff, "_base_identity", lambda *a, **k: base)
    monkeypatch.setattr(dense_eval_handoff, "_restore_decision", lambda *a, **k: decision)
    monkeypatch.setattr(dense_eval_handoff, "_existing_result", lambda args: None)
    monkeypatch.setattr(dense_eval_handoff, "_cleanup_resumed_attempt", lambda *a: None)
    monkeypatch.setattr(dense_eval_handoff, "_deep_audit_confirmatory", lambda args: {"ok": True})
    monkeypatch.setattr(dense_eval_handoff, "_evaluation_command", lambda *a: ["evaluate"])
    monkeypatch.setattr(
        dense_eval_handoff,
        "inspect_handoff_condition",
        lambda args: pytest.fail("resume must not recheck the now-zero remaining condition"),
    )
    monkeypatch.setattr(
        dense_eval_handoff,
        "_run_with_timeout",
        lambda command, args, on_started: 0,
    )
    monkeypatch.setattr(dense_eval_handoff, "_result_receipt", lambda args: {"valid_units": 72})
    args = Namespace(
        plan=plan,
        scope_amendment=scope,
        handoff_ledger=ledger_path,
        supervisor_lock=tmp_path / "handoff.lock",
        resume=True,
        gpus_a="0,1,2,3",
        gpus_b="4,5,6,7",
        queue_pid_a=111,
        queue_pid_b=222,
    )

    assert dense_eval_handoff.run_handoff(args) == 0
    assert json.loads(ledger_path.read_text())["complete"] is True
