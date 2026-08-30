from __future__ import annotations

import json
from argparse import Namespace
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
    assert "late" not in command


def test_condition_timeout_is_durably_recorded(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    scope = tmp_path / "scope.json"
    scope.write_text("{}")
    monkeypatch.setattr(
        dense_eval_handoff, "load_queue_plan", lambda path: (plan.resolve(), {}, {})
    )
    monkeypatch.setattr(dense_eval_handoff, "resolve_scope", lambda *a, **k: (("dense",), {}))
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
