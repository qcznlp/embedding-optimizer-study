import hashlib
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim.config import OptimizerConfig, RunConfig
from embed_optim.family_training_queue import (
    CommandOutcome,
    QueueJob,
    QueueTermination,
    _exclusive_pool_lease,
    _pid_command,
    _quarantine_invalid_completion,
    _repository,
    _run_command_with_watchdog,
    load_queue_plan,
    parse_args,
    run_pool,
)


def test_frozen_dense_queue_covers_each_output_once():
    _, payload, pools = load_queue_plan("configs/dense_training_queue.json")
    jobs = [job for queue in pools.values() for job in queue]

    assert len(jobs) == payload["expected"]["total_runs"] == 18
    assert sum(job.phase == "confirmatory" for job in jobs) == 9
    assert sum(job.phase == "short-branch" for job in jobs) == 9
    assert {job.config.model_family for job in jobs} == {"dense"}
    assert len({job.config.output_dir.resolve() for job in jobs}) == 18


def test_queue_job_identity_includes_matrix_seed(tmp_path):
    config = SimpleNamespace(model_family="dense", run_id="muon-selected")
    first = QueueJob("confirmatory", tmp_path / "seed1.yaml", config)
    second = QueueJob("confirmatory", tmp_path / "seed2.yaml", config)

    assert first.identity != second.identity


def test_queue_cli_requires_four_gpus():
    with pytest.raises(SystemExit):
        parse_args(["--pool", "a", "--gpus", "0,1", "--port", "30100"])


@pytest.mark.parametrize(
    "gpus",
    (
        "0,1,2,2",
        "0,1,2,-3",
        "0,1,2,three",
        "0,1,2,03",
        "0,1,2,",
    ),
)
def test_queue_cli_rejects_noncanonical_or_duplicate_gpu_tokens(gpus):
    with pytest.raises(SystemExit):
        parse_args(["--pool", "a", "--gpus", gpus, "--port", "30100"])


def test_queue_cli_normalizes_valid_gpu_tokens():
    args = parse_args(["--pool", "a", "--gpus", " 0, 2,4, 6 ", "--port", "30100"])

    assert args.gpus == "0,2,4,6"
    assert args.job_timeout_seconds == 24 * 60 * 60


def test_pid_command_returns_none_for_absent_pid():
    assert _pid_command(2**30) is None


def test_queue_plan_rejects_changed_bound_matrix(tmp_path, monkeypatch):
    source = Path("configs/dense_training_queue.json")
    payload = json.loads(source.read_text())
    plan = tmp_path / "configs" / source.name
    plan.parent.mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    payload["source_bindings"][0]["sha256"] = "0" * 64
    plan.write_text(json.dumps(payload))
    monkeypatch.setattr("embed_optim.family_training_queue.resolve_matrix_path", lambda _: plan)
    monkeypatch.setattr("embed_optim.family_training_queue._repository", lambda _: Path.cwd())

    with pytest.raises(ValueError, match="Frozen queue source differs"):
        load_queue_plan(plan)


def test_repository_for_external_absolute_plan_uses_current_repo(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    (repository / "src" / "embed_optim").mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname='test'\n")
    external = tmp_path / "external" / "queue.json"
    external.parent.mkdir()
    external.write_text("{}")
    monkeypatch.chdir(repository)

    assert _repository(external.resolve()) == repository.resolve()


def test_external_absolute_plan_resolves_all_relative_bindings_from_repo(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    (repository / "src" / "embed_optim").mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname='test'\n")
    configs = repository / "configs"
    configs.mkdir()
    amendment = configs / "dense_scope_amendment.json"
    amendment.write_text(
        json.dumps(
            {
                "status": "user_directed_post_hoc_scope_amendment",
                "active_scope": {"families": ["dense"]},
            }
        )
    )
    bindings = []
    records = {"a": [], "b": []}
    configs_by_path = {}
    for index in range(6):
        matrix = configs / f"matrix-{index}.yaml"
        matrix.write_text(f"matrix-{index}\n")
        relative = str(matrix.relative_to(repository))
        run_id = f"run-{index}"
        bindings.append(
            {"path": relative, "sha256": hashlib.sha256(matrix.read_bytes()).hexdigest()}
        )
        records["a" if index % 2 == 0 else "b"].append(
            {
                "phase": "confirmatory" if index < 3 else "short-branch",
                "matrix": relative,
                "run_id": run_id,
            }
        )
        configs_by_path[matrix.resolve()] = RunConfig(
            run_id=run_id,
            model_family="dense",
            optimizer=OptimizerConfig(name="adamw", lr=3e-5),
            model_name="test/model",
            dataset_path="data/test",
            output_root=f"outputs/{run_id}",
        )
    plan_payload = {
        "schema_version": 1,
        "status": "frozen_before_dense_confirmatory_or_short_branch_training",
        "family": "dense",
        "scope_amendment": {"path": "configs/dense_scope_amendment.json"},
        "source_bindings": bindings,
        "expected": {"total_runs": 6, "confirmatory_runs": 3, "short_branch_runs": 3},
        "pools": records,
    }
    external = tmp_path / "external" / "queue.json"
    external.parent.mkdir()
    external.write_text(json.dumps(plan_payload))
    monkeypatch.chdir(repository)
    monkeypatch.setattr(
        "embed_optim.family_training_queue.load_matrix",
        lambda path: [configs_by_path[Path(path).resolve()]],
    )

    resolved, _, pools = load_queue_plan(external.resolve())

    assert resolved == external.resolve()
    assert {job.matrix for jobs in pools.values() for job in jobs} == set(configs_by_path)
    assert all(job.config.output_dir.is_absolute() for jobs in pools.values() for job in jobs)


def test_pool_lease_fails_closed_for_duplicate_holder(tmp_path):
    lease = tmp_path / "pool-a.lease"
    with _exclusive_pool_lease(lease, {"pool": "a"}):
        with pytest.raises(RuntimeError, match="already held"):
            with _exclusive_pool_lease(lease, {"pool": "a"}):
                pytest.fail("duplicate lease unexpectedly acquired")


def test_child_inherited_lease_blocks_duplicate_after_parent_copy_closes(tmp_path):
    lease = tmp_path / "pool-a.lease"
    child = None
    try:
        with _exclusive_pool_lease(lease, {"pool": "a"}) as lease_fd:
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                pass_fds=(lease_fd,),
            )
        with pytest.raises(RuntimeError, match="already held"):
            with _exclusive_pool_lease(lease, {"pool": "a"}):
                pytest.fail("child-held lease unexpectedly released")
    finally:
        if child is not None:
            child.terminate()
            child.wait(timeout=5)

    with _exclusive_pool_lease(lease, {"pool": "a"}):
        pass


def test_watchdog_terminates_timed_out_process_group(tmp_path):
    outcome = _run_command_with_watchdog(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        cwd=tmp_path,
        timeout_seconds=0.05,
        termination_grace_seconds=0.05,
    )

    assert outcome == CommandOutcome(
        return_code=124,
        timed_out=True,
        process_group_cleaned=True,
    )


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), RuntimeError("wait failed")])
def test_watchdog_cleans_process_group_before_reraising_base_exception(
    tmp_path, monkeypatch, failure
):
    signals = []
    popen_kwargs = {}

    class Process:
        pid = 424242
        calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise failure
            return -signal.SIGTERM

    def popen(*_args, **kwargs):
        popen_kwargs.update(kwargs)
        return Process()

    monkeypatch.setattr("embed_optim.family_training_queue.subprocess.Popen", popen)

    def killpg(pid, signum):
        signals.append((pid, signum))
        if signum == 0:
            raise ProcessLookupError

    monkeypatch.setattr("embed_optim.family_training_queue.os.killpg", killpg)

    with pytest.raises(type(failure), match=str(failure) if str(failure) else None):
        _run_command_with_watchdog(
            [sys.executable, "-c", "pass"],
            cwd=tmp_path,
            timeout_seconds=10,
            termination_grace_seconds=1,
            lease_fd=17,
        )

    assert signals == [(424242, signal.SIGTERM), (424242, 0)]
    assert popen_kwargs["start_new_session"] is True
    assert popen_kwargs["pass_fds"] == (17,)


def test_watchdog_sigterm_handler_cleans_process_group(tmp_path, monkeypatch):
    signals = []

    class Process:
        pid = 434343
        calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                handler = signal.getsignal(signal.SIGTERM)
                assert callable(handler)
                handler(signal.SIGTERM, None)
            return -signal.SIGTERM

    monkeypatch.setattr(
        "embed_optim.family_training_queue.subprocess.Popen", lambda *_args, **_kwargs: Process()
    )

    def killpg(pid, signum):
        signals.append((pid, signum))
        if signum == 0:
            raise ProcessLookupError

    monkeypatch.setattr("embed_optim.family_training_queue.os.killpg", killpg)

    with pytest.raises(QueueTermination) as raised:
        _run_command_with_watchdog(
            [sys.executable, "-c", "pass"],
            cwd=tmp_path,
            timeout_seconds=10,
            termination_grace_seconds=1,
        )

    assert raised.value.signum == signal.SIGTERM
    assert signals == [(434343, signal.SIGTERM), (434343, 0)]


@pytest.mark.parametrize("return_code", [1, -signal.SIGTERM])
def test_nonzero_matrix_exit_kills_residual_group_and_releases_lease(tmp_path, return_code):
    lease = tmp_path / "pool-a.lease"
    child_pid_path = tmp_path / "child.pid"
    source = (
        "import os, pathlib, signal, time; "
        "pid=os.fork(); "
        f"path=pathlib.Path({str(child_pid_path)!r}); "
        "(signal.signal(signal.SIGTERM, signal.SIG_IGN), "
        "path.write_text(str(os.getpid())), time.sleep(60)) "
        "if pid == 0 else "
        "(time.sleep(0.2), os.kill(os.getpid(), "
        f"{signal.SIGTERM}) if {return_code} < 0 else None, "
        f"raise_exit({return_code}))"
    )
    # A small helper avoids statement-level branching restrictions in the
    # compact subprocess expression above.
    source = "def raise_exit(code): raise SystemExit(code)\n" + source

    with _exclusive_pool_lease(lease, {"pool": "a"}) as lease_fd:
        outcome = _run_command_with_watchdog(
            [sys.executable, "-c", source],
            cwd=tmp_path,
            timeout_seconds=5,
            termination_grace_seconds=0.1,
            lease_fd=lease_fd,
        )

    assert outcome.return_code == return_code
    assert outcome.process_group_cleaned
    assert child_pid_path.is_file()
    with _exclusive_pool_lease(lease, {"pool": "a"}):
        pass


def test_invalid_completed_output_is_preserved_for_clean_recovery(tmp_path):
    output = tmp_path / "outputs" / "dense" / "run"
    output.mkdir(parents=True)
    (output / "corrupt-marker").write_text("evidence")
    config = SimpleNamespace(model_family="dense", run_id="run", output_dir=output)
    job = QueueJob("confirmatory", tmp_path / "seed.yaml", config)
    audit = {"complete": False, "problems": ["checkpoint-10: corrupt optimizer"]}

    quarantined = _quarantine_invalid_completion(job, audit)

    assert quarantined is not None
    assert not output.exists()
    assert (quarantined / "corrupt-marker").read_text() == "evidence"
    receipt = quarantined.with_name(f"{quarantined.name}.queue-recovery.json")
    assert json.loads(receipt.read_text())["audit"] == audit


def _run_pool_args(plan, ledger):
    return SimpleNamespace(
        plan=plan,
        pool="a",
        ledger=ledger,
        gpus="0,1,2,3",
        port=30100,
        python=sys.executable,
        max_retries=2,
        wait_pid=None,
        wait_command_fragment=None,
        poll_seconds=1.0,
        job_timeout_seconds=100.0,
        termination_grace_seconds=1.0,
    )


def test_queue_clears_sticky_complete_before_waiting(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    (repository / "src" / "embed_optim").mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname='test'\n")
    plan = repository / "configs" / "queue.json"
    plan.parent.mkdir()
    plan.write_text("{}")
    ledger = tmp_path / "ledger.json"
    identity = {
        "path": str(plan.resolve()),
        "sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
    }
    ledger.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": True,
                "started_at": "earlier",
                "finished_at": "earlier",
                "plan": identity,
                "pool": "a",
                "family": "dense",
                "gpus": "0,1,2,3",
                "jobs": [],
            }
        )
    )
    monkeypatch.setattr(
        "embed_optim.family_training_queue.load_queue_plan",
        lambda _: (plan.resolve(), {}, {"a": [], "b": []}),
    )

    class StopAfterObservation(RuntimeError):
        pass

    def observe_reset(*_):
        payload = json.loads(ledger.read_text())
        assert payload["complete"] is False
        assert payload["active_attempt"]["pid"] > 0
        raise StopAfterObservation

    monkeypatch.setattr("embed_optim.family_training_queue._wait_for_process", observe_reset)

    with pytest.raises(StopAfterObservation):
        run_pool(_run_pool_args(plan, ledger))


def test_deeply_invalid_completed_job_is_rerun_instead_of_shallow_skipped(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    (repository / "src" / "embed_optim").mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname='test'\n")
    plan = repository / "configs" / "queue.json"
    plan.parent.mkdir()
    plan.write_text("{}")
    output = repository / "outputs" / "dense" / "run"
    config = SimpleNamespace(model_family="dense", run_id="run", output_dir=output)
    job = QueueJob("confirmatory", repository / "configs" / "seed.yaml", config)
    ledger = tmp_path / "ledger.json"
    audits = iter(
        [
            {"complete": False, "verified_checkpoints": 4, "problems": ["corrupt"]},
            {"complete": True, "verified_checkpoints": 5, "problems": []},
        ]
    )
    commands = []
    quarantines = []
    monkeypatch.setattr(
        "embed_optim.family_training_queue.load_queue_plan",
        lambda _: (plan.resolve(), {}, {"a": [job], "b": []}),
    )
    monkeypatch.setattr("embed_optim.family_training_queue._run_is_complete", lambda _: True)
    monkeypatch.setattr(
        "embed_optim.family_training_queue._deep_completion_audit", lambda _: next(audits)
    )
    monkeypatch.setattr(
        "embed_optim.family_training_queue._quarantine_invalid_completion",
        lambda *_: quarantines.append(job.identity) or (tmp_path / "quarantined"),
    )
    monkeypatch.setattr(
        "embed_optim.family_training_queue._run_command_with_watchdog",
        lambda command, **_: commands.append(command) or CommandOutcome(0),
    )

    assert run_pool(_run_pool_args(plan, ledger)) == 0
    assert quarantines == [job.identity]
    assert len(commands) == 1
    payload = json.loads(ledger.read_text())
    assert payload["complete"] is True
    assert payload["jobs"][0]["complete"] is True
