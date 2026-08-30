import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim.family_training_queue import (
    QueueJob,
    _pid_command,
    load_queue_plan,
    parse_args,
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
