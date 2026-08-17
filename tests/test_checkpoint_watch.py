import json
from types import SimpleNamespace

import pytest

from embed_optim.checkpoint_watch import _validate_formal_runtime, audit_once


def _config(tmp_path):
    return SimpleNamespace(
        model_family="late",
        run_id="muon-test",
        output_dir=tmp_path / "late" / "muon-test",
    )


def _write_schedule(config):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "checkpoint_schedule.json").write_text(
        json.dumps({"steps": [10, 20, 30, 40, 50]})
    )


def _write_checkpoint(config, step):
    checkpoint = config.output_dir / f"checkpoint-{step}"
    checkpoint.mkdir()
    (checkpoint / "optimizer.pt").write_bytes(f"optimizer-{step}".encode())
    (checkpoint / "model.safetensors").write_bytes(f"model-{step}".encode())
    return checkpoint


def test_watcher_audits_new_and_changed_payload_once(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _write_schedule(config)
    checkpoint = _write_checkpoint(config, 10)
    calls = []

    monkeypatch.setattr(
        "embed_optim.matrix._checkpoint_is_resumable", lambda path, world_size: path.is_dir()
    )

    def deep_problems(path, step, world_size, config, final_step):
        calls.append((path.name, step, world_size, final_step))
        return []

    monkeypatch.setattr("embed_optim.aggregate._deep_checkpoint_problems", deep_problems)
    monkeypatch.setattr(
        "embed_optim.aggregate._safetensors_digest", lambda path: f"digest-{path.name}"
    )
    state_path = tmp_path / "audit.json"

    state, events = audit_once([config], state_path)
    assert state["audited_checkpoints"] == 1
    assert state["expected_checkpoints"] == 5
    assert not state["audit_complete"]
    assert not state["training_complete"]
    assert [event["status"] for event in events] == ["passed"]
    assert calls == [("checkpoint-10", 10, 4, 50)]

    _, events = audit_once([config], state_path)
    assert events == []
    assert len(calls) == 1

    (checkpoint / "optimizer.pt").write_bytes(b"changed-and-longer")
    _, events = audit_once([config], state_path)
    assert [event["status"] for event in events] == ["passed"]
    assert len(calls) == 2


def test_watcher_requires_changed_models_and_training_marker(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _write_schedule(config)
    for step in (10, 20, 30, 40, 50):
        _write_checkpoint(config, step)
    (config.output_dir / "completed.json").write_text("{}")

    monkeypatch.setattr(
        "embed_optim.matrix._checkpoint_is_resumable", lambda path, world_size: path.is_dir()
    )
    monkeypatch.setattr(
        "embed_optim.aggregate._deep_checkpoint_problems", lambda *args, **kwargs: []
    )
    monkeypatch.setattr("embed_optim.aggregate._safetensors_digest", lambda path: path.name)

    state, events = audit_once([config], tmp_path / "audit.json")

    assert len(events) == 5
    assert state["audited_checkpoints"] == 5
    assert state["audit_complete"]
    assert state["training_complete"]


def test_watcher_rejects_unchanged_model_payload(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _write_schedule(config)
    _write_checkpoint(config, 10)
    _write_checkpoint(config, 20)

    monkeypatch.setattr(
        "embed_optim.matrix._checkpoint_is_resumable", lambda path, world_size: path.is_dir()
    )
    monkeypatch.setattr(
        "embed_optim.aggregate._deep_checkpoint_problems", lambda *args, **kwargs: []
    )
    monkeypatch.setattr("embed_optim.aggregate._safetensors_digest", lambda path: "same")

    state, events = audit_once([config], tmp_path / "audit.json")

    assert [event["status"] for event in events] == ["passed", "failed"]
    assert state["audited_checkpoints"] == 1
    assert state["runs"]["late/muon-test"]["checkpoints"]["20"]["problems"] == [
        "model payload is unchanged from the previous checkpoint"
    ]


def test_watcher_rejects_invalid_schedule(tmp_path):
    config = _config(tmp_path)
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "checkpoint_schedule.json").write_text(json.dumps({"steps": [10, 20, 30]}))

    with pytest.raises(RuntimeError, match="five increasing steps"):
        audit_once([config], tmp_path / "audit.json")


def test_watcher_invalidates_deleted_checkpoint_and_reaudits_when_restored(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _write_schedule(config)
    checkpoint = _write_checkpoint(config, 10)
    monkeypatch.setattr(
        "embed_optim.matrix._checkpoint_is_resumable", lambda path, world_size: path.is_dir()
    )
    calls = []
    monkeypatch.setattr(
        "embed_optim.aggregate._deep_checkpoint_problems",
        lambda *args, **kwargs: calls.append(args[1]) or [],
    )
    monkeypatch.setattr("embed_optim.aggregate._safetensors_digest", lambda path: "digest")
    state_path = tmp_path / "audit.json"

    audit_once([config], state_path)
    for item in checkpoint.iterdir():
        item.unlink()
    checkpoint.rmdir()
    state, events = audit_once([config], state_path)
    assert events[0]["status"] == "missing"
    assert state["audited_checkpoints"] == 0

    _write_checkpoint(config, 10)
    state, events = audit_once([config], state_path)
    assert events[0]["status"] == "passed"
    assert state["audited_checkpoints"] == 1
    assert calls == [10, 10]


def test_watcher_reaudits_when_world_size_changes(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _write_schedule(config)
    _write_checkpoint(config, 10)
    monkeypatch.setattr(
        "embed_optim.matrix._checkpoint_is_resumable", lambda path, world_size: path.is_dir()
    )
    calls = []
    monkeypatch.setattr(
        "embed_optim.aggregate._deep_checkpoint_problems",
        lambda *args, **kwargs: calls.append(args[2]) or [],
    )
    monkeypatch.setattr("embed_optim.aggregate._safetensors_digest", lambda path: "digest")
    state_path = tmp_path / "audit.json"

    audit_once([config], state_path, world_size=4)
    audit_once([config], state_path, world_size=2)

    assert calls == [4, 2]


def test_watcher_validates_declared_formal_runtime_before_audit(tmp_path, monkeypatch):
    spec = tmp_path / "formal_runtime.json"
    observed = []
    runtime = {
        "python_executable": "/formal/python",
        "packages": {"torch": "2.9.1+cu129"},
        "torch_cuda": "12.9",
    }
    monkeypatch.setattr("embed_optim.checkpoint_watch.matrix_runtime_spec", lambda matrix: spec)
    monkeypatch.setattr(
        "embed_optim.runtime.verify_runtime_spec",
        lambda path: observed.append(path) or runtime,
    )

    assert _validate_formal_runtime("experiment.yaml") == runtime
    assert observed == [spec]

    monkeypatch.setattr("embed_optim.checkpoint_watch.matrix_runtime_spec", lambda matrix: None)
    assert _validate_formal_runtime("portable.yaml") is None
