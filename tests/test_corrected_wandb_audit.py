from types import SimpleNamespace

from embed_optim.corrected_wandb_audit import audit_run, build_audit


class Config(SimpleNamespace):
    def as_dict(self):
        return {
            "run_id": self.run_id,
            "model_family": self.model_family,
            "optimizer": {"name": self.optimizer.name, "lr": self.optimizer.lr},
            "micro_batch_size": self.micro_batch_size,
            "global_batch_size": self.global_batch_size,
            "epochs": self.epochs,
            "dense_can_flatten_inputs": self.dense_can_flatten_inputs,
            "seed": self.seed,
            "wandb_entity": self.wandb_entity,
            "wandb_project": self.wandb_project,
        }


def _config(run_id="padded-adamw-1e-6"):
    return Config(
        run_id=run_id,
        model_family="dense",
        optimizer=SimpleNamespace(name="adamw", lr=1e-6),
        micro_batch_size=8,
        global_batch_size=128,
        epochs=1.0,
        dense_can_flatten_inputs=False,
        seed=42,
        wandb_entity="owner",
        wandb_project="project",
        checkpoint_fractions=(0.2, 0.4, 0.6, 0.8, 1.0),
    )


def _remote(config, *, state="finished", step=3907, epoch=1.0):
    remote_config = config.as_dict()
    remote_config.update(
        {
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 1.0,
            "per_device_train_batch_size": 8,
            "run_name": f"dense-{config.run_id}",
        }
    )
    return SimpleNamespace(
        id=f"study-v2-dense-{config.run_id}-seed42",
        name=f"dense-{config.run_id}",
        group="dense",
        tags=["adamw", "dense", "seed-42"],
        state=state,
        config=remote_config,
        summary={"train/global_step": step, "train/epoch": epoch},
        url=f"https://wandb.example/{config.run_id}",
    )


def _identity(name):
    return {"path": name, "bytes": 1, "sha256": "a" * 64}


def test_finished_run_requires_exact_id_config_tags_and_terminal_summary():
    config = _config()
    record = audit_run(
        config,
        _remote(config),
        local_complete=True,
        expected_steps=3907,
        world_size=4,
    )

    assert record["status"] == "valid"
    assert not record["problems"]
    assert record["summary_global_step"] == 3907


def test_finished_run_rejects_config_and_terminal_drift():
    config = _config()
    remote = _remote(config, state="running", step=3900, epoch=0.99)
    remote.config["dense_can_flatten_inputs"] = True
    remote.tags = ["dense"]
    record = audit_run(
        config,
        remote,
        local_complete=True,
        expected_steps=3907,
        world_size=4,
    )

    assert record["status"] == "invalid"
    assert "dense_can_flatten_inputs" in record["config_mismatches"]
    assert len(record["problems"]) == 5


def test_partial_audit_accepts_not_started_and_valid_running_runs(monkeypatch):
    first = _config()
    second = _config("padded-adamw-3e-6")
    monkeypatch.setattr("embed_optim.corrected_wandb_audit._run_is_complete", lambda config: False)
    running = _remote(first, state="running", step=100, epoch=0.02)
    result = build_audit(
        [first, second],
        {running.id: running},
        expected_steps=3907,
        world_size=4,
        allow_partial=True,
        matrix_identity=_identity("matrix.yaml"),
        protocol_identity=_identity("protocol.json"),
        source_identity=_identity("audit.py"),
    )

    assert result["status"] == "partial"
    assert result["remote_visible_runs"] == 1
    assert result["valid_remote_runs"] == 1
    assert not result["problems"]


def test_local_completion_without_remote_is_invalid(monkeypatch):
    config = _config()
    monkeypatch.setattr("embed_optim.corrected_wandb_audit._run_is_complete", lambda _: True)
    result = build_audit(
        [config],
        {},
        expected_steps=3907,
        world_size=4,
        allow_partial=True,
        matrix_identity=_identity("matrix.yaml"),
        protocol_identity=_identity("protocol.json"),
        source_identity=_identity("audit.py"),
    )

    assert result["status"] == "invalid"
    assert result["problems"] == ["padded-adamw-1e-6: locally complete run is missing from W&B"]


def test_complete_audit_requires_all_runs_finished(monkeypatch):
    config = _config()
    remote = _remote(config)
    monkeypatch.setattr("embed_optim.corrected_wandb_audit._run_is_complete", lambda _: True)
    result = build_audit(
        [config],
        {remote.id: remote},
        expected_steps=3907,
        world_size=4,
        allow_partial=False,
        matrix_identity=_identity("matrix.yaml"),
        protocol_identity=_identity("protocol.json"),
        source_identity=_identity("audit.py"),
    )

    assert result["status"] == "complete"
    assert result["complete"] is True
