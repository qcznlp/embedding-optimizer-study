import json
from types import SimpleNamespace

import pytest

import embed_optim.corrected_checkpoint_backup as backup
from embed_optim.corrected_checkpoint_backup import (
    _inventory_digest,
    backup_run,
    compare_inventories,
)


def test_corrected_checkpoint_backup_inventory_is_exact():
    local = {"completed.json": 10, "checkpoint-1/model.safetensors": 20}

    audit = compare_inventories(local, dict(local))

    assert audit == {
        "complete": True,
        "local_files": 2,
        "local_bytes": 30,
        "remote_files": 2,
        "remote_bytes": 30,
        "missing": [],
        "extra": [],
        "size_mismatch": [],
    }
    assert _inventory_digest(local) == _inventory_digest(dict(reversed(list(local.items()))))


def test_corrected_checkpoint_backup_rejects_missing_extra_or_wrong_size():
    audit = compare_inventories(
        {"a": 1, "b": 2, "d": 4},
        {"a": 9, "c": 3, "d": 4},
    )

    assert audit["complete"] is False
    assert audit["missing"] == ["b"]
    assert audit["extra"] == ["c"]
    assert audit["size_mismatch"] == ["a"]


def test_audit_only_preserves_the_original_upload_commit(monkeypatch, tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "model.safetensors").write_bytes(b"model")
    config = SimpleNamespace(run_id="padded-adamw-1e-6", output_dir=output)
    receipt_root = tmp_path / "receipts"
    receipt_root.mkdir()
    prefix = "corrected/dense/padded-adamw-1e-6"
    inventory = backup.local_inventory(output)
    prior = {
        "schema_version": 1,
        "status": "complete",
        "run_id": config.run_id,
        "local_root": str(output),
        "repo_id": "owner/checkpoints",
        "repo_type": "model",
        "remote_prefix": prefix,
        "inventory_sha256": _inventory_digest(inventory),
        "commit_url": "https://huggingface.co/owner/checkpoints/commit/abc",
        "commit_oid": "abc",
    }
    (receipt_root / f"{config.run_id}.json").write_text(json.dumps(prior), encoding="utf-8")
    monkeypatch.setattr(backup, "_run_is_complete", lambda _: True)
    monkeypatch.setattr(backup, "remote_inventory", lambda *_: inventory)

    receipt = backup_run(
        SimpleNamespace(),
        config,
        repo_id="owner/checkpoints",
        remote_prefix="corrected/dense",
        receipt_root=receipt_root,
        audit_only=True,
    )

    assert receipt["commit_url"] == prior["commit_url"]
    assert receipt["commit_oid"] == prior["commit_oid"]


def test_audit_only_rejects_commit_provenance_from_a_different_inventory(monkeypatch, tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "model.safetensors").write_bytes(b"model")
    config = SimpleNamespace(run_id="padded-adamw-1e-6", output_dir=output)
    receipt_root = tmp_path / "receipts"
    receipt_root.mkdir()
    prior = {
        "schema_version": 1,
        "status": "complete",
        "run_id": config.run_id,
        "local_root": str(output),
        "repo_id": "owner/checkpoints",
        "repo_type": "model",
        "remote_prefix": "corrected/dense/padded-adamw-1e-6",
        "inventory_sha256": "0" * 64,
        "commit_url": "https://huggingface.co/owner/checkpoints/commit/abc",
        "commit_oid": "abc",
    }
    (receipt_root / f"{config.run_id}.json").write_text(json.dumps(prior), encoding="utf-8")
    inventory = backup.local_inventory(output)
    monkeypatch.setattr(backup, "_run_is_complete", lambda _: True)
    monkeypatch.setattr(backup, "remote_inventory", lambda *_: inventory)

    with pytest.raises(RuntimeError, match="mismatched receipt"):
        backup_run(
            SimpleNamespace(),
            config,
            repo_id="owner/checkpoints",
            remote_prefix="corrected/dense",
            receipt_root=receipt_root,
            audit_only=True,
        )
