import json
from pathlib import Path

import pytest

import embed_optim.completion_backup_contract_migration as migration


def _identity(path: str, digest: str) -> dict[str, object]:
    return {"path": path, "bytes": 10, "sha256": digest}


def _contracts() -> tuple[dict[str, object], dict[str, object]]:
    arguments = {
        "matrix": "/repo/configs/matrix.yaml",
        "python": "/usr/bin/python3",
        "gpus": "0,1,2,3,4,5,6,7",
        "training_log_dir": "/repo/logs/training",
        "checkpoint_repo": "owner/checkpoints",
        "checkpoint_prefix": "corrected/dense",
    }
    steps = [{"index": 1, "name": "one", "command": ["python", "one"]}]
    old = {
        "schema_version": 1,
        "sha256": "a" * 64,
        "arguments": arguments,
        "steps": steps,
        "sources": [
            _identity("controller.py", "1" * 64),
            _identity("protocol.json", "2" * 64),
        ],
    }
    new = {
        **old,
        "sha256": "b" * 64,
        "sources": [
            _identity("controller.py", "3" * 64),
            _identity("backup.py", "4" * 64),
            _identity("protocol.json", "2" * 64),
        ],
    }
    return old, new


def _protocol(root: Path) -> tuple[Path, dict[str, object]]:
    payload = {
        "schema_version": 1,
        "reason": "bind hardened backup source",
        "scientific_contract_changed": False,
        "from_contract_sha256": "a" * 64,
        "to_contract_sha256": "b" * 64,
        "required_unchanged_sources": ["protocol.json"],
        "allowed_changed_sources": ["controller.py"],
        "allowed_added_sources": ["backup.py"],
        "target_source_identities": [
            _identity("controller.py", "3" * 64),
            _identity("backup.py", "4" * 64),
        ],
        "archive_basename": "pipeline-ledger.pre-backup-hardening.json",
    }
    path = root / "migration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_transition_requires_exact_changed_added_and_unchanged_sources(tmp_path: Path):
    old, new = _contracts()
    _, protocol = _protocol(tmp_path)

    migration.validate_transition(old, new, protocol)

    protocol["allowed_added_sources"] = []
    with pytest.raises(ValueError, match="Added completion sources differ"):
        migration.validate_transition(old, new, protocol)


def test_transition_rejects_command_or_target_identity_drift(tmp_path: Path):
    old, new = _contracts()
    _, protocol = _protocol(tmp_path)
    drifted = {**new, "steps": [{"index": 1, "name": "different", "command": ["x"]}]}
    with pytest.raises(ValueError, match="command order changed"):
        migration.validate_transition(old, drifted, protocol)

    protocol["target_source_identities"][0]["sha256"] = "9" * 64
    with pytest.raises(ValueError, match="Target source identity differs"):
        migration.validate_transition(old, new, protocol)


def test_migration_preserves_prior_history_and_archives_exact_ledger(monkeypatch, tmp_path: Path):
    old, new = _contracts()
    protocol_path, _ = _protocol(tmp_path)
    ledger_path = tmp_path / "logs" / "pipeline-ledger.json"
    ledger_path.parent.mkdir()
    ledger = {
        "schema_version": 1,
        "complete": False,
        "status": "waiting_for_training",
        "contract": old,
        "contract_migrations": [{"from_contract_sha256": "0" * 64}],
        "backups": {"complete-run": {"complete": True}},
    }
    original = json.dumps(ledger, indent=2).encode()
    ledger_path.write_bytes(original)
    monkeypatch.setattr(migration, "current_contract", lambda *_: new)

    result = migration.migrate_ledger(ledger_path, protocol_path, tmp_path)

    assert result["status"] == "migrated"
    assert (tmp_path / "logs/pipeline-ledger.pre-backup-hardening.json").read_bytes() == original
    migrated = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert migrated["contract"] == new
    assert len(migrated["contract_migrations"]) == 2
    assert migrated["backups"] == ledger["backups"]
    assert migration.migrate_ledger(ledger_path, protocol_path, tmp_path)["status"] == (
        "already_migrated"
    )
