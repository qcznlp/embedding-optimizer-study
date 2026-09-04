import json
from pathlib import Path

import pytest

import embed_optim.completion_contract_migration as migration


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
            _identity("controller.py", "1" * 64),
            _identity("protocol.json", "3" * 64),
        ],
    }
    return old, new


def _protocol(root: Path) -> tuple[Path, dict[str, object]]:
    (root / "protocol.json").write_text(
        json.dumps({"paper_only_amendment": {"scientific_changed": False}}),
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "reason": "paper-only publication amendment",
        "scientific_contract_changed": False,
        "from_contract_sha256": "a" * 64,
        "to_contract_sha256": "b" * 64,
        "required_unchanged_sources": ["controller.py"],
        "allowed_changed_sources": ["protocol.json"],
        "amendment_assertions": [
            {
                "path": "protocol.json",
                "field": "paper_only_amendment.scientific_changed",
                "expected": False,
            }
        ],
        "archive_basename": "pipeline-ledger.pre-paper-only.json",
    }
    path = root / "migration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_transition_accepts_only_the_exact_non_scientific_source_partition(tmp_path: Path):
    old, new = _contracts()
    _, protocol = _protocol(tmp_path)

    migration.validate_transition(old, new, protocol, tmp_path)

    drifted = {**new, "steps": [{"index": 1, "name": "different", "command": ["x"]}]}
    with pytest.raises(ValueError, match="command order changed"):
        migration.validate_transition(old, drifted, protocol, tmp_path)

    protocol["scientific_contract_changed"] = True
    with pytest.raises(ValueError, match="preserve the scientific contract"):
        migration.validate_transition(old, new, protocol, tmp_path)


def test_transition_rejects_missing_or_false_amendment_evidence(tmp_path: Path):
    old, new = _contracts()
    _, protocol = _protocol(tmp_path)
    (tmp_path / "protocol.json").write_text(
        json.dumps({"paper_only_amendment": {"scientific_changed": True}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="assertion failed"):
        migration.validate_transition(old, new, protocol, tmp_path)


def test_migration_archives_exact_ledger_and_is_idempotent(monkeypatch, tmp_path: Path):
    old, new = _contracts()
    protocol_path, _ = _protocol(tmp_path)
    ledger_path = tmp_path / "logs" / "pipeline-ledger.json"
    ledger_path.parent.mkdir()
    ledger = {
        "schema_version": 1,
        "complete": False,
        "status": "waiting_for_training",
        "contract": old,
        "backups": {"complete-run": {"complete": True, "attempts": [{"return_code": 0}]}},
        "steps": [],
    }
    original = json.dumps(ledger, indent=2).encode()
    ledger_path.write_bytes(original)
    monkeypatch.setattr(migration, "current_contract", lambda *_: new)

    result = migration.migrate_ledger(ledger_path, protocol_path, tmp_path)

    assert result["status"] == "migrated"
    archive = tmp_path / "logs" / "pipeline-ledger.pre-paper-only.json"
    assert archive.read_bytes() == original
    migrated = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert migrated["contract"] == new
    assert migrated["backups"] == ledger["backups"]
    assert migrated["contract_migrations"][0]["scientific_contract_changed"] is False
    assert migration.migrate_ledger(ledger_path, protocol_path, tmp_path)["status"] == (
        "already_migrated"
    )


def test_complete_ledger_is_never_migrated(monkeypatch, tmp_path: Path):
    old, new = _contracts()
    protocol_path, _ = _protocol(tmp_path)
    ledger_path = tmp_path / "pipeline-ledger.json"
    ledger_path.write_text(
        json.dumps({"complete": True, "contract": old}),
        encoding="utf-8",
    )
    monkeypatch.setattr(migration, "current_contract", lambda *_: new)

    with pytest.raises(ValueError, match="complete completion ledger"):
        migration.migrate_ledger(ledger_path, protocol_path, tmp_path)
