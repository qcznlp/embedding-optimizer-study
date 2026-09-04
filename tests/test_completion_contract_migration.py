import hashlib
import json
from pathlib import Path

import pytest

import embed_optim.completion_contract_migration as migration


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_checked_in_migration_receipt_binds_implementation_and_protocol():
    root = Path.cwd()
    receipt = json.loads(
        (root / "reports/dense-no-packing/completion-contract-migration.json").read_text(
            encoding="utf-8"
        )
    )

    assert receipt["complete"] is True
    assert receipt["migration"]["scientific_contract_changed"] is False
    assert receipt["resume"]["status"] == "waiting_for_training"
    assert receipt["resume"]["failed_step"] is None
    for name in ("implementation", "protocol"):
        identity = receipt["migration"][name]
        path = root / identity["path"]
        assert path.stat().st_size == identity["bytes"]
        assert _sha256(path) == identity["sha256"]


def test_checked_in_publication_layout_migration_receipt_binds_exact_transition():
    root = Path.cwd()
    receipt = json.loads(
        (root / "reports/dense-no-packing/publication-layout-migration.json").read_text(
            encoding="utf-8"
        )
    )

    assert receipt["complete"] is True
    assert receipt["scientific_completion"] is False
    assert receipt["preflight"]["performed_before_corrected_results"] is True
    assert receipt["migration"]["scientific_contract_changed"] is False
    assert receipt["migration"]["from_contract_sha256"] == (
        "25eefbe52b4cc275600dae631860d2aa69a0734c1c143813b4e7e4a9190f3c13"
    )
    assert receipt["migration"]["to_contract_sha256"] == (
        "abb5973ab7247ec427195ab9afa6f60add579e9e33829e3cc08a61f4947d5a67"
    )
    assert receipt["resume"]["controller_lease"] == "held"
    assert receipt["resume"]["status"] == "waiting_for_training"
    assert receipt["resume"]["failed_step"] is None
    assert receipt["resume"]["preflight_rejection"]["ledger_changed"] is False
    assert receipt["layout_verification"] == {
        "main_end_page": 8,
        "corrected_primary_page": 6,
        "corrected_bridge_page": 11,
        "corrected_sensitivity_page": 11,
        "overfull_boxes": 0,
    }
    identities = [
        receipt["migration"]["implementation"],
        receipt["migration"]["protocol"],
        *receipt["migration"]["paper_topology"],
    ]
    for identity in identities:
        path = root / identity["path"]
        assert path.stat().st_size == identity["bytes"]
        assert _sha256(path) == identity["sha256"]
    expected_commits = {
        "padded-adamw-1e-6": "71c58f98367eea5a15464163600a22c2005f7c76",
        "padded-adamw-3e-6": "fd47604c588deae679e17411a6acfc0d455613f9",
    }
    assert {
        row["run_id"]: row["upload_commit_oid"]
        for row in receipt["resume"]["remote_backup_reaudits"]
    } == expected_commits
    for row in receipt["resume"]["remote_backup_reaudits"]:
        backup_receipt = (
            root / "reports/dense-no-packing/checkpoint-backup" / f"{row['run_id']}.json"
        )
        payload = json.loads(backup_receipt.read_text(encoding="utf-8"))
        assert payload["status"] == "complete"
        assert payload["commit_oid"] == row["upload_commit_oid"]
        assert backup_receipt.stat().st_size == row["receipt_bytes"]
        assert _sha256(backup_receipt) == row["receipt_sha256"]
