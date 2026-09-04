"""Migrate the corrected finalizer to a source-bound backup implementation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .completion_contract_migration import (
    _contract_sources,
    _object,
    _timestamp,
    current_contract,
)
from .corrected_completion_pipeline import _atomic_json, _exclusive_lease, _file_identity

DEFAULT_PROTOCOL = Path("configs/dense_no_packing_backup_provenance_migration.json")
DEFAULT_LEDGER = Path("logs/dense-no-packing-finalization/pipeline-ledger.json")


def validate_transition(
    old_contract: dict[str, Any],
    new_contract: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    if (
        protocol.get("schema_version") != 1
        or protocol.get("scientific_contract_changed") is not False
    ):
        raise ValueError("Migration protocol must explicitly preserve the scientific contract")
    if old_contract.get("sha256") != protocol.get("from_contract_sha256"):
        raise ValueError("Existing completion contract is not the frozen migration source")
    if new_contract.get("sha256") != protocol.get("to_contract_sha256"):
        raise ValueError("Current completion contract is not the frozen migration target")
    if old_contract.get("arguments") != new_contract.get("arguments"):
        raise ValueError("Completion arguments changed across the proposed migration")
    if old_contract.get("steps") != new_contract.get("steps"):
        raise ValueError("Completion command order changed across the proposed migration")

    old_sources = _contract_sources(old_contract)
    new_sources = _contract_sources(new_contract)
    old_paths = set(old_sources)
    new_paths = set(new_sources)
    unchanged = set(protocol.get("required_unchanged_sources") or [])
    changed = set(protocol.get("allowed_changed_sources") or [])
    added = set(protocol.get("allowed_added_sources") or [])
    if old_paths - new_paths:
        raise ValueError("The migration must not remove a completion contract source")
    if new_paths - old_paths != added:
        raise ValueError("Added completion sources differ from the exact migration allowlist")
    if unchanged & changed or unchanged | changed != old_paths:
        raise ValueError("Migration source partition is incomplete or overlapping")
    observed_changed = {
        path for path in old_paths & new_paths if old_sources[path] != new_sources[path]
    }
    if observed_changed != changed:
        raise ValueError("Observed contract drift differs from the hardening allowlist")
    if any(old_sources[path] != new_sources[path] for path in unchanged):
        raise ValueError("A required-unchanged completion source drifted")

    identities = protocol.get("target_source_identities")
    if not isinstance(identities, list):
        raise ValueError("Migration protocol has no target source identities")
    declared = {
        identity.get("path"): identity for identity in identities if isinstance(identity, dict)
    }
    if set(declared) != changed | added:
        raise ValueError("Target identities do not cover the changed and added sources")
    for path, identity in declared.items():
        if new_sources.get(path) != identity:
            raise ValueError(f"Target source identity differs from the frozen migration: {path}")


def migrate_ledger(
    ledger_path: Path,
    protocol_path: Path,
    repository: Path,
) -> dict[str, Any]:
    repository = repository.resolve()
    ledger_path = ledger_path.resolve()
    protocol_path = protocol_path.resolve()
    protocol = _object(protocol_path)
    if not ledger_path.is_file():
        raise FileNotFoundError(f"Completion ledger does not exist: {ledger_path}")

    with _exclusive_lease(ledger_path.parent / "controller.lease"):
        original_bytes = ledger_path.read_bytes()
        ledger = json.loads(original_bytes)
        if not isinstance(ledger, dict) or not isinstance(ledger.get("contract"), dict):
            raise ValueError("Completion ledger has no valid contract")
        if ledger.get("complete") is True:
            raise ValueError("A complete completion ledger must not be migrated")

        old_contract = ledger["contract"]
        implementation = _file_identity(Path(__file__).resolve(), repository)
        protocol_identity = _file_identity(protocol_path, repository)
        if old_contract.get("sha256") == protocol.get("to_contract_sha256"):
            migrations = ledger.get("contract_migrations") or []
            latest = migrations[-1] if migrations else {}
            if (
                latest.get("protocol", {}).get("sha256") != protocol_identity["sha256"]
                or latest.get("implementation", {}).get("sha256") != implementation["sha256"]
            ):
                raise ValueError("Ledger has the target contract without the required receipt")
            return {
                "complete": True,
                "status": "already_migrated",
                "from_contract_sha256": protocol["from_contract_sha256"],
                "to_contract_sha256": protocol["to_contract_sha256"],
            }

        new_contract = current_contract(old_contract, repository)
        validate_transition(old_contract, new_contract, protocol)

        archive_basename = protocol.get("archive_basename")
        if not isinstance(archive_basename, str) or Path(archive_basename).name != archive_basename:
            raise ValueError("Migration archive basename must be a local filename")
        archive_path = ledger_path.parent / archive_basename
        if archive_path.exists():
            if archive_path.read_bytes() != original_bytes:
                raise ValueError("Existing migration archive differs from the source ledger")
        else:
            with archive_path.open("xb") as handle:
                handle.write(original_bytes)
                handle.flush()
                os.fsync(handle.fileno())

        migration = {
            "migrated_at_utc": _timestamp(),
            "reason": protocol["reason"],
            "scientific_contract_changed": False,
            "from_contract_sha256": old_contract["sha256"],
            "to_contract_sha256": new_contract["sha256"],
            "source_ledger_archive": _file_identity(archive_path, repository),
            "implementation": implementation,
            "protocol": protocol_identity,
        }
        ledger["contract"] = new_contract
        ledger.setdefault("contract_migrations", []).append(migration)
        ledger["status"] = "waiting_for_training"
        ledger["observed_at_utc"] = _timestamp()
        ledger.pop("failed_step", None)
        _atomic_json(ledger_path, ledger)
        return {
            "complete": True,
            "status": "migrated",
            "from_contract_sha256": migration["from_contract_sha256"],
            "to_contract_sha256": migration["to_contract_sha256"],
            "source_ledger_archive": migration["source_ledger_archive"],
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = migrate_ledger(args.ledger, args.protocol, args.workdir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
