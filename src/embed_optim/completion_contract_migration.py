"""Audit and migrate the corrected completion ledger across a paper-only amendment.

The completion controller intentionally fails closed when any bound source changes.  This utility
provides the narrower recovery path needed after a publication-only amendment: it proves that the
matrix, execution protocol, controller implementation, arguments, and command order are unchanged;
archives the original ledger byte-for-byte; and records the exact old and new contract hashes.
"""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .corrected_completion_pipeline import (
    _atomic_json,
    _contract,
    _exclusive_lease,
    _file_identity,
    pipeline_steps,
)

DEFAULT_PROTOCOL = Path("configs/dense_no_packing_completion_contract_migration.json")
DEFAULT_LEDGER = Path("logs/dense-no-packing-finalization/pipeline-ledger.json")


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _contract_sources(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = contract.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Completion contract has no source list")
    by_path: dict[str, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            raise ValueError("Completion contract contains an invalid source identity")
        path = source["path"]
        if path in by_path:
            raise ValueError(f"Completion contract repeats source path: {path}")
        by_path[path] = source
    return by_path


def _nested_field(payload: dict[str, Any], field: str) -> Any:
    value: Any = payload
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Missing amendment assertion field: {field}")
        value = value[part]
    return value


def _arguments_from_contract(contract: dict[str, Any], repository: Path) -> Namespace:
    arguments = contract.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("Completion contract has no arguments")
    required = {
        "matrix",
        "python",
        "gpus",
        "training_log_dir",
        "checkpoint_repo",
        "checkpoint_prefix",
    }
    if set(arguments) != required:
        raise ValueError("Completion contract arguments differ from the expected exact interface")
    return Namespace(
        workdir=repository,
        matrix=Path(arguments["matrix"]).resolve(),
        python=str(arguments["python"]),
        gpus=str(arguments["gpus"]),
        training_log_dir=Path(arguments["training_log_dir"]).resolve(),
        checkpoint_repo=str(arguments["checkpoint_repo"]),
        checkpoint_prefix=str(arguments["checkpoint_prefix"]),
    )


def current_contract(old_contract: dict[str, Any], repository: Path) -> dict[str, Any]:
    args = _arguments_from_contract(old_contract, repository)
    return _contract(args, repository, pipeline_steps(args, repository))


def validate_transition(
    old_contract: dict[str, Any],
    new_contract: dict[str, Any],
    protocol: dict[str, Any],
    repository: Path,
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
    if set(old_sources) != set(new_sources):
        raise ValueError("Completion source paths changed across the proposed migration")

    unchanged = set(protocol.get("required_unchanged_sources") or [])
    allowed = set(protocol.get("allowed_changed_sources") or [])
    if not unchanged or unchanged & allowed or unchanged | allowed != set(old_sources):
        raise ValueError("Migration source partition is incomplete or overlapping")
    observed_changed = {path for path in old_sources if old_sources[path] != new_sources[path]}
    if observed_changed != allowed:
        raise ValueError("Observed contract drift differs from the paper-only allowlist")
    if any(old_sources[path] != new_sources[path] for path in unchanged):
        raise ValueError("A required-unchanged completion source drifted")

    assertions = protocol.get("amendment_assertions")
    if not isinstance(assertions, list) or {item.get("path") for item in assertions} != allowed:
        raise ValueError("Migration amendment assertions do not cover every changed source")
    for assertion in assertions:
        path = assertion.get("path")
        field = assertion.get("field")
        if not isinstance(path, str) or not isinstance(field, str):
            raise ValueError("Invalid migration amendment assertion")
        observed = _nested_field(_object(repository / path), field)
        if observed != assertion.get("expected"):
            raise ValueError(f"Migration amendment assertion failed: {path}:{field}")


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
        if old_contract.get("sha256") == protocol.get("to_contract_sha256"):
            migrations = ledger.get("contract_migrations") or []
            if (
                not migrations
                or migrations[-1].get("protocol", {}).get("sha256")
                != _file_identity(protocol_path, repository)["sha256"]
            ):
                raise ValueError(
                    "Ledger has the target contract without the required migration receipt"
                )
            return {
                "complete": True,
                "status": "already_migrated",
                "from_contract_sha256": protocol["from_contract_sha256"],
                "to_contract_sha256": protocol["to_contract_sha256"],
            }

        new_contract = current_contract(old_contract, repository)
        validate_transition(old_contract, new_contract, protocol, repository)

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
            "protocol": _file_identity(protocol_path, repository),
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
