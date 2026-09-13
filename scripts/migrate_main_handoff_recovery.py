"""Exact seven-file, zero-step main transition; read-only unless --apply is given.

This is not a deployment tool. The reviewed candidate files must already be in place,
and applying the ledger transition requires the existing controller lease to be free.
It cannot stop a job, patch source, upload artifacts, or migrate a started pipeline.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

OLD_CONTRACT = "4152531e354bdf325411c7f4d6de044ea68b6d5f26a59fbf5914f8b55f3c9789"
NEW_CONTRACT = "4380a3631ec196307c775cdf4ca766298075de0312ad9d7158c2795577424a56"
CONTROLLER = "src/embed_optim/corrected_completion_pipeline.py"
LEDGER = "logs/dense-no-packing-finalization/pipeline-ledger.json"
ARCHIVE = "pipeline-ledger.pre-main-handoff-recovery-4152531e.json"
IMPLEMENTATION = "scripts/migrate_main_handoff_recovery.py"
CHANGED = {
    CONTROLLER,
    "src/embed_optim/corrected_publication.py",
    "src/embed_optim/evaluation_source_provenance.py",
    "configs/dense_no_packing_outcome_protocol.json",
    "configs/dense_no_packing_bridge_implementation_protocol_v2.json",
    "configs/dense_no_packing_sensitivity_implementation_protocol.json",
    "configs/dense_no_packing_publication_protocol.json",
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def identity(label: str, raw: bytes) -> dict:
    return {"path": label, "bytes": len(raw), "sha256": digest(raw)}


def contract_digest(value: dict) -> str:
    return digest(
        json.dumps(
            {k: v for k, v in value.items() if k != "sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


def local_path(repository: Path, label: str) -> Path:
    relative = Path(label)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != label:
        raise ValueError("Expected an exact repository-relative path")
    path = repository / relative
    if path.resolve() != path or not path.is_relative_to(repository):
        raise ValueError("Symlinked or escaping transition path")
    return path


def load_protocol(path: Path, trusted_sha256: str, repository: Path) -> dict:
    raw = path.read_bytes()
    if len(trusted_sha256) != 64 or digest(raw) != trusted_sha256:
        raise ValueError("Transition protocol does not match the trusted SHA-256")
    protocol = json.loads(raw)
    if (
        protocol.get("schema_version") != 1
        or protocol.get("scope") != "exact_zero_step_main_handoff_recovery_transition"
        or protocol.get("scientific_contract_changed") is not False
        or protocol.get("numerical_implementation_changed") is not False
        or protocol.get("from_contract_sha256") != OLD_CONTRACT
        or protocol.get("to_contract_sha256") != NEW_CONTRACT
        or protocol.get("ledger_path") != LEDGER
        or protocol.get("archive_basename") != ARCHIVE
    ):
        raise ValueError("Not the exact reviewed zero-step transition")
    implementation = protocol["implementation"]
    if implementation != identity(IMPLEMENTATION, Path(__file__).read_bytes()):
        raise ValueError("Migration implementation changed")
    if implementation != identity(
        IMPLEMENTATION, local_path(repository, IMPLEMENTATION).read_bytes()
    ):
        raise ValueError("Candidate migration implementation differs")
    before, after = protocol["before_sources"], protocol["after_sources"]
    if (
        not isinstance(before, dict)
        or set(before) != set(after)
        or len(before) != 12
        or {p for p in before if before[p] != after[p]} != CHANGED
    ):
        raise ValueError("Not the exact seven-file source transition")
    for group in (before, after):
        for label, record in group.items():
            local_path(repository, label)
            if record.get("path") != label or set(record) != {"path", "bytes", "sha256"}:
                raise ValueError("Invalid transition source identity")
    return protocol


def observed_contract(repository: Path, source: dict, protocol: dict) -> dict:
    """Rehash the physical candidate; generate real commands at the frozen logical root.

    The logical root retains every original absolute argument. This also permits a
    temporary-copy rehearsal without reading or writing files at the live root.
    """
    if source.get("sha256") != OLD_CONTRACT or contract_digest(source) != OLD_CONTRACT:
        raise ValueError("Invalid original main contract, including its content digest")
    before, after = protocol["before_sources"], protocol["after_sources"]
    for label, record in after.items():
        if identity(label, local_path(repository, label).read_bytes()) != record:
            raise ValueError(f"Candidate source drift: {label}")
    sources = source["sources"]
    if len(sources) != 10 or len({record["path"] for record in sources}) != 10:
        raise ValueError("Unexpected main contract source topology")
    if any(before.get(record["path"]) != record for record in sources):
        raise ValueError("Original source records differ from the reviewed predecessor")
    module_name = "embed_optim._reviewed_main_handoff_recovery"
    spec = importlib.util.spec_from_file_location(module_name, local_path(repository, CONTROLLER))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        # Execute the verified source without creating a bytecode cache in the
        # candidate checkout during a read-only preflight.
        exec(
            compile(local_path(repository, CONTROLLER).read_bytes(), spec.origin, "exec"),
            module.__dict__,
        )
        arguments = source["arguments"]
        logical_root = Path(arguments["matrix"]).parents[1]
        args = SimpleNamespace(
            **{
                key: Path(value) if key in {"matrix", "training_log_dir"} else value
                for key, value in arguments.items()
            }
        )
        steps = module.pipeline_steps(args, logical_root)
    finally:
        sys.modules.pop(module_name, None)
    generated = [
        {"index": index, "name": step.name, "command": list(step.command)}
        for index, step in enumerate(steps, 1)
    ]
    if len(generated) != 17 or generated != source["steps"]:
        raise ValueError("Candidate changed a main command or argument")
    result = deepcopy(source)
    result["sources"] = [after[record["path"]] for record in sources]
    result["sha256"] = contract_digest(result)
    if result["sha256"] != NEW_CONTRACT:
        raise ValueError("Reconstructed target differs from the exact reviewed target")
    return result


def check_ledger(ledger: dict) -> None:
    if (
        ledger.get("scope") != "corrected_dense_no_packing_completion"
        or ledger.get("complete") is not False
        or ledger.get("status") != "waiting_for_training"
        or ledger.get("steps") != []
        or not isinstance(ledger.get("backups"), dict)
        or "active_step" in ledger
        or "active_run" in ledger
    ):
        raise ValueError("Transition requires an inactive, waiting, zero-step ledger")


@contextmanager
def exclusive_lease(path: Path):
    # The lease content is retained; no PID inspection, signalling or takeover.
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "The existing completion controller still holds its lease"
            ) from error
        yield


def atomic_ledger(path: Path, payload: dict) -> None:
    descriptor, name = tempfile.mkstemp(prefix=".main-recovery-transition-", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(name, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def transition(repository: Path, protocol_path: Path, trusted_sha256: str, *, apply=False) -> dict:
    repository = repository.resolve()
    protocol_path = protocol_path.resolve()
    protocol = load_protocol(protocol_path, trusted_sha256, repository)
    ledger_path = local_path(repository, LEDGER)
    archive_path = local_path(repository, str(Path(LEDGER).parent / ARCHIVE))
    protocol_identity = identity(
        str(protocol_path.relative_to(repository)), protocol_path.read_bytes()
    )

    def inspect() -> tuple[bytes, dict, dict, bool]:
        if load_protocol(protocol_path, trusted_sha256, repository) != protocol:
            raise ValueError("Transition protocol changed during inspection")
        raw = ledger_path.read_bytes()
        ledger = json.loads(raw)
        check_ledger(ledger)
        already = ledger["contract"].get("sha256") == NEW_CONTRACT
        if already:
            original = json.loads(archive_path.read_bytes())
            history = ledger.get("contract_migrations", [])
            if not history or history[-1].get("protocol") != protocol_identity:
                raise ValueError("Target ledger lacks the exact migration receipt")
            receipt = history[-1]
            if receipt.get("source_ledger_archive") != identity(
                str(archive_path.relative_to(repository)), archive_path.read_bytes()
            ):
                raise ValueError("Migration archive identity changed")
            if history[:-1] != original.get("contract_migrations", []):
                raise ValueError("Prior migration history changed")
            excluded = {"contract", "contract_migrations", "observed_at_utc"}
            if {k: v for k, v in ledger.items() if k not in excluded} != {
                k: v for k, v in original.items() if k not in excluded
            }:
                raise ValueError("Migrated ledger did not preserve the original records")
            source = original["contract"]
        else:
            source = ledger["contract"]
        target = observed_contract(repository, source, protocol)
        if already and ledger["contract"] != target:
            raise ValueError("Target contract content changed")
        return raw, ledger, target, already

    raw, ledger, target, already = inspect()
    result = {
        "scope": "engineering_main_handoff_recovery_migration",
        "scientific_completion": False,
        "from_contract_sha256": OLD_CONTRACT,
        "to_contract_sha256": NEW_CONTRACT,
        "backup_records_preserved": len(ledger["backups"]),
        "post_training_steps": len(ledger["steps"]),
        "commands_and_arguments_unchanged": True,
        "controller_started": False,
        "source_files_changed": False,
    }
    if not apply:
        return {**result, "status": "already_migrated" if already else "ready", "applied": False}
    with exclusive_lease(local_path(repository, str(Path(LEDGER).parent / "controller.lease"))):
        raw, ledger, target, already = inspect()
        result["backup_records_preserved"] = len(ledger["backups"])
        if already:
            return {**result, "status": "already_migrated", "applied": False}
        if archive_path.exists():
            if archive_path.read_bytes() != raw:
                raise ValueError("Existing archive differs from the original ledger")
        else:
            with archive_path.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        # Source and ledger drift between preflight and the locked write fails closed.
        if (
            ledger_path.read_bytes() != raw
            or observed_contract(repository, ledger["contract"], protocol) != target
        ):
            raise ValueError("Transition inputs changed before the ledger write")
        moment = datetime.now(UTC).isoformat()
        ledger["contract"] = target
        ledger.setdefault("contract_migrations", []).append(
            {
                "migrated_at_utc": moment,
                "scope": protocol["scope"],
                "scientific_contract_changed": False,
                "numerical_implementation_changed": False,
                "from_contract_sha256": OLD_CONTRACT,
                "to_contract_sha256": NEW_CONTRACT,
                "protocol": protocol_identity,
                "implementation": protocol["implementation"],
                "source_ledger_archive": identity(str(archive_path.relative_to(repository)), raw),
            }
        )
        ledger["observed_at_utc"] = moment
        atomic_ledger(ledger_path, ledger)
        inspect()
        return {**result, "status": "migrated", "applied": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            transition(
                args.workdir, args.protocol, args.expected_protocol_sha256, apply=args.apply
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
