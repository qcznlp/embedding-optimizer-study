"""Exercise the real exact transition on small temporary source/ledger copies."""

import fcntl
import json
import shutil
from pathlib import Path

import pytest

from scripts import migrate_main_handoff_recovery as migration

ROOT = Path(__file__).resolve().parents[1]
SIX_FILE = ROOT / "reports/engineering-archive/main-handoff-v1"
RESUME = ROOT / "reports/engineering-archive/main-resume-v1"
PROTOCOL = RESUME / "migration-protocol.json"


@pytest.fixture
def candidate(tmp_path):
    root = tmp_path / "candidate"
    protocol = json.loads(PROTOCOL.read_bytes())
    for label in protocol["after_sources"]:
        source = SIX_FILE / "after" / label
        if not source.exists():
            source = SIX_FILE / "before" / label
        if label == migration.CONTROLLER:
            source = RESUME / "after/corrected_completion_pipeline.py"
        target = root / label
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    script = root / migration.IMPLEMENTATION
    script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / migration.IMPLEMENTATION, script)
    protocol_path = root / "configs/main-handoff-recovery-transition.json"
    shutil.copyfile(PROTOCOL, protocol_path)
    ledger = root / migration.LEDGER
    ledger.parent.mkdir(parents=True)
    shutil.copyfile(SIX_FILE / "before/main-ledger.json", ledger)
    return root, protocol_path, migration.digest(PROTOCOL.read_bytes())


def inventory(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def run(candidate, *, apply=False):
    return migration.transition(*candidate, apply=apply)


def test_read_only_preflight_reconstructs_real_17_step_target_without_writes(candidate):
    root, path, sha = candidate
    before = inventory(root)
    report = run(candidate)
    assert report["status"] == "ready"
    assert report["applied"] is False
    assert report["backup_records_preserved"] == 8
    assert report["commands_and_arguments_unchanged"] is True
    assert inventory(root) == before
    old = json.loads((root / migration.LEDGER).read_bytes())["contract"]
    protocol = migration.load_protocol(path, sha, root)
    actual = migration.observed_contract(root, old, protocol)
    assert actual == json.loads((RESUME / "projected-main-contract.json").read_bytes())
    assert len(actual["steps"]) == 17
    assert actual["steps"] == old["steps"]
    assert actual["arguments"] == old["arguments"]


def test_applied_transition_preserves_full_original_and_every_prior_record(candidate):
    root, _, _ = candidate
    ledger = root / migration.LEDGER
    raw = ledger.read_bytes()
    original = json.loads(raw)
    before = inventory(root)
    result = run(candidate, apply=True)
    assert result["status"] == "migrated"
    assert result["applied"] is True
    assert (ledger.parent / migration.ARCHIVE).read_bytes() == raw
    after = json.loads(ledger.read_bytes())
    excluded = {"contract", "contract_migrations", "observed_at_utc"}
    assert {k: v for k, v in after.items() if k not in excluded} == {
        k: v for k, v in original.items() if k not in excluded
    }
    assert after["contract_migrations"][:-1] == original["contract_migrations"]
    assert after["contract"]["sha256"] == migration.NEW_CONTRACT
    for label, payload in before.items():
        if label != migration.LEDGER:
            assert (root / label).read_bytes() == payload
    all_after = inventory(root)
    assert run(candidate, apply=True)["status"] == "already_migrated"
    assert inventory(root) == all_after


@pytest.mark.parametrize("after_write", [False, True])
def test_interrupted_archive_or_commit_recovers_without_discarding_evidence(
    candidate, monkeypatch, after_write
):
    root, _, _ = candidate
    ledger = root / migration.LEDGER
    original = ledger.read_bytes()
    real = migration.atomic_ledger

    def interrupted(path, value):
        if after_write:
            real(path, value)
        raise OSError("simulated interruption")

    monkeypatch.setattr(migration, "atomic_ledger", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        run(candidate, apply=True)
    assert (ledger.parent / migration.ARCHIVE).read_bytes() == original
    if not after_write:
        assert ledger.read_bytes() == original
    monkeypatch.setattr(migration, "atomic_ledger", real)
    expected = "already_migrated" if after_write else "migrated"
    assert run(candidate, apply=True)["status"] == expected
    assert (ledger.parent / migration.ARCHIVE).read_bytes() == original


def test_held_existing_lease_prevents_archive_and_ledger_mutation(candidate):
    root, _, _ = candidate
    lease = (root / migration.LEDGER).parent / "controller.lease"
    lease.write_text("existing controller evidence\n")
    with lease.open("r+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = inventory(root)
        with pytest.raises(RuntimeError, match="still holds its lease"):
            run(candidate, apply=True)
        assert inventory(root) == before


@pytest.mark.parametrize(
    "label", sorted(migration.CHANGED | {"configs/dense_no_packing_retrain.yaml"})
)
def test_source_drift_is_rejected_before_any_write(candidate, label):
    root, _, _ = candidate
    source = root / label
    source.write_bytes(source.read_bytes() + b"\n")
    before = inventory(root)
    with pytest.raises(ValueError, match="Candidate source drift"):
        run(candidate, apply=True)
    assert inventory(root) == before


@pytest.mark.parametrize("field", ["steps", "status", "complete", "contract", "active_step"])
def test_started_or_modified_ledger_is_rejected_before_any_write(candidate, field):
    root, _, _ = candidate
    path = root / migration.LEDGER
    ledger = json.loads(path.read_bytes())
    if field == "steps":
        ledger[field] = [{"name": "training-progress-receipt", "complete": True}]
    elif field == "status":
        ledger[field] = "finalizing"
    elif field == "complete":
        ledger[field] = True
    elif field == "contract":
        ledger[field]["arguments"]["gpus"] = "0,1,2,3"
    else:
        ledger[field] = "decontaminated-beir"
    path.write_text(json.dumps(ledger))
    before = inventory(root)
    with pytest.raises(ValueError):
        run(candidate, apply=True)
    assert inventory(root) == before


def test_unknown_protocol_or_changed_executor_cannot_authorize_a_transition(candidate):
    root, path, sha = candidate
    with pytest.raises(ValueError, match="trusted SHA-256"):
        migration.transition(root, path, "0" * 64, apply=True)
    script = root / migration.IMPLEMENTATION
    script.write_bytes(script.read_bytes() + b"\n")
    before = inventory(root)
    with pytest.raises(ValueError, match="migration implementation differs"):
        migration.transition(root, path, sha, apply=True)
    assert inventory(root) == before


def test_symlinked_archive_is_rejected_without_touching_its_target(candidate, tmp_path):
    root, _, _ = candidate
    target = tmp_path / "unrelated"
    target.write_text("retained unrelated data")
    archive = (root / migration.LEDGER).parent / migration.ARCHIVE
    archive.symlink_to(target)
    with pytest.raises(ValueError, match="Symlinked or escaping"):
        run(candidate, apply=True)
    assert target.read_text() == "retained unrelated data"


@pytest.mark.parametrize("mutation", ["source", "archive", "backup", "contract", "prior_history"])
def test_already_migrated_path_revalidates_sources_archive_and_preservation(candidate, mutation):
    root, _, _ = candidate
    run(candidate, apply=True)
    ledger = root / migration.LEDGER
    value = json.loads(ledger.read_bytes())
    if mutation == "source":
        path = root / migration.CONTROLLER
        path.write_bytes(path.read_bytes() + b"\n")
    elif mutation == "archive":
        path = ledger.parent / migration.ARCHIVE
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        if mutation == "backup":
            value["backups"] = {}
        elif mutation == "contract":
            value["contract"]["arguments"]["gpus"] = "0"
        else:
            value["contract_migrations"] = value["contract_migrations"][-1:]
        ledger.write_text(json.dumps(value))
    before = inventory(root)
    with pytest.raises(ValueError):
        run(candidate, apply=True)
    assert inventory(root) == before
