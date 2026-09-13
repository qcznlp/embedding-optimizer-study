"""Rehearse the exact combined repair on copies; never access a live job handle."""

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from embed_optim import completion_contract_migration as migration
from scripts.prepare_main_handoff_repair import (
    BRIDGE,
    CHANGED_PATHS,
    HELPER,
    OUTCOME,
    PUBLICATION,
    RENDERER,
    SENSITIVITY,
    build_repair,
    digest,
    identity,
    prepare,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reports/engineering-archive/main-handoff-v1"


def read(path):
    return json.loads(path.read_bytes())


@pytest.fixture
def repair():
    receipt = read(ARCHIVE / "preparation.json")
    before = {
        entry["path"]: (ARCHIVE / "before" / entry["path"]).read_bytes()
        for entry in receipt["before"]
    }
    for entry in receipt["before"]:
        assert identity(entry["path"], before[entry["path"]]) == entry
    ledger_raw = (ARCHIVE / "before/main-ledger.json").read_bytes()
    assert identity("before/main-ledger.json", ledger_raw) == receipt["source_ledger"]
    plan = build_repair(before, (ARCHIVE / "after" / HELPER).read_bytes(), ledger_raw)
    return receipt, before, ledger_raw, plan


@pytest.fixture
def candidate_tree(repair, tmp_path):
    _, before, _, plan = repair
    for label, raw in {**before, **plan["changed"]}.items():
        path = tmp_path / label
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    # Materialize only the small immediate protocol dependencies required by the
    # real loaders. This never copies outputs, logs or any full experiment tree.
    for label in (OUTCOME, BRIDGE, SENSITIVITY, PUBLICATION):
        protocol = read(tmp_path / label)
        for group in ("source_bindings", "parent_bindings", "historical_bindings"):
            for binding in protocol.get(group, {}).values():
                path = tmp_path / binding["path"]
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / binding["path"], path)
                assert digest(path.read_bytes()) == binding["sha256"]
    return tmp_path


def test_prepared_patch_reconstructs_all_six_files_and_exact_contract(repair):
    receipt, _, ledger_raw, plan = repair
    assert tuple(plan["changed"]) == CHANGED_PATHS
    assert plan["patch"] == (ARCHIVE / "main-handoff-repair.patch").read_text()
    assert identity("main-handoff-repair.patch", plan["patch"].encode()) == receipt["patch"]
    assert plan["contract"] == read(ARCHIVE / "projected-main-contract.json")
    assert plan["migration"] == read(ARCHIVE / "migration-draft.json")
    for record in receipt["after"]:
        assert identity(record["path"], plan["changed"][record["path"]]) == record
        assert (ARCHIVE / "after" / record["path"]).read_bytes() == plan["changed"][record["path"]]
    old = json.loads(ledger_raw)["contract"]
    assert plan["contract"]["steps"] == old["steps"]
    assert plan["contract"]["arguments"] == old["arguments"]
    assert receipt["controller_lease_acquired"] is receipt["runtime_deployed"] is False


def test_combined_repair_preserves_every_scientific_field(repair):
    _, before, _, plan = repair
    for label in (OUTCOME, BRIDGE, SENSITIVITY, PUBLICATION):
        original = json.loads(before[label])
        proposed = json.loads(plan["changed"][label])
        amendment = proposed.pop("evaluation_handoff_amendment")
        assert amendment["scientific_contract_changed"] is False
        assert amendment["numerical_implementation_changed"] is False
        if label == OUTCOME:
            assert proposed["source_bindings"].pop("evaluation_source_provenance") == identity(
                HELPER, plan["changed"][HELPER]
            )
        if label == PUBLICATION:
            assert proposed.pop("latex_header_syntax_amendment")["affected_headers"] == 3
            assert proposed["source_bindings"]["corrected_publication"] == identity(
                RENDERER, plan["changed"][RENDERER]
            )
            proposed["source_bindings"]["corrected_publication"] = original["source_bindings"][
                "corrected_publication"
            ]
        for key, binding in proposed["parent_bindings"].items():
            old_binding = original["parent_bindings"][key]
            assert {k: v for k, v in binding.items() if k not in {"sha256", "bytes"}} == {
                k: v for k, v in old_binding.items() if k not in {"sha256", "bytes"}
            }
            if binding["path"] in plan["changed"]:
                assert binding["sha256"] == digest(plan["changed"][binding["path"]])
                proposed["parent_bindings"][key] = old_binding
        assert proposed == original


def test_actual_protocol_loaders_accept_combined_candidate_and_reject_helper_drift(candidate_tree):
    from embed_optim.corrected_execution_sensitivity import _load_protocol
    from embed_optim.corrected_outcome_summary import _load_outcome_protocol
    from embed_optim.corrected_publication import _load_publication_protocol
    from embed_optim.corrected_retrieval_bridge import _load_implementation_protocol

    for label, loader in (
        (OUTCOME, _load_outcome_protocol),
        (BRIDGE, _load_implementation_protocol),
        (SENSITIVITY, _load_protocol),
        (PUBLICATION, _load_publication_protocol),
    ):
        assert loader(candidate_tree / label, candidate_tree)
    helper = candidate_tree / HELPER
    helper.write_bytes(helper.read_bytes() + b"# unreviewed drift\n")
    with pytest.raises(ValueError, match="source_bindings mismatch"):
        _load_outcome_protocol(candidate_tree / OUTCOME, candidate_tree)


def test_real_migration_on_copy_preserves_every_old_backup_and_prior_record(
    candidate_tree, repair, monkeypatch
):
    _, _, raw, plan = repair
    before = json.loads(raw)
    path = candidate_tree / "logs/pipeline-ledger.json"
    path.parent.mkdir()
    path.write_bytes(raw)
    protocol = candidate_tree / "migration.json"
    protocol.write_text(json.dumps(plan["migration"]))
    monkeypatch.setattr(migration, "current_contract", lambda *_: plan["contract"])
    result = migration.migrate_ledger(path, protocol, candidate_tree)
    assert result["status"] == "migrated"
    assert (path.parent / plan["migration"]["archive_basename"]).read_bytes() == raw
    after = read(path)
    for key in ("backups", "complete_runs", "steps", "complete", "started_at_utc"):
        assert after[key] == before[key]
    assert len(after["backups"]) == 8
    assert after["contract_migrations"][:-1] == before["contract_migrations"]
    assert migration.migrate_ledger(path, protocol, candidate_tree)["status"] == "already_migrated"


@pytest.mark.parametrize("mutation", ["source", "helper", "status", "steps"])
def test_preparation_rejects_a_different_live_predecessor(repair, mutation):
    _, before, raw, _ = repair
    before = dict(before)
    ledger = json.loads(raw)
    helper = (ARCHIVE / "after" / HELPER).read_bytes()
    if mutation == "source":
        before[OUTCOME] += b"\n"
    elif mutation == "helper":
        helper += b"\n"
    elif mutation == "status":
        ledger["status"] = "finalizing"
    else:
        ledger["steps"] = [{"name": "already-started"}]
    with pytest.raises(ValueError):
        build_repair(before, helper, json.dumps(ledger).encode())


def test_existing_prepared_successor_must_rebind_without_weakening_main_gate(repair, tmp_path):
    from embed_optim.state_operator_factorial_completion import _main_complete

    _, _, _, plan = repair
    # This proof freezes the predecessor at repair preparation; the active
    # isolated successor is subsequently rebound under a separate amendment.
    current = read(
        ROOT
        / "reports/engineering-archive/successor-handoff-v1/before/configs"
        / "dense_no_packing_state_operator_factorial_completion_protocol.json"
    )
    candidate = deepcopy(current)
    candidate["main_completion_gate"]["contract_sha256"] = plan["contract"]["sha256"]
    gate = candidate["main_completion_gate"]
    ledger = {
        "scope": gate["scope"],
        "status": "complete",
        "complete": True,
        "training_runs_complete": 12,
        "training_runs_expected": 12,
        "contract": {"sha256": gate["contract_sha256"]},
        "steps": [{"name": name, "complete": True} for name in gate["required_steps"]],
        "backups": {run: {"complete": True} for run in gate["required_run_ids"]},
    }
    path = tmp_path / "synthetic-complete-main.json"
    path.write_text(json.dumps(ledger))
    assert _main_complete(path, current) is False
    assert _main_complete(path, candidate) is True
    ledger["steps"][-1]["complete"] = False
    path.write_text(json.dumps(ledger))
    assert _main_complete(path, candidate) is False


def test_preparation_cannot_write_inside_or_above_live_root(tmp_path):
    live = tmp_path / "experiment"
    with pytest.raises(ValueError, match="outside the live"):
        prepare(live, tmp_path / "helper", live / "reports")
    with pytest.raises(ValueError, match="outside the live"):
        prepare(live, tmp_path / "helper", tmp_path)
