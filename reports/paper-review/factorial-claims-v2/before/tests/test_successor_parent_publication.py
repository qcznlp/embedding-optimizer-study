"""The prepared successor tracks the repaired main without weaker completion gates."""

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from embed_optim import state_operator_factorial_completion as controller
from scripts.audit_successor_contract import (
    COMPLETION,
    FACTORIAL_PUBLICATION,
    HANDOFF,
    PUBLICATION,
    build_plan,
    project_steps,
    validate_amendment_chain,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / controller.COMPLETION_PROTOCOL
AMENDMENT = ROOT / "configs/dense_no_packing_state_operator_parent_publication_amendment.json"
COMBINED = ROOT / "configs/dense_no_packing_state_operator_handoff_repairs_amendment.json"
RECOVERY = ROOT / "configs/dense_no_packing_state_operator_main_recovery_amendment.json"


def read(path):
    return json.loads(path.read_text())


def identity(path):
    raw = path.read_bytes()
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def complete_parent(protocol):
    gate = protocol["main_completion_gate"]
    return {
        "scope": gate["scope"],
        "status": "complete",
        "complete": True,
        "training_runs_complete": 12,
        "training_runs_expected": 12,
        "complete_runs": gate["required_run_ids"],
        "contract": {"sha256": gate["contract_sha256"]},
        "steps": [{"name": name, "complete": True} for name in gate["required_steps"]],
        "backups": {run: {"complete": True} for run in gate["required_run_ids"]},
    }


def test_parent_amendment_changes_only_declared_operational_fields():
    amendment = read(AMENDMENT)
    for binding in [
        amendment["prior_prepared_completion_protocol"],
        *amendment["evidence_bindings"],
    ]:
        assert identity(ROOT / binding["path"]) == binding
    previous = read(ROOT / amendment["prior_prepared_completion_protocol"]["path"])
    assert (
        previous["main_completion_gate"]["contract_sha256"]
        == amendment["main_contract_transition"]["from_contract_sha256"]
    )
    proposed = deepcopy(previous)
    proposed["main_completion_gate"]["contract_sha256"] = amendment["main_contract_transition"][
        "to_contract_sha256"
    ]
    proposed["parent_bindings"]["main_publication_header_dependency"] = identity(AMENDMENT)
    proposed["amended_at_utc"] = amendment["frozen_at_utc"]
    second = read(COMBINED)
    actual = read(ROOT / second["prior_prepared_protocols"][COMPLETION]["path"])
    assert actual == proposed
    assert amendment["runtime_deployed"] is False
    assert amendment["visibility_at_freeze"]["factorial_steps_started"] == 0
    main_fix = read(
        ROOT / "reports/experiment-integrity/live-publication-header-migration-draft.json"
    )
    assert actual["main_completion_gate"]["contract_sha256"] == main_fix["to_contract_sha256"]


def test_combined_amendment_reconstructs_both_layers_without_changing_science():
    protocol = read(ROOT / read(RECOVERY)["prior_prepared_completion_protocol"]["path"])
    first, binding = validate_amendment_chain(ROOT, protocol)
    assert first == read(AMENDMENT)
    assert binding == identity(COMBINED)
    second = read(COMBINED)
    prior = read(ROOT / second["prior_prepared_protocols"][COMPLETION]["path"])
    expected = deepcopy(prior)
    expected["main_completion_gate"]["contract_sha256"] = second["main_contract_transition"][
        "required_main_contract_sha256"
    ]
    expected["parent_bindings"]["primary_dimension_handoff"] = identity(ROOT / HANDOFF)
    expected["parent_bindings"]["publication_protocol"] = identity(ROOT / FACTORIAL_PUBLICATION)
    expected["parent_bindings"]["main_handoff_repairs"] = identity(COMBINED)
    expected["amended_at_utc"] = second["frozen_at_utc"]
    assert expected == protocol
    assert second["runtime_deployed"] is False
    assert second["visibility_at_freeze"]["factorial_steps_started"] == 0
    assert (
        read(ROOT / PUBLICATION)["source_bindings"]
        == read(ROOT / second["prior_prepared_protocols"][PUBLICATION]["path"])["source_bindings"]
    )


def test_recovery_amendment_adds_only_the_exact_main_target_binding_and_timestamp():
    protocol = controller.load_completion_protocol(PROTOCOL, ROOT)
    amendment = read(RECOVERY)
    previous = read(ROOT / amendment["prior_prepared_completion_protocol"]["path"])
    expected = deepcopy(previous)
    expected["main_completion_gate"]["contract_sha256"] = amendment["main_contract_transition"][
        "required_main_contract_sha256"
    ]
    expected["parent_bindings"]["main_resume_recovery"] = identity(RECOVERY)
    expected["amended_at_utc"] = amendment["frozen_at_utc"]
    assert protocol == expected
    _, binding = validate_amendment_chain(ROOT, protocol)
    assert binding == identity(RECOVERY)
    assert amendment["scientific_contract_changed"] is False
    assert amendment["numerical_implementation_changed"] is False
    assert amendment["runtime_deployed"] is False


def test_prior_combined_sources_are_archived_at_their_original_receipt_digests():
    receipt = read(ROOT / "reports/experiment-integrity/combined-successor-validation.json")
    labels = {
        "scripts/audit_successor_contract.py",
        "tests/test_successor_parent_publication.py",
        COMPLETION,
    }
    for binding in receipt["bindings"]:
        if binding["path"] in labels:
            archived = (
                ROOT / "reports/engineering-archive/successor-resume-v1/before" / binding["path"]
            )
            raw = archived.read_bytes()
            assert len(raw) == binding["bytes"]
            assert hashlib.sha256(raw).hexdigest() == binding["sha256"]


@pytest.fixture
def amendment_copy(tmp_path):
    # Only small declared evidence, never checkpoints or a live control plane.
    first, second, third = read(AMENDMENT), read(COMBINED), read(RECOVERY)
    paths = {
        str(AMENDMENT.relative_to(ROOT)),
        str(COMBINED.relative_to(ROOT)),
        str(RECOVERY.relative_to(ROOT)),
    }
    for item in [
        first["prior_prepared_completion_protocol"],
        *first["evidence_bindings"],
        *second["prior_prepared_protocols"].values(),
        *second["refreshed_protocol_bindings"].values(),
        *second["evidence_bindings"],
        third["prior_prepared_completion_protocol"],
        *third["evidence_bindings"],
    ]:
        paths.add(item["path"])
    for label in (
        "configs/dense_no_packing_outcome_protocol.json",
        "configs/dense_no_packing_bridge_implementation_protocol_v2.json",
        "configs/dense_no_packing_sensitivity_implementation_protocol.json",
    ):
        for side in ("before", "after"):
            paths.add(f"reports/engineering-archive/main-handoff-v1/{side}/{label}")
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    protocol = read(PROTOCOL)
    validate_amendment_chain(tmp_path, protocol)
    return tmp_path, protocol


@pytest.mark.parametrize("mutation", ["weaker_gate", "source_state", "extra_parent"])
def test_combined_rejects_undeclared_completion_changes(amendment_copy, mutation):
    root, protocol = amendment_copy
    if mutation == "weaker_gate":
        protocol["main_completion_gate"]["training_runs_complete"] = 11
    elif mutation == "source_state":
        protocol["source_checkpoint_gate"]["checkpoint_step"] = 3907
    else:
        protocol["parent_bindings"]["undeclared"] = {"path": "unreviewed.json"}
    with pytest.raises(ValueError, match="Undeclared completion"):
        validate_amendment_chain(root, protocol)


@pytest.mark.parametrize("label", [PUBLICATION, HANDOFF, FACTORIAL_PUBLICATION])
def test_combined_rejects_rehashed_scientific_or_command_drift(amendment_copy, label):
    root, protocol = amendment_copy
    changed = read(root / label)
    if label == PUBLICATION:
        changed["rendering_contract"]["primary"] = "Select the most favorable rate."
    elif label == FACTORIAL_PUBLICATION:
        changed["rendering_contract"]["abstract"] = "Report only the favorable estimand."
    else:
        changed["steps"][1]["command"][-1] = "7"
    (root / label).write_text(json.dumps(changed))
    second_path = root / COMBINED.relative_to(ROOT)
    second = read(second_path)
    second["refreshed_protocol_bindings"][label] = controller._identity(root / label, root)
    second_path.write_text(json.dumps(second))
    protocol["parent_bindings"]["main_handoff_repairs"] = controller._identity(second_path, root)
    with pytest.raises(
        ValueError,
        match="Undeclared (narrative publication|dimension handoff|factorial publication)",
    ):
        validate_amendment_chain(root, protocol)


def test_combined_rejects_changed_archived_evidence(amendment_copy):
    root, protocol = amendment_copy
    evidence = read(COMBINED)["evidence_bindings"][0]["path"]
    (root / evidence).write_bytes((root / evidence).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Combined amendment evidence"):
        validate_amendment_chain(root, protocol)


@pytest.mark.parametrize("mutation", ["evidence", "target", "scientific_flag", "prior_protocol"])
def test_recovery_rejects_modified_or_rehashed_dependency_changes(amendment_copy, mutation):
    root, protocol = amendment_copy
    path = root / RECOVERY.relative_to(ROOT)
    amendment = read(path)
    if mutation == "evidence":
        label = amendment["evidence_bindings"][0]["path"]
        source = root / label
        source.write_bytes(source.read_bytes() + b"\n")
    elif mutation == "target":
        amendment["main_contract_transition"]["required_main_contract_sha256"] = "0" * 64
    elif mutation == "scientific_flag":
        amendment["scientific_contract_changed"] = True
    else:
        source = root / amendment["prior_prepared_completion_protocol"]["path"]
        previous = read(source)
        previous["main_completion_gate"]["training_runs_complete"] = 11
        source.write_text(json.dumps(previous))
        amendment["prior_prepared_completion_protocol"] = controller._identity(source, root)
    if mutation != "evidence":
        path.write_text(json.dumps(amendment))
        protocol["parent_bindings"]["main_resume_recovery"] = controller._identity(path, root)
    with pytest.raises(
        ValueError, match="(Main recovery|Unexpected main recovery|Undeclared completion)"
    ):
        validate_amendment_chain(root, protocol)


def test_repaired_parent_preserves_all_47_commands_and_arguments_exactly():
    old = read(ROOT / "reports/experiment-integrity/dimension-figure-factorial-plan.json")[
        "result"
    ]["contract"]
    args = controller.parse_args(
        [
            "--workdir",
            str(ROOT),
            "--main-ledger",
            old["arguments"]["main_ledger"],
            "--python",
            old["arguments"]["python"],
            "--dry-run",
        ]
    )
    protocol = controller.load_completion_protocol(PROTOCOL, ROOT)
    steps = controller.pipeline_steps(args, ROOT, protocol)
    actual = controller._contract(args, ROOT, PROTOCOL, steps)
    assert len(actual["steps"]) == 47
    assert actual["steps"] == old["steps"]
    assert actual["arguments"] == old["arguments"]
    assert actual["sha256"] != old["sha256"]
    assert actual["protocol"] == identity(PROTOCOL)


def test_live_handoff_preserves_original_36_commands_and_explicit_interpreter():
    old = read(
        ROOT
        / "reports/engineering-archive/publication-header/factorial-live-ledger-before-parent-fix.json"
    )
    amendment = read(AMENDMENT)
    assert (
        old["contract"]["sha256"]
        == amendment["runtime_transition"]["from_live_factorial_contract_sha256"]
    )
    assert old["steps"] == [] and old["status"] == "waiting_for_main_completion"
    expected = amendment["runtime_transition"]["required_arguments"]
    assert expected == old["contract"]["arguments"]
    assert expected["python"] == "/usr/bin/python3"
    args = controller.parse_args(
        [
            "--workdir",
            str(ROOT),
            "--main-ledger",
            expected["main_ledger"],
            "--python",
            expected["python"],
            "--dry-run",
        ]
    )
    protocol = controller.load_completion_protocol(PROTOCOL, ROOT)
    new = controller._contract(
        args, ROOT, PROTOCOL, controller.pipeline_steps(args, ROOT, protocol)
    )
    projected_steps = project_steps(
        controller.pipeline_steps(args, ROOT, protocol),
        ROOT,
        Path(expected["main_ledger"]).parents[2],
    )
    new = controller._contract(args, ROOT, PROTOCOL, projected_steps)
    retained = [step for step in new["steps"] if not step["name"].startswith("primary-")]
    assert len(retained) == 36
    assert [{k: v for k, v in step.items() if k != "index"} for step in retained] == [
        {k: v for k, v in step.items() if k != "index"} for step in old["contract"]["steps"]
    ]
    assert new["arguments"] == old["contract"]["arguments"]


def test_real_projection_has_distinct_local_and_deployment_contracts():
    result = build_plan(ROOT)
    assert result["original_36_commands_and_arguments_preserved"] is True
    assert len(result["exact_path_relocations"]) == 4
    assert result["local_contract"]["sha256"] != result["projected_live_contract"]["sha256"]
    assert result["controller_started"] is result["runtime_deployed"] is False
    assert result["both_prepared_amendment_layers_verified"] is True
    assert result["main_recovery_dependency_verified"] is True
    assert result["actual_consumer_protocol_loaders_pass"] is True


def test_projection_rejects_undeclared_worktree_path_change():
    from dataclasses import replace

    args = controller.parse_args(["--workdir", str(ROOT), "--dry-run"])
    protocol = controller.load_completion_protocol(PROTOCOL, ROOT)
    steps = controller.pipeline_steps(args, ROOT, protocol)
    step = next(s for s in steps if s.name == "paper-release")
    modified = replace(step, command=("make", "-C", "/unrelated/paper", "release"))
    with pytest.raises(ValueError, match="relocation"):
        project_steps(
            [modified if s.name == step.name else s for s in steps], ROOT, Path("/target")
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "old_parent",
        "header_only_parent",
        "six_file_parent",
        "incomplete_run",
        "missing_backup",
        "unfinished_step",
        "wrong_step_order",
    ],
)
def test_new_parent_identity_does_not_admit_incomplete_or_old_main(tmp_path, mutation):
    protocol = controller.load_completion_protocol(PROTOCOL, ROOT)
    ledger = complete_parent(protocol)
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger))
    assert controller._main_complete(path, protocol)
    if mutation == "old_parent":
        ledger["contract"]["sha256"] = read(AMENDMENT)["main_contract_transition"][
            "from_contract_sha256"
        ]
    elif mutation == "header_only_parent":
        ledger["contract"]["sha256"] = read(AMENDMENT)["main_contract_transition"][
            "to_contract_sha256"
        ]
    elif mutation == "six_file_parent":
        ledger["contract"]["sha256"] = read(RECOVERY)["main_contract_transition"][
            "prior_prepared_required_main_contract_sha256"
        ]
    elif mutation == "incomplete_run":
        ledger["training_runs_complete"] = 11
    elif mutation == "missing_backup":
        ledger["backups"].pop(next(iter(ledger["backups"])))
    elif mutation == "unfinished_step":
        ledger["steps"][-1]["complete"] = False
    else:
        ledger["steps"] = list(reversed(ledger["steps"]))
    path.write_text(json.dumps(ledger))
    assert not controller._main_complete(path, protocol)


def test_matching_parent_dry_run_never_takes_a_lease_or_launches_work(
    tmp_path, monkeypatch, capsys
):
    protocol = controller.load_completion_protocol(PROTOCOL, ROOT)
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(complete_parent(protocol)))
    args = controller.parse_args(
        [
            "--workdir",
            str(ROOT),
            "--main-ledger",
            str(ledger),
            "--log-dir",
            str(tmp_path / "logs"),
            "--dry-run",
        ]
    )
    monkeypatch.setattr(controller, "_exclusive_lease", lambda *a: pytest.fail("Unexpected lease"))
    assert (
        controller.run_pipeline(args, run_command=lambda *a, **kw: pytest.fail("Unexpected work"))
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["main_completion_ready"] is True
    assert result["controller_started"] is False
    assert result["scientific_completion"] is False
    assert not (tmp_path / "logs").exists()
