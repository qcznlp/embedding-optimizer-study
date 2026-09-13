"""Project the isolated successor onto the live root without running or migrating it."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from embed_optim import state_operator_factorial_completion as controller
from scripts.prepare_main_handoff_repair import (
    BRIDGE,
    OUTCOME,
    PUBLICATION,
    SENSITIVITY,
    contract_digest,
)

HANDOFF = "configs/dense_primary_dimension_handoff_protocol.json"
COMPLETION = str(controller.COMPLETION_PROTOCOL)
FACTORIAL_PUBLICATION = (
    "configs/dense_no_packing_state_operator_factorial_publication_protocol.json"
)
MAIN_REPAIR_ARCHIVE = "reports/engineering-archive/main-handoff-v1"
HANDOFF_NOTE = {
    "reason": "Refresh only the primary publication parent identity after binding the corrected evaluation-source verifier. All eleven steps, science, sources and completion requirements remain unchanged.",
    "scientific_contract_changed": False,
    "numerical_implementation_changed": False,
    "runtime_deployed": False,
}

PATH_RELOCATIONS = {
    "paper-release": (2, "paper"),
    "portable-evidence-refresh": (1, "scripts/portable_evidence.py"),
    "paper-audit": (7, "configs/dense_scope_amendment.json"),
    "portable-evidence-audit": (1, "scripts/portable_evidence.py"),
}


def project_steps(steps, source_root: Path, live_root: Path):
    projected = []
    seen = set()
    for step in steps:
        if step.name in PATH_RELOCATIONS:
            index, relative = PATH_RELOCATIONS[step.name]
            if step.command[index] != str(source_root / relative):
                raise ValueError(f"Unexpected relocation argument: {step.name}")
            command = list(step.command)
            command[index] = str(live_root / relative)
            step = replace(step, command=tuple(command))
            seen.add(step.name)
        projected.append(step)
    if seen != set(PATH_RELOCATIONS):
        raise ValueError("Missing declared worktree-local command")
    return projected


def validate_amendment_chain(repository: Path, protocol: dict) -> tuple[dict, dict]:
    """Reconstruct both exact preparation layers; neither is a live transition."""
    binding = protocol["parent_bindings"]["main_publication_header_dependency"]
    amendment = json.loads((repository / binding["path"]).read_text())
    for item in [
        binding,
        amendment["prior_prepared_completion_protocol"],
        *amendment["evidence_bindings"],
    ]:
        if not controller._binding_valid(item, repository):
            raise ValueError("Parent amendment evidence changed")
    prior = json.loads(
        (repository / amendment["prior_prepared_completion_protocol"]["path"]).read_text()
    )
    expected_protocol = deepcopy(prior)
    expected_protocol["main_completion_gate"]["contract_sha256"] = amendment[
        "main_contract_transition"
    ]["to_contract_sha256"]
    expected_protocol["parent_bindings"]["main_publication_header_dependency"] = binding
    expected_protocol["amended_at_utc"] = amendment["frozen_at_utc"]

    combined_binding = protocol["parent_bindings"]["main_handoff_repairs"]
    if not controller._binding_valid(combined_binding, repository):
        raise ValueError("Combined amendment identity changed")
    combined = json.loads((repository / combined_binding["path"]).read_text())
    before = combined["prior_prepared_protocols"]
    after = combined["refreshed_protocol_bindings"]
    if (
        combined.get("status") != "prospective_combined_handoff_dependency_amendment"
        or combined.get("runtime_deployed") is not False
        or set(before)
        != {OUTCOME, BRIDGE, SENSITIVITY, PUBLICATION, HANDOFF, COMPLETION, FACTORIAL_PUBLICATION}
        or set(after) != set(before) - {COMPLETION}
    ):
        raise ValueError("Unexpected combined dependency scope")
    for item in [*before.values(), *after.values(), *combined["evidence_bindings"]]:
        if not controller._binding_valid(item, repository):
            raise ValueError("Combined amendment evidence changed")
    if any(item["path"] != label for label, item in after.items()):
        raise ValueError("Refreshed binding must name the actual protocol")
    previous = {
        label: json.loads((repository / item["path"]).read_text()) for label, item in before.items()
    }
    if previous[COMPLETION] != expected_protocol:
        raise ValueError("The archived header-only predecessor has undeclared changes")

    main_archive = repository / MAIN_REPAIR_ARCHIVE
    main_fix = json.loads((main_archive / "migration-draft.json").read_text())
    main_contract = json.loads((main_archive / "projected-main-contract.json").read_text())
    transition = combined["main_contract_transition"]
    if (
        transition["live_main_contract_sha256"] != main_fix["from_contract_sha256"]
        or transition["prior_prepared_required_main_contract_sha256"]
        != amendment["main_contract_transition"]["to_contract_sha256"]
        or transition["required_main_contract_sha256"] != main_fix["to_contract_sha256"]
        or contract_digest(main_contract) != main_fix["to_contract_sha256"]
        or main_contract["sha256"] != main_fix["to_contract_sha256"]
    ):
        raise ValueError("Combined main repair identity differs")
    for label in (OUTCOME, BRIDGE, SENSITIVITY):
        if (repository / before[label]["path"]).read_bytes() != (
            main_archive / "before" / label
        ).read_bytes() or (repository / label).read_bytes() != (
            main_archive / "after" / label
        ).read_bytes():
            raise ValueError("Combined main dependency is not the exact reviewed candidate")
    note = json.loads((repository / OUTCOME).read_text())["evaluation_handoff_amendment"]
    expected_publication = deepcopy(previous[PUBLICATION])
    for item in expected_publication["parent_bindings"].values():
        if item["path"] in (OUTCOME, BRIDGE, SENSITIVITY):
            item["sha256"] = after[item["path"]]["sha256"]
            if "bytes" in item:
                item["bytes"] = after[item["path"]]["bytes"]
    expected_publication["evaluation_handoff_amendment"] = note
    if json.loads((repository / PUBLICATION).read_text()) != expected_publication:
        raise ValueError("Undeclared narrative publication change")
    expected_handoff = deepcopy(previous[HANDOFF])
    expected_handoff["parent_bindings"][PUBLICATION] = after[PUBLICATION]
    expected_handoff["evaluation_handoff_amendment"] = HANDOFF_NOTE
    if json.loads((repository / HANDOFF).read_text()) != expected_handoff:
        raise ValueError("Undeclared dimension handoff change")
    expected_factorial_publication = deepcopy(previous[FACTORIAL_PUBLICATION])
    expected_factorial_publication["amendments"] = [
        after[HANDOFF] if item["path"] == HANDOFF else item
        for item in expected_factorial_publication["amendments"]
    ]
    if (
        json.loads((repository / FACTORIAL_PUBLICATION).read_text())
        != expected_factorial_publication
    ):
        raise ValueError("Undeclared factorial publication change")

    expected_protocol["main_completion_gate"]["contract_sha256"] = transition[
        "required_main_contract_sha256"
    ]
    expected_protocol["parent_bindings"]["primary_dimension_handoff"] = after[HANDOFF]
    expected_protocol["parent_bindings"]["publication_protocol"] = after[FACTORIAL_PUBLICATION]
    expected_protocol["parent_bindings"]["main_handoff_repairs"] = combined_binding
    expected_protocol["amended_at_utc"] = combined["frozen_at_utc"]
    if protocol != expected_protocol:
        raise ValueError("Undeclared completion protocol change")
    return amendment, combined_binding


def build_plan(repository: Path) -> dict:
    repository = repository.resolve()
    protocol_path = repository / controller.COMPLETION_PROTOCOL
    protocol = controller.load_completion_protocol(protocol_path, repository)
    amendment, combined_binding = validate_amendment_chain(repository, protocol)
    # Exercise the actual consumer loaders, not just the controller's immediate
    # source identities. This includes the previously omitted transitive helper.
    from embed_optim.corrected_execution_sensitivity import _load_protocol
    from embed_optim.corrected_outcome_summary import _load_outcome_protocol
    from embed_optim.corrected_publication import _load_publication_protocol
    from embed_optim.corrected_retrieval_bridge import _load_implementation_protocol
    from embed_optim.state_operator_factorial_publication import _load_protocol as load_factorial

    for label, loader in (
        (OUTCOME, _load_outcome_protocol),
        (BRIDGE, _load_implementation_protocol),
        (SENSITIVITY, _load_protocol),
        (PUBLICATION, _load_publication_protocol),
        (FACTORIAL_PUBLICATION, load_factorial),
    ):
        loader(repository / label, repository)
    old = json.loads(
        (
            repository
            / "reports/engineering-archive/publication-header/factorial-live-ledger-before-parent-fix.json"
        ).read_text()
    )
    transition = amendment["runtime_transition"]
    if (
        old["contract"]["sha256"] != transition["from_live_factorial_contract_sha256"]
        or old["steps"] != []
        or old["status"] != "waiting_for_main_completion"
    ):
        raise ValueError("The archived live controller is not the exact zero-step predecessor")
    expected = transition["required_arguments"]
    if expected != old["contract"]["arguments"]:
        raise ValueError("Original live arguments changed")
    args = controller.parse_args(
        [
            "--workdir",
            str(repository),
            "--main-ledger",
            expected["main_ledger"],
            "--python",
            expected["python"],
            "--dry-run",
        ]
    )
    steps = controller.pipeline_steps(args, repository, protocol)
    local = controller._contract(args, repository, protocol_path, steps)
    live_root = Path(expected["main_ledger"]).parents[2]
    projected = controller._contract(
        args, repository, protocol_path, project_steps(steps, repository, live_root)
    )
    retained = [step for step in projected["steps"] if not step["name"].startswith("primary-")]

    def strip_index(entries):
        return [{k: v for k, v in step.items() if k != "index"} for step in entries]

    if (
        len(projected["steps"]) != 47
        or len(retained) != 36
        or strip_index(retained) != strip_index(old["contract"]["steps"])
    ):
        raise ValueError("Original factorial commands changed")
    if projected["arguments"] != old["contract"]["arguments"]:
        raise ValueError("Original factorial arguments changed")
    return {
        "scope": "read_only_successor_contract_projection",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "implementation": controller._identity(Path(__file__), repository),
        "amendment": combined_binding,
        "prior_header_amendment": protocol["parent_bindings"]["main_publication_header_dependency"],
        "both_prepared_amendment_layers_verified": True,
        "actual_consumer_protocol_loaders_pass": True,
        "local_contract": local,
        "projected_live_contract": projected,
        "live_root": str(live_root),
        "exact_path_relocations": PATH_RELOCATIONS,
        "original_36_commands_and_arguments_preserved": True,
        "required_main_contract_sha256": protocol["main_completion_gate"]["contract_sha256"],
        "controller_started": False,
        "runtime_deployed": False,
        "scientific_completion": False,
        "boundary": "Read-only source validation and exact four-command path relocation. It neither executes a command nor verifies a real deployment. Recompute the contract on the experiment checkout after approved deployment before restarting the zero-step successor.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_plan(args.repository), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
