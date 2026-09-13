"""Prepare an exact, non-numerical main repair; never deploy or control a job."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.completion_contract_migration import validate_transition

OLD_CONTRACT = "4152531e354bdf325411c7f4d6de044ea68b6d5f26a59fbf5914f8b55f3c9789"
RENDERER = "src/embed_optim/corrected_publication.py"
HELPER = "src/embed_optim/evaluation_source_provenance.py"
OUTCOME = "configs/dense_no_packing_outcome_protocol.json"
BRIDGE = "configs/dense_no_packing_bridge_implementation_protocol_v2.json"
SENSITIVITY = "configs/dense_no_packing_sensitivity_implementation_protocol.json"
PUBLICATION = "configs/dense_no_packing_publication_protocol.json"
CHANGED_PATHS = (RENDERER, HELPER, OUTCOME, BRIDGE, SENSITIVITY, PUBLICATION)
OLD_HELPER_SHA = "fec4c185af25e973894f69626fe6361ba59a211a2044986e2edad69862125f6d"
NEW_HELPER_SHA = "235a9c1cf78597b11e41436003dba5a19eca689b1c4d8394c3da4e0a307f3cba"
OLD_RENDERER_SHA = "a2f9aaed850373a0bc54466769f0513944eb7c8e54c76c2a6d0c1aaddcc364e6"
NEW_RENDERER_SHA = "ec299f29bb76b73228eb5d026cc014a78e35bcacaed688906ad0b02668dd1af6"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def identity(label: str, raw: bytes) -> dict:
    return {"path": label, "bytes": len(raw), "sha256": digest(raw)}


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


def contract_digest(value: dict) -> str:
    return digest(
        json.dumps(
            {k: v for k, v in value.items() if k != "sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


def build_repair(original: dict[str, bytes], helper: bytes, ledger_raw: bytes) -> dict:
    """Derive every changed byte from the exact archived main, not the story branch."""
    ledger = json.loads(ledger_raw)
    old = ledger["contract"]
    if (
        old.get("sha256") != OLD_CONTRACT
        or contract_digest(old) != OLD_CONTRACT
        or ledger.get("status") != "waiting_for_training"
        or ledger.get("complete") is not False
        or ledger.get("steps") != []
    ):
        raise ValueError("Main is not the exact waiting, zero-step predecessor")
    for record in old["sources"]:
        if identity(record["path"], original[record["path"]]) != record:
            raise ValueError(f"Original contract source drift: {record['path']}")
    if digest(original[HELPER]) != OLD_HELPER_SHA or digest(helper) != NEW_HELPER_SHA:
        raise ValueError("Source-provenance helper is not the exact reviewed repair")
    if digest(original[RENDERER]) != OLD_RENDERER_SHA:
        raise ValueError("Original renderer is not the reviewed main version")
    old_renderer = original[RENDERER].decode()
    if old_renderer.count(r"\\\\n\\midrule") != 3:
        raise ValueError("Unexpected table-header syntax topology")
    renderer = old_renderer.replace(r"\\\\n\\midrule", r"\\\\\n\\midrule").encode()
    if digest(renderer) != NEW_RENDERER_SHA:
        raise ValueError("Renderer repair changed more than the three reviewed separators")
    changed = {RENDERER: renderer, HELPER: helper}
    note = {
        "reason": "Authenticate the exact corrected Dense evaluation source mapping before outcome aggregation; preserve all scientific and numerical definitions.",
        "scientific_contract_changed": False,
        "numerical_implementation_changed": False,
        "runtime_deployed": False,
    }
    outcome = json.loads(original[OUTCOME])
    if "evaluation_source_provenance" in outcome["source_bindings"]:
        raise ValueError("Unexpected existing source-provenance binding")
    outcome["source_bindings"]["evaluation_source_provenance"] = identity(HELPER, helper)
    outcome["evaluation_handoff_amendment"] = note
    changed[OUTCOME] = json_bytes(outcome)
    for label in (BRIDGE, SENSITIVITY, PUBLICATION):
        value = json.loads(original[label])
        for binding in value["parent_bindings"].values():
            target = binding["path"]
            if target in changed:
                binding["sha256"] = digest(changed[target])
                if "bytes" in binding:
                    binding["bytes"] = len(changed[target])
        value["evaluation_handoff_amendment"] = note
        if label == PUBLICATION:
            value["source_bindings"]["corrected_publication"] = identity(RENDERER, renderer)
            value["latex_header_syntax_amendment"] = {
                "reason": "Replace literal backslash-n after all three table-header row terminators with an actual newline. Publication syntax only.",
                "affected_headers": 3,
                "scientific_contract_changed": False,
                "primary_publication_outputs_visible": False,
            }
        changed[label] = json_bytes(value)
    projected = deepcopy(old)
    for index, record in enumerate(projected["sources"]):
        if record["path"] in changed:
            projected["sources"][index] = identity(record["path"], changed[record["path"]])
    projected["sha256"] = contract_digest(projected)
    changed_protocols = (OUTCOME, BRIDGE, SENSITIVITY, PUBLICATION)
    migration = {
        "schema_version": 1,
        "scope": "draft_evaluation_and_publication_handoff_transition",
        "runtime_deployed": False,
        "reason": "Bind the corrected evaluation source verifier and repair three publication header separators, with all numeric sources, commands and arguments unchanged.",
        "from_contract_sha256": OLD_CONTRACT,
        "to_contract_sha256": projected["sha256"],
        "scientific_contract_changed": False,
        "required_unchanged_sources": [
            r["path"] for r in old["sources"] if r["path"] not in changed_protocols
        ],
        "allowed_changed_sources": list(changed_protocols),
        "amendment_assertions": [
            {
                "path": label,
                "field": "evaluation_handoff_amendment.scientific_contract_changed",
                "expected": False,
            }
            for label in changed_protocols
        ],
        "archive_basename": "pipeline-ledger.pre-evaluation-handoff-4152531e.json",
        "draft_boundary": "Not deployed. Require a verified sole-controller handoff, archived ledger, audited six-file patch and consistent successor parent before runtime transition. This does not authorize deletion, publication, training changes or a competing controller.",
    }
    patch = "".join(
        "".join(
            difflib.unified_diff(
                original[label].decode().splitlines(True),
                changed[label].decode().splitlines(True),
                fromfile=f"a/{label}",
                tofile=f"b/{label}",
            )
        )
        for label in CHANGED_PATHS
    )
    return {"changed": changed, "contract": projected, "migration": migration, "patch": patch}


def prepare(repository: Path, helper_root: Path, output: Path) -> dict:
    repository, helper_root, output = (p.resolve() for p in (repository, helper_root, output))
    if output.is_relative_to(repository) or repository.is_relative_to(output):
        raise ValueError("Preparation output must be outside the live experiment checkout")
    if output.exists():
        raise FileExistsError("Use a new preparation output directory")
    ledger_path = repository / "logs/dense-no-packing-finalization/pipeline-ledger.json"
    ledger_raw = ledger_path.read_bytes()
    ledger = json.loads(ledger_raw)
    paths = {r["path"] for r in ledger["contract"]["sources"]} | set(CHANGED_PATHS)
    if any(Path(p).is_absolute() or ".." in Path(p).parts for p in paths):
        raise ValueError("Unsafe original source path")
    original = {p: (repository / p).read_bytes() for p in sorted(paths)}
    helper = (helper_root / HELPER).read_bytes()
    if any((repository / "results/dense-no-packing-beir").rglob("*Decontaminated.json")):
        raise ValueError("Primary retrieval outputs already exist; re-review transition timing")
    plan = build_repair(original, helper, ledger_raw)
    if any((repository / p).read_bytes() != raw for p, raw in original.items()):
        raise ValueError("Live source changed during preparation")
    output.mkdir(parents=True)

    def save(path: Path, raw: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(raw)

    for label, raw in original.items():
        save(output / "before" / label, raw)
    save(output / "before/main-ledger.json", ledger_raw)
    for label, raw in plan["changed"].items():
        save(output / "after" / label, raw)
    validate_transition(ledger["contract"], plan["contract"], plan["migration"], output / "after")
    save(output / "migration-draft.json", json_bytes(plan["migration"]))
    save(output / "projected-main-contract.json", json_bytes(plan["contract"]))
    save(output / "main-handoff-repair.patch", plan["patch"].encode())
    report = {
        "schema_version": 1,
        "scope": "engineering_main_handoff_repair_preparation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "repository": str(repository),
        "runtime_deployed": False,
        "scientific_completion": False,
        "implementation": identity(str(Path(__file__).resolve()), Path(__file__).read_bytes()),
        "source_ledger": identity("before/main-ledger.json", ledger_raw),
        "before": [identity(p, raw) for p, raw in original.items()],
        "after": [identity(p, raw) for p, raw in plan["changed"].items()],
        "patch": identity("main-handoff-repair.patch", plan["patch"].encode()),
        "from_contract_sha256": OLD_CONTRACT,
        "to_contract_sha256": plan["contract"]["sha256"],
        "commands_and_arguments_unchanged": True,
        "controller_lease_acquired": False,
        "current_main_steps": len(ledger["steps"]),
        "current_backups_preserved_in_snapshot": len(ledger["backups"]),
        "successor_requires_new_parent_binding": True,
    }
    save(output / "preparation.json", json_bytes(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--helper-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.repository, args.helper_root, args.output), indent=2))


if __name__ == "__main__":
    main()
