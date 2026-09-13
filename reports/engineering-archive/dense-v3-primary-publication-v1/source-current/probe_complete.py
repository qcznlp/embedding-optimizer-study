"""Generate from the complete accepted synthetic joint population, never primary evidence."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from embed_optim import primary_v3_publication as publication
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_publication_contract import PARENTS, SOURCES, PublicationContract
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.prepare_dense_v3_publication import payload

AUDIT = Path("/tmp/dense-v3-joint-cold.yR0bGv/result.json")
AUDIT_SHA = "9e1a06975185aa92b1e1114cef164f1df41637470a58f3a230055698d798fa77"


def run(args):
    root, work = args.repository.resolve(), args.workdir.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or not work.is_dir() or list(work.iterdir()):
        raise ValueError("Require CPU-only verification and a new empty work directory")
    before = handoff()
    parent_path, parent_sha = PARENTS["joint_reconstruction_acceptance"]
    assert file_identity(root / parent_path)["sha256"] == parent_sha
    parent = read_json(root / parent_path)
    assert parent["artifact_validation_passed"] is True
    assert parent["scientific_completion"] is False
    assert file_identity(AUDIT)["sha256"] == AUDIT_SHA
    audit = read_json(AUDIT)
    assert audit["passed"] is True and audit["upstream_primary_admission_simulated"] is True
    archived = {row["path"]: row for row in audit["artifacts"]}
    inputs = []

    def checked(path):
        binding = archived[str(path)]
        verify_file(path, binding)
        inputs.append(binding)
        return path

    raw = AUDIT.parent / "cold-reconstruction/recomputed"
    tables = {}
    for kind, relative, counts in (
        ("outcomes", "outcomes/outcomes", publication.TABLE_COUNTS),
        ("geometry", "geometry/geometry", publication.geometry.TABLE_COUNTS),
        ("bridge", "bridge", publication.inference.original.TABLE_COUNTS),
        ("functional", "inference", publication.inference.TABLE_COUNTS),
    ):
        for name in counts:
            checked(raw / relative / (name + ".csv"))
        tables[kind] = publication.read_tables(raw / relative, counts)
    outcome = read_json(checked(raw / "outcomes/outcomes/evidence.json"))
    functional = read_json(checked(raw / "inference/evidence.json"))
    sources = [{"path": str(root / name), **file_identity(root / name)} for name in SOURCES]
    sources.append({"path": str(Path(__file__).resolve()), **file_identity(Path(__file__))})
    primary = PrimaryV3Contract.load(root / PARENTS["primary"][0], root, args.training_root)
    draft = work / "draft.json"
    write_new(draft, payload(root))
    contract = PublicationContract.load(draft, primary)
    try:
        publication.gather(contract, SimpleNamespace(experiment_root=work / "missing-primary"))
    except ValueError as error:
        missing_refusal = str(error)
        assert "ordinary retained run" in missing_refusal
    else:
        raise AssertionError("Actual missing primary publication was accepted")
    evidence = {
        "scope": "engineering_complete_synthetic_publication_generation",
        "upstream_primary_admission_simulated": True,
        "source_bindings": inputs,
        "scientific_completion": False,
    }
    contents = publication.generate(
        primary,
        tables,
        outcome["validation_selection"],
        outcome["grid"]["runs"],
        functional["decisions"],
        evidence,
    )
    assert contents["functional-inference.tex"].decode() == functional["rendered_latex"]
    assert contents[
        "dimension-utilization.tex"
    ].decode() == publication.rendering.bind_functional_table(functional["rendered_latex"])
    plan = {
        "scope": evidence["scope"],
        "scientific_completion": False,
        "upstream_primary_admission_simulated": True,
    }
    output = work / "synthetic-evidence"
    saved = publication.save(output, plan, contents)
    assert saved["recomputed_bytes_verified"] is True
    repeated = publication.generate(
        primary,
        tables,
        outcome["validation_selection"],
        outcome["grid"]["runs"],
        functional["decisions"],
        evidence,
    )
    require_same(saved, publication.inspect(output, plan, repeated))
    previous_match = False
    if args.previous:
        assert file_identity(args.previous)["sha256"] == args.previous_sha256
        previous = read_json(args.previous)
        assert previous["passed"] is True and previous["scientific_completion"] is False
        recorded = {row["path"]: row for row in previous["generated_artifacts"]}
        for name, content in contents.items():
            old_name = "dimension-utilization.tex" if name == "functional-inference.tex" else name
            path = args.previous.parent / "synthetic-evidence" / old_name
            verify_file(path, recorded[str(path)])
            expected = path.read_bytes()
            if name == "dimension-utilization.tex":
                expected = publication.rendering.bind_functional_table(expected.decode()).encode()
            assert content == expected, name
        previous_match = True
    try:
        publication.reject_simulation(evidence)
    except ValueError as error:
        simulated_refusal = str(error)
        assert "Synthetic admission" in simulated_refusal
    else:
        raise AssertionError("Synthetic fixture was accepted for primary publication")
    for row in (*inputs, *sources):
        verify_file(row["path"], row)
    for row in parent["unchanged_live_core"]:
        for tree in (root, Path("/root/embedding-optimizer-study")):
            verify_file(tree / row["path"], row)
    verify_file(root / "paper/main.tex", parent["unchanged_manuscript_source"])
    require_same(handoff(), before)
    result = {
        "scope": "engineering_complete_primary_publication_generation_check",
        "passed": True,
        "joint_acceptance": {"path": parent_path, "sha256": parent_sha},
        "joint_first_audit": {"path": str(AUDIT), "sha256": AUDIT_SHA},
        "all_table_counts": {
            kind: {name: len(rows) for name, rows in group.items()}
            for kind, group in tables.items()
        },
        "functional_latex_exactly_matches_accepted_joint_reconstruction": True,
        "same_process_generation_and_exact_readback_verified": True,
        "previous_evidence_preserved_except_declared_functional_table_label": previous_match,
        "previous_result": None
        if args.previous is None
        else {
            "path": str(args.previous),
            "sha256": args.previous_sha256,
        },
        "actual_missing_primary_refusal": missing_refusal,
        "synthetic_primary_admission_refusal": simulated_refusal,
        "upstream_primary_admission_simulated": True,
        "actual_checkpoint_backed_positive_authoring_verified": False,
        "portable_primary_publication_verified": False,
        "manuscript_installed": False,
        "complete_paper_layout_verified": False,
        "scientific_completion": False,
        "sources": sources,
        "inputs": inputs,
        "generated_artifacts": [
            {"path": str(path), **file_identity(path)}
            for path in sorted(work.rglob("*"))
            if path.is_file()
        ],
        "unchanged_live_core": parent["unchanged_live_core"],
        "unchanged_manuscript_source": parent["unchanged_manuscript_source"],
        "post_execution_dispatchers": before,
    }
    write_new(work / "result.json", result)
    print(
        {"passed": True, "scientific_completion": False, **file_identity(work / "result.json")},
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "training-root", "workdir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--previous-sha256")
    args = parser.parse_args()
    if bool(args.previous) != bool(args.previous_sha256):
        raise ValueError("A previous-result comparison requires an external result hash")
    run(args)


if __name__ == "__main__":
    main()
