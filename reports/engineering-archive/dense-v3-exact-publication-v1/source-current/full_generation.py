"""Check complete presentation from accepted synthetic raw reconstruction, not primary runs."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from embed_optim import primary_v3_exact_publication as publication
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_exact_publication_contract import (
    OUTPUTS,
    PARENTS,
    SOURCES,
    ExactPublicationContract,
)
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.prepare_dense_v3_exact_publication import payload

PARENT = "reports/engineering-archive/dense-v3-exact-reconstruction-v1/validation.json"
PARENT_SHA = "d9888b172648555b54895c8a0b0acc751a3b8a345b7872ebf929ebf7038f5c59"
AUDIT = Path("/tmp/dense-v3-exact-reconstruction-cold.LVS9GV/result.json")
AUDIT_SHA = "80d0727dc4422b90255027dba4d2c28e433d40667561d438154898bc6a21717a"


def run(args):
    root, work = args.repository.resolve(), args.workdir.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or not work.is_dir() or list(work.iterdir()):
        raise ValueError("Require a new empty CPU-only complete presentation directory")
    before = handoff()
    assert file_identity(root / PARENT)["sha256"] == PARENT_SHA
    parent = read_json(root / PARENT)
    assert parent["artifact_validation_passed"] is True and parent["scientific_completion"] is False
    assert parent["accepted_results"]["first"]["sha256"] == AUDIT_SHA
    assert file_identity(AUDIT)["sha256"] == AUDIT_SHA
    audit = read_json(AUDIT)
    assert audit["passed"] is True and audit["upstream_primary_admission_simulated"] is True
    archived = {row["path"]: row for row in audit["artifacts"]}
    inputs = []

    def checked(path):
        record = archived[str(path)]
        verify_file(path, record)
        inputs.append(record)
        return path

    raw = AUDIT.parent / "cold/recomputed"
    old_root = raw / "original-functional/publication"
    old = {name: checked(old_root / name).read_bytes() for name in publication.original.OUTPUTS}
    exact_root, geometry_root = raw / "exact-bridge", raw / "exact-geometry"
    for directory, counts in (
        (exact_root, publication.exact.TABLE_COUNTS),
        (geometry_root, publication.exact.exact_geometry.TABLE_COUNTS),
    ):
        for name in counts:
            checked(directory / (name + ".csv"))
    geometry = publication.original.read_tables(
        geometry_root, publication.exact.exact_geometry.TABLE_COUNTS
    )
    bridge = publication.read_exact_tables(exact_root)
    exact_evidence = read_json(checked(exact_root / "evidence.json"))
    original_evidence = json.loads(old["evidence.json"])
    require_same(
        original_evidence["functional_evidence"]["original_bridge_evidence"],
        exact_evidence["original_bridge_evidence"],
    )
    sources = [{"path": str(root / name), **file_identity(root / name)} for name in SOURCES]
    sources.append({"path": str(Path(__file__).resolve()), **file_identity(Path(__file__))})
    primary = PrimaryV3Contract.load(root / PARENTS["primary"][0], root, args.training_root)
    draft = work / "draft.json"
    write_new(draft, payload(root))
    contract = ExactPublicationContract.load(draft, primary)
    try:
        publication.gather(contract, SimpleNamespace(experiment_root=work / "absent-primary"))
    except ValueError as error:
        missing_refusal = str(error)
        assert "ordinary retained run" in missing_refusal
    else:
        raise AssertionError("Actual missing primary evidence was admitted")
    evidence = {
        "scope": "engineering_complete_synthetic_exact_publication",
        "upstream_primary_admission_simulated": True,
        "original_publication_evidence": original_evidence,
        "exact_bridge_evidence": exact_evidence,
        "source_bindings": inputs,
        "scientific_completion": False,
    }
    contents = publication.generate(primary, old, geometry, bridge, evidence)
    summary = json.loads(contents["exact-summary.json"])
    require_same(
        summary["diagnostics"],
        {key: exact_evidence[key] for key in ("numerical_diagnostics", "measurement_coverage")},
    )
    assert contents["original-optimizer-primary.tex"] == old["optimizer-primary.tex"]
    assert contents["optimizer-primary.tex"].startswith(old["optimizer-primary.tex"])
    for name in ("functional-inference.tex", "dimension-utilization.tex", "primary-summary.json"):
        assert contents[name] == old[name], name
    plan = {
        "scope": evidence["scope"],
        "scientific_completion": False,
        "upstream_primary_admission_simulated": True,
    }
    output = work / "synthetic-evidence"
    saved = publication.save(output, plan, contents)
    repeated = publication.generate(primary, old, geometry, bridge, evidence)
    assert repeated == contents
    require_same(publication.inspect(output, plan, repeated), saved)
    controls = []
    for name in OUTPUTS:
        directory = work / ("altered-" + name.replace(".", "-"))
        publication.save(directory, plan, contents)
        (directory / name).write_bytes(contents[name] + b"\nrehashed alteration\n")
        manifest = read_json(directory / "manifest.json")
        manifest["outputs"][name] = {"path": name, **file_identity(directory / name)}
        (directory / "manifest.json").write_text(json.dumps(manifest))
        try:
            publication.inspect(directory, plan, repeated)
        except ValueError as error:
            assert "freshly reconstructed" in str(error)
            controls.append({"file": name, "refused": True, "reason": str(error)})
        else:
            raise AssertionError("Rehashed complete output was admitted: " + name)
    try:
        publication.original.reject_simulation(evidence)
    except ValueError as error:
        simulated_refusal = str(error)
    else:
        raise AssertionError("Synthetic evidence was accepted for primary publication")
    contract.recheck()
    for row in (*inputs, *sources):
        verify_file(row["path"], row)
    for row in parent["unchanged_live_core"]:
        for tree in (root, Path("/root/embedding-optimizer-study")):
            verify_file(tree / row["path"], row)
    verify_file(root / "paper/main.tex", parent["unchanged_manuscript_source"])
    verify_file(parent["unchanged_main_ledger"]["path"], parent["unchanged_main_ledger"])
    require_same(handoff(), before)
    result = {
        "scope": "engineering_complete_exact_publication_generation_check",
        "passed": True,
        "parent_acceptance": {"path": PARENT, "sha256": PARENT_SHA},
        "first_complete_raw_audit": {"path": str(AUDIT), "sha256": AUDIT_SHA},
        "all_table_counts": {
            kind: {name: len(rows) for name, rows in tables.items()}
            for kind, tables in json.loads(contents["tables.json"]).items()
        },
        "original_primary_include_retained_as_exact_copy_and_prefix": True,
        "original_functional_and_primary_summary_bytes_preserved": True,
        "all_original_and_exact_decisions_recomputed": True,
        "same_process_full_generation_repeated": True,
        "rehashed_output_controls": controls,
        "raw_reconstruction_repeated_for_output_controls": False,
        "actual_missing_primary_refusal": missing_refusal,
        "synthetic_primary_admission_refusal": simulated_refusal,
        "upstream_primary_admission_simulated": True,
        "actual_checkpoint_backed_positive_authoring_verified": False,
        "portable_primary_publication_verified": False,
        "complete_paper_layout_verified": False,
        "manuscript_installed": False,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "training-root", "workdir"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
