"""Accept completed synthetic authoring checks, not publication or real experiments."""

import os
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_publication_contract import PARENTS, PublicationContract
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = Path(__file__).resolve().parent
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")
RECORDS = (
    (
        "complete-first.json",
        "source-first",
        "878940c7a59f4418022f01af80b3365be8ef751e2cfd0580b69048917c024497",
    ),
    (
        "complete-retry.json",
        "layout-attempt-1/source",
        "ca672b515fa520aa132f5bab0faf826029e2684e2690426b11cb09c536fa2cde",
    ),
    (
        "complete-final.json",
        "source-current",
        "b4b09cd1af09772636c88cc47ab8da3c0551a00031a99a89f1a85a4dbf89b1d0",
    ),
)


def binding(path):
    name = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    return {"path": name, **file_identity(path)}


def checked(path, expected, checks):
    verify_file(path, expected)
    checks.append(binding(path))


def tests(path, count, failures=0):
    suites = list(ET.parse(path).iter("testsuite"))
    assert len(suites) == 1
    row = suites[0].attrib
    assert int(row["tests"]) == count and int(row["failures"]) == failures
    assert int(row["errors"]) == int(row["skipped"]) == 0
    return row


def validate():
    if sys.flags.optimize or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Require assertions and CPU-only validation")
    before = handoff()
    parent_name, parent_sha = PARENTS["joint_reconstruction_acceptance"]
    parent_path = ROOT / parent_name
    assert file_identity(parent_path)["sha256"] == parent_sha
    parent = read_json(parent_path)
    assert parent["artifact_validation_passed"] is True
    assert parent["scientific_completion"] is False
    preserved, checks = [], []
    for row in parent["bindings"]:
        path = ARCHIVE / "before" / row["path"] if row["path"] in DOCS else ROOT / row["path"]
        checked(path, row, preserved)
    records = {}
    for name, stage, sha in RECORDS:
        path = ARCHIVE / name
        assert file_identity(path)["sha256"] == sha
        value = records[name] = read_json(path)
        assert value["passed"] is True and value["upstream_primary_admission_simulated"] is True
        for flag in (
            "scientific_completion",
            "actual_checkpoint_backed_positive_authoring_verified",
            "portable_primary_publication_verified",
            "manuscript_installed",
        ):
            assert value[flag] is False
        assert "ordinary retained run" in value["actual_missing_primary_refusal"]
        assert "Synthetic admission" in value["synthetic_primary_admission_refusal"]
        for row in value["sources"]:
            source = Path(row["path"])
            archived = ARCHIVE / stage / source.relative_to(ROOT)
            if source.parent == ARCHIVE:
                archived = ARCHIVE / stage / source.name
            checked(archived if archived.exists() else source, row, checks)
        for row in (*value["inputs"], *value["generated_artifacts"]):
            checked(Path(row["path"]), row, checks)
        require_same(value["post_execution_dispatchers"], before)
    for path in (ARCHIVE / "source-current").rglob("*"):
        if path.is_file():
            relative = path.relative_to(ARCHIVE / "source-current")
            current = ARCHIVE / relative if len(relative.parts) == 1 else ROOT / relative
            checked(current, file_identity(path), checks)
    layout_path = ARCHIVE / "layout-final.json"
    assert (
        file_identity(layout_path)["sha256"]
        == "20b8dabeede997426e0a90b336d215da7f300894718e62650df3c03e226da6de"
    )
    layout = read_json(layout_path)
    assert layout["passed"] is True and layout["layout"]["main_end_page"] == 7
    assert layout["abstract_conservative_budget"]["complete"] is True
    for flag in (
        "overfull_boxes",
        "type3_fonts",
        "actual_primary_results",
        "manuscript_installed",
        "strict_manuscript_publication_audit_passed",
        "scientific_completion",
    ):
        assert layout[flag] is False
    for row in (*layout["sources"], *layout["original_paper_bindings"], *layout["artifacts"]):
        checked(Path(row["path"]), row, checks)
    require_same(layout["post_execution_dispatchers"], before)
    return finish(parent, preserved, checks, records, layout, before)


def finish(parent, preserved, checks, records, layout, before):
    suites = {
        name: tests(ARCHIVE / name, count, failures)
        for name, count, failures in (
            ("focused-first.xml", 50, 0),
            ("exact-csv-attempt-1/roundtrip.xml", 4, 3),
            ("focused-retry.xml", 57, 0),
            ("focused-final.xml", 63, 0),
            ("full-final.xml", 2636, 0),
        )
    }
    assert (
        file_identity(ARCHIVE / "full-final.xml")["sha256"]
        == "8c46fda787ba7417fcf9954ba62deb9fdc5a664bff70c5671831e2e2fbd70ee5"
    )
    primary = PrimaryV3Contract.load(
        ROOT / PARENTS["primary"][0], ROOT, Path("/tmp/dense-identity-source.8rUgLF")
    )
    protocol = ROOT / "configs/dense_primary_v3_publication_protocol.json"
    assert (
        file_identity(protocol)["sha256"]
        == "39c8740962fbfa57770bc21058bfc18b3f862cf97f7b12f48ad29dcd04b43f81"
    )
    PublicationContract.load(protocol, primary).recheck()
    final = records["complete-final.json"]
    assert final["previous_evidence_preserved_except_declared_functional_table_label"] is True
    assert final["functional_latex_exactly_matches_accepted_joint_reconstruction"] is True
    assert layout["primary_and_functional_include_source"]["sha256"] == RECORDS[-1][2]
    for row in parent["unchanged_live_core"]:
        for tree in (ROOT, Path("/root/embedding-optimizer-study")):
            checked(tree / row["path"], row, checks)
    checked(ROOT / "paper/main.tex", parent["unchanged_manuscript_source"], checks)
    checked(Path(parent["unchanged_main_ledger"]["path"]), parent["unchanged_main_ledger"], checks)
    require_same(handoff(), before)
    return {
        "scope": "engineering_synthetic_primary_publication_preparation_acceptance",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "full_development_regression_passed": True,
        "complete_synthetic_generation_and_layout_verified": True,
        "raw_functional_latex_preserved_with_separate_manuscript_label_adapter": True,
        "upstream_primary_admission_simulated": True,
        "actual_checkpoint_backed_positive_authoring_verified": False,
        "portable_primary_publication_verified": False,
        "strict_manuscript_consumer_integrated": False,
        "physical_cross_host_execution_verified": False,
        "manuscript_installed": False,
        "formal_replication_ready": False,
        "scientific_completion": False,
        "tests_including_preserved_failures": suites,
        "prior_acceptance": {
            "path": PARENTS["joint_reconstruction_acceptance"][0],
            "sha256": PARENTS["joint_reconstruction_acceptance"][1],
        },
        "prior_bindings_preserved": preserved,
        "prior_external_payloads_rechecked": False,
        "source_and_artifact_checks": checks,
        "all_table_counts": final["all_table_counts"],
        "unchanged_live_core": parent["unchanged_live_core"],
        "unchanged_manuscript_source": parent["unchanged_manuscript_source"],
        "unchanged_main_ledger": parent["unchanged_main_ledger"],
        "post_execution_dispatchers": before,
        "mutable_handoff_documents_are_not_new_acceptance_inputs": True,
        "bindings": [
            binding(p)
            for p in sorted(ARCHIVE.rglob("*"))
            if p.is_file() and p.name != "validation.json"
        ],
    }


if __name__ == "__main__":
    output = ARCHIVE / "validation.json"
    if output.exists():
        raise ValueError("Never overwrite the accepted receipt")
    write_new(output, validate())
    print({"artifact_validation_passed": True, **file_identity(output)}, flush=True)
