"""Validate the bounded entry-measurement milestone, preserving full scientific holds."""

import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_validation_io import write_new
from embed_optim.primary_weight_entries import load_amendment
from scripts.audit_dense_natural_data import handoff
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-v3-geometry-v1")
PRIOR = Path("reports/engineering-archive/dense-v3-outcomes-v1/validation.json")
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")
AMENDMENT_SHA = "a4b4f6839b185931d2de7f0771487123b971a1e7eb69ef34bb5f9e4350060ec5"
REHEARSAL_SHA = "a269aa6f60ad931a5bd7998c6009b9af8f5a5e93d7f23acc33fd3ea586f84b49"
REPLAY_SHA = "233ebd74d5645b0cc02213c77990be5827c6992032b307887aa0330627127625"


def validate(root):
    if sys.flags.optimize:
        raise ValueError("Do not disable evidence assertions")
    archive = root / ARCHIVE
    assert (
        file_identity(root / PRIOR)["sha256"]
        == "58d328af52a6fc94e06c5ef514c9f53c2fb28e71dbb833f569cea7861c5ee5db"
    )
    prior = read_json(root / PRIOR)
    preserved = [
        compare_binding(
            row,
            archive / "before" / row["path"] if row["path"] in DOCS else root / row["path"],
            root,
        )
        for row in prior["bindings"]
    ]
    for path in (archive / "source").rglob("*"):
        if path.is_file():
            verify_file(root / path.relative_to(archive / "source"), file_identity(path))
    assert file_identity(archive / "amendment.json")["sha256"] == AMENDMENT_SHA
    assert file_identity(archive / "rehearsal.json")["sha256"] == REHEARSAL_SHA
    assert file_identity(archive / "fresh-cpu-replay.json")["sha256"] == REPLAY_SHA
    primary = PrimaryV3Contract.load(
        root / "configs/dense_primary_v3_protocol.json", root, "/tmp/dense-identity-source.8rUgLF"
    )
    amendment = load_amendment(
        root / "configs/dense_weight_entry_measurement_amendment.json", primary
    )
    require_same(amendment, read_json(archive / "amendment.json"))
    failure = read_json(archive / "attempt-1/failure.json")
    assert failure["exit_code"] == 1 and failure["model_census_started"] is False
    assert failure["failure_overridden"] is False
    assert not any(Path(failure["workdir"]).iterdir())
    assert (
        file_identity(archive / "attempt-1/source/scripts/audit_dense_weight_entries.py")["sha256"]
        == failure["source_sha256"]
    )
    rehearsal = read_json(archive / "rehearsal.json")
    replay = read_json(archive / "fresh-cpu-replay.json")
    assert rehearsal["actual_diagnostic_census_passed"] is True
    assert replay["fresh_cpu_replay_passed"] is True
    for result in (rehearsal, replay):
        assert result["models"] == 3 and result["checkpoints"] == 9
        assert result["independent_numpy_count_comparisons"] == 1584
        assert result["scientific_completion"] is result["formal_geometry_produced"] is False
        assert result["model_updates"] == result["gpu_workers"] == 0
        assert len(result["changed_cases"]) == 6
        assert all(row["rejected"] is True for row in result["changed_cases"])
    assert rehearsal["hidden_matrices_per_checkpoint"] == 88
    assert rehearsal["hidden_parameters_per_checkpoint"] == 110297088
    assert rehearsal["reference_copy"]["all_declared_files_match"] is True
    assert rehearsal["reference_copy"]["symlink_policy_relaxed"] is False
    assert rehearsal["full_geometry_integration_complete"] is False
    require_same(rehearsal["counterexample"], read_json(archive / "counterexample.json"))
    assert rehearsal["counterexample"]["legacy_fraction"] == 1.0
    assert rehearsal["counterexample"]["actual_entry_fraction"] == 0.25
    assert rehearsal["actual_missing_primary_cli"]["returncode"] != 0
    assert (
        "Require an ordinary retained run directory"
        in rehearsal["actual_missing_primary_cli"]["stderr"]
    )
    for row in rehearsal["records"]:
        verify_file(archive / f"{row['algorithm']}-census.json", row["census"])
        assert row["independent_numpy_comparisons"] == 528
        for stage in row["independent_totals"]:
            assert stage["parameters"] == 110297088
            for kind in ("saved_segment", "cumulative"):
                assert (
                    stage[kind]["entry_fraction"]
                    == stage[kind]["individual_entries_changed"] / stage["parameters"]
                )
                if stage["step"] == 1:
                    assert stage[kind]["individual_entries_changed"] == 0
                else:
                    assert 0 < stage[kind]["entry_fraction"] < 1
                    assert stage[kind]["legacy_matrix_mass_fraction"] == 1
    checks = {row["path"]: row for row in prior["external_payload_checks"]}
    for row in (
        *rehearsal["unchanged_input_and_source_bindings"],
        *rehearsal["artifacts"],
        *replay["artifacts"],
        replay["rehearsal"],
        replay["source"],
    ):
        checks[row["path"]] = row
    for path in (
        Path("/tmp/dense-weight-entry-census-retry.DgbmCc/result.json"),
        Path("/tmp/dense-weight-entry-replay.a9BFoN/result.json"),
    ):
        checks[str(path)] = {"path": str(path), **file_identity(path)}
    external = [compare_binding(row, Path(path), root) for path, row in sorted(checks.items())]
    tests = {}
    for name, count, failures in (
        ("focused-tests.xml", 44, 0),
        ("full-tests.xml", 1890, 0),
        ("focused-tests-final.xml", 45, 0),
        ("full-tests-final.xml", 1891, 2),
        ("full-tests-final-absolute-path.xml", 1891, 0),
    ):
        suite = next(ET.parse(archive / name).iter("testsuite"))
        value = suite.attrib
        assert int(value["tests"]) == count and int(value["failures"]) == failures
        assert int(value["errors"]) == int(value["skipped"]) == 0
        tests[name] = value
        if failures:
            assert {row.attrib["name"] for row in suite if row.find("failure") is not None} == {
                "test_complete_synthetic_findings_fit_the_naacl_paper_layout",
                "test_complete_synthetic_candidate_paper_compiles_within_layout_gate",
            }
    for row in prior["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            verify_file(base / row["path"], row)
    verify_file(root / "paper/main.tex", prior["unchanged_manuscript_source"])
    verify_file(Path(prior["unchanged_main_ledger"]["path"]), prior["unchanged_main_ledger"])
    require_same(handoff(), rehearsal["post_execution_dispatchers"])
    return {
        "scope": "engineering_dense_weight_entry_evidence_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "changed_entry_measurement_component_verified": True,
        "real_diagnostic_numpy_replay_verified": True,
        "first_reference_preflight_failure_preserved": True,
        "relative_path_test_failure_preserved": True,
        "scientific_completion": False,
        "formal_geometry_produced": False,
        "full_geometry_integration_complete": False,
        "formal_replication_ready": False,
        "production_deployed": False,
        "source_published": False,
        "prior_bindings_preserved": preserved,
        "external_payload_checks": external,
        "tests": tests,
        "unchanged_live_core": prior["unchanged_live_core"],
        "unchanged_manuscript_source": prior["unchanged_manuscript_source"],
        "unchanged_main_ledger": prior["unchanged_main_ledger"],
        "post_execution_dispatchers": handoff(),
        "bindings": [
            identity(p, root)
            for p in sorted(archive.rglob("*"))
            if p.is_file() and p.name != "validation.json"
        ]
        + [
            identity(root / p, root)
            for p in (
                *DOCS,
                "configs/dense_weight_entry_measurement_amendment.json",
                "scripts/validate_dense_weight_entries.py",
            )
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("Require a new receipt under an existing explicit parent")
    value = validate(args.repository.resolve())
    write_new(args.output, value)
    print(
        {
            key: value[key]
            for key in (
                "artifact_validation_passed",
                "changed_entry_measurement_component_verified",
                "full_geometry_integration_complete",
                "scientific_completion",
            )
        }
    )


if __name__ == "__main__":
    main()
