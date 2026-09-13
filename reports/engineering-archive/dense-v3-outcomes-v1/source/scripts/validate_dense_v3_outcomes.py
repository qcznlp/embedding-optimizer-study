"""Verify outcome integration evidence without promoting any synthetic or held primary result."""

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_outcomes import OutcomeContract, inspect_bundle, outcome_tables
from embed_optim.primary_v3_validation import ValidationContract
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-v3-outcomes-v1")
PRIOR = Path("reports/engineering-archive/dense-v3-validation-v1/validation.json")
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")
PROTOCOL_SHA = "2a12b78a1a309cd6cc4a56593e4b1b1030370fc579045d1708d572eb5577d68e"
REHEARSAL_SHA = "82aa59dde630634a2224f3f12023646e1886d797ede98b64f3db8f8cd80874f2"


def validate(root):
    if sys.flags.optimize:
        raise ValueError("Do not disable evidence assertions")
    archive = root / ARCHIVE
    assert (
        file_identity(root / PRIOR)["sha256"]
        == "1967ca78e216df0934e0b379808fd10c98598b8054e936f3f2ae2f3806f363b4"
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
    assert file_identity(archive / "protocol-v2.json")["sha256"] == PROTOCOL_SHA
    assert file_identity(archive / "rehearsal.json")["sha256"] == REHEARSAL_SHA
    old_path = root / "configs/dense_primary_v3_outcome_protocol.json"
    assert (
        file_identity(old_path)["sha256"]
        == "914996b3423ba01329c03844419cca61e2a33452fe026665bb940d5b82e2cffe"
    )
    require_same(read_json(old_path), read_json(archive / "attempt-1/protocol.json"))
    old = read_json(old_path)
    for name, binding in old["sources"].items():
        preserved_path = archive / "attempt-1/source" / name
        verify_file(preserved_path if preserved_path.exists() else root / name, binding)
    primary = PrimaryV3Contract.load(
        root / "configs/dense_primary_v3_protocol.json", root, "/tmp/dense-identity-source.8rUgLF"
    )
    validation = ValidationContract.load(
        root / "configs/dense_primary_v3_validation_protocol.json", primary
    )
    contract = OutcomeContract.load(
        root / "configs/dense_primary_v3_outcome_protocol_v2.json", validation
    )
    require_same(contract.payload, read_json(archive / "protocol-v2.json"))
    failure = read_json(archive / "attempt-1/failure.json")
    assert failure["exit_code"] == 1 and failure["failure_overridden"] is False
    old_work = Path(failure["workdir"])
    expected = read_json(old_work / "fixture-input.json")
    expected_rows = sorted(
        expected["score_rows"], key=lambda r: (r["run_id"], r["stage"], r["task"])
    )
    with (old_work / "fixture-bundle/all_task_scores.csv").open() as stream:
        stored = list(csv.DictReader(stream))
    assert len(stored) == 840
    require_same(stored, [{k: str(v) for k, v in row.items()} for row in expected_rows])
    assert list(stored[0]) != list(expected_rows[0])
    rehearsal = read_json(archive / "rehearsal.json")
    assert rehearsal["rehearsal_passed"] is True
    assert rehearsal["all_positive_tables_are_synthetic"] is True
    assert rehearsal["synthetic_grid_cells"] == 840
    assert rehearsal["bootstrap_samples"] == 50_000 and rehearsal["bootstrap_seed"] == 20260903
    assert len(rehearsal["original_kernel_scalar_comparisons"]) == 21
    assert rehearsal["scalar_atol"] == rehearsal["scalar_rtol"] == 1e-12
    assert (
        max(r["absolute_error"] for r in rehearsal["original_kernel_scalar_comparisons"]) <= 1e-12
    )
    assert rehearsal["primary_unchanged_when_validation_selection_changes"] is True
    assert rehearsal["secondary_changes_when_selection_changes"] is True
    assert rehearsal["replay_cli"]["returncode"] == 0
    assert rehearsal["actual_missing_primary_cli"]["returncode"] != 0
    assert (
        "Require an ordinary retained run directory"
        in rehearsal["actual_missing_primary_cli"]["stderr"]
    )
    assert len(rehearsal["changed_cases"]) == 18
    assert all(r["rejected"] for r in rehearsal["changed_cases"])
    assert all(rehearsal[k] == 0 for k in ("model_updates", "gpu_workers", "network_uploads"))
    assert rehearsal["formal_outcome_produced"] is False
    fresh_work = Path("/tmp/dense-v3-outcomes-replay.EEVh60")
    input_value = read_json(fresh_work / "fixture-input.json")
    plan = read_json(fresh_work / "fixture-plan.json")
    assert plan["not_primary_evidence"] is True
    tables = outcome_tables(primary, input_value["score_rows"], input_value["selection"])
    replayed = inspect_bundle(fresh_work / "fixture-bundle", plan, input_value, tables)
    require_same(replayed, rehearsal["fresh_cpu_fixture_readback"])
    for name in (
        "primary_summary.csv",
        "secondary_summary.csv",
        "run_stage_scores.csv",
        "evidence.json",
    ):
        target = fresh_work / f"invalid-bundle-{name.replace('.', '-')}"
        try:
            inspect_bundle(target, plan, input_value, tables)
        except ValueError as error:
            assert "fresh evidence/statistical recomputation" in str(error)
        else:
            raise AssertionError("Rehashed invalid bundle accepted")
    checks = {row["path"]: row for row in prior["external_payload_checks"]}
    for row in (*rehearsal["unchanged_inputs_and_sources"], *rehearsal["artifacts"]):
        checks[row["path"]] = row
    for path in sorted(old_work.rglob("*")):
        if path.is_file():
            row = {"path": str(path), **file_identity(path)}
            checks[row["path"]] = row
    for path in (fresh_work / "result.json",):
        row = {"path": str(path), **file_identity(path)}
        checks[row["path"]] = row
    external = [compare_binding(row, Path(path), root) for path, row in sorted(checks.items())]
    tests = {}
    for name, count in (
        ("focused-tests.xml", 65),
        ("focused-tests-final.xml", 67),
        ("full-tests.xml", 1844),
        ("focused-tests-v2.xml", 69),
        ("full-tests-v2.xml", 1846),
    ):
        value = next(ET.parse(archive / name).iter("testsuite")).attrib
        assert int(value["tests"]) == count
        assert all(int(value[k]) == 0 for k in ("failures", "errors", "skipped"))
        tests[name] = value
    for row in prior["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            verify_file(base / row["path"], row)
    verify_file(root / "paper/main.tex", prior["unchanged_manuscript_source"])
    verify_file(Path(prior["unchanged_main_ledger"]["path"]), prior["unchanged_main_ledger"])
    return {
        "scope": "engineering_dense_v3_outcome_evidence_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "prepared_outcome_integration_rehearsed": True,
        "original_replay_failure_preserved": True,
        "all_positive_outcome_tables_are_synthetic": True,
        "scientific_completion": False,
        "formal_outcome_produced": False,
        "formal_replication_ready": False,
        "full_downstream_integration_complete": False,
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
                "configs/dense_primary_v3_outcome_protocol.json",
                "configs/dense_primary_v3_outcome_protocol_v2.json",
                "scripts/validate_dense_v3_outcomes.py",
            )
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("Require a new receipt under an existing parent")
    value = validate(args.repository.resolve())
    write_new(args.output, value)
    print(
        {
            k: value[k]
            for k in (
                "artifact_validation_passed",
                "prepared_outcome_integration_rehearsed",
                "scientific_completion",
                "formal_replication_ready",
            )
        }
    )


if __name__ == "__main__":
    main()
