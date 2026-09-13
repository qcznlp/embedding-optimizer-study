"""Validate v3 validation scoring/readback/admission evidence without scientific promotion."""

import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_validation import ValidationContract
from scripts.audit_dense_natural_data import handoff, write_new
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-v3-validation-v1")
PRIOR = Path("reports/engineering-archive/dense-primary-v3-chain-v1/validation.json")
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")


def validate(root):
    if sys.flags.optimize:
        raise ValueError("Do not disable evidence assertions")
    archive = root / ARCHIVE
    assert (
        file_identity(root / PRIOR)["sha256"]
        == "efa06bd154e6e67035654e214b1a278eede3a74e6c24cb8f5ff48183feb4bdf8"
    )
    prior = read_json(root / PRIOR)
    preserved = [
        compare_binding(
            r, archive / "before" / r["path"] if r["path"] in DOCS else root / r["path"], root
        )
        for r in prior["bindings"]
    ]
    for path in (archive / "source").rglob("*"):
        if path.is_file():
            verify_file(root / path.relative_to(archive / "source"), file_identity(path))
    for name, sha in (
        ("protocol.json", "96a8aa2fc684d29ec67649c2e637c0cd4c7dcdd0f31f2e354b7d0955a18fb1c3"),
        ("gpu-result.json", "51936005e30ddd2519fba66e5d6847abaaee25cb0ad3d6eb0d4e2e6294b88200"),
        ("admission.json", "40107d077d154da07e3707b02624d9be4757dcf8f45f6a5caa72e51b3a6752ae"),
    ):
        assert file_identity(archive / name)["sha256"] == sha
    primary = PrimaryV3Contract.load(
        root / "configs/dense_primary_v3_protocol.json", root, "/tmp/dense-identity-source.8rUgLF"
    )
    validation = ValidationContract.load(
        root / "configs/dense_primary_v3_validation_protocol.json", primary
    )
    require_same(validation.payload, read_json(archive / "protocol.json"))
    _, data = validation.data("/tmp/dense-partition-candidate.kHXoGW/validation")
    plan = read_json(archive / "plan.json")
    require_same(data, plan["complete_validation_admission"])
    assert len(data["row_identities"]) == 4096
    selection = plan["selection"]
    assert selection["rows"] == 59 and len(selection["row_identities"]) == 59
    assert len(selection["mandatory_replacement_positions"]) == 52
    assert set(selection["mandatory_replacement_positions"]).issubset(selection["positions"])
    assert len(selection["source_counts"]) == 7
    assert selection["max_truncated_token_length"] == 1379
    gpu = read_json(archive / "gpu-result.json")
    admission = read_json(archive / "admission.json")
    assert gpu["readiness_passed"] is True and gpu["worker_returncode"] == 0
    assert gpu["formal_validation_executed"] is False and gpu["model_updates"] == 0
    assert len(gpu["worker"]["records"]) == len(gpu["cpu_readback"]) == 3
    for row in gpu["worker"]["records"]:
        assert row["rows"] == 59
        assert row["all_scores_and_metrics_equal_unchanged_scorer"] is True
        assert row["all_134_parameter_values_unchanged"] is True
        assert row["saved"]["summary"]["raw_metrics_preserved"] is True
    assert len(admission["fresh_cpu_diagnostic_readbacks"]) == 3
    assert len(admission["changed_cases"]) == 18 and admission["all_changed_cases_rejected"] is True
    assert all(r["rejected"] is True for r in admission["changed_cases"])
    assert admission["actual_missing_primary_selection"]["returncode"] != 0
    assert (
        "Require an ordinary retained run directory"
        in admission["actual_missing_primary_selection"]["stderr"]
    )
    assert admission["primary_recipe_selection_produced"] is False
    assert admission["model_updates"] == admission["gpu_workers"] == 0
    checks = {r["path"]: r for r in prior["external_payload_checks"]}
    for row in (
        *admission["unchanged_inputs_and_producer_artifacts"],
        admission["source"],
        *[a for r in admission["changed_cases"] for a in r["artifacts"]],
    ):
        checks[row["path"]] = row
    external = [compare_binding(r, Path(p), root) for p, r in sorted(checks.items())]
    tests = {}
    for name, count in (
        ("focused-tests.xml", 65),
        ("focused-tests-final.xml", 67),
        ("full-tests.xml", 1777),
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
        "scope": "engineering_dense_v3_validation_evidence_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "prepared_validation_scorer_reader_selector_verified": True,
        "diagnostic_models": 3,
        "diagnostic_rows_per_model": 59,
        "maximum_diagnostic_input_length": 1379,
        "scientific_completion": False,
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
                "configs/dense_primary_v3_validation_protocol.json",
                "scripts/validate_dense_v3_validation.py",
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
            k: value[k]
            for k in (
                "artifact_validation_passed",
                "prepared_validation_scorer_reader_selector_verified",
                "scientific_completion",
                "formal_replication_ready",
            )
        }
    )


if __name__ == "__main__":
    main()
