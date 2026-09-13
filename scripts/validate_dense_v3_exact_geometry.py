"""Validate complete exact-geometry preparation and all-hidden diagnostic evidence."""

import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_exact_geometry import ExactGeometryContract
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-v3-exact-geometry-v1")
PRIOR = Path("reports/engineering-archive/dense-geometry-robustness-v1/validation.json")
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")
PROTOCOL_SHA = "b13351152e6bd4b8ec2562d46c4e8bd2c1c2496bfd4bae7d4dabd093c47cb9f5"
COUNTS = {
    "checkpoint_exact_geometry": 9,
    "run_pair_exact_subspace_overlap": 18,
    "optimizer_pair_exact_subspace_summary": 18,
    "exact_matrix_geometry": 1584,
}


def validate(root, audit_path, audit_sha, replay_path, replay_sha):
    if sys.flags.optimize:
        raise ValueError("Evidence assertions must not be disabled")
    archive = root / ARCHIVE
    assert (
        file_identity(root / PRIOR)["sha256"]
        == "03165391e649ad37db2cb182a92165999f0682fe30898154da69206a60681dfb"
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
    primary = PrimaryV3Contract.load(
        root / "configs/dense_primary_v3_protocol.json", root, "/tmp/dense-identity-source.8rUgLF"
    )
    contract = ExactGeometryContract.load(
        root / "configs/dense_primary_v3_exact_geometry_protocol.json", primary
    )
    assert contract.sha256 == PROTOCOL_SHA
    require_same(contract.payload, read_json(archive / "protocol.json"))
    assert file_identity(audit_path)["sha256"] == audit_sha
    assert file_identity(replay_path)["sha256"] == replay_sha
    verify_file(archive / "audit.json", file_identity(audit_path))
    verify_file(archive / "fresh-replay.json", file_identity(replay_path))
    audit, replay = read_json(audit_path), read_json(replay_path)
    assert (
        audit["actual_exact_geometry_verified"] is replay["actual_exact_geometry_verified"] is True
    )
    assert (
        audit["fresh_numeric_replay_passed"] is None
        and replay["fresh_numeric_replay_passed"] is True
    )
    assert replay["replay_parent"]["sha256"] == audit_sha
    for result in (audit, replay):
        assert result["models"] == 3 and result["checkpoints"] == 9
        require_same(result["table_counts"], COUNTS)
        assert result["protocol"]["sha256"] == PROTOCOL_SHA
        assert result["model_updates"] == result["gpu_workers"] == 0
        assert result["formal_exact_geometry_produced"] is result["scientific_completion"] is False
        assert len(result["readers"]) == 3
        assert sum(len(row["matrix_rows"]) for row in result["readers"]) == 1584
        assert sum(len(row["checkpoint_rows"]) for row in result["readers"]) == 9
        assert sum(result["basis_status_counts"].values()) == 1584
        assert result["basis_status_counts"]["zero"] == 528
        require_same(handoff(), result["post_execution_dispatchers"])
    require_same(audit["readers"], replay["readers"])
    require_same(audit["basis_status_counts"], replay["basis_status_counts"])
    independent = audit["independent_full_matrix_controls"]
    assert len(independent) == 3 and sum(len(row["stages"]) for row in independent) == 9
    assert sum(row["matrix_kind_records"] for row in independent) == 1584
    assert sum(row["zero_records"] for row in independent) == 528
    assert sum(row["nonzero_spectra"] for row in independent) == 1056
    assert (
        sum(row["resolved_projectors"] for row in independent)
        == audit["basis_status_counts"]["resolved"]
    )
    assert (
        sum(row["unresolved_nonzero"] for row in independent)
        == audit["basis_status_counts"]["insufficient_signal_rank"]
        + audit["basis_status_counts"]["unresolved_boundary"]
    )
    assert max(row["max_projector_rms_sine"] for row in independent) <= 2e-7
    for result, expected_changes in ((audit, 8), (replay, 2)):
        assert len(result["changed_cases"]) == expected_changes
        assert all(row["rejected"] for row in result["changed_cases"])
    assert {row["action"] for row in audit["actual_missing_primary_cli"]} == {"produce", "inspect"}
    for row in audit["actual_missing_primary_cli"]:
        assert row["returncode"] != 0 and "ordinary retained run" in row["stderr"]
        assert row["output_created"] is False
    checks = {row["path"]: row for row in prior["external_payload_checks"]}
    probe_path = archive / "bridge-null-feature.json"
    assert (
        file_identity(probe_path)["sha256"]
        == "5c8d8e3fbd0f9a13c3bd6a4fbcd23a2bcc6f2f05d40ac42aa12fca07e77067b0"
    )
    probe = read_json(probe_path)
    assert (
        probe["bridge_correctness_accepted"]
        is probe["bridge_source_changed"]
        is probe["scientific_completion"]
        is False
    )
    assert [case["case"] for case in probe["cases"]] == [
        "initial_stage_float_outcome",
        "bounded_dyadic_null",
    ]
    for case in probe["cases"]:
        assert len(case["rows"]) == 60 and case["baseline_rank"] == case["augmented_rank"] == 8
        baseline = np.array(
            [
                [
                    1.0,
                    float(row["optimizer"] == "muon"),
                    float(row["optimizer"] == "normuon"),
                    *[float(row["stage"] == stage) for stage in range(2, 6)],
                    row["centered_log10_learning_rate"],
                ]
                for row in case["rows"]
            ]
        )
        values = np.array([row["log_saved_segment_to_weight_ratio"] for row in case["rows"]])
        assert np.array_equal(baseline @ np.array(case["closed_form_feature_coefficients"]), values)
        if case["closed_form_outcome_coefficients"] is not None:
            assert np.array_equal(
                baseline @ np.array(case["closed_form_outcome_coefficients"]),
                np.array([row["mean_ndcg_at_10"] for row in case["rows"]]),
            )
    assert probe["cases"][0]["legacy_predicted_useful_count"] == 0
    assert probe["cases"][1]["legacy_predicted_useful_count"] == 9
    assert all(case["legacy_finite_residual_associations"] == 9 for case in probe["cases"])
    for row in (probe["source"], probe["legacy_source"]):
        verify_file(row["path"], row)
        checks[row["path"]] = row
    for result in (audit, replay):
        for row in (*result["inputs"], *result["artifacts"]):
            checks[row["path"]] = row
    for path in (audit_path, replay_path):
        checks[str(path)] = {"path": str(path), **file_identity(path)}
    external = [compare_binding(row, Path(path), root) for path, row in sorted(checks.items())]
    for name in COUNTS:
        verify_file(archive / f"{name}.csv", file_identity(audit_path.parent / f"{name}.csv"))
    tests = {}
    for name, count in (("focused-tests.xml", 45), ("full-tests.xml", 2051)):
        values = next(ET.parse(archive / name).iter("testsuite")).attrib
        assert int(values["tests"]) == count
        assert all(int(values[key]) == 0 for key in ("failures", "errors", "skipped"))
        tests[name] = values
    for row in prior["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            verify_file(base / row["path"], row)
    verify_file(root / "paper/main.tex", prior["unchanged_manuscript_source"])
    verify_file(Path(prior["unchanged_main_ledger"]["path"]), prior["unchanged_main_ledger"])
    return {
        "scope": "engineering_dense_v3_exact_geometry_evidence_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "complete_exact_geometry_consumer_prepared": True,
        "all_hidden_diagnostic_spectra_independently_verified": True,
        "fresh_numerical_replay_verified": True,
        "legacy_bridge_null_feature_issue_confirmed": True,
        "legacy_bridge_null_feature_issue_repaired": False,
        "formal_primary_exact_geometry_observed": False,
        "formal_replication_ready": False,
        "scientific_completion": False,
        "accepted_audit_sha256": audit_sha,
        "accepted_replay_sha256": replay_sha,
        "prior_bindings_preserved": preserved,
        "external_payload_checks": external,
        "basis_status_counts": audit["basis_status_counts"],
        "tests": tests,
        "unchanged_live_core": prior["unchanged_live_core"],
        "unchanged_manuscript_source": prior["unchanged_manuscript_source"],
        "unchanged_main_ledger": prior["unchanged_main_ledger"],
        "post_execution_dispatchers": handoff(),
        "bindings": [
            identity(path, root)
            for path in sorted(archive.rglob("*"))
            if path.is_file() and path.name != "validation.json"
        ]
        + [
            identity(root / name, root)
            for name in (
                *DOCS,
                "configs/dense_primary_v3_exact_geometry_protocol.json",
                "scripts/validate_dense_v3_exact_geometry.py",
            )
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "audit", "replay", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("audit-sha256", "replay-sha256"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("Require a new receipt under an explicit existing parent")
    result = validate(
        args.repository.resolve(),
        args.audit.resolve(),
        args.audit_sha256,
        args.replay.resolve(),
        args.replay_sha256,
    )
    write_new(args.output, result)
    print(
        {
            key: result[key]
            for key in (
                "artifact_validation_passed",
                "complete_exact_geometry_consumer_prepared",
                "scientific_completion",
            )
        }
    )


if __name__ == "__main__":
    main()
