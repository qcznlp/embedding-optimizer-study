"""Bind prepared data, independent reconstruction and all preserved predecessor evidence."""

import argparse
import hashlib
import itertools
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import file_identity, read_json, verify_file
from scripts.audit_dense_natural_data import handoff, write_new
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-data-candidate-v1")
PRIOR = Path("reports/engineering-archive/dense-data-partition-v1/validation.json")
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")


def validate(root, producer_sha, replay_sha):
    if sys.flags.optimize:
        raise ValueError("Do not disable audit assertions")
    archive = root / ARCHIVE
    assert (
        file_identity(root / PRIOR)["sha256"]
        == "ef667f2d8c3e5b860c55dbe7b08cdc2f6977214d37abd5dd40fa60855d85c730"
    )
    prior = read_json(root / PRIOR)
    preserved = [
        compare_binding(
            r, archive / "before" / r["path"] if r["path"] in DOCS else root / r["path"], root
        )
        for r in prior["bindings"]
    ]
    for p in sorted((archive / "source").rglob("*")):
        if p.is_file():
            verify_file(root / p.relative_to(archive / "source"), file_identity(p))
    assert file_identity(archive / "producer.json")["sha256"] == producer_sha
    assert file_identity(archive / "reconstruction.json")["sha256"] == replay_sha
    producer, replay = (
        read_json(archive / "producer.json"),
        read_json(archive / "reconstruction.json"),
    )
    assert replay["producer"]["sha256"] == producer_sha and replay["replay_passed"] is True
    assert all(v == 0 for v in producer["isolation"].values())
    assert all(v == 0 for v in replay["full_string_isolation"].values())
    assert all(
        r["execution_authorized"] is False
        and r["scientific_completion"] is False
        and r["gpu_workers_started"] == 0
        for r in (producer, replay)
    )
    selection = read_json(Path(producer["selection"]["path"]))
    assert len(selection["source_files"]) == 54
    checks = {r["path"]: r for r in prior["external_payload_checks"]}
    checks.update({r["path"]: r for r in replay["upstream_and_prepared_payloads_unchanged"]})
    for r in (
        producer["source"],
        producer["selection"],
        replay["source"],
        replay["producer"],
        replay["original_scanner"],
        selection["decisions"],
    ):
        checks[r["path"]] = r
    for partition, count, changes in (("training", 500000, 6), ("validation", 4096, 52)):
        left, right = producer["partitions"][partition], replay["partitions"][partition]
        assert left["rows"] == right["rows"] == count
        assert left["changed_sample_ids"] == right["changed_sample_ids"]
        assert len(left["changed_sample_ids"]) == changes
        assert left["every_non_target_value_preserved"] is True
        assert right["every_non_target_value_matches_original"] is True
        assert right["every_prepared_value_matches_independent_materialization"] is True
        assert left["content"]["affected_rows_union"] == 0
        assert left["linkage"]["every_materialized_id_matches_manifest"] is True
        for r in left["files"] + right["files"]:
            checks[r["path"]] = r
    initial = read_json(archive / "attempt-1/terminal-observation.json")
    assert initial["observed_exit_code"] == 1 and initial["datasets_materialized"] == 0
    verify_file(
        archive / "attempt-1/source/materialize_dense_partition_revision.py",
        {
            "bytes": (archive / "attempt-1/source/materialize_dense_partition_revision.py")
            .stat()
            .st_size,
            "sha256": initial["source_sha256"],
        },
    )
    verify_file(archive / "attempt-1/decisions.jsonl", initial["partial_decisions"])
    verify_file(Path(initial["partial_decisions"]["path"]), initial["partial_decisions"])
    prefix = hashlib.sha256()
    with Path(selection["decisions"]["path"]).open("rb") as stream:
        for line in itertools.islice(stream, initial["partial_decisions"]["rows"]):
            prefix.update(line)
    assert prefix.hexdigest() == initial["partial_decisions"]["sha256"]
    tests = {}
    for name, count in (
        ("revision-tests.xml", 20),
        ("revision-cached-tests.xml", 21),
        ("reconstruction-tests.xml", 5),
        ("full-revision-tests.xml", 1626),
        ("full-candidate-tests.xml", 1632),
    ):
        result = next(ET.parse(archive / "tests" / name).iter("testsuite")).attrib
        assert int(result["tests"]) == count
        assert all(int(result[k]) == 0 for k in ("failures", "errors", "skipped"))
        tests[name] = result
    for r in prior["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            verify_file(base / r["path"], r)
    verify_file(root / "paper/main.tex", prior["unchanged_manuscript_source"])
    verify_file(Path(prior["unchanged_main_ledger"]["path"]), prior["unchanged_main_ledger"])
    external = [compare_binding(r, Path(p), root) for p, r in sorted(checks.items())]
    return {
        "scope": "engineering_dense_revised_data_evidence_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "revised_full_data_content_and_exact_query_isolation_passed": True,
        "independent_source_and_materialization_replay_passed": True,
        "scientific_completion": False,
        "formal_replication_ready": False,
        "production_deployed": False,
        "source_published": False,
        "gpu_workers_started": 0,
        "replacement_groups_prepared": {"training": 6, "validation": 52},
        "interrupted_prefix_preserved_and_reproduced": True,
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
        + [identity(root / p, root) for p in (*DOCS, "scripts/validate_dense_data_candidate.py")],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--producer-sha256", required=True)
    parser.add_argument("--replay-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("Require a new receipt under an existing explicit parent")
    value = validate(args.repository.resolve(), args.producer_sha256, args.replay_sha256)
    write_new(args.output, value)
    print(
        {
            k: value[k]
            for k in (
                "artifact_validation_passed",
                "replacement_groups_prepared",
                "scientific_completion",
                "formal_replication_ready",
            )
        }
    )


if __name__ == "__main__":
    main()
