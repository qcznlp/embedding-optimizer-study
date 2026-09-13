"""Validate actual query-partition counterexamples, preserving prior evidence."""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset

from embed_optim.primary_contract import file_identity, read_json, verify_file
from scripts.audit_dense_natural_data import handoff, write_new
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-data-partition-v1")
PRIOR = Path("reports/engineering-archive/dense-natural-readiness-v1/validation.json")
PRIOR_SHA = "3f59ac72ca18bb4614cb2df1233e9d10743b7d75f6e31a702eca9f9d85717a62"
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")


def validate(root):
    if sys.flags.optimize:
        raise ValueError("Do not disable audit assertions")
    archive = root / ARCHIVE
    assert file_identity(root / PRIOR)["sha256"] == PRIOR_SHA
    prior = read_json(root / PRIOR)
    preserved = [
        compare_binding(
            row,
            archive / "before" / row["path"] if row["path"] in DOCS else root / row["path"],
            root,
        )
        for row in prior["bindings"]
    ]
    for path in sorted((archive / "source").rglob("*")):
        if path.is_file():
            verify_file(root / path.relative_to(archive / "source"), file_identity(path))
    values = []
    for filename, sha in (
        (
            "protected-queries.json",
            "f20353ce500807a7bfe4ddf30169560bfe6ef415b082d2291060102946e37f66",
        ),
        ("string-replay.json", "4afa162215c5f16138d71d96d6e1444e9b25221032b295e3ff0c37dd53deec3f"),
        (
            "revision-proposal.json",
            "f05c32c367f2adc531eb90aa89b4a96a6b3b6af0ee0766729e38036eb8181d76",
        ),
    ):
        assert file_identity(archive / filename)["sha256"] == sha
        values.append(read_json(archive / filename))
    protection, replay, proposal = values
    assert len(protection["beir"]) == 14
    assert sum(t["query_count"] for t in protection["beir"]) == 19385
    assert len(protection["training_validation_text_overlaps"]) == 84
    assert len(protection["training_beir_query_overlaps"]) == 1
    assert not protection["validation_beir_query_overlaps"]
    assert replay["replay_passed"] is True
    assert replay["raw_equal_train_validation_pairs"] == 84
    assert replay["raw_equal_train_beir_pairs"] == 1
    assert len(replay["affected_validation_rows"]) == 51
    assert not replay["validation_internal_normalized_duplicates"]
    checks = {row["path"]: row for row in prior["external_payload_checks"]}
    for row in protection["original_inputs_unchanged"] + protection["source_bindings"]:
        checks[row["path"]] = row
    for task in protection["beir"]:
        for row in task["files"]:
            assert row["sha256"] == row["independent_hf_lfs_sha256"]
            checks[row["path"]] = row
    for row in [replay["source"], proposal["source"], *proposal["deciding_audits"]]:
        checks[row["path"]] = row
    for subset in proposal["derived_subset_impact"]:
        assert (
            subset["parent_row_values_exact"] is True and not subset["affected_parent_sample_ids"]
        )
        for row in subset["files"]:
            checks[row["path"]] = row
    external = [compare_binding(row, Path(path), root) for path, row in sorted(checks.items())]
    data = Path("/root/embedding-optimizer-study/data")
    train = Dataset.load_from_disk(str(data / "denseon-sft-500k-seed42/dataset"))
    validation = Dataset.load_from_disk(str(data / "validation-4096-seed20260826/dataset"))
    for row in replay["training_validation_pairs"]:
        left, right = (
            train[row["training"]["sample_id"]],
            validation[row["validation"]["sample_id"]],
        )
        assert left["query"] == right["query"] and row["raw_text_equal"] is True
        assert (
            left["query_id"] == row["training"]["query_id"]
            and right["query_id"] == row["validation"]["query_id"]
        )
    fever = next(t for t in protection["beir"] if t["task"] == "FEVER")
    qfile = next(r["path"] for r in fever["files"] if r["repo_path"] == "queries.parquet")
    queries = {
        str(r["_id"]): r["text"] for r in pq.read_table(qfile, columns=["_id", "text"]).to_pylist()
    }
    assert train[389365]["query"] == queries["154383"]
    assert proposal["replacement_queries_selected"] == 0 and proposal["data_modified"] is False
    assert proposal["partitions"]["training"]["replace_rows"] == 6
    assert proposal["partitions"]["validation"]["replace_rows"] == 52
    for partition, dataset in (("training", train), ("validation", validation)):
        from embed_optim.primary_contract import digest

        for row in proposal["partitions"][partition]["rows"]:
            assert digest(dataset[row["sample_id"]]) == row["original_row_sha256"]
    tests = {}
    for name, count in (("protection-tests.xml", 9), ("full-protection-tests.xml", 1606)):
        result = next(ET.parse(archive / "tests" / name).iter("testsuite")).attrib
        assert int(result["tests"]) == count
        assert all(int(result[key]) == 0 for key in ("failures", "errors", "skipped"))
        tests[name] = result
    for row in prior["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            verify_file(base / row["path"], row)
    verify_file(root / "paper/main.tex", prior["unchanged_manuscript_source"])
    verify_file(Path(prior["unchanged_main_ledger"]["path"]), prior["unchanged_main_ledger"])
    return {
        "scope": "engineering_dense_data_partition_evidence_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "query_text_partitions_disjoint": False,
        "scientific_completion": False,
        "formal_replication_ready": False,
        "production_deployed": False,
        "source_published": False,
        "replacement_queries_selected": 0,
        "gpu_workers_started": 0,
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
        + [identity(root / p, root) for p in (*DOCS, "scripts/validate_dense_partition_audit.py")],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("Require a new receipt under an existing explicit output parent")
    value = validate(args.repository.resolve())
    write_new(args.output, value)
    print(
        json.dumps(
            {
                k: value[k]
                for k in (
                    "artifact_validation_passed",
                    "query_text_partitions_disjoint",
                    "replacement_queries_selected",
                    "gpu_workers_started",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
