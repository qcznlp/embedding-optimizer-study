"""Read-only handoff validation retaining the failed natural-data readiness gate."""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import Dataset

from embed_optim.primary_contract import file_identity, read_json, verify_file
from scripts.audit_dense_natural_data import handoff, select_prefix, write_new
from scripts.validate_dense_gpu_replay_localization import compare_binding, identity

ARCHIVE = Path("reports/engineering-archive/dense-natural-readiness-v1")
PRIOR = Path("reports/engineering-archive/dense-primary-completion-v1/validation.json")
PRIOR_SHA = "516f7412375c5e6a4411c0a0293e0711fca1a84e27499e1ba036f9f444482701"
DOCS = ("AGENTS.md", "PROJECT_STATUS.md", "README.md", "CURRENT_PROGRESS.json")


def validate(root):
    if sys.flags.optimize:
        raise ValueError("Do not disable validation assertions")
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
    receipts = {}
    for name, sha in (
        (
            "full-data-content.json",
            "fbfc06240961e943bfa6869cec9a70b40c7ea98415fb8963e495915faa1ac66d",
        ),
        ("empty-origins.json", "f04322b8e59879796f2454710a9537d60dec29eef8abb2ed0b9585dc9048acaa"),
        (
            "validation-content.json",
            "a26438d638b0320f1303454e324697166c129aa512e38b632cb5f363320de5b2",
        ),
    ):
        assert file_identity(archive / name)["sha256"] == sha
        receipts[name] = read_json(archive / name)
    training, origin, validation = (
        receipts[k]
        for k in ("full-data-content.json", "empty-origins.json", "validation-content.json")
    )
    assert training["content"]["rows"] == 500000 and training["content"]["text_cells"] == 4500000
    assert training["content"]["invalid_text_rows"] == 5
    assert [r["index"] for r in training["content"]["affected_rows"]] == [
        230,
        634,
        750,
        13096,
        336061,
    ]
    assert validation["content"]["rows"] == 4096 and validation["content"]["invalid_text_rows"] == 1
    assert validation["content"]["affected_rows"][0]["index"] == 46
    assert validation["frozen_validation_audit"]["query_overlap_with_training"] == 0
    for value, count in ((training, 500000), (validation, 4096)):
        assert value["scientific_completion"] is False
        assert value["actual_materialized_id_linkage"]["rows_compared"] == count
        assert (
            value["actual_materialized_id_linkage"]["every_materialized_id_matches_manifest"]
            is True
        )
        assert value["content"]["positive_negative_identical_text_rows"] == 0
        assert value["content"]["repeated_negative_text_rows"] == 0
    external = []
    for row in (
        training["authenticated_inputs_unchanged"] + validation["authenticated_files_unchanged"]
    ):
        external.append(compare_binding(row, Path(row["path"]), root))
    for row in [
        training["source"],
        training["source_materializer"],
        origin["audit_source"],
        *validation["source_bindings"],
    ]:
        verify_file(Path(row["path"]), row)
    assert origin["origin_audit_passed"] is True and len(origin["upstream_documents"]) == 3
    assert len(origin["affected_training_cells"]) == 5
    for relative, row in origin["authenticated_upstream_files"].items():
        assert row["independent_hf_lfs_sha256"] == row["sha256"]
        verify_file(Path(row["local_path"]), row)
        external.append({"path": row["local_path"], **file_identity(Path(row["local_path"]))})
        for document in (d for d in origin["upstream_documents"] if d["path"] == relative):
            table = pq.ParquetFile(row["local_path"]).read_row_group(
                document["row_group"], columns=["document_id", "document"]
            )
            match = table.filter(
                pc.equal(table["document_id"], document["document_id"])
            ).to_pylist()
            assert len(match) == 1 and match[0]["document"] == ""
    data = Dataset.load_from_disk(
        "/root/embedding-optimizer-study/data/denseon-sft-500k-seed42/dataset"
    )
    assert data[230]["positive"] == "" and data[230]["positive_id"] == 741
    try:
        select_prefix(data)
    except ValueError as error:
        assert str(error) == "A diagnostic row lacks one of its nine original texts"
    else:
        raise AssertionError("The real invalid prefix no longer fails")
    attempt = read_json(archive / "failed-prefix-attempt.json")
    assert attempt["observed_exit_code"] == 1 and attempt["gpu_workers_started"] == 0
    assert not list(Path(attempt["workdir"]).iterdir())
    tests = {}
    for filename, count in (("focused-content.xml", 59), ("full-content.xml", 1597)):
        attributes = next(ET.parse(archive / "tests" / filename).iter("testsuite")).attrib
        assert int(attributes["tests"]) == count
        assert all(int(attributes[key]) == 0 for key in ("errors", "failures", "skipped"))
        tests[filename] = attributes
    for row in prior["unchanged_live_core"]:
        for base in (root, Path("/root/embedding-optimizer-study")):
            verify_file(base / row["path"], row)
    verify_file(root / "paper/main.tex", prior["unchanged_manuscript_source"])
    verify_file(Path(prior["unchanged_main_ledger"]["path"]), prior["unchanged_main_ledger"])
    dispatchers = handoff()
    return {
        "scope": "engineering_dense_natural_readiness_evidence_validation",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "artifact_validation_passed": True,
        "natural_data_readiness_passed": False,
        "scientific_completion": False,
        "production_deployed": False,
        "source_published": False,
        "formal_replication_ready": False,
        "gpu_workers_started": 0,
        "summary": {
            "training_rows_checked": 500000,
            "validation_rows_checked": 4096,
            "invalid_training_groups": 5,
            "invalid_validation_groups": 1,
            "upstream_empty_documents": 3,
            "primary_results_added": 0,
        },
        "prior_bindings_preserved": preserved,
        "external_payload_checks": external,
        "tests": tests,
        "unchanged_live_core": prior["unchanged_live_core"],
        "unchanged_manuscript_source": prior["unchanged_manuscript_source"],
        "unchanged_main_ledger": prior["unchanged_main_ledger"],
        "post_execution_dispatchers": dispatchers,
        "bindings": [
            identity(p, root)
            for p in sorted(archive.rglob("*"))
            if p.is_file() and p.name != "validation.json"
        ]
        + [
            identity(root / p, root) for p in (*DOCS, "scripts/validate_dense_natural_readiness.py")
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("Use a new evidence receipt")
    value = validate(args.repository.resolve())
    write_new(args.output, value)
    print(
        json.dumps(
            {
                k: value[k]
                for k in (
                    "artifact_validation_passed",
                    "natural_data_readiness_passed",
                    "gpu_workers_started",
                    "summary",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
