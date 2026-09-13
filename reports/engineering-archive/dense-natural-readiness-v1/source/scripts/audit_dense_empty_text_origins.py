"""Authenticate the exact pinned upstream document files for observed empty texts."""

import argparse
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download

from embed_optim.primary_contract import file_identity, read_json
from scripts.audit_dense_natural_data import write_new

REPO = "lightonai/embeddings-fine-tuning"
REVISION = "1ca463331ed637d25c1058567e932e0d3bad2983"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content-audit", type=Path, required=True)
    parser.add_argument("--content-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Require a CPU-only original-source audit and new receipt")
    if file_identity(args.content_audit)["sha256"] != args.content_sha256:
        raise ValueError("Content audit differs from the requested trusted receipt")
    content = read_json(args.content_audit)
    root = Path(
        snapshot_download(REPO, revision=REVISION, repo_type="dataset", local_files_only=True)
    )
    targets, relations = defaultdict(set), []
    for row in content["content"]["affected_rows"]:
        for column in row["invalid_text_columns"]:
            if column == "query":
                raise ValueError(
                    "This diagnostic supports document-origin checks, not query repair"
                )
            docid = (
                row["positive_id"] if column == "positive" else row["negative_ids"][int(column[9:])]
            )
            targets[row["source"]].add(docid)
            relations.append(
                {
                    "index": row["index"],
                    "source": row["source"],
                    "column": column,
                    "document_id": docid,
                }
            )
    found, files = [], {}
    for source, ids in targets.items():
        for path in sorted((root / "documents").glob(f"{source}-*.parquet")):
            parquet = pq.ParquetFile(path, memory_map=True)
            for index in range(parquet.num_row_groups):
                ids_table = parquet.read_row_group(index, columns=["document_id"])
                wanted = set(ids_table["document_id"].to_pylist()) & ids
                if not wanted:
                    continue
                values = parquet.read_row_group(index, columns=["document_id", "document"])
                for docid in sorted(wanted):
                    records = values.filter(pc.equal(values["document_id"], docid)).to_pylist()
                    if len(records) != 1 or records[0]["document"] != "":
                        raise ValueError(
                            "The observed empty text is not an exact empty upstream document"
                        )
                    relative = path.relative_to(root).as_posix()
                    found.append(
                        {
                            "source": source,
                            "document_id": docid,
                            "path": relative,
                            "row_group": index,
                            "document_is_exact_empty_string": True,
                        }
                    )
                    files[relative] = {
                        "local_path": str(path.resolve()),
                        **file_identity(path.resolve()),
                    }
    if sorted((r["source"], r["document_id"]) for r in found) != sorted(
        (source, docid) for source, ids in targets.items() for docid in ids
    ):
        raise ValueError("Upstream document identity is missing or duplicated")
    remote = HfApi().get_paths_info(
        REPO, paths=sorted(files), revision=REVISION, repo_type="dataset", expand=True
    )
    if sorted(item.path for item in remote) != sorted(files):
        raise ValueError("Immutable HF file inventory is incomplete")
    for item in remote:
        local = files[item.path]
        if item.lfs is None or item.lfs.sha256 != local["sha256"] or item.size != local["bytes"]:
            raise ValueError("Pinned HF source digest differs from the local parquet file")
        local["independent_hf_lfs_sha256"] = item.lfs.sha256
    result = {
        "scope": "engineering_exact_empty_document_origins",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "origin_audit_passed": True,
        "scientific_completion": False,
        "training_executed": False,
        "files_modified": False,
        "source_repo": REPO,
        "source_revision": REVISION,
        "content_audit": {"path": str(args.content_audit), **file_identity(args.content_audit)},
        "affected_training_cells": relations,
        "upstream_documents": found,
        "authenticated_upstream_files": files,
        "audit_source": {"path": str(Path(__file__).resolve()), **file_identity(Path(__file__))},
        "boundary": "The five observed empty document cells map to three distinct exact-empty upstream documents. Authentication is against immutable HF file LFS digests, not only local metadata. No resampling, alternative-positive selection or dataset repair is performed, and no optimizer effect is inferred.",
    }
    write_new(args.output, result)
    print(
        json.dumps(
            {
                "passed": True,
                "training_cells": len(relations),
                "unique_documents": len(found),
                "upstream_files": len(files),
                "output": str(args.output),
                **file_identity(args.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
