#!/usr/bin/env python3
"""Compare completed local runs to their immutable Hugging Face upload commits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

from embed_optim.incremental_checkpoint_backup import (
    compare_checkpoint_inventories,
    local_checkpoint_inventory,
    stat_signature,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--training-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    training_bytes = args.training_audit.read_bytes()
    training = json.loads(training_bytes)
    if not training.get("audited_portion_valid"):
        raise ValueError("The completed local training audit must pass first")
    api = HfApi()
    records = []
    for run_id in training["completed_run_ids"]:
        receipt_path = root / "reports/dense-no-packing/checkpoint-backup" / f"{run_id}.json"
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
        if receipt.get("status") != "complete" or receipt.get("run_id") != run_id:
            raise ValueError(f"Invalid original upload receipt for {run_id}")
        local_root = root / receipt["local_root"]
        before = stat_signature(local_root)
        print(f"Hashing local payloads and verifying upload commit for {run_id}", flush=True)
        local = local_checkpoint_inventory(local_root)
        remote = {}
        for entry in api.list_repo_tree(
            receipt["repo_id"],
            revision=receipt["commit_oid"],
            path_in_repo=receipt["remote_prefix"],
            recursive=True,
            expand=True,
            repo_type="model",
        ):
            if not isinstance(entry, RepoFile):
                continue
            key = str(Path(entry.path).relative_to(receipt["remote_prefix"]))
            kind = "sha256" if entry.lfs is not None else "git_blob_sha1"
            remote[key] = {
                "size": entry.size,
                "digest_kind": kind,
                "digest": entry.lfs.sha256 if entry.lfs is not None else entry.blob_id,
            }
        if stat_signature(local_root) != before:
            raise RuntimeError(f"Completed local run changed during audit: {run_id}")
        comparison = compare_checkpoint_inventories(local, remote)
        records.append(
            {
                "run_id": run_id,
                "repo_id": receipt["repo_id"],
                "revision": receipt["commit_oid"],
                "prefix": receipt["remote_prefix"],
                "original_upload_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                "comparison": comparison,
                "local_inventory": local,
                "remote_inventory": remote,
            }
        )
        print(json.dumps({"run_id": run_id, **comparison}), flush=True)
    complete = bool(records) and all(row["comparison"]["complete"] for row in records)
    output = {
        "schema_version": 1,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "audit_scope": "immutable remote file content versus completed local runs",
        "complete_for_audited_runs": complete,
        "scientific_completion": False,
        "runs": len(records),
        "files": sum(row["comparison"]["local_files"] for row in records),
        "bytes": sum(row["comparison"]["local_bytes"] for row in records),
        "training_audit_sha256": hashlib.sha256(training_bytes).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(
            (root / "src/embed_optim/incremental_checkpoint_backup.py").read_bytes()
        ).hexdigest(),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
