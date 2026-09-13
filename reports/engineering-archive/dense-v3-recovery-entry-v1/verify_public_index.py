"""Anonymous read-only metadata verification of every exact indexed HF payload."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

from scripts.restore_primary_v3 import REPO, file_identity, load_index, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve previous metadata audit")
    value = load_index(args.index, args.index_sha256)
    started = datetime.now(timezone.utc).isoformat()

    def inspect(row):
        api = HfApi(endpoint="https://huggingface.co", token=False)
        paths = [f"{row['prefix']}/{entry['path']}" for entry in row["files"]]
        actual = api.get_paths_info(
            REPO, paths, revision=row["revision"], repo_type="model", token=False
        )
        require(
            {item.path for item in actual} == set(paths) and len(actual) == 20,
            "Anonymous remote path coverage differs",
        )
        files = {}
        for item in actual:
            name = item.path.removeprefix(row["prefix"] + "/")
            files[name] = {
                "bytes": item.size,
                "kind": "sha256" if item.lfs else "git_blob_sha1",
                "digest": item.lfs.sha256 if item.lfs else item.blob_id,
            }
        require(
            files == {entry["path"]: entry["remote"] for entry in row["files"]},
            "Fresh remote metadata differs from the original index",
        )
        return {
            "run_id": row["run_id"],
            "step": row["step"],
            "revision": row["revision"],
            "prefix": row["prefix"],
            "anonymous_access": True,
            "files": files,
        }

    with ThreadPoolExecutor(max_workers=2) as executor:
        records = list(executor.map(inspect, value["checkpoints"]))
    result = {
        "scope": "primary-v3-sixty-checkpoint-anonymous-metadata-read",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "index_sha256": args.index_sha256,
        "anonymous_checkpoint_revisions": len(records),
        "remote_files": sum(len(r["files"]) for r in records),
        "records": records,
        "actual_payload_download_or_hash": False,
        "remote_mutation": False,
        "gpu_access": False,
        "scientific_completion": False,
        "source": file_identity(Path(__file__)),
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "anonymous_checkpoint_revisions": len(records),
                "remote_files": result["remote_files"],
                "receipt": file_identity(args.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
