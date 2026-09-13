"""Derive the portable sixty-checkpoint index from authenticated original receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.restore_primary_v3 import PREFIX, PROTOCOL, REPO, RUNS, SCOPE, STEPS, validate_index

COMPLETION_NAME = (
    "observations/all-twelve-training-sixty-checkpoints-evaluation-running-20260910.json"
)
COMPLETION_SHA = "1d5aa49c6ec8c89c4454b4b9b641aa5e1bffc09a9967dbac6320cbe129d505f3"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def authenticated(path, sha):
    require(path.is_file() and not path.is_symlink(), "Require an ordinary original receipt")
    content = path.read_bytes()
    require(hashlib.sha256(content).hexdigest() == sha, f"Original receipt changed: {path}")
    return json.loads(content)


def build(launch):
    completion = authenticated(launch / COMPLETION_NAME, COMPLETION_SHA)
    accepted = completion["checkpoint_receipts_and_final_local_payloads"]["accepted"]
    require(
        accepted["all_receipts_inventory_identities_matched"] is True,
        "The original source inventory did not pass",
    )
    require(
        accepted["verified_checkpoint_receipts"] == 60 and accepted["total_files"] == 1200,
        "Incomplete primary backup source",
    )
    rows, bindings = [], []
    for cell in accepted["checkpoints"]:
        run, step = cell["run_id"], cell["step"]
        require(run in RUNS and step in STEPS, "Original source includes another campaign")
        prefix = f"{PREFIX}/{run}/checkpoint-{step}"
        name = f"{run}-step-{step}"
        receipts = {}
        for suffix in ("json", "upload.json", "audit.json"):
            filename = f"{name}.{suffix}"
            sha = cell["receipt_sha256"][filename]
            receipts[suffix] = authenticated(launch / "backup-receipts" / filename, sha)
            bindings.append({"path": f"backup-receipts/{filename}", "sha256": sha})
        native, upload, audit = (receipts[k] for k in ("json", "upload.json", "audit.json"))
        require(
            all(r["commit_oid"] == cell["commit_oid"] for r in (native, upload, audit)),
            "Immutable checkpoint commits disagree",
        )
        require(
            audit["durability_verified"] is True and audit["scientific_completion"] is False,
            "Missing real remote verification",
        )
        require(
            native["repo_id"] == REPO
            and native["repo_type"] == "model"
            and native["prefix"] == prefix,
            "Wrong checkpoint remote namespace",
        )
        require(
            native["scope"] == "dense_primary_v3_checkpoint_durability", "Wrong native backup scope"
        )
        checkpoint = native["checkpoint"]
        require(
            checkpoint["protocol_sha256"] == PROTOCOL
            and checkpoint["run_id"] == run
            and checkpoint["step"] == step,
            "Native checkpoint identity differs",
        )
        require(
            checkpoint["run_identity_sha256"] == cell["run_identity_sha256"],
            "Native run identity differs",
        )
        require(
            set(native["remote_inventory"]) == {f["path"] for f in checkpoint["files"]},
            "Native local/remote inventories differ",
        )
        files = [
            {**f, "remote": native["remote_inventory"][f["path"]]} for f in checkpoint["files"]
        ]
        require(
            sum(f["bytes"] for f in files) == cell["total_bytes"]
            and len(files) == cell["file_count"],
            "Native file totals differ",
        )
        rows.append(
            {
                "run_id": run,
                "step": step,
                "revision": native["commit_oid"],
                "prefix": prefix,
                "run_identity_sha256": checkpoint["run_identity_sha256"],
                "checkpoint_seal": checkpoint["checkpoint_seal"],
                "total_bytes": cell["total_bytes"],
                "files": sorted(files, key=lambda r: r["path"]),
                "original_receipt_sha256": cell["receipt_sha256"],
            }
        )
    result = {
        "scope": SCOPE,
        "schema_version": 1,
        "scientific_completion": False,
        "repo_id": REPO,
        "repo_type": "model",
        "protocol_sha256": PROTOCOL,
        "checkpoints": sorted(rows, key=lambda r: (r["run_id"], r["step"])),
        "total_files": 1200,
        "total_bytes": sum(r["total_bytes"] for r in rows),
        "provenance": {
            "original_completion": {
                "relative_to_experiment_launch": COMPLETION_NAME,
                "sha256": COMPLETION_SHA,
            },
            "original_receipts": bindings,
            "original_verified_at_utc": accepted["observed_at_utc"],
            "new_remote_reaudit": False,
        },
        "boundary": "Complete checkpoint-file recovery index; no main-branch download, pickle execution, training resume, full-run reconstruction or scientific admission.",
    }
    require(
        result["total_bytes"] == accepted["total_logical_bytes"] == 89889820336,
        "Whole sixty-checkpoint byte count differs",
    )
    return validate_index(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.launch)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    content = args.output.read_bytes()
    print(
        json.dumps(
            {
                "checkpoints": 60,
                "files": 1200,
                "logical_bytes": result["total_bytes"],
                "index_bytes": len(content),
                "index_sha256": hashlib.sha256(content).hexdigest(),
                "network_or_gpu_access": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
