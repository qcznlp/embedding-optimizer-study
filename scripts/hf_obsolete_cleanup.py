"""Owner-directed HF cleanup. Inventory is read-only; execution needs an exact plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

REPOSITORIES = {
    "model": "qcz/embedding-optimizer-study-checkpoints",
    "dataset": "qcz/embedding-optimizer-study-analysis-artifacts",
}

MODEL_REMOVE = (
    "dense/",
    "confirmatory/",
    "short-branch/",
    "hybrid-adamw/dense/",
    "quarantine/",
    "ddp-smoke/dense/",
    "smoke/dense/",
    "stress-long/dense/",
)
DATASET_REMOVE = (
    "candidate-breadth/",
    "common-state/dense/",
    "common-state-spectra/dense/",
    "common-state-spectra/summary/",
    "confirmatory-beir/",
    "confirmatory-beir-dynamics/",
    "decontaminated-beir/dense/",
    "functional-intervention/dense/",
    "hybrid-adamw-beir/",
    "hybrid-adamw-beir-dynamics/",
    "recipe-validation/dense/",
    "short-branch/",
    "smoke-dense/",
    "spectral-transplant/dense/",
    "weight-space/dense/",
)
OLD_DENSE_LOG_GROUPS = {
    "candidate-breadth-release",
    "candidate-breadth",
    "confirmatory-dynamics-evaluation",
    "confirmatory-evaluation",
    "confirmatory-training",
    "dense-completion-pipeline",
    "dense-finalization-pipeline",
    "dense-only-runtime",
    "hybrid-adamw-dynamics-evaluation",
    "hybrid-adamw-evaluation",
    "hybrid-adamw-training",
    "post-eval-pipeline",
    "short-branch-training",
    "short-branch",
    "spectral-transplant",
    "tail-stability",
    "evaluation",
}


def removal_reason(repo_type, path):
    """Conservative path rules compiled into an explicit, digest-bound file plan."""
    parts = path.split("/")
    if path.startswith("/") or any(p in {"", ".", ".."} for p in parts):
        raise ValueError(f"Unsafe repository path: {path!r}")
    if path.startswith(
        (
            "corrected-dense-no-packing-v1/",
            "primary-dimension/",
            "state-operator-factorial/",
            "project/data/",
            "project/configs/",
        )
    ):
        return None
    if repo_type == "model":
        return (
            "invalidated historical training or quarantined implementation"
            if path.startswith(MODEL_REMOVE)
            else None
        )
    if repo_type != "dataset":
        raise ValueError("Unsupported repository type")
    if path.startswith(DATASET_REMOVE):
        return "derived from invalidated Dense training or its affected analysis path"
    if path.startswith("representation-space/"):
        # These independently encoded, untrained baseline files are not invalidated runs.
        if (
            path.startswith("representation-space/decontaminated-beir/")
            and "/dense/pretrained" in path
        ):
            return None
        if "/dense/" in path or "/summary/" in path or "/smoke/dense-" in path:
            return "historical Dense representation output or mixed summary containing it"
    if path.startswith("project/reports/"):
        if path.startswith(("project/reports/confirmatory-data/", "project/reports/figures/late-")):
            return None
        return "historical Dense or mixed result-report snapshot"
    if path.startswith("project/logs/"):
        relative = path[len("project/logs/") :]
        # Preserve separately named LateOn log payloads even inside mixed controllers.
        if "late" in relative.lower():
            return None
        if (
            "dense" in relative.lower()
            or parts[2] in OLD_DENSE_LOG_GROUPS
            or "/quarantine/" in path
            or relative.startswith(
                (
                    "formal-checkpoint-",
                    "evaluation-handoff.",
                    "evaluation-supervisor.",
                    "evaluation-live-audit.",
                )
            )
            or relative in {"training/final-six-supervisor.log", "training/matrix-supervisor.log"}
        ):
            return "historical Dense result-bearing execution log"
    return None


def save_new(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError(f"Refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
    return hashlib.sha256(raw).hexdigest()


def build_plan(before, before_sha):
    repositories = []
    for snapshot in before["repositories"]:
        delete = [
            dict(row, reason=reason)
            for row in snapshot["files"]
            if (reason := removal_reason(snapshot["repo_type"], row["path"]))
        ]
        repositories.append(
            {
                **snapshot,
                "delete": delete,
                "delete_files": len(delete),
                "delete_bytes": sum(row["size"] for row in delete),
            }
        )
    return {
        "operation": "delete_explicit_invalidated_files_from_main_only",
        "history_purge_authorized": False,
        "before_sha256": before_sha,
        "repositories": repositories,
    }


def execute(api, plan, output_dir, cards_dir):
    """One compare-and-swap commit per repo, then full preserved-file verification."""
    for entry in plan["repositories"]:
        kind = entry["repo_type"]
        if entry["repo_id"] != REPOSITORIES[kind]:
            raise ValueError("Repository does not match the fixed allowlist")
        receipt_path = output_dir / f"{kind}-execution.json"
        if receipt_path.exists():
            raise ValueError("Existing execution receipt; audit it rather than executing twice")
        snapshot = inventory(api, kind)
        now = {row["path"]: row for row in snapshot["files"]}
        planned = {row["path"]: row for row in entry["delete"]}
        if len(planned) != len(entry["delete"]) or not planned:
            raise ValueError("Duplicate or empty deletion plan")
        for path, row in planned.items():
            if not removal_reason(kind, path):
                raise ValueError(f"Plan tries to delete a protected/unknown path: {path}")
            if now.get(path) != {key: value for key, value in row.items() if key != "reason"}:
                raise ValueError(f"Deletion target changed or disappeared: {path}")
        if {p for p in now if removal_reason(kind, p)} != set(planned):
            raise ValueError("Invalidated namespace changed; create a new reviewed inventory")
        card = (cards_dir / f"{kind}-README.md").read_bytes()
        save_new(output_dir / f"{kind}-execution-before.json", snapshot)
        operations = [
            CommitOperationDelete(path_in_repo=p, is_folder=False) for p in sorted(planned)
        ]
        operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card))
        commit = api.create_commit(
            repo_id=entry["repo_id"],
            repo_type=kind,
            revision="main",
            parent_commit=snapshot["revision"],
            operations=operations,
            commit_message="Remove owner-withdrawn invalidated Dense artifacts; preserve current replication",
            commit_description="Exact reviewed paths only. Current replication, shared data and separately retained LateOn artifacts are unchanged. History is not purged by this commit.",
        )
        # Write the commit immediately so a later network failure cannot obscure its identity.
        save_new(
            output_dir / f"{kind}-commit.json",
            {"commit_oid": commit.oid, "parent": snapshot["revision"], "repo_id": entry["repo_id"]},
        )
        after = inventory(api, kind, commit.oid)
        saved_after = save_new(output_dir / f"{kind}-execution-after.json", after)
        remaining = {row["path"]: row for row in after["files"]}
        preserved = {p: r for p, r in now.items() if p not in planned and p != "README.md"}
        mismatches = [p for p, row in preserved.items() if remaining.get(p) != row]
        unexpected = sorted(set(remaining) - set(preserved) - {"README.md"})
        surviving = sorted(set(planned) & set(remaining))
        card_blob = hashlib.sha1(b"blob " + str(len(card)).encode() + b"\0" + card).hexdigest()
        card_valid = remaining.get("README.md", {}).get("blob_id") == card_blob
        verified = not (mismatches or unexpected or surviving) and card_valid
        receipt = {
            "verified": verified,
            "repo_id": entry["repo_id"],
            "repo_type": kind,
            "commit_oid": commit.oid,
            "parent_revision": snapshot["revision"],
            "deleted_files": len(planned),
            "deleted_bytes": sum(r["size"] for r in planned.values()),
            "preserved_files_verified": len(preserved),
            "preserved_bytes_verified": sum(r["size"] for r in preserved.values()),
            "preserved_mismatches": mismatches,
            "unexpected_paths": unexpected,
            "surviving_deleted_paths": surviving,
            "card_verified": card_valid,
            "after_inventory_sha256": saved_after,
            "history_purged": False,
            "local_artifacts_deleted": False,
        }
        save_new(receipt_path, receipt)
        print(json.dumps(receipt), flush=True)
        if not verified:
            raise ValueError("Post-deletion verification failed; stop and inspect the receipt")


def inventory(api, repo_type, revision=None):
    repo_id = REPOSITORIES[repo_type]
    revision = revision or api.repo_info(repo_id, repo_type=repo_type).sha
    files = []
    for item in api.list_repo_tree(repo_id, repo_type=repo_type, revision=revision, recursive=True):
        if not hasattr(item, "size"):
            continue
        lfs = getattr(item, "lfs", None)
        files.append(
            {
                "path": item.path,
                "size": item.size,
                "blob_id": item.blob_id,
                "lfs_sha256": getattr(lfs, "sha256", None),
            }
        )
    return {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision,
        "files": sorted(files, key=lambda row: row["path"]),
    }


def summarize(snapshot, depth):
    groups = defaultdict(lambda: {"files": 0, "bytes": 0})
    for row in snapshot["files"]:
        group = "/".join(row["path"].split("/")[:depth])
        groups[group]["files"] += 1
        groups[group]["bytes"] += row["size"]
    return dict(sorted(groups.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--plan-from", type=Path)
    parser.add_argument("--execute-plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--receipt-dir", type=Path)
    parser.add_argument("--cards-dir", type=Path)
    args = parser.parse_args()
    if args.execute_plan:
        raw = args.execute_plan.read_bytes()
        if not args.plan_sha256 or hashlib.sha256(raw).hexdigest() != args.plan_sha256:
            raise ValueError("Execution requires the exact reviewed plan SHA-256")
        if not args.receipt_dir or not args.cards_dir:
            raise ValueError("Receipt and card directories are required")
        execute(HfApi(), json.loads(raw), args.receipt_dir, args.cards_dir)
        return
    if not args.output:
        parser.error("--output is required for an inventory or plan")
    if args.plan_from:
        raw = args.plan_from.read_bytes()
        plan = build_plan(json.loads(raw), hashlib.sha256(raw).hexdigest())
        digest = save_new(args.output, plan)
        print(
            json.dumps(
                {
                    "plan": str(args.output),
                    "sha256": digest,
                    "repositories": [
                        {k: e[k] for k in ("repo_id", "delete_files", "delete_bytes")}
                        for e in plan["repositories"]
                    ],
                },
                indent=2,
            )
        )
        return
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing inventory receipt")
    api = HfApi()
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "read_only_inventory",
        "repositories": [inventory(api, kind) for kind in REPOSITORIES],
    }
    raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({"receipt": str(args.output), "sha256": hashlib.sha256(raw).hexdigest()}))
    for snapshot in report["repositories"]:
        print(
            json.dumps(
                {
                    "repo_id": snapshot["repo_id"],
                    "revision": snapshot["revision"],
                    "groups": summarize(snapshot, args.depth),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
