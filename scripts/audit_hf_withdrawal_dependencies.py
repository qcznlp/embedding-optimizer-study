"""Read-only withdrawal preflight; no deletion, history rewrite, or receipt migration."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi

from scripts.hf_obsolete_cleanup import REPOSITORIES, inventory, removal_reason, save_new

RECEIPT_DIRS = (
    "reports/dense-no-packing/checkpoint-backup",
    "reports/dense-no-packing/incremental-checkpoint-backup",
    "reports/dense-no-packing-weight-space",
)


def identity(path: Path, root: Path) -> dict:
    raw = path.read_bytes()
    return {
        "path": str(path.relative_to(root)),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def repository_dependencies(planned: dict, snapshot: dict) -> dict:
    """Find shared objects, never infer purge permission from disjoint path names."""
    kind = planned["repo_type"]
    if snapshot["repo_type"] != kind or planned["repo_id"] != REPOSITORIES[kind]:
        raise ValueError("Repository identity mismatch")
    if snapshot["repo_id"] != planned["repo_id"]:
        raise ValueError("Snapshot repository mismatch")
    rows = {r["path"]: r for r in snapshot["files"]}
    targets = {r["path"]: r for r in planned["delete"]}
    if len(rows) != len(snapshot["files"]) or len(targets) != len(planned["delete"]):
        raise ValueError("Duplicate path in snapshot or plan")
    for path, row in targets.items():
        if not removal_reason(kind, path):
            raise ValueError(f"Protected deletion target: {path}")
        if rows.get(path) != {k: v for k, v in row.items() if k != "reason"}:
            raise ValueError(f"Changed deletion target: {path}")
    if {p for p in rows if removal_reason(kind, p)} != set(targets):
        raise ValueError("Invalidated namespace changed since the reviewed plan")
    groups = defaultdict(list)
    for row in rows.values():
        if row.get("lfs_sha256"):
            groups[row["lfs_sha256"]].append(row)
    shared = []
    for digest, uses in sorted(groups.items()):
        if len({row["size"] for row in uses}) != 1:
            raise ValueError("One LFS identity has conflicting byte counts")
        removed = sorted(row["path"] for row in uses if row["path"] in targets)
        retained = sorted(row["path"] for row in uses if row["path"] not in targets)
        if removed and retained:
            shared.append(
                {
                    "sha256": digest,
                    "bytes": uses[0]["size"],
                    "withdrawn_paths": removed,
                    "retained_paths": retained,
                }
            )
    return {
        "repo_id": snapshot["repo_id"],
        "repo_type": kind,
        "observed_revision": snapshot["revision"],
        "reviewed_targets_unchanged": True,
        "reviewed_delete_files": len(targets),
        "reviewed_delete_logical_bytes": sum(row["size"] for row in targets.values()),
        "protected_files": len(rows) - len(targets),
        "shared_lfs_objects": shared,
        "shared_lfs_count": len(shared),
        "shared_lfs_unique_bytes": sum(row["bytes"] for row in shared),
        "unreferenced_history_objects_enumerated": False,
        "safe_to_purge": False,
    }


def declared_reference(path: Path, root: Path) -> dict | None:
    raw = path.read_bytes()
    receipt = json.loads(raw)
    if "repo_id" not in receipt:
        return None  # Non-backup report in the bounded analysis directory.
    kind = receipt.get("repo_type")
    if kind not in REPOSITORIES or receipt["repo_id"] != REPOSITORIES[kind]:
        raise ValueError(f"Unexpected receipt repository: {path}")
    prefix = receipt.get("remote_prefix", "")
    if not prefix.startswith("corrected-dense-no-packing-v1/"):
        raise ValueError(f"Unexpected receipt namespace: {path}")
    if any(part in {"", ".", ".."} for part in prefix.split("/")):
        raise ValueError(f"Unsafe receipt namespace: {path}")
    url_prefix = "https://huggingface.co/" + ("datasets/" if kind == "dataset" else "")
    url_prefix += receipt["repo_id"] + "/commit/"
    url = receipt.get("commit_url", "")
    if not url.startswith(url_prefix):
        raise ValueError(f"Receipt commit URL mismatch: {path}")
    commit = url[len(url_prefix) :]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"Receipt does not pin an immutable commit: {path}")
    if receipt.get("commit_oid", commit) != commit:
        raise ValueError(f"Receipt commit identity mismatch: {path}")
    check = receipt.get("inventory", {})
    if (
        receipt.get("status") not in {"complete", "partial_complete"}
        or check.get("complete") is not True
    ):
        raise ValueError(f"Receipt does not report a complete upload: {path}")
    if (
        not isinstance(check.get("remote_files"), int)
        or check["remote_files"] <= 0
        or not isinstance(check.get("remote_bytes"), int)
        or check["remote_bytes"] <= 0
        or check.get("local_files") != check["remote_files"]
        or check.get("local_bytes") != check.get("remote_bytes")
    ):
        raise ValueError(f"Receipt inventory counts mismatch: {path}")
    if any(check.get(k) != [] for k in ("missing", "extra", "size_mismatch")):
        raise ValueError(f"Receipt inventory has mismatches: {path}")
    if check.get("digest_mismatch", []) != []:
        raise ValueError(f"Receipt reports corrupt digests: {path}")
    return {
        "source": {
            "path": str(path.relative_to(root)),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "repo_id": receipt["repo_id"],
        "repo_type": kind,
        "commit_oid": commit,
        "remote_prefix": prefix,
        "declared_files": check["remote_files"],
        "declared_bytes": check["remote_bytes"],
        "original_receipt_includes_digest_check": "digest_mismatch" in check,
    }


def check_reference(api, reference: dict, snapshot: dict) -> dict:
    prefix = reference["remote_prefix"]
    pinned = []
    for row in api.list_repo_tree(
        reference["repo_id"],
        repo_type=reference["repo_type"],
        revision=reference["commit_oid"],
        path_in_repo=prefix,
        recursive=True,
    ):
        if not hasattr(row, "size"):
            continue
        if not row.path.startswith(prefix + "/"):
            raise ValueError("Pinned inventory escaped the requested prefix")
        pinned.append(
            {
                "path": row.path,
                "size": row.size,
                "blob_id": row.blob_id,
                "lfs_sha256": getattr(getattr(row, "lfs", None), "sha256", None),
            }
        )
    pinned.sort(key=lambda r: r["path"])
    if len({r["path"] for r in pinned}) != len(pinned):
        raise ValueError("Duplicate path at pinned commit")
    if (
        len(pinned) != reference["declared_files"]
        or sum(r["size"] for r in pinned) != reference["declared_bytes"]
    ):
        raise ValueError("Pinned inventory disagrees with the original receipt")
    current = {row["path"]: row for row in snapshot["files"]}
    changed = [row["path"] for row in pinned if current.get(row["path"]) != row]
    return {
        **reference,
        "immutable_prefix_readable": True,
        "pinned_inventory": pinned,
        "pinned_files_unchanged_at_observed_head": not changed,
        "changed_or_missing_at_observed_head": changed,
        "payload_downloaded_or_locally_rehashed": False,
        "requires_new_verified_reference_before_history_rewrite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite a dependency audit")
    raw = args.plan.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.plan_sha256:
        raise ValueError("Reviewed plan SHA-256 mismatch")
    plan = json.loads(raw)
    if {r["repo_type"] for r in plan["repositories"]} != set(REPOSITORIES) or len(
        plan["repositories"]
    ) != 2:
        raise ValueError("Expected exactly the two reviewed repositories")
    root = args.repository.resolve()
    api = HfApi()
    snapshots = {kind: inventory(api, kind) for kind in REPOSITORIES}
    dependencies = [
        repository_dependencies(row, snapshots[row["repo_type"]]) for row in plan["repositories"]
    ]
    refs = {}
    for kind, repo_id in REPOSITORIES.items():
        found = api.list_repo_refs(repo_id, repo_type=kind, include_pull_requests=True)
        refs[kind] = {
            group: [vars(r) for r in getattr(found, group, []) or []]
            for group in ("branches", "tags", "converts", "pull_requests")
        }
    references = []
    for relative in RECEIPT_DIRS:
        for path in sorted((root / relative).glob("*.json")):
            if (ref := declared_reference(path, root)) is not None:
                references.append(ref)
    if not references or not any(r["repo_type"] == "model" for r in references):
        raise ValueError("No current checkpoint reference receipts found")
    with ThreadPoolExecutor(max_workers=4) as pool:
        checked = list(
            pool.map(lambda ref: check_reference(api, ref, snapshots[ref["repo_type"]]), references)
        )
    # Preserve an honest snapshot if a writer adds or replaces a receipt during the network checks.
    changed_receipts = [
        r["source"]["path"]
        for r in checked
        if identity(root / r["source"]["path"], root) != r["source"]
    ]
    if changed_receipts:
        raise ValueError(f"Receipt changed during audit: {changed_receipts}")
    report = {
        "schema_version": 1,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "scope": "read_only_hf_withdrawal_dependency_audit",
        "plan_sha256": args.plan_sha256,
        "implementation": identity(Path(__file__).resolve(), Path(__file__).resolve().parents[1]),
        "inventory_helper": identity(
            Path(__file__).with_name("hf_obsolete_cleanup.py").resolve(),
            Path(__file__).resolve().parents[1],
        ),
        "local_receipt_repository": str(root),
        "receipt_scope": list(RECEIPT_DIRS),
        "snapshots": snapshots,
        "named_remote_refs": refs,
        "repositories": dependencies,
        "declared_backup_references": checked,
        "reference_receipts": len(checked),
        "unique_pinned_commits": len({(r["repo_type"], r["commit_oid"]) for r in checked}),
        "all_pinned_files_unchanged_at_observed_heads": all(
            r["pinned_files_unchanged_at_observed_head"] for r in checked
        ),
        "safe_to_purge": False,
        "hf_mutations": False,
        "local_receipts_modified": False,
        "scientific_completion": False,
        "boundary": "Metadata/digest comparison of named current trees and bounded primary backup receipts only; not a whole-history LFS inventory, physical-storage reclaim estimate, local payload rehash, authorization, or completed dependency migration. New uploads can add dependencies after this snapshot.",
    }
    digest = save_new(args.output, report)
    print(
        json.dumps(
            {
                "report": str(args.output),
                "sha256": digest,
                "reference_receipts": len(checked),
                "safe_to_purge": False,
                "shared_lfs_counts": {r["repo_type"]: r["shared_lfs_count"] for r in dependencies},
            }
        )
    )


if __name__ == "__main__":
    main()
