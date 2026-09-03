"""Upload and size-audit completed corrected Dense runs on Hugging Face Hub."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import RemoteEntryNotFoundError
from huggingface_hub.hf_api import RepoFile

from .config import RunConfig, load_matrix
from .matrix import _run_is_complete

DEFAULT_REPO = "qcz/embedding-optimizer-study-checkpoints"
DEFAULT_PREFIX = "corrected-dense-no-packing-v1/dense"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def local_inventory(root: Path) -> dict[str, int]:
    return {
        str(path.relative_to(root)): path.stat().st_size
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    }


def remote_inventory(api: HfApi, repo_id: str, prefix: str) -> dict[str, int]:
    try:
        entries = api.list_repo_tree(
            repo_id,
            path_in_repo=prefix,
            recursive=True,
            expand=True,
            repo_type="model",
        )
        return {
            str(Path(entry.path).relative_to(prefix)): int(entry.size)
            for entry in entries
            if isinstance(entry, RepoFile)
        }
    except RemoteEntryNotFoundError:
        return {}


def compare_inventories(local: dict[str, int], remote: dict[str, int]) -> dict:
    missing = sorted(set(local) - set(remote))
    extra = sorted(set(remote) - set(local))
    size_mismatch = sorted(path for path in set(local) & set(remote) if local[path] != remote[path])
    return {
        "complete": not missing and not extra and not size_mismatch,
        "local_files": len(local),
        "local_bytes": sum(local.values()),
        "remote_files": len(remote),
        "remote_bytes": sum(remote.values()),
        "missing": missing,
        "extra": extra,
        "size_mismatch": size_mismatch,
    }


def _inventory_digest(inventory: dict[str, int]) -> str:
    digest = hashlib.sha256()
    for path, size in sorted(inventory.items()):
        digest.update(f"{path}\0{size}\n".encode())
    return digest.hexdigest()


def _selected_configs(matrix: Path, run_ids: list[str]) -> list[RunConfig]:
    configs = load_matrix(matrix)
    if (
        len(configs) != 12
        or any(config.model_family != "dense" for config in configs)
        or any(config.dense_can_flatten_inputs for config in configs)
    ):
        raise ValueError("Corrected backup requires the 12-run padded Dense matrix")
    requested = set(run_ids)
    unknown = requested - {config.run_id for config in configs}
    if unknown:
        raise ValueError(f"Unknown run IDs: {sorted(unknown)}")
    return [config for config in configs if not requested or config.run_id in requested]


def backup_run(
    api: HfApi,
    config: RunConfig,
    *,
    repo_id: str,
    remote_prefix: str,
    receipt_root: Path,
    audit_only: bool,
) -> dict:
    if not _run_is_complete(config):
        raise RuntimeError(f"Refusing to back up incomplete run {config.run_id}")
    local = local_inventory(config.output_dir)
    if not local:
        raise RuntimeError(f"Completed run has no files: {config.output_dir}")
    prefix = f"{remote_prefix.rstrip('/')}/{config.run_id}"
    commit = None
    if not audit_only:
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=config.output_dir,
            path_in_repo=prefix,
            ignore_patterns=[".cache/**", "*.tmp"],
            commit_message=f"Back up corrected Dense run {config.run_id}",
        )
    remote = remote_inventory(api, repo_id, prefix)
    audit = compare_inventories(local, remote)
    if not audit["complete"]:
        raise RuntimeError(
            f"Remote inventory differs for {config.run_id}: "
            f"missing={audit['missing'][:3]} extra={audit['extra'][:3]} "
            f"size_mismatch={audit['size_mismatch'][:3]}"
        )
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "audited_at_utc": _utc_now(),
        "run_id": config.run_id,
        "local_root": str(config.output_dir),
        "repo_id": repo_id,
        "repo_type": "model",
        "remote_prefix": prefix,
        "inventory_sha256": _inventory_digest(local),
        "inventory": audit,
        "commit_url": str(getattr(commit, "commit_url", "")) or None,
        "commit_oid": str(getattr(commit, "oid", "")) or None,
    }
    receipt_root.mkdir(parents=True, exist_ok=True)
    path = receipt_root / f"{config.run_id}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return receipt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix", type=Path, default=Path("configs/dense_no_packing_retrain.yaml")
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--remote-prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--receipt-root", type=Path, default=Path("reports/dense-no-packing/checkpoint-backup")
    )
    parser.add_argument("--run-ids", nargs="*", default=[])
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    api = HfApi()
    receipts = [
        backup_run(
            api,
            config,
            repo_id=args.repo_id,
            remote_prefix=args.remote_prefix,
            receipt_root=args.receipt_root,
            audit_only=args.audit_only,
        )
        for config in _selected_configs(args.matrix, args.run_ids)
    ]
    print(
        json.dumps(
            {
                "status": "complete",
                "runs": len(receipts),
                "files": sum(item["inventory"]["local_files"] for item in receipts),
                "bytes": sum(item["inventory"]["local_bytes"] for item in receipts),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
