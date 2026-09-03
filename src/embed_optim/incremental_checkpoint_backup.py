"""Back up sealed checkpoints from corrected runs before the run finishes.

The completion controller intentionally uploads a corrected run only after all five
checkpoints are deeply complete.  This module closes the durability window for an
already sealed intermediate checkpoint without treating the run as scientifically
complete or changing the controller's behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.errors import RemoteEntryNotFoundError
from huggingface_hub.hf_api import RepoFile
from safetensors import safe_open

from .config import RunConfig, load_matrix
from .corrected_checkpoint_backup import DEFAULT_PREFIX, DEFAULT_REPO

DEFAULT_RECEIPT_ROOT = Path("reports/dense-no-packing/incremental-checkpoint-backup")
REQUIRED_CHECKPOINT_FILES = frozenset(
    {
        "config.json",
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        "rng_state_0.pth",
        "rng_state_1.pth",
        "rng_state_2.pth",
        "rng_state_3.pth",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _file_digests(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(usedforsecurity=False)
    git_blob.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            git_blob.update(chunk)
    return sha256.hexdigest(), git_blob.hexdigest()


def _iter_payload_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".cache" not in path.relative_to(root).parts:
            yield path


def stat_signature(root: Path) -> dict[str, tuple[int, int]]:
    """Return the immutable-file fields used to detect a concurrent mutation."""

    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in _iter_payload_files(root)
    }


def local_checkpoint_inventory(root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in _iter_payload_files(root):
        sha256, git_blob_sha1 = _file_digests(path)
        inventory[str(path.relative_to(root))] = {
            "size": path.stat().st_size,
            "sha256": sha256,
            "git_blob_sha1": git_blob_sha1,
        }
    return inventory


def remote_checkpoint_inventory(
    api: HfApi,
    *,
    repo_id: str,
    prefix: str,
) -> dict[str, dict[str, Any]]:
    try:
        entries = api.list_repo_tree(
            repo_id,
            path_in_repo=prefix,
            recursive=True,
            expand=True,
            repo_type="model",
        )
    except RemoteEntryNotFoundError:
        return {}

    inventory: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, RepoFile):
            continue
        relative = str(Path(entry.path).relative_to(prefix))
        if entry.lfs is not None:
            digest_kind = "sha256"
            digest = entry.lfs.sha256
        else:
            digest_kind = "git_blob_sha1"
            digest = entry.blob_id
        inventory[relative] = {
            "size": int(entry.size),
            "digest_kind": digest_kind,
            "digest": str(digest),
        }
    return inventory


def compare_checkpoint_inventories(
    local: dict[str, dict[str, Any]],
    remote: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    missing = sorted(set(local) - set(remote))
    extra = sorted(set(remote) - set(local))
    common = set(local) & set(remote)
    size_mismatch = sorted(
        path for path in common if int(local[path]["size"]) != int(remote[path]["size"])
    )
    digest_mismatch = sorted(
        path
        for path in common
        if path not in size_mismatch
        and str(local[path][str(remote[path]["digest_kind"])]) != str(remote[path]["digest"])
    )
    return {
        "complete": not missing and not extra and not size_mismatch and not digest_mismatch,
        "local_files": len(local),
        "local_bytes": sum(int(item["size"]) for item in local.values()),
        "remote_files": len(remote),
        "remote_bytes": sum(int(item["size"]) for item in remote.values()),
        "missing": missing,
        "extra": extra,
        "size_mismatch": size_mismatch,
        "digest_mismatch": digest_mismatch,
    }


def inventory_digest(inventory: dict[str, dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for path, item in sorted(inventory.items()):
        digest.update(
            (f"{path}\0{int(item['size'])}\0{item['sha256']}\0{item['git_blob_sha1']}\n").encode()
        )
    return digest.hexdigest()


def _recover_upload_commit(
    api: HfApi,
    *,
    repo_id: str,
    title: str,
) -> tuple[str | None, str | None, str | None]:
    for item in api.list_repo_commits(repo_id, repo_type="model"):
        if item.title != title:
            continue
        oid = str(item.commit_id)
        created_at = item.created_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return oid, f"https://huggingface.co/{repo_id}/commit/{oid}", created_at
    return None, None, None


def _selected_configs(matrix: Path, run_ids: list[str]) -> list[RunConfig]:
    configs = load_matrix(matrix)
    if (
        len(configs) != 12
        or any(config.model_family != "dense" for config in configs)
        or any(config.dense_can_flatten_inputs for config in configs)
    ):
        raise ValueError("Incremental backup requires the 12-run padded Dense matrix")
    requested = set(run_ids)
    unknown = requested - {config.run_id for config in configs}
    if unknown:
        raise ValueError(f"Unknown run IDs: {sorted(unknown)}")
    return [config for config in configs if config.run_id in requested]


def validate_sealed_checkpoint(config: RunConfig, step: int) -> Path:
    checkpoint = config.output_dir / f"checkpoint-{step}"
    if not checkpoint.is_dir():
        raise RuntimeError(f"Checkpoint does not exist: {checkpoint}")

    schedule_path = config.output_dir / "checkpoint_schedule.json"
    if not schedule_path.is_file():
        raise RuntimeError(f"Missing checkpoint schedule: {schedule_path}")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if step not in [int(value) for value in schedule.get("steps", [])]:
        raise RuntimeError(f"Step {step} is not in the frozen checkpoint schedule")

    observed = {str(path.relative_to(checkpoint)) for path in _iter_payload_files(checkpoint)}
    missing = sorted(REQUIRED_CHECKPOINT_FILES - observed)
    if missing:
        raise RuntimeError(f"Checkpoint {checkpoint} is missing required files: {missing}")
    empty = sorted(
        str(path.relative_to(checkpoint))
        for path in _iter_payload_files(checkpoint)
        if path.stat().st_size == 0
    )
    if empty:
        raise RuntimeError(f"Checkpoint {checkpoint} has empty payloads: {empty}")

    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if int(trainer_state.get("global_step", -1)) != step:
        raise RuntimeError(
            f"Checkpoint trainer_state has step {trainer_state.get('global_step')}, expected {step}"
        )
    with safe_open(checkpoint / "model.safetensors", framework="numpy") as model:
        if not list(model.keys()):
            raise RuntimeError(f"Checkpoint has an empty safetensors index: {checkpoint}")
    return checkpoint


def backup_checkpoint(
    api: HfApi,
    config: RunConfig,
    step: int,
    *,
    matrix_path: Path,
    repo_id: str,
    remote_prefix: str,
    receipt_root: Path,
    stability_seconds: float,
    audit_only: bool,
) -> dict[str, Any]:
    checkpoint = validate_sealed_checkpoint(config, step)
    before = stat_signature(checkpoint)
    if stability_seconds:
        time.sleep(stability_seconds)
    stable = stat_signature(checkpoint)
    if not before or before != stable:
        raise RuntimeError(f"Checkpoint changed during the stability window: {checkpoint}")

    local = local_checkpoint_inventory(checkpoint)
    if stat_signature(checkpoint) != stable:
        raise RuntimeError(f"Checkpoint changed while hashing: {checkpoint}")

    prefix = f"{remote_prefix.rstrip('/')}/{config.run_id}/checkpoint-{step}"
    commit_message = f"Back up sealed corrected checkpoint {config.run_id} step {step}"
    commit = None
    if not audit_only:
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=checkpoint,
            path_in_repo=prefix,
            ignore_patterns=[".cache/**", "*.tmp"],
            commit_message=commit_message,
        )
    if stat_signature(checkpoint) != stable:
        raise RuntimeError(f"Checkpoint changed during upload: {checkpoint}")

    remote = remote_checkpoint_inventory(api, repo_id=repo_id, prefix=prefix)
    audit = compare_checkpoint_inventories(local, remote)
    if not audit["complete"]:
        raise RuntimeError(
            f"Remote inventory differs for {config.run_id} checkpoint {step}: "
            f"missing={audit['missing'][:3]} extra={audit['extra'][:3]} "
            f"size={audit['size_mismatch'][:3]} digest={audit['digest_mismatch'][:3]}"
        )

    receipt_root.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_root / f"{config.run_id}-checkpoint-{step}.json"
    previous: dict[str, Any] = {}
    if receipt_path.is_file():
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))

    commit_oid = str(getattr(commit, "oid", "")) or previous.get("commit_oid")
    commit_url = str(getattr(commit, "commit_url", "")) or previous.get("commit_url")
    uploaded_at_utc = previous.get("uploaded_at_utc")
    if not commit_oid or not commit_url:
        recovered_oid, recovered_url, recovered_at = _recover_upload_commit(
            api, repo_id=repo_id, title=commit_message
        )
        commit_oid = commit_oid or recovered_oid
        commit_url = commit_url or recovered_url
        uploaded_at_utc = uploaded_at_utc or recovered_at
    if not audit_only and uploaded_at_utc is None:
        uploaded_at_utc = _utc_now()
    if not commit_oid or not commit_url or not uploaded_at_utc:
        raise RuntimeError(
            f"Could not bind checkpoint {config.run_id} step {step} to its upload commit"
        )

    source = Path(__file__).resolve()
    matrix = matrix_path.resolve()
    audited_at_utc = _utc_now()
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "scientific_completion": False,
        "role": "sealed_intermediate_checkpoint_durability",
        "uploaded_at_utc": uploaded_at_utc,
        "audited_at_utc": audited_at_utc,
        "run_id": config.run_id,
        "optimizer": config.optimizer.name,
        "checkpoint_step": step,
        "local_root": str(checkpoint),
        "repo_id": repo_id,
        "repo_type": "model",
        "remote_prefix": prefix,
        "stability_seconds": stability_seconds,
        "required_files": sorted(REQUIRED_CHECKPOINT_FILES),
        "inventory_sha256": inventory_digest(local),
        "inventory": audit,
        "source": {
            "path": _display_path(source),
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
        },
        "matrix": {
            "path": _display_path(matrix),
            "bytes": matrix.stat().st_size,
            "sha256": _sha256(matrix),
        },
        "commit_url": commit_url,
        "commit_oid": commit_oid,
    }
    temporary = receipt_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(receipt_path)
    return receipt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Upload hash-verified sealed checkpoints from active corrected Dense runs"
    )
    parser.add_argument(
        "--matrix", type=Path, default=Path("configs/dense_no_packing_retrain.yaml")
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--remote-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--receipt-root", type=Path, default=DEFAULT_RECEIPT_ROOT)
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--stability-seconds", type=float, default=2.0)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    if args.stability_seconds < 0:
        parser.error("--stability-seconds must be non-negative")

    api = HfApi()
    receipts = [
        backup_checkpoint(
            api,
            config,
            step,
            matrix_path=args.matrix,
            repo_id=args.repo_id,
            remote_prefix=args.remote_prefix,
            receipt_root=args.receipt_root,
            stability_seconds=args.stability_seconds,
            audit_only=args.audit_only,
        )
        for config in _selected_configs(args.matrix, args.run_ids)
        for step in args.steps
    ]
    print(
        json.dumps(
            {
                "status": "complete",
                "scientific_completion": False,
                "checkpoints": len(receipts),
                "files": sum(item["inventory"]["local_files"] for item in receipts),
                "bytes": sum(item["inventory"]["local_bytes"] for item in receipts),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
