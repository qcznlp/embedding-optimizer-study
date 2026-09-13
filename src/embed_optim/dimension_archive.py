"""Content-addressed HF archival of primary dimension vectors and publication evidence.

The archive is a reconstruction input, not a scientific-completion certificate.
Only an explicit source-derived file set is uploaded; no training/model tree,
credentials, cache directory, or unrelated repository file is scanned for upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import RemoteEntryNotFoundError
from huggingface_hub.hf_api import RepoFile

from . import dimension_publication as publication
from . import dimension_release as release
from .artifact_inventory import compare_inventories, file_inventory

PROTOCOL = Path("configs/dense_primary_dimension_archive_protocol.json")
EXPORT_ROOT = Path("results/dense-primary-dimension-probe/exports/dense")
RECEIPT_ROOT = Path("reports/dimension-utilization-archive")
REPO_ID = "qcz/embedding-optimizer-study-analysis-artifacts"
REMOTE_ROOT = "primary-dimension/v1"
SOURCES = (
    "src/embed_optim/dimension_archive.py",
    "src/embed_optim/dimension_release.py",
    "src/embed_optim/dimension_publication.py",
    "src/embed_optim/dimension_utilization.py",
    "src/embed_optim/primary_dimension_probe.py",
    "src/embed_optim/artifact_inventory.py",
    "src/embed_optim/dimension_replay.py",
    "src/embed_optim/config.py",
)
PARENTS = (
    "configs/dense_primary_dimension_export_protocol.json",
    "configs/dense_dimension_utilization_protocol.json",
)
BOUNDARY = (
    "Primary fixed-probe vectors and publication reconstruction evidence only. "
    "Checkpoint payloads and raw training/probe text are not included. Re-encoding requires "
    "the separately archived model states and pinned public probe sources. This receipt "
    "does not certify whole-study scientific completion."
)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _bytes_inventory(value: bytes) -> dict[str, Any]:
    return {
        "size": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
        "git_blob_sha1": hashlib.sha1(
            f"blob {len(value)}\0".encode() + value, usedforsecurity=False
        ).hexdigest(),
    }


def _path_inventory(path: Path) -> dict[str, Any]:
    return file_inventory(path)


def load_contract(repository: Path) -> dict[str, Any]:
    payload = json.loads((repository / PROTOCOL).read_text())
    if (
        payload.get("status") != "prospective_primary_dimension_archive_lock"
        or payload.get("destination")
        != {"repo_id": REPO_ID, "repo_type": "dataset", "prefix": REMOTE_ROOT}
        or set(payload.get("source_bindings", {})) != set(SOURCES)
        or set(payload.get("parent_bindings", {})) != set(PARENTS)
    ):
        raise ValueError("Primary dimension archive contract differs")
    for group in ("source_bindings", "parent_bindings"):
        for name, record in payload[group].items():
            if publication._identity(repository / name, repository) != record:
                raise ValueError(f"Primary dimension archive binding changed: {name}")
    return payload


def _safe_relative(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or name in {".", ".."}
        or path.is_absolute()
        or path.as_posix() != name
        or any(part in {".", "..", ".cache"} for part in path.parts)
        or "\\" in name
    ):
        raise ValueError(f"Unsafe archive path: {name!r}")
    return name


def build_bundle(repository: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    """Reaudit the closed statistics and every raw-vector identity before selection."""
    repository = repository.resolve()
    load_contract(repository)
    publication.audit_report(release.publication_args(repository), portable=True)
    handoff = json.loads((repository / EXPORT_ROOT / "primary_exports.json").read_text())
    from .config import load_matrix
    from .primary_dimension_probe import planned_cells

    scientific = json.loads((repository / release.PROTOCOL).read_text())
    expected_cells = planned_cells(
        load_matrix(repository / "configs/dense_no_packing_retrain.yaml"),
        scientific["inputs"]["checkpoint_stages"],
    )
    outputs = handoff.get("outputs", [])
    if (
        handoff.get("status") != "complete"
        or handoff.get("scope") != "primary_dense_fixed_probe_exports"
        or handoff.get("cells") != expected_cells
        or [row.get("cell") for row in outputs] != expected_cells
    ):
        raise ValueError("Archive requires the exact 61-state primary export handoff")
    paths = set(release.selected_files(repository))
    paths.update(repository / name for name in (*SOURCES, *PARENTS, str(PROTOCOL)))
    paths.update(
        repository / name
        for name in (str(release.MANIFEST), "pyproject.toml", "src/embed_optim/__init__.py")
    )
    for row in outputs:
        path = publication._resolve_identity(row["export"], repository)
        expected = repository / EXPORT_ROOT / f"{row['cell']}.npz"
        if path != expected:
            raise ValueError(f"Primary vector path differs: {row['cell']}")
        paths.add(path)
    selected: dict[str, Path] = {}
    for path in sorted(paths):
        relative = path.relative_to(repository).as_posix()
        _safe_relative(relative)
        # No links, including links in parent directories, enter the payload.
        if path.resolve() != path or not path.is_file():
            raise ValueError(f"Archive payload is not a regular repository file: {path}")
        selected[relative] = path
    inventory = {name: _path_inventory(path) for name, path in selected.items()}
    payload = {
        "schema_version": 1,
        "scope": "primary_dense_dimension_reconstruction",
        "scientific_completion": False,
        "protocol": publication._identity(repository / PROTOCOL, repository),
        "cells": expected_cells,
        "files": inventory,
        "summary": {"files": len(inventory), "bytes": sum(v["size"] for v in inventory.values())},
        "boundary": BOUNDARY,
    }
    return payload, selected


def _remote_inventory(api: HfApi, prefix: str, revision: str) -> dict[str, dict[str, Any]]:
    try:
        entries = list(
            api.list_repo_tree(
                REPO_ID,
                path_in_repo=prefix,
                repo_type="dataset",
                revision=revision,
                recursive=True,
                expand=True,
            )
        )
    except RemoteEntryNotFoundError:
        return {}
    inventory = {}
    for entry in entries:
        if not isinstance(entry, RepoFile):
            continue
        name = PurePosixPath(entry.path).relative_to(prefix).as_posix()
        _safe_relative(name)
        if name in inventory:
            raise ValueError(f"Duplicate remote archive path: {name}")
        kind = "sha256" if entry.lfs is not None else "git_blob_sha1"
        digest = entry.lfs.sha256 if entry.lfs is not None else entry.blob_id
        if not digest:
            raise ValueError(f"Missing remote digest: {name}")
        inventory[name] = {"size": int(entry.size), "digest_kind": kind, "digest": digest}
    return inventory


def _require_match(local: dict[str, Any], remote: dict[str, Any]) -> dict[str, Any]:
    result = compare_inventories(local, remote)
    if not result["complete"]:
        raise ValueError(f"Dimension archive inventory differs: {result}")
    return result


def archive(
    repository: Path, *, audit_only: bool = False, api: HfApi | None = None
) -> dict[str, Any]:
    repository = repository.resolve()
    if not audit_only:
        # This expensive checkpoint-backed audit must finish before any remote operation.
        publication.audit_report(release.publication_args(repository))
    payload, selected = build_bundle(repository)
    manifest_bytes = _json_bytes(payload)
    archive_sha = hashlib.sha256(manifest_bytes).hexdigest()
    prefix = f"{REMOTE_ROOT}/{archive_sha}"
    receipt_path = repository / RECEIPT_ROOT / f"{archive_sha}.json"
    local = {f"payload/{name}": record for name, record in payload["files"].items()}
    local["archive_manifest.json"] = _bytes_inventory(manifest_bytes)
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
    if audit_only and receipt is None:
        raise FileNotFoundError(f"No verified immutable archive receipt: {receipt_path}")
    if receipt is not None and (
        receipt.get("archive_sha256") != archive_sha
        or receipt.get("repo_id") != REPO_ID
        or receipt.get("remote_prefix") != prefix
        or receipt.get("local_inventory") != local
        or not receipt.get("commit_oid")
        or receipt.get("scientific_completion") is not False
    ):
        raise ValueError("Stored dimension archive receipt differs")
    api = HfApi() if api is None else api
    if receipt is not None:
        revision = receipt["commit_oid"]
        provenance = receipt["commit_provenance"]
    else:
        # This immutable snapshot also serves as the optimistic concurrency precondition.
        revision = str(api.repo_info(REPO_ID, repo_type="dataset").sha)
        provenance = "recovered_verified_snapshot"
        existing = _remote_inventory(api, prefix, revision)
        if existing:
            _require_match(local, existing)  # Never overwrite a conflicting prefix.
        else:
            with tempfile.TemporaryDirectory(prefix="dense-dimension-archive-") as temporary:
                stage = Path(temporary)
                operations = []
                for name, source in selected.items():
                    target = stage / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    if _path_inventory(target) != payload["files"][name]:
                        raise ValueError(f"Dimension source changed while staging: {name}")
                    operations.append(CommitOperationAdd(f"{prefix}/payload/{name}", target))
                operations.append(
                    CommitOperationAdd(f"{prefix}/archive_manifest.json", manifest_bytes)
                )
                if build_bundle(repository)[0] != payload:
                    raise ValueError("Dimension source changed before upload")
                commit = api.create_commit(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"Archive primary dimension evidence {archive_sha}",
                    parent_commit=revision,
                )
                revision = str(commit.oid)
                provenance = "created_atomic_archive_commit"
    remote = _remote_inventory(api, prefix, revision)
    comparison = _require_match(local, remote)
    if build_bundle(repository)[0] != payload:
        raise ValueError("Dimension source changed during archive verification")
    observed = datetime.now(timezone.utc).isoformat()
    result = {
        "schema_version": 1,
        "complete": True,
        "scientific_completion": False,
        "scope": "primary_dense_dimension_archive",
        "repo_id": REPO_ID,
        "repo_type": "dataset",
        "archive_sha256": archive_sha,
        "remote_prefix": prefix,
        "commit_oid": revision,
        "commit_provenance": provenance,
        "commit_url": f"https://huggingface.co/datasets/{REPO_ID}/commit/{revision}",
        "first_verified_at_utc": receipt["first_verified_at_utc"] if receipt else observed,
        "last_audited_at_utc": observed,
        "local_inventory": local,
        "remote_inventory": remote,
        "inventory": comparison,
        "boundary": BOUNDARY,
    }
    publication._atomic_json(receipt_path, result)
    return result


def audit_download(root: Path, expected_sha256: str) -> dict[str, Any]:
    """Verify a downloaded prefix without network, producer tree or model payloads."""
    root = root.resolve()
    manifest = root / "archive_manifest.json"
    if manifest.resolve() != manifest or not manifest.is_file():
        raise ValueError("Downloaded archive manifest must be a regular local file")
    data = manifest.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Downloaded archive manifest differs from the trusted receipt hash")
    payload = json.loads(data)
    if payload.get("scope") != "primary_dense_dimension_reconstruction":
        raise ValueError("Downloaded archive scope differs")
    expected = {"archive_manifest.json"}
    for name, record in payload["files"].items():
        _safe_relative(name)
        relative = f"payload/{name}"
        path = root / relative
        if path.resolve() != path or not path.is_file() or _path_inventory(path) != record:
            raise ValueError(f"Downloaded dimension payload differs: {relative}")
        expected.add(relative)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if (path.is_file() or path.is_symlink()) and path.relative_to(root).parts[0] != ".cache"
    }
    if actual != expected:
        raise ValueError("Downloaded archive file set differs")
    return {
        "complete": True,
        "scientific_completion": False,
        "scope": "downloaded_primary_dimension_archive",
        "archive_sha256": expected_sha256,
        "files": len(expected),
        "boundary": BOUNDARY,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--verify-download", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args(argv)
    if args.verify_download:
        if not args.expected_sha256:
            parser.error("--verify-download requires the trusted receipt's --expected-sha256")
        result = audit_download(args.verify_download, args.expected_sha256)
    elif args.dry_run:
        load_contract(args.repository.resolve())
        required = [
            release.MANIFEST,
            release.DIMENSION_DIR / "summary_manifest.json",
            EXPORT_ROOT / "primary_exports.json",
        ]
        missing = [str(path) for path in required if not (args.repository / path).is_file()]
        result = {
            "status": "dry_run",
            "network_used": False,
            "scientific_completion": False,
            "missing_inputs": missing,
            "upload_ready": False,
        }
        if not missing:
            payload, _ = build_bundle(args.repository)
            result.update(
                archive_sha256=hashlib.sha256(_json_bytes(payload)).hexdigest(),
                **payload["summary"],
            )
    else:
        result = archive(args.repository, audit_only=args.audit_only)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
