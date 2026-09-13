"""Lightweight file integrity primitives, independent of model/training imports."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def file_inventory(path: Path) -> dict[str, Any]:
    before = path.stat()
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(usedforsecurity=False)
    blob.update(f"blob {before.st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha256.update(block)
            blob.update(block)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in fields):
        raise ValueError(f"File changed while hashing: {path}")
    return {"size": after.st_size, "sha256": sha256.hexdigest(), "git_blob_sha1": blob.hexdigest()}


def compare_inventories(local: dict[str, Any], remote: dict[str, Any]) -> dict[str, Any]:
    for record in remote.values():
        if record["digest_kind"] not in {"sha256", "git_blob_sha1"}:
            raise ValueError("Unsupported remote digest kind")
    missing = sorted(set(local) - set(remote))
    extra = sorted(set(remote) - set(local))
    common = set(local) & set(remote)
    sizes = sorted(name for name in common if int(local[name]["size"]) != int(remote[name]["size"]))
    digests = sorted(
        name
        for name in common
        if name not in sizes and local[name][remote[name]["digest_kind"]] != remote[name]["digest"]
    )
    return {
        "complete": not missing and not extra and not sizes and not digests,
        "local_files": len(local),
        "local_bytes": sum(int(v["size"]) for v in local.values()),
        "remote_files": len(remote),
        "remote_bytes": sum(int(v["size"]) for v in remote.values()),
        "missing": missing,
        "extra": extra,
        "size_mismatch": sizes,
        "digest_mismatch": digests,
    }
