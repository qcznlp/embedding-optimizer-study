"""Externally anchored local reconstruction payloads; no primary admission or network.

Only explicitly selected ordinary files enter a new tree. Stored metadata may
retain original producer locations as evidence, but this module never resolves
or opens a path taken from that metadata. All consumed paths are local roles.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path, PurePosixPath

from .primary_contract import canonical, file_identity, read_json, require_same, verify_file
from .primary_v3_validation_io import write_new

SCOPE = "dense_primary_v3_local_reconstruction_payload"
ROLES = {
    "source",
    "training-source",
    "vectors",
    "features",
    "inference",
    "bridge",
    "geometry",
    "outcomes",
    "validation",
    "beir",
    "provenance",
}
RESERVED = {".git", ".cache", ".env", ".netrc", "gpu.py"}


def identity(record):
    if (
        type(record.get("bytes")) is not int
        or record["bytes"] < 0
        or not isinstance(record.get("sha256"), str)
        or not re.fullmatch("[0-9a-f]{64}", record["sha256"])
    ):
        raise ValueError("Invalid typed reconstruction file identity")
    return {"bytes": record["bytes"], "sha256": record["sha256"]}


def safe_name(name):
    if not isinstance(name, str):
        raise ValueError("A reconstruction role path must be a string")
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or path.as_posix() != name
        or len(path.parts) < 2
        or path.parts[0] not in ROLES
        or any(part in {".", "..", *RESERVED} for part in path.parts)
        or "\\" in name
        or "\x00" in name
    ):
        raise ValueError("Unsafe or undeclared reconstruction role path")
    return name


def ordinary(path):
    path = Path(path).absolute()
    if path.resolve() != path or not path.is_file():
        raise ValueError("Reconstruction sources must be ordinary files without parent links")
    return path


def inventory(root):
    root = Path(root).absolute()
    if root.resolve() != root or not root.is_dir():
        raise ValueError("Require an ordinary local reconstruction root")
    paths = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Reconstruction tree contains a symlink")
    if any(not path.is_dir() and not path.is_file() for path in paths):
        raise ValueError("Reconstruction tree contains a nonordinary entry")
    return [path.relative_to(root).as_posix() for path in paths if path.is_file()]


def inspect(root, expected_manifest_sha256):
    """The expected hash must come from trusted authoring, never the inspected tree."""
    root = Path(root).absolute()
    if not isinstance(expected_manifest_sha256, str) or not re.fullmatch(
        "[0-9a-f]{64}", expected_manifest_sha256
    ):
        raise ValueError("Require an explicit external SHA-256 reconstruction anchor")
    names = inventory(root)
    manifest = root / "manifest.json"
    if file_identity(manifest)["sha256"] != expected_manifest_sha256:
        raise ValueError("Reconstruction manifest differs from its external trusted anchor")
    value = read_json(manifest)
    if (
        set(value)
        != {"schema_version", "scope", "metadata", "files", "summary", "scientific_completion"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["scope"] != SCOPE
        or value["scientific_completion"] is not False
        or not isinstance(value["files"], dict)
        or not value["files"]
        or not isinstance(value["metadata"], dict)
    ):
        raise ValueError("Invalid reconstruction-only manifest")
    expected = ["manifest.json"]
    for name, record in value["files"].items():
        safe_name(name)
        if set(record) != {"bytes", "sha256"}:
            raise ValueError("A reconstruction file has unexpected identity fields")
        identity(record)
        expected.append(name)
        verify_file(ordinary(root / name), record)
    require_same(names, sorted(expected))
    wanted_dirs = {
        str(parent)
        for name in expected
        for parent in PurePosixPath(name).parents
        if str(parent) != "."
    }
    actual_dirs = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()}
    require_same(sorted(actual_dirs), sorted(wanted_dirs))
    require_same(
        value["summary"],
        {
            "files": len(value["files"]),
            "bytes": sum(row["bytes"] for row in value["files"].values()),
        },
    )
    for name, record in value["files"].items():
        verify_file(ordinary(root / name), record)
    require_same(inventory(root), names)
    if file_identity(manifest)["sha256"] != expected_manifest_sha256:
        raise ValueError("Reconstruction manifest changed during verification")
    return value


def select(selected, role, relative, path, expected):
    """An authenticated caller supplies both the original path and expected identity."""
    name = safe_name(role + "/" + str(relative))
    path = ordinary(path)
    verify_file(path, expected)
    record = identity(expected)
    if name in selected:
        previous = selected[name]
        if previous != (path, record):
            raise ValueError("A logical reconstruction role has conflicting sources")
    selected[name] = path, record


def write(root, selected, metadata):
    """Pure transport, not an authoring gate; only the admitted top-level builder calls it."""
    root = Path(root).absolute()
    if (
        root.exists()
        or root.is_symlink()
        or root.parent.resolve() != root.parent
        or not selected
        or not isinstance(metadata, dict)
    ):
        raise ValueError("Require a new ordinary reconstruction directory and explicit sources")
    canonical(metadata)
    files = {}
    for name, (path, record) in sorted(selected.items()):
        safe_name(name)
        path = ordinary(path)
        if root == path or root in path.parents:
            raise ValueError("A reconstruction input cannot live inside its new output")
        verify_file(path, record)
        files[name] = identity(record)
    # No output side effects before the complete selection has passed.
    root.mkdir(parents=True, exist_ok=False)
    for name, (path, record) in sorted(selected.items()):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with ordinary(path).open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        verify_file(destination, record)
        verify_file(ordinary(path), record)
    write_new(
        root / "manifest.json",
        {
            "schema_version": 1,
            "scope": SCOPE,
            "metadata": metadata,
            "files": files,
            "summary": {"files": len(files), "bytes": sum(row["bytes"] for row in files.values())},
            "scientific_completion": False,
        },
    )
    anchor = file_identity(root / "manifest.json")["sha256"]
    inspect(root, anchor)
    return {"manifest_sha256": anchor, "files": len(files), "scientific_completion": False}
