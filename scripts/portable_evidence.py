#!/usr/bin/env python3
"""Build or audit the minimal result-file closure required by the paper audit.

The full checkpoint archive belongs on Hugging Face, not in Git.  This manifest instead records
the small, content-addressed evaluation files that make the checked-in paper auditable in a clean
clone.  It is deterministic: rebuilding it without source changes produces identical bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
STATUS = "portable-paper-evidence-closure"
LEGACY_CHECKOUT_NAME = "embedding-optimizer-study"
MANIFEST_PATH = Path("configs/portable_paper_evidence.json")
AUDIT_IMPLEMENTATION = Path("src/embed_optim/paper_audit.py")
SOURCE_MANIFESTS = (
    Path("reports/retrieval-dynamics/summary_manifest.json"),
    Path("reports/tail-stability/summary_manifest.json"),
    Path("reports/spectral-transplant/summary_manifest.json"),
    Path("reports/dense-retrieval-dynamics/summary_manifest.json"),
)
FIXED_RESULT_FILES = (
    Path("results/common-state-spectra/summary/summary_manifest.json"),
    Path("results/common-state-spectra/summary/spectrum_metrics.csv"),
    Path("results/representation-space/training/summary/summary_manifest.json"),
    Path("results/representation-space/decontaminated-beir/summary/summary_manifest.json"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    relative = resolved.relative_to(root).as_posix()
    return {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _declared_result_path(root: Path, record: Any) -> Path | None:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        return None
    declared = Path(record["path"])
    if declared.is_absolute():
        indexes = [
            index for index, part in enumerate(declared.parts) if part == LEGACY_CHECKOUT_NAME
        ]
        if not indexes or indexes[-1] + 1 >= len(declared.parts):
            raise ValueError(f"Cannot port unrelated absolute evidence path: {declared}")
        declared = Path(*declared.parts[indexes[-1] + 1 :])
    resolved = (root / declared).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Evidence path escapes the repository: {declared}") from error
    if not relative.parts or relative.parts[0] != "results":
        return None
    return resolved


def _nested_result_paths(value: Any, root: Path) -> set[Path]:
    paths: set[Path] = set()
    if isinstance(value, list):
        for item in value:
            paths.update(_nested_result_paths(item, root))
    elif isinstance(value, dict):
        path = _declared_result_path(root, value)
        if path is not None and "sha256" in value:
            paths.add(path)
        for name, item in value.items():
            if name not in {"path", "bytes", "sha256", "rows"}:
                paths.update(_nested_result_paths(item, root))
    return paths


def _selected_result_paths(root: Path) -> set[Path]:
    retrieval = _load(root / SOURCE_MANIFESTS[0])
    tail = _load(root / SOURCE_MANIFESTS[1])
    spectral = _load(root / SOURCE_MANIFESTS[2])
    dynamics = _load(root / SOURCE_MANIFESTS[3])
    selected = {(root / relative).resolve() for relative in FIXED_RESULT_FILES}
    selected.update(
        _nested_result_paths(retrieval.get("sources", {}).get("evaluation_results"), root)
    )
    selected.update(_nested_result_paths(tail.get("discovery_sources"), root))
    selected.update(_nested_result_paths(tail.get("short_branch_sources"), root))
    selected.update(_nested_result_paths(spectral.get("sources"), root))
    selected.update(_nested_result_paths(dynamics.get("sources", {}).get("partitions"), root))
    return selected


def build_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    generator = Path(__file__).resolve()
    files = sorted(_selected_result_paths(root))
    if not files or any(not path.is_file() for path in files):
        missing = [str(path) for path in files if not path.is_file()]
        raise ValueError(f"Portable paper evidence is missing source files: {missing[:3]}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "complete": True,
        "purpose": (
            "Minimal content-addressed evaluation closure for strict paper auditing in a clean "
            "Git clone; full checkpoints remain in the public Hugging Face archive."
        ),
        "generator": _identity(generator, root),
        "audit_implementation": _identity(root / AUDIT_IMPLEMENTATION, root),
        "source_manifests": [_identity(root / relative, root) for relative in SOURCE_MANIFESTS],
        "files": [_identity(path, root) for path in files],
        "summary": {
            "files": len(files),
            "bytes": sum(path.stat().st_size for path in files),
        },
    }


def audit_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    observed = _load(root / MANIFEST_PATH)
    expected = build_manifest(root)
    if observed != expected:
        raise ValueError("Portable paper evidence manifest differs from its source-bound closure")
    return {
        "complete": True,
        "manifest": MANIFEST_PATH.as_posix(),
        **expected["summary"],
    }


def _repository_root() -> Path:
    for candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError("Cannot locate repository root")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    root = _repository_root()
    if args.audit_only:
        result = audit_manifest(root)
    else:
        result = build_manifest(root)
        target = root / MANIFEST_PATH
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = audit_manifest(root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
