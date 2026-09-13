"""Seal a distinct preparation-only full-identity integration, preserving its parent."""

import argparse
import hashlib
import json
from pathlib import Path


def file_identity(path):
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("Manifest output must be new")
    if file_identity(args.parent_manifest)["sha256"] != args.parent_sha256:
        raise ValueError("Parent manifest is not authenticated")
    parent = json.loads(args.parent_manifest.read_text())
    root = args.candidate_root.resolve()
    if root == Path(parent["candidate_root"]):
        raise ValueError("Do not modify the sealed parent checkout")
    source_rows = []
    for row in parent["candidate_sources"]:
        if file_identity(Path(row["identity"]["path"])) != row["identity"]:
            raise ValueError("Sealed parent source changed")
        relative = row["relative_path"]
        actual = file_identity(root / relative)
        if relative not in ("src/embed_optim/train.py", "tests/test_dense_numerical_contract.py"):
            if actual["sha256"] != row["identity"]["sha256"]:
                raise ValueError("Integration changed a source outside its declared surface")
        source_rows.append({"relative_path": relative, "identity": actual})
    for relative in (
        "src/embed_optim/__init__.py",
        "src/embed_optim/dense_run_contract.py",
        "tests/test_dense_run_contract.py",
    ):
        source_rows.append({"relative_path": relative, "identity": file_identity(root / relative)})
    result = {
        "scope": parent["scope"],
        "candidate_root": str(root),
        "base_commit": parent["base_commit"],
        "candidate_sources": source_rows,
        "parent_manifest": file_identity(args.parent_manifest),
        "integration": "Complete actual recipe/data/model/runtime/source admission and all-rank checkpoint seals",
        "scientific_completion": False,
        "production_deployed": False,
        "formal_execution_protocol_reviewed": False,
        "matrix_field_checks": parent["matrix_field_checks"],
        "matrix_unchanged_from_parent": True,
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(file_identity(args.output)))


if __name__ == "__main__":
    main()
