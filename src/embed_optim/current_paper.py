"""Rebuild the reviewed complete Dense paper from explicitly supplied local files.

This is document reproduction, not a replacement for scientific reconstruction,
checkpoint admission or source release. No producer directory or network fallback
is used. The complete document component is promoted byte-for-byte from the
verified native/revised builds; its original checks remain in force.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from . import complete_paper_document as document

SNAPSHOT_NAME = "document-snapshot.json"
SNAPSHOT_SHA256 = "6ef28bcf1f2cd873bcb0a663c2baea57ec6dc98a4e441d0850608a61f46ae6b9"
COMPONENT_SHA256 = "dbed8e2d4d418ce89d5d56d75da4e433e9b9b0888194a9c44ca053dd03d2684b"


def _identity(raw: bytes) -> dict[str, Any]:
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read_snapshot(paper_dir: str | Path) -> tuple[Path, bytes, dict[str, Any]]:
    """Authenticate all twelve inputs before creating an output directory."""
    paper = document.original.ordinary(paper_dir, directory=True)
    component = document.original.ordinary(document.__file__)
    if _identity(component.read_bytes())["sha256"] != COMPONENT_SHA256:
        raise ValueError("Complete document component differs from the reviewed source")
    snapshot = document.original.ordinary(paper / SNAPSHOT_NAME)
    raw = snapshot.read_bytes()
    if _identity(raw)["sha256"] != SNAPSHOT_SHA256:
        raise ValueError("Not the reviewed complete-document snapshot")
    value = json.loads(raw)
    if (
        value["scope"] != "reviewed_complete_dense_document_v1"
        or value["document_component_sha256"] != COMPONENT_SHA256
        or set(value["inputs"]) != set(document.BUILD_INPUTS)
        or value["upstream_scientific_recomputation"] is not False
        or value["source_release_verified"] is not False
    ):
        raise ValueError("Invalid reviewed document scope or input population")
    for name, binding in value["inputs"].items():
        path = document.original.ordinary(paper / name)
        if _identity(path.read_bytes()) != binding:
            raise ValueError("Reviewed document input differs: " + name)
    if snapshot.read_bytes() != raw:
        raise ValueError("Document snapshot changed during reading")
    return paper, raw, value


def build_current_paper(paper_dir: str | Path, output: str | Path) -> dict[str, Any]:
    """Compile into a new directory, preserving any failure and previous build."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Set CUDA_VISIBLE_DEVICES='' for document-only work")
    paper, snapshot_raw, snapshot = read_snapshot(paper_dir)
    root = Path(output)
    if not root.is_absolute() or root != root.resolve():
        raise ValueError("Output must be an explicit canonical absolute new directory")
    document.original.ordinary(root.parent, directory=True)
    if root.exists() or root.is_symlink():
        raise ValueError("Preserve the previous output; supply a new output directory")
    if root.is_relative_to(paper):
        raise ValueError("Do not write inside the authenticated document inputs")
    root.mkdir()
    try:
        for name, binding in snapshot["inputs"].items():
            source = document.original.ordinary(paper / name)
            raw = source.read_bytes()
            if _identity(raw) != binding:
                raise ValueError("Document input changed before copy: " + name)
            target = root / "paper" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
        with (root / SNAPSHOT_NAME).open("xb") as stream:
            stream.write(snapshot_raw)
        if read_snapshot(paper)[1] != snapshot_raw:
            raise ValueError("Source snapshot changed during copying")
        includes = {name: (root / "paper" / name).read_bytes() for name in document.RESULT_FILES}
        receipt = document.compile_fresh(root, includes, (root / "paper/results.tex").read_bytes())
        read_snapshot(paper)
        # Recheck actual compiler inputs against the reviewed snapshot, not only
        # against bytes observed at the beginning of this particular compilation.
        if receipt["source_inspection"]["inputs"] != snapshot["inputs"]:
            raise ValueError("Compiled source differs from the reviewed snapshot")
        result = {
            "scope": "reviewed_dense_document_reproduction_v1",
            "document_reproduction_complete": True,
            "snapshot_sha256": SNAPSHOT_SHA256,
            "document_receipt": _identity((root / "document.json").read_bytes()),
            "pdf": str(root / "paper/build/main.pdf"),
            "main_end_page": receipt["layout"]["main_end_page"],
            "abstract_words": receipt["source_inspection"]["actual_abstract"]["words_conservative"],
            "pdf_pages": receipt["pdf_pages"],
            "font_count": receipt["font_count"],
            "scientific_numerical_recomputation": False,
            "source_release_verified": False,
        }
        document.write_new(root / "current-paper.json", result)
        return result
    except BaseException as error:
        document.write_new(
            root / "failed.json",
            {
                "exception_type": type(error).__name__,
                "message": str(error),
                "outputs_preserved": True,
                "automatic_retry": False,
            },
        )
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New absolute output directory")
    args = parser.parse_args(argv)
    print(json.dumps(build_current_paper(args.paper_dir, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
