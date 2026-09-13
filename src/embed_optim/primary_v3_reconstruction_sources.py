"""Explicit source selection after full v3 authoring; no whole-workspace traversal."""

from __future__ import annotations

from pathlib import Path

from . import reconstruction_files as files
from .primary_contract import file_identity, read_json, relative_path, require_same, verify_file
from .primary_v3_dimension_exports import inventory

INFERENCE_PROTOCOL = "configs/dense_primary_v3_dimension_inference_protocol.json"
TOP_REFERENCES = (
    "primary",
    "data_amendment",
    "input_bindings",
    "natural_acceptance",
    "primary_parent",
    "core_acceptance",
)


def source_selection(contract, selected):
    """Copy the metadata actually loaded by the current contracts, not arbitrary ancestors."""
    root = contract.primary.repository
    objects = [
        contract.primary,
        contract,
        contract.bridge,
        contract.bridge.outcomes,
        contract.bridge.outcomes.validation,
        contract.bridge.geometry,
        contract.dimensions,
    ]
    for current in objects:
        path = current.path
        if current is contract:
            path = root / INFERENCE_PROTOCOL
            if file_identity(path)["sha256"] != current.sha256:
                raise ValueError("Canonical inference protocol differs from the admitted draft")
        files.select(
            selected, "source", path.relative_to(root).as_posix(), path, file_identity(path)
        )
        payload_sources(selected, root, current.payload)
    # The geometry loader also explicitly opens this measurement amendment.
    amendment = root / contract.bridge.geometry.payload["parents"]["entry_amendment"]["path"]
    payload_sources(selected, root, read_json(amendment))
    for name, record in contract.primary.payload["training_sources"].items():
        files.select(
            selected,
            "training-source",
            relative_path(name),
            contract.primary.training_root / name,
            record,
        )
    # A complete Python package avoids import fallbacks into an installed/live checkout.
    # This bounded, code-only selection never scans data, logs, credentials or the workspace.
    for path in sorted((root / "src/embed_optim").glob("*.py")):
        files.select(
            selected, "source", path.relative_to(root).as_posix(), path, file_identity(path)
        )
    for name in (
        "pyproject.toml",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "requirements-formal.lock",
        "requirements-formal-flash.txt",
    ):
        files.select(selected, "source", name, root / name, file_identity(root / name))
    return selected


def payload_sources(selected, root, payload):
    for group in ("sources", "consumer_sources"):
        for name, record in payload.get(group, {}).items():
            files.select(selected, "source", relative_path(name), root / name, record)
    references = list(payload.get("parents", {}).values())
    references += [payload[name] for name in TOP_REFERENCES if name in payload]
    for record in references:
        name = relative_path(record["path"])
        files.select(selected, "source", name, root / name, record)


def add_tree(selected, role, root, bound):
    root = Path(root).absolute()
    for relative in inventory(root):
        path = root / relative
        record = bound.get(str(path))
        if record is None:
            raise ValueError("A selected analysis file lacks the completed authoring binding")
        files.select(selected, role, relative, path, record)


def under(root, path):
    return files.ordinary(path).relative_to(Path(root).absolute()).as_posix()


def analysis_selection(args, evidence, inference_reader):
    """Close every stored input; embedded producer paths are never rewritten."""
    selected = {}
    bound = {str(Path(row["path"]).absolute()): row for row in evidence["source_bindings"]}
    for role, attr in (
        ("vectors", "dimension_vectors"),
        ("features", "dimension_features"),
        ("bridge", "bridge_root"),
        ("geometry", "geometry_root"),
        ("outcomes", "outcomes_root"),
    ):
        add_tree(selected, role, getattr(args, attr), bound)
    inference_root = Path(args.inference_root).absolute()
    for name, record in inference_reader["outputs"].items():
        files.select(selected, "inference", name, inference_root / name, record)
    files.select(
        selected,
        "inference",
        "manifest.json",
        inference_root / "manifest.json",
        file_identity(inference_root / "manifest.json"),
    )
    original = evidence["original_bridge_evidence"]["outcome_evidence"]
    for item in original["grid"]["evaluations"]:
        source_plan = item["plan"]
        path = Path(source_plan["results_root"]) / "primary_admission.json"
        require_same(read_json(path), source_plan)
        files.select(selected, "beir", under(args.results_root, path), path, file_identity(path))
        for task in item["tasks"]:
            for row in task["files"]:
                files.select(
                    selected, "beir", under(args.results_root, row["path"]), row["path"], row
                )
    for item in original["validation_selection"]["source_receipts"]:
        manifest = Path(item["manifest"]["path"])
        files.select(
            selected,
            "validation",
            under(args.validation_root, manifest),
            manifest,
            item["manifest"],
        )
        path = manifest.parent / "admission.json"
        require_same(read_json(path), item["plan"])
        files.select(
            selected, "validation", under(args.validation_root, path), path, file_identity(path)
        )
        for row in item["outputs"].values():
            path = manifest.parent / relative_path(row["path"])
            files.select(selected, "validation", under(args.validation_root, path), path, row)
    return selected


def recheck_selection(selected):
    for path, record in selected.values():
        verify_file(files.ordinary(path), record)
