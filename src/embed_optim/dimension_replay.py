"""Recompute dimension features from a trusted downloaded archive, without models.

Reconstruction has its own scope. It cannot become a primary authoring receipt,
certify training, or silently replace the archived evidence it is checking.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import dimension_archive as archive
from . import dimension_publication as publication
from . import dimension_utilization as dimensions
from .artifact_inventory import file_inventory
from .config import load_matrix
from .primary_dimension_probe import load_contract as load_export_contract
from .primary_dimension_probe import planned_cells

TOLERANCES = {
    "csv": {"rtol": 1e-9, "atol": 1e-12},
    "float32_arrays": {"rtol": 1e-6, "atol": 1e-12},
}
IDENTITY_FIELDS = {
    "run_id",
    "optimizer",
    "learning_rate",
    "step",
    "tasks",
    "task",
    "rotation_seed",
    "removed_fraction",
    "draw",
}
OUTPUTS = {
    "checkpoint_summary",
    "task_summary",
    "random_removal",
    "rotation_summary",
    "coordinate_attribution",
}
BOUNDARY = (
    "Verified-archive feature reconstruction only. Model encoding, checkpoint validation, "
    "training and publication inference were not repeated. The original archived outputs "
    "are preserved. A matching reconstruction is not new primary evidence or scientific completion."
)


def _verify_sources(payload: Path, manifest: dict[str, Any]) -> None:
    # Import only the locally defined, hash-matched module set; never execute names
    # or shell commands supplied by an archive manifest.
    for name in archive.SOURCES:
        module = importlib.import_module(f"embed_optim.{Path(name).stem}")
        if file_inventory(Path(module.__file__)) != manifest["files"][name]:
            raise ValueError(f"Running replay implementation differs from archived source: {name}")
    archive.load_contract(payload)
    load_export_contract(payload)


def _verify_record_at(record: dict[str, Any], path: Path) -> Path:
    """Bind an old producer identity to its declared canonical archive role.

    Never consult the producer's absolute path, even when it still exists.
    """
    if path.resolve() != path or not path.is_file():
        raise ValueError(f"Missing canonical replay input: {path}")
    current = file_inventory(path)
    if current["size"] != record.get("bytes") or current["sha256"] != record.get("sha256"):
        raise ValueError(f"Canonical replay input identity differs: {path}")
    return path


def _validate_inputs(payload: Path) -> tuple[dict[str, Any], list[Path], dict[str, Any]]:
    protocol_path = payload / archive.release.PROTOCOL
    protocol = dimensions._load_protocol(protocol_path)
    cells = planned_cells(
        load_matrix(payload / "configs/dense_no_packing_retrain.yaml"),
        protocol["inputs"]["checkpoint_stages"],
    )
    export_root = payload / archive.EXPORT_ROOT
    handoff = json.loads((export_root / "primary_exports.json").read_text())
    records = handoff.get("outputs", [])
    if (
        handoff.get("status") != "complete"
        or handoff.get("scope") != "primary_dense_fixed_probe_exports"
        or handoff.get("cells") != cells
        or [row.get("cell") for row in records] != cells
    ):
        raise ValueError("Replay requires the exact primary export cells")
    expected = {export_root / f"{cell}.npz" for cell in cells}
    if set(export_root.rglob("*.npz")) != expected:
        raise ValueError("Replay vector file set differs from the primary matrix")
    expected_pairs = {}
    for row in records:
        path = _verify_record_at(row["export"], export_root / f"{row['cell']}.npz")
        manifest_path = _verify_record_at(
            row["export_manifest"], path.with_suffix(".npz.manifest.json")
        )
        _verify_record_at(row["receipt"], path.with_suffix(".npz.primary.json"))
        key = (row["export"]["sha256"], row["export_manifest"]["sha256"])
        if key in expected_pairs:
            raise ValueError("Replay repeats a vector/manifest identity pair")
        expected_pairs[key] = (path, manifest_path)
    original_path = payload / archive.release.DIMENSION_DIR / "summary_manifest.json"
    original = json.loads(original_path.read_text())
    coverage = original.get("coverage", {})
    if (
        original.get("scope") != "corrected_dense_dimension_utilization"
        or original.get("status") != "complete"
        or coverage.get("checkpoint_exports") != 60
        or coverage.get("pretrained_exports") != 1
        or coverage.get("tasks") != 14
        or coverage.get("dimensions") != protocol["inputs"]["embedding_dimension"]
        or set(original.get("outputs", {})) != OUTPUTS
    ):
        raise ValueError("Replay source analysis is incomplete or outside the primary scope")
    _verify_record_at(original["protocol"], protocol_path)
    _verify_record_at(
        original["implementation"], payload / "src/embed_optim/dimension_utilization.py"
    )
    _verify_record_at(original["primary_export_handoff"], export_root / "primary_exports.json")
    if len(original.get("inputs", [])) != 61:
        raise ValueError("Replay analysis source coverage differs")
    observed = set()
    for pair in original["inputs"]:
        key = (pair["archive"]["sha256"], pair["manifest"]["sha256"])
        if key not in expected_pairs:
            raise ValueError("Replay analysis uses an unknown vector/manifest pair")
        path, manifest_path = expected_pairs[key]
        _verify_record_at(pair["archive"], path)
        _verify_record_at(pair["manifest"], manifest_path)
        if path in observed:
            raise ValueError("Replay analysis repeats an input state")
        observed.add(path)
    if observed != expected:
        raise ValueError("Replay analysis inputs differ from archived vectors")
    paths = [export_root / "pretrained.npz", *sorted(export_root.glob("*/checkpoint-*.npz"))]
    return protocol, paths, original


def compare_csv(expected: Path, observed: Path) -> dict[str, Any]:
    with expected.open(newline="", encoding="utf-8") as handle:
        left_reader = csv.DictReader(handle)
        fields = left_reader.fieldnames
        left = list(left_reader)
    with observed.open(newline="", encoding="utf-8") as handle:
        right_reader = csv.DictReader(handle)
        if fields != right_reader.fieldnames:
            raise ValueError("Replayed CSV columns differ")
        right = list(right_reader)
    if len(left) != len(right):
        raise ValueError("Replayed CSV row count differs")
    mismatches, examples = 0, []
    maximum_error = 0.0
    exact = True
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        for key in fields or []:
            exact = exact and a[key] == b[key]
            if key in IDENTITY_FIELDS:
                matches = a[key] == b[key]
            else:
                x, y = float(a[key]), float(b[key])
                if math.isnan(x) or math.isnan(y):
                    matches = key == "relative_margin" and math.isnan(x) and math.isnan(y)
                else:
                    if not math.isfinite(x) or not math.isfinite(y):
                        raise ValueError("Non-finite replay comparison")
                    maximum_error = max(maximum_error, abs(x - y))
                    matches = math.isclose(
                        x, y, rel_tol=TOLERANCES["csv"]["rtol"], abs_tol=TOLERANCES["csv"]["atol"]
                    )
            if not matches:
                mismatches += 1
                if len(examples) < 8:
                    examples.append(
                        {"row": index, "field": key, "expected": a[key], "observed": b[key]}
                    )
    return {
        "complete": mismatches == 0,
        "rows": len(left),
        "exact_cells": exact,
        "maximum_absolute_error": maximum_error,
        "mismatched_cells": mismatches,
        "first_mismatches": examples,
        "tolerances": TOLERANCES["csv"],
    }


def compare_arrays(expected: Path, observed: Path) -> dict[str, Any]:
    results = {}
    with (
        np.load(expected, allow_pickle=False) as left,
        np.load(observed, allow_pickle=False) as right,
    ):
        if set(left.files) != set(right.files):
            raise ValueError("Replayed attribution array names differ")
        for name in sorted(left.files):
            a, b = left[name], right[name]
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"Replayed attribution array shape/dtype differs: {name}")
            exact = np.array_equal(a, b)
            if a.dtype.kind in "f":
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError("Non-finite attribution comparison")
                matches = bool(np.allclose(a, b, **TOLERANCES["float32_arrays"]))
                error = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            else:
                matches, error = exact, None
            results[name] = {"complete": matches, "exact": exact, "maximum_absolute_error": error}
    return {
        "complete": all(row["complete"] for row in results.values()),
        "arrays": results,
        "tolerances": TOLERANCES["float32_arrays"],
    }


def replay(root: Path, expected_sha256: str, output_dir: Path) -> dict[str, Any]:
    root, output_dir = root.resolve(), output_dir.resolve()
    if output_dir.exists() or output_dir.is_relative_to(root) or root.is_relative_to(output_dir):
        raise FileExistsError("Replay needs a new output directory outside the immutable archive")
    archive.audit_download(root, expected_sha256)
    archive_manifest = json.loads((root / "archive_manifest.json").read_text())
    payload = root / "payload"
    _verify_sources(payload, archive_manifest)
    protocol, paths, original = _validate_inputs(payload)
    originals = {
        name: _verify_record_at(
            record,
            payload
            / archive.release.DIMENSION_DIR
            / f"{name}.{'npz' if name == 'coordinate_attribution' else 'csv'}",
        )
        for name, record in original["outputs"].items()
    }
    args = argparse.Namespace(
        repository=payload,
        protocol=payload / archive.release.PROTOCOL,
        input_root=payload / archive.EXPORT_ROOT,
        output_dir=output_dir,
        analysis_scope="reconstruction",
        skip_rotation=False,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        rebuilt = dimensions._compute_features(args, protocol, paths, 60, None)
        comparisons = {}
        for name, original_path in originals.items():
            record = rebuilt["outputs"][name]
            generated = (payload / record["path"]).resolve()
            if dimensions._identity(generated, payload) != record:
                raise ValueError(f"Reconstructed output changed: {name}")
            comparisons[name] = (
                compare_arrays(original_path, generated)
                if name == "coordinate_attribution"
                else compare_csv(original_path, generated)
            )
        archive.audit_download(root, expected_sha256)
        _verify_sources(payload, archive_manifest)
        complete = all(row["complete"] for row in comparisons.values())
        result = {
            "schema_version": 1,
            "status": "matched" if complete else "mismatch",
            "complete": complete,
            "scope": "primary_dimension_archive_reconstruction",
            "scientific_completion": False,
            "archive_sha256": expected_sha256,
            "features_recomputed": True,
            "model_encoding_repeated": False,
            "checkpoint_revalidated": False,
            "publication_inference_repeated": False,
            "comparisons": comparisons,
            "reconstructed_manifest": publication._identity(
                output_dir / "summary_manifest.json", output_dir
            ),
            "boundary": BOUNDARY,
        }
    except BaseException as error:
        dimensions._atomic_json(
            output_dir / "replay.json",
            {
                "schema_version": 1,
                "status": "error",
                "complete": False,
                "scientific_completion": False,
                "archive_sha256": expected_sha256,
                "error_type": type(error).__name__,
                "error": str(error),
                "boundary": BOUNDARY,
            },
        )
        raise
    dimensions._atomic_json(output_dir / "replay.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = replay(args.archive_root, args.expected_sha256, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
