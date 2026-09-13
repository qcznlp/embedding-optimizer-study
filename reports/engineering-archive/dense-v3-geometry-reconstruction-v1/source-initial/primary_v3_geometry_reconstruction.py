"""Reconstruct original geometry aggregates from authenticated local primitives.

No raw weight metric, singular spectrum or spectral health is remeasured. Those
records are authenticated inputs; checkpoint rows and basis overlaps are recomputed.
"""

import argparse
import os
from pathlib import Path

import torch

from . import reconstruction_files as files
from .corrected_geometry_summary import OPTIMIZER_ORDER
from .primary_contract import file_identity, require_same, verify_file
from .primary_v3_geometry import SCOPE, TABLE_COUNTS, tables_from_verified_runs
from .primary_v3_geometry_primitives import inspect_run_primitives
from .primary_v3_outcome_primitives import producer_path, recorded_runs
from .primary_v3_outcomes import csv_bytes
from .primary_v3_reconstruction_inputs import ReconstructionInput
from .primary_v3_validation_io import write_new

BOUNDARY = "Complete geometry only; retrieval bridge, useful dimensions, factorial, publication and formal release remain separate acceptance gates."


def geometry_addresses(inputs):
    """The unique original geometry summary locates a lexical provenance subtree only."""
    original = inputs.evidence["original_bridge_evidence"]["source_bindings"]
    candidates = [row for row in original if producer_path(row["path"]).name == "summary.json"]
    if len(candidates) != 1:
        raise ValueError("Ambiguous or absent original geometry location")
    source_root = producer_path(candidates[0]["path"]).parent
    addresses = []
    for row in original:
        source = producer_path(row["path"])
        if not source.is_relative_to(source_root):
            continue
        if set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("Invalid original geometry source binding")
        name = "geometry/" + source.relative_to(source_root).as_posix()
        files.safe_name(name)
        verify_file(files.ordinary(inputs.root / name), row)
        addresses.append({"original": row["path"], "local": name})
    require_same(
        sorted(row["local"] for row in addresses),
        ["geometry/" + name for name in files.inventory(inputs.root / "geometry")],
    )
    return addresses


def expected_geometry(inputs):
    contract, primary = inputs.contract.bridge.geometry, inputs.contract.primary
    torch.set_num_threads(contract.payload["settings"]["cpu_threads"])
    runs = recorded_runs(primary, inputs.admitted)
    reference = inputs.admitted["reference"]
    root = inputs.root / "geometry"
    run_root = root / "runs"
    require_same(sorted(path.name for path in run_root.iterdir()), sorted(runs))
    admitted = [
        (run_root / run_id, primary.expected_identity(run_id), checked)
        for run_id, checked in runs.items()
    ]
    admitted.sort(
        key=lambda row: (
            OPTIMIZER_ORDER[row[1]["recipe"]["optimizer"]["name"]],
            row[1]["recipe"]["optimizer"]["lr"],
            row[1]["recipe"]["run_id"],
        )
    )
    results, bindings = [], []
    for path, expected, checked in admitted:
        result = inspect_run_primitives(path, expected, checked, reference, contract)
        results.append(result)
        for row in result["manifest"]["outputs"]:
            for key in ("records", "bases"):
                file = path / row[key]["path"]
                bindings.append({"path": file.relative_to(root).as_posix(), **file_identity(file)})
        bindings.append(
            {
                "path": (path / "manifest.json").relative_to(root).as_posix(),
                **file_identity(path / "manifest.json"),
            }
        )
    tables = tables_from_verified_runs(admitted, run_root, reference, results)
    require_same({name: len(rows) for name, rows in tables.items()}, TABLE_COUNTS)
    summary = {
        "scope": SCOPE,
        "primary_protocol_sha256": primary.sha256,
        "geometry_protocol_sha256": contract.sha256,
        "raw_bindings": bindings,
        "table_counts": TABLE_COUNTS,
        "scientific_completion": False,
        "boundary": BOUNDARY,
    }
    require_same(
        sorted(path.name for path in root.iterdir()),
        sorted(["runs", "summary.json", *(name + ".csv" for name in TABLE_COUNTS)]),
    )
    return summary, tables, results, geometry_addresses(inputs)


def reconstruct(inputs, output):
    output = Path(output).absolute()
    if (
        output.exists()
        or output.is_symlink()
        or output.parent.resolve() != output.parent
        or output == inputs.root
        or inputs.root in output.parents
    ):
        raise ValueError("Use a new geometry reconstruction directory outside the archive")
    inputs.recheck()
    summary, tables, records, addresses = expected_geometry(inputs)
    inputs.recheck()
    output.mkdir(parents=True, exist_ok=False)
    geometry = output / "geometry"
    geometry.mkdir()
    write_new(
        output / "reconstruction_plan.json",
        {
            "scope": "dense_primary_v3_geometry_reconstruction",
            "external_archive_sha256": inputs.anchor,
            "geometry_protocol_sha256": inputs.contract.bridge.geometry.sha256,
            "scientific_completion": False,
        },
    )
    write_new(geometry / "summary.json", summary)
    for name, rows in tables.items():
        with (geometry / (name + ".csv")).open("xb") as stream:
            stream.write(csv_bytes(rows))
    # Save new computed aggregates before refusing any contradictory stored derived value.
    write_new(output / "fresh_checkpoint_rows.json", [r["checkpoint_rows"] for r in records])
    for result in records:
        require_same(result["checkpoint_rows"], result["stored_checkpoint_rows"])
    require_same(summary, inputs.evidence["original_bridge_evidence"]["geometry_reader"])
    for name in ("summary.json", *(key + ".csv" for key in TABLE_COUNTS)):
        if (geometry / name).read_bytes() != (inputs.root / "geometry" / name).read_bytes():
            raise ValueError("Stored geometry differs from fresh aggregate recomputation")
    write_new(output / "local_addresses.json", addresses)
    inputs.recheck()
    receipt = {
        "scope": "dense_primary_v3_geometry_reconstruction",
        "external_archive_sha256": inputs.anchor,
        "original_geometry_aggregate_numerics_verified": True,
        "table_counts": TABLE_COUNTS,
        "checkpoints_reconstructed": 60,
        "raw_matrix_records_authenticated": 5280,
        "spectral_health_records_authenticated": 10560,
        "run_pairs_recomputed": 660,
        "original_spectral_and_entry_definitions_preserved": True,
        "upstream_admission_simulated": inputs.evidence.get("upstream_primary_admission_simulated")
        is True,
        "raw_weight_metrics_recomputed": False,
        "spectral_health_remeasured": False,
        "basis_orthogonality_remeasured": False,
        "checkpoint_tensors_revalidated": False,
        "model_encoding_repeated": False,
        "retrieval_repeated": False,
        "original_outcomes_recomputed": False,
        "raw_vector_features_recomputed": False,
        "bridge_and_functional_inference_recomputed": False,
        "primary_scientific_admission": False,
        "manuscript_installed": False,
        "scientific_completion": False,
        "outputs": {name: file_identity(output / name) for name in files.inventory(output)},
    }
    write_new(output / "reconstruction.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Geometry reconstruction is CPU only")
    inputs = ReconstructionInput.load(args.archive_root, args.expected_manifest_sha256)
    verify_file(
        Path(__file__),
        inputs.manifest["files"]["source/src/embed_optim/primary_v3_geometry_reconstruction.py"],
    )
    result = reconstruct(inputs, args.output)
    print(
        {
            "original_geometry_aggregate_numerics_verified": result[
                "original_geometry_aggregate_numerics_verified"
            ],
            "scientific_completion": False,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
