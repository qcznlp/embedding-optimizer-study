"""Exact CPU reconstruction of all 61 archived vector states into literal features.

No producer path, checkpoint payload, raw text, network or model encoder is used.
Inference, manuscript publication and primary eligibility are deliberately not
certified by this raw-vector-to-feature reader.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import primary_v3_dimensions as feature
from . import reconstruction_files as files
from .dimension_intervention_io import inspect_state, save_state
from .dimension_interventions import compute_state
from .primary_contract import file_identity, require_same
from .primary_v3_dimension_contract import TABLE_COUNTS
from .primary_v3_dimension_exports import _manifest
from .primary_v3_outcomes import csv_bytes
from .primary_v3_reconstruction_inputs import ReconstructionInput, raw_vectors
from .primary_v3_validation_io import write_new


def reconstruct(inputs, output, *, progress=None):
    """Caller supplies authenticated local inputs; every numerical state is recomputed."""
    output = Path(output).absolute()
    if (
        output.exists()
        or output.is_symlink()
        or output.parent.resolve() != output.parent
        or output == inputs.root
        or inputs.root in output.parents
    ):
        raise ValueError("Use a new reconstruction directory outside the immutable archive")
    inputs.recheck()
    root, admitted, jobs = inputs.root, inputs.admitted, inputs.jobs
    contract = inputs.contract.dimensions
    anchor = inputs.manifest["metadata"]["trusted_vector_manifest_sha256"]
    vectors = _manifest(root / "vectors", admitted, anchor)
    plan = feature.feature_plan(contract, admitted, anchor)
    original = feature._feature_manifest(root / "features", plan, jobs)
    expected_reader = inputs.evidence["dimension_feature_reader"]
    require_same(
        expected_reader,
        {
            "status": "complete",
            "manifest_sha256": file_identity(root / "features/manifest.json")["sha256"],
            "plan": plan,
            "table_counts": TABLE_COUNTS,
            "raw_vector_states_recomputed": 61,
            "model_encoding_repeated": False,
            "scientific_completion": False,
        },
    )
    output.mkdir(parents=True, exist_ok=False)
    write_new(
        output / "reconstruction_plan.json",
        {
            "scope": "dense_primary_v3_raw_vector_reconstruction",
            "external_archive_sha256": inputs.anchor,
            "original_feature_plan": plan,
            "comparison": "Exact CSV bytes and FP64 array elements; no numerical tolerance",
            "scientific_completion": False,
        },
    )
    tables = {name: [] for name in TABLE_COUNTS}
    identities = admitted["probe"]["row_identities"]
    task_names = sorted({row["source"] for row in identities})
    receipts = []
    for index, job in enumerate(jobs, 1):
        state = job["plan"]["state"]
        cell = state["cell"]
        arrays = raw_vectors(
            root / "vectors/states" / cell,
            job["plan"],
            identities,
            admitted["reference"]["all_shapes"],
            vectors["states"][cell],
        )
        result = compute_state(state["meta"], arrays, contract.scientific)
        feature.check_result(result, state, task_names)
        current = feature.state_plan(plan, job, vectors["states"][cell])
        target = output / "states" / cell
        target.parent.mkdir(parents=True, exist_ok=True)
        # Retain newly computed evidence before comparison, including a failing state.
        save_state(target, current, result)
        checked = inspect_state(root / "features/states" / cell, current, result)
        for name, rows in result["tables"].items():
            tables[name].extend(rows)
        receipts.append(
            {
                "cell": cell,
                "raw_vector_manifest": vectors["states"][cell],
                "original_feature_manifest": original["states"][cell],
                "fresh_numerics_verified": checked["fresh_numerics_verified"],
            }
        )
        if progress:
            progress({"completed_states": index, "total_states": 61, "cell": cell})
        del arrays, result
    require_same({key: len(rows) for key, rows in tables.items()}, TABLE_COUNTS)
    for name, rows in tables.items():
        data = csv_bytes(rows)
        with (output / (name + ".csv")).open("xb") as stream:
            stream.write(data)
        if (root / "features" / (name + ".csv")).read_bytes() != data:
            raise ValueError("Archived aggregate differs from exact full raw-vector reconstruction")
    require_same(feature._feature_manifest(root / "features", plan, jobs), original)
    inputs.recheck()
    receipt = {
        "scope": "dense_primary_v3_raw_vector_reconstruction",
        "external_archive_sha256": inputs.anchor,
        "raw_vector_states_recomputed": len(receipts),
        "table_counts": TABLE_COUNTS,
        "state_checks": receipts,
        "exact_numerics_verified": True,
        "comparison_tolerance": None,
        "runtime": inputs.runtime,
        "upstream_admission_simulated": inputs.evidence.get("upstream_primary_admission_simulated")
        is True,
        "model_encoding_repeated": False,
        "checkpoint_tensors_revalidated": False,
        "raw_probe_text_revalidated": False,
        "outcome_geometry_primitives_recomputed": False,
        "publication_inference_recomputed": False,
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
        raise ValueError("Raw-vector reconstruction is CPU only")
    inputs = ReconstructionInput.load(args.archive_root, args.expected_manifest_sha256)
    # -m executes this module under __main__, outside the package-name import check.
    name = "source/src/embed_optim/primary_v3_vector_reconstruction.py"
    from .primary_contract import verify_file

    verify_file(Path(__file__), inputs.manifest["files"][name])
    result = reconstruct(inputs, args.output, progress=lambda row: print(row, flush=True))
    print(
        {
            "exact_numerics_verified": result["exact_numerics_verified"],
            "scientific_completion": False,
        }
    )


if __name__ == "__main__":
    main()
