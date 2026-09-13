"""CPU-only original-outcome reconstruction from externally anchored local roles.

This repeats validation metric checking/selection and original outcome statistics,
not model inference, corpus retrieval, checkpoint verification or paper publication.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import reconstruction_files as files
from .primary_contract import digest, file_identity, read_json, require_same, verify_file
from .primary_v3_outcome_primitives import grid_from_tasks, recorded_runs, selection_from_records
from .primary_v3_outcomes import (
    BOUNDARY,
    SCOPE,
    TABLE_COUNTS,
    inspect_bundle,
    outcome_tables,
    save_bundle,
    system_rows,
)
from .primary_v3_reconstruction_inputs import ReconstructionInput
from .primary_v3_validation_io import write_new


def expected_outcomes(inputs):
    contract = inputs.contract.bridge.outcomes
    primary = contract.primary
    runs = recorded_runs(primary, inputs.admitted)
    original = inputs.evidence["original_bridge_evidence"]["outcome_evidence"]
    require_same(read_json(inputs.root / "outcomes/evidence.json"), original)
    # Each function receives only its required branch; no BEIR value enters selection.
    selected, validation_addresses = selection_from_records(
        inputs, runs, original["validation_selection"]
    )
    grid, beir_addresses = grid_from_tasks(inputs, runs, original["grid"])
    tables = outcome_tables(primary, grid["score_rows"], selected)
    tables["system_metrics"] = system_rows(primary, grid["runs"])
    require_same({name: len(rows) for name, rows in tables.items()}, TABLE_COUNTS)
    evidence = {"grid": grid, "validation_selection": selected}
    plan = {
        "scope": SCOPE,
        "outcome_protocol_sha256": contract.sha256,
        "primary_protocol_sha256": primary.sha256,
        "validation_protocol_sha256": contract.validation.sha256,
        "evidence_sha256": digest(evidence),
        "scientific_completion": False,
        "scientific_rules": contract.payload["scientific_rules"],
        "boundary": BOUNDARY,
    }
    return plan, evidence, tables, validation_addresses + beir_addresses


def reconstruct(inputs, output):
    output = Path(output).absolute()
    if (
        output.exists()
        or output.is_symlink()
        or output.parent.resolve() != output.parent
        or output == inputs.root
        or inputs.root in output.parents
    ):
        raise ValueError("Use a new outcome reconstruction directory outside the immutable archive")
    inputs.recheck()
    plan, evidence, tables, addresses = expected_outcomes(inputs)
    inputs.recheck()
    output.mkdir(parents=True, exist_ok=False)
    write_new(
        output / "reconstruction_plan.json",
        {
            "scope": "dense_primary_v3_outcome_reconstruction",
            "external_archive_sha256": inputs.anchor,
            "original_outcome_plan": plan,
            "scientific_completion": False,
        },
    )
    # A failing stored aggregate still leaves the separately computed new numerical evidence.
    fresh = save_bundle(output / "outcomes", plan, evidence, tables)
    original = inspect_bundle(inputs.root / "outcomes", plan, evidence, tables)
    require_same(fresh, original)
    write_new(output / "local_addresses.json", addresses)
    inputs.recheck()
    receipt = {
        "scope": "dense_primary_v3_outcome_reconstruction",
        "external_archive_sha256": inputs.anchor,
        "original_outcome_numerics_verified": True,
        "table_counts": TABLE_COUNTS,
        "validation_jobs": 12,
        "validation_records_replayed": 49152,
        "validation_selection_recomputed": True,
        "beir_task_results_reparsed": 840,
        "comparison": "Exact original evidence and outcome table bytes",
        "validation_replay_changes_recorded_metrics": False,
        "upstream_admission_simulated": inputs.evidence.get("upstream_primary_admission_simulated")
        is True,
        "checkpoint_tensors_revalidated": False,
        "model_encoding_repeated": False,
        "retrieval_repeated": False,
        "raw_validation_text_revalidated": False,
        "timing_remeasured": False,
        "raw_vector_features_recomputed": False,
        "geometry_primitives_recomputed": False,
        "functional_inference_recomputed": False,
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
        raise ValueError("Outcome reconstruction is CPU only")
    inputs = ReconstructionInput.load(args.archive_root, args.expected_manifest_sha256)
    verify_file(
        Path(__file__),
        inputs.manifest["files"]["source/src/embed_optim/primary_v3_outcome_reconstruction.py"],
    )
    result = reconstruct(inputs, args.output)
    print(
        {
            "original_outcome_numerics_verified": result["original_outcome_numerics_verified"],
            "scientific_completion": False,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
