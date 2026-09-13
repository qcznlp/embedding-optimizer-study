"""Checkpoint-backed local reconstruction selection, never remote or paper publication."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import primary_v3_dimension_inference as inference
from . import reconstruction_files as files
from .primary_contract import digest, file_identity, read_json, require_same
from .primary_v3_contract import PrimaryV3Contract
from .primary_v3_outcomes import inspect_bundle
from .primary_v3_reconstruction_sources import (
    INFERENCE_PROTOCOL,
    analysis_selection,
    recheck_selection,
    source_selection,
)
from .runtime import runtime_snapshot

ACCEPTANCE = (
    "reports/engineering-archive/dense-v3-dimension-inference-v1/validation.json",
    "ad04bba6f2bec212a35ad6aee6a71aa725bec1f7673a2b524d9fb7cb11e7989b",
)
INFERENCE_SHA = "d66274878d90e463b74a03cf6e09518ae01dcddfe76584542e3bece435713545"
BOUNDARY = {
    "authoring": "Require complete actual primary checkpoint, validation, BEIR, weight-geometry and raw-vector feature readers before selecting any output",
    "addressing": "External trusted manifest SHA-256 and exact local roles; retain original files unchanged, never use their producer paths as a fallback",
    "payload": "Declared source/protocol files, all 61 raw-vector states and features, full geometry/outcome/bridge/inference bundles, validation candidate scores and BEIR task-result provenance",
    "excluded": "Checkpoint payloads, raw training/probe text, credentials, unrelated files, remote operations and manuscript installation",
    "reconstruction": "Separate raw-vector/statistical reconstruction is required after transport; transport does not repeat model encoding, retrieval or weight-derived measurements",
    "authority": "Preparation only; this does not release any draft parent or authorize formal training/deployment/publication",
}


def require_parent(contract):
    if contract.sha256 != INFERENCE_SHA:
        raise ValueError("Reconstruction requires the accepted functional inference revision")
    path = contract.primary.repository / ACCEPTANCE[0]
    if file_identity(path)["sha256"] != ACCEPTANCE[1]:
        raise ValueError("Changed functional inference acceptance")
    value = read_json(path)
    if (
        value["artifact_validation_passed"] is not True
        or value["scientific_completion"] is not False
    ):
        raise ValueError("Missing bounded inference foundation")
    return path


def prepare(contract, args):
    """Complete source-backed authoring occurs before output creation, not after copying."""
    contract = contract.recheck()
    acceptance = require_parent(contract)
    plan, evidence, tables = inference.gather(contract, args)
    current = inspect_bundle(args.inference_root, plan, evidence, tables)
    _, validation_data = contract.bridge.outcomes.validation.data(args.validation_data)
    selected = source_selection(contract, analysis_selection(args, evidence, current))
    files.select(selected, "source", ACCEPTANCE[0], acceptance, file_identity(acceptance))
    metadata = {
        "scope": "dense_primary_v3_checkpoint_backed_reconstruction_selection",
        "primary_protocol_sha256": contract.primary.sha256,
        "inference_protocol_sha256": contract.sha256,
        "inference_acceptance_sha256": ACCEPTANCE[1],
        "dimension_protocol_sha256": contract.dimensions.sha256,
        "trusted_vector_manifest_sha256": args.vector_manifest_sha256,
        "authoring_plan": plan,
        "authoring_evidence_sha256": digest(evidence),
        "validation_identity": validation_data,
        "original_location_fields_preserved": True,
        "boundary": BOUNDARY,
        "authoring_runtime": runtime_snapshot(
            [
                "torch",
                "numpy",
                "scipy",
                "sympy",
                "sentence-transformers",
                "transformers",
                "accelerate",
                "mteb",
            ]
        ),
        "full_raw_vector_reconstruction_repeated_by_transport": False,
        "manuscript_installed": False,
        "scientific_completion": False,
    }
    inference.original.verify_snapshot(evidence["source_bindings"])
    inspect_bundle(args.inference_root, plan, evidence, tables)
    contract.recheck()
    recheck_selection(selected)
    return selected, metadata


def build(contract, args):
    selected, metadata = prepare(contract, args)
    result = files.write(args.output, selected, metadata)
    recheck_selection(selected)
    contract.recheck()
    require_same(files.inspect(args.output, result["manifest_sha256"])["metadata"], metadata)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repository",
        "training-root",
        "experiment-root",
        "results-root",
        "validation-data",
        "validation-root",
        "outcomes-root",
        "geometry-root",
        "reference",
        "bridge-root",
        "dimension-vectors",
        "dimension-features",
        "probe-root",
        "inference-root",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--vector-manifest-sha256", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Reconstruction authoring is CPU only")
    primary = PrimaryV3Contract.load(
        args.repository / inference.PARENTS["primary"][0], args.repository, args.training_root
    )
    contract = inference.FunctionalInferenceContract.load(
        args.repository / INFERENCE_PROTOCOL, primary
    )
    print(build(contract, args))


if __name__ == "__main__":
    main()
