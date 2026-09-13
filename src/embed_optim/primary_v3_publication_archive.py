"""Checkpoint-backed publication transport; no training, installation or release."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import primary_v3_publication as publication
from . import primary_v3_reconstruction_authoring as authoring
from . import reconstruction_files as files
from .primary_contract import digest, file_identity, read_json, require_same, verify_file
from .primary_v3_contract import PrimaryV3Contract
from .primary_v3_publication_contract import OUTPUTS, PublicationContract
from .primary_v3_reconstruction_sources import (
    analysis_selection,
    payload_sources,
    recheck_selection,
    source_selection,
)
from .runtime import runtime_snapshot

PROTOCOL = "configs/dense_primary_v3_publication_protocol.json"
PROTOCOL_SHA = "39c8740962fbfa57770bc21058bfc18b3f862cf97f7b12f48ad29dcd04b43f81"
ACCEPTANCE = "reports/engineering-archive/dense-v3-primary-publication-v1/validation.json"
ACCEPTANCE_SHA = "34a28795f0a6b1682971e8e0424137224a18b4ffafa80f4a5f3bae38e7d65e8c"
ROLE = "provenance/primary-publication"
META = "primary_publication_preparation"
BOUNDARY = {
    "scope": "dense_primary_v3_publication_reconstruction_selection",
    "source": "Complete archived package and actual loaded primary/inference/publication parents",
    "reconstruction": "All original joint raw branches, inference and seven publication outputs",
    "authority": "Preparation only; no draft release, manuscript installation or primary admission by transport",
    "producer_paths": "Preserved lexical provenance; never open as a reconstruction fallback",
}


def require_foundation(contract):
    root = contract.primary.repository
    if contract.sha256 != PROTOCOL_SHA:
        raise ValueError("Publication reconstruction requires its frozen preparation protocol")
    verify_file(root / PROTOCOL, {"bytes": 12180, "sha256": PROTOCOL_SHA})
    verify_file(root / ACCEPTANCE, {"bytes": 107271, "sha256": ACCEPTANCE_SHA})
    accepted = read_json(root / ACCEPTANCE)
    if (
        accepted["artifact_validation_passed"] is not True
        or accepted["scientific_completion"] is not False
    ):
        raise ValueError("Missing bounded publication preparation foundation")


def extension(contract, plan, evidence):
    """Pure metadata, not a primary-admission helper; the authoring entry checks inputs."""
    return {
        "boundary": BOUNDARY,
        "publication_protocol_sha256": contract.sha256,
        "preparation_acceptance_sha256": ACCEPTANCE_SHA,
        "publication_plan": plan,
        "publication_evidence_sha256": digest(evidence),
        "manuscript_installed": False,
        "scientific_completion": False,
    }


def add_publication_sources(contract, selected):
    require_foundation(contract)
    root = contract.primary.repository
    payload_sources(selected, root, contract.payload)
    for name in (PROTOCOL, ACCEPTANCE):
        files.select(selected, "source", name, root / name, file_identity(root / name))
    return selected


def prepare(contract, args):
    """One actual full gather precedes all publication selection and output creation."""
    contract = contract.recheck()
    require_foundation(contract)
    plan, contents, bound = publication.gather(contract, args)
    evidence = json.loads(contents["evidence.json"])
    publication.reject_simulation((plan, evidence))
    pub_root = Path(args.publication_root).absolute()
    paper = contract.primary.repository / "paper"
    if pub_root == paper or paper in pub_root.parents:
        raise ValueError("Publication preparation cannot be a manuscript destination")
    current = publication.inspect(pub_root, plan, contents)
    functional = evidence["functional_evidence"]
    functional_contract = contract.inference
    acceptance = authoring.require_parent(functional_contract)
    _, validation_data = functional_contract.bridge.outcomes.validation.data(args.validation_data)
    selected = source_selection(
        functional_contract,
        analysis_selection(args, functional, evidence["functional_reader"]),
    )
    files.select(selected, "source", authoring.ACCEPTANCE[0], acceptance, file_identity(acceptance))
    add_publication_sources(contract, selected)
    for name in (*OUTPUTS, "manifest.json"):
        record = (
            file_identity(pub_root / name)
            if name == "manifest.json"
            else current["manifest"]["outputs"][name]
        )
        files.select(selected, "provenance", "primary-publication/" + name, pub_root / name, record)
    metadata = {
        "scope": "dense_primary_v3_checkpoint_backed_reconstruction_selection",
        "primary_protocol_sha256": contract.primary.sha256,
        "inference_protocol_sha256": functional_contract.sha256,
        "inference_acceptance_sha256": authoring.ACCEPTANCE[1],
        "dimension_protocol_sha256": functional_contract.dimensions.sha256,
        "trusted_vector_manifest_sha256": args.vector_manifest_sha256,
        "authoring_plan": evidence["functional_plan"],
        "authoring_evidence_sha256": digest(functional),
        "validation_identity": validation_data,
        "original_location_fields_preserved": True,
        "boundary": authoring.BOUNDARY,
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
        META: extension(contract, plan, evidence),
    }
    publication.reject_simulation(metadata)
    publication.inference.original.verify_snapshot(bound)
    publication.inspect(pub_root, plan, contents)
    contract.recheck()
    recheck_selection(selected)
    verify_file(
        Path(__file__), selected["source/src/embed_optim/primary_v3_publication_archive.py"][1]
    )
    return selected, metadata


def build(contract, args):
    selected, metadata = prepare(contract, args)
    output = Path(args.output).absolute()
    paper = contract.primary.repository / "paper"
    if output == paper or paper in output.parents:
        raise ValueError("Archive preparation never writes manuscript destinations")
    result = files.write(output, selected, metadata)
    recheck_selection(selected)
    contract.recheck()
    require_same(files.inspect(output, result["manifest_sha256"])["metadata"], metadata)
    return result


def main(argv=None):
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
        "publication-root",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--vector-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Publication archive preparation is CPU only")
    primary = PrimaryV3Contract.load(
        args.repository / publication.PARENTS["primary"][0], args.repository, args.training_root
    )
    contract = PublicationContract.load(args.repository / PROTOCOL, primary)
    print(build(contract, args))


if __name__ == "__main__":
    main()
