"""Recompute the full local publication closure; never admit models or release a paper."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import primary_v3_joint_reconstruction as joint
from . import primary_v3_publication as publication
from . import primary_v3_publication_archive as archive
from . import reconstruction_files as files
from .primary_contract import digest, file_identity, read_json, require_same, verify_file
from .primary_v3_outcome_primitives import producer_path
from .primary_v3_publication_contract import OUTPUTS, RULES, SCOPE, PublicationContract
from .primary_v3_reconstruction_inputs import ReconstructionInput
from .primary_v3_validation_io import write_new


def simulation(value):
    """Keep synthetic provenance explicit even on the numerical-only reader."""
    found = False
    if isinstance(value, dict):
        for key, item in value.items():
            if key in publication.SIMULATION_KEYS:
                if type(item) is not bool:
                    raise ValueError("A simulation annotation must be an explicit Boolean")
                found |= item
            found |= simulation(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            found |= simulation(item)
    return found


def publication_plan(contract, evidence):
    return {
        "scope": SCOPE,
        "publication_protocol_sha256": contract.sha256,
        "primary_protocol_sha256": contract.primary.sha256,
        "inference_protocol_sha256": contract.inference.sha256,
        "evidence_sha256": digest(evidence),
        "scientific_completion": False,
        "manuscript_installed": False,
        "final_portable_publication_verified": False,
    }


def provenance(inputs, evidence):
    prefix = inputs.evidence["source_bindings"]
    names = joint.bundle_names(publication.inference.TABLE_COUNTS)
    bound = evidence["source_bindings"]
    if not isinstance(bound, list) or len(bound) != len(prefix) + len(names):
        raise ValueError("Incomplete publication provenance population")
    require_same(bound[: len(prefix)], prefix)
    inferred_root, addresses = joint.local_bindings(
        inputs, bound[len(prefix) :], "inference", names
    )
    previous = joint.provenance(
        inputs, inputs.evidence["original_bridge_evidence"]["geometry_reader"]
    )
    for row in previous:
        local = producer_path(row["local"])
        original = producer_path(row["original"]).parents[len(local.parts) - 2]
        if inferred_root.is_relative_to(original) or original.is_relative_to(inferred_root):
            raise ValueError("Original publication inference role overlaps another analysis role")
    return addresses


def admitted_structure(inputs):
    """Validate stored authoring identities; do not claim new checkpoint admission."""
    contract = PublicationContract.load(
        inputs.root / "source" / archive.PROTOCOL, inputs.contract.primary
    )
    archive.require_foundation(contract)
    require_same(contract.inference.sha256, inputs.contract.sha256)
    evidence = read_json(inputs.root / archive.ROLE / "evidence.json")
    expected = {
        "functional_plan": inputs.manifest["metadata"]["authoring_plan"],
        "functional_evidence": inputs.evidence,
        "functional_reader": publication.inspect_bundle(
            inputs.root / "inference",
            inputs.manifest["metadata"]["authoring_plan"],
            inputs.evidence,
            publication.read_tables(inputs.root / "inference", publication.inference.TABLE_COUNTS),
        ),
        "source_bindings": evidence["source_bindings"],
        "publication_rules": RULES,
        "manuscript_installed": False,
        "scientific_completion": False,
    }
    require_same(evidence, expected)
    plan = publication_plan(contract, evidence)
    require_same(
        inputs.manifest["metadata"][archive.META], archive.extension(contract, plan, evidence)
    )
    require_same(files.inventory(inputs.root / archive.ROLE), sorted([*OUTPUTS, "manifest.json"]))
    require_same(read_json(inputs.root / archive.ROLE / "manifest.json")["plan"], plan)
    addresses = provenance(inputs, evidence)
    simulation(inputs.manifest["metadata"])
    simulation(evidence)
    return contract, plan, evidence, addresses


def reconstruct_publication(inputs, root, joint_receipt, contract, plan, original):
    """Only the current joint invocation's complete, bound outputs feed generation."""
    require_same(joint_receipt["external_archive_sha256"], inputs.anchor)
    for name, record in joint_receipt["outputs"].items():
        verify_file(files.ordinary(root / "joint" / name), record)
    tables = {}
    for kind, relative, counts in (
        ("outcomes", "outcomes/outcomes", publication.TABLE_COUNTS),
        ("geometry", "geometry/geometry", publication.geometry.TABLE_COUNTS),
        ("bridge", "bridge", publication.inference.original.TABLE_COUNTS),
        ("functional", "inference", publication.inference.TABLE_COUNTS),
    ):
        tables[kind] = publication.read_tables(root / "joint" / relative, counts)
    functional = read_json(root / "joint/inference/evidence.json")
    outcomes = read_json(root / "joint/outcomes/outcomes/evidence.json")
    evidence = {
        "functional_plan": joint_receipt["functional_inference_reader"]["plan"],
        "functional_evidence": functional,
        "functional_reader": joint_receipt["functional_inference_reader"],
        "source_bindings": original["source_bindings"],
        "publication_rules": RULES,
        "manuscript_installed": False,
        "scientific_completion": False,
    }
    require_same(evidence, original)
    contents = publication.generate(
        contract.primary,
        tables,
        outcomes["validation_selection"],
        outcomes["grid"]["runs"],
        functional["decisions"],
        evidence,
    )
    checked = publication.inspect(inputs.root / archive.ROLE, plan, contents)
    require_same(publication.save(root / "publication", plan, contents), checked)
    return checked


def reconstruct(inputs, output, *, progress=None):
    output = Path(output).absolute()
    if (
        output.exists()
        or output.is_symlink()
        or output.parent.resolve() != output.parent
        or output == inputs.root
        or inputs.root in output.parents
    ):
        raise ValueError(
            "Use a new publication reconstruction directory outside the immutable archive"
        )
    inputs.recheck()
    contract, plan, evidence, addresses = admitted_structure(inputs)
    output.mkdir(parents=True, exist_ok=False)
    current = joint.reconstruct(inputs, output / "joint", progress=progress)
    require_same(read_json(output / "joint/reconstruction.json"), current)
    for flag in (
        "all_three_raw_branches_reconstructed",
        "original_and_functional_inference_recomputed",
        "original_nine_geometry_features_retained",
        "exact_rendered_latex_verified",
    ):
        if current[flag] is not True:
            raise ValueError("Publication requires the complete fresh joint reconstruction")
    checked = reconstruct_publication(inputs, output, current, contract, plan, evidence)
    write_new(output / "publication_addresses.json", addresses)
    inputs.recheck()
    contract.recheck()
    receipt = {
        "scope": "dense_primary_v3_publication_numerical_reconstruction",
        "external_archive_sha256": inputs.anchor,
        "publication_protocol_sha256": contract.sha256,
        "joint_reconstruction": file_identity(output / "joint/reconstruction.json"),
        "publication_reader": checked,
        "all_joint_raw_branches_and_inference_recomputed": True,
        "all_seven_publication_outputs_recomputed": True,
        "raw_functional_latex_and_label_only_include_preserved": True,
        "upstream_admission_simulated": simulation((inputs.manifest["metadata"], evidence)),
        "checkpoint_tensors_revalidated": False,
        "weight_spectra_or_health_remeasured": False,
        "model_encoding_repeated": False,
        "retrieval_repeated": False,
        "physical_cross_host_execution_verified": False,
        "primary_scientific_admission": False,
        "manuscript_installed": False,
        "reviewed_source_runtime_release_verified": False,
        "complete_primary_publication_pipeline_verified": False,
        "scientific_completion": False,
        "outputs": {name: file_identity(output / name) for name in files.inventory(output)},
    }
    write_new(output / "reconstruction.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Publication reconstruction is CPU only")
    inputs = ReconstructionInput.load(args.archive_root, args.expected_manifest_sha256)
    verify_file(
        Path(__file__),
        inputs.manifest["files"]["source/src/embed_optim/primary_v3_publication_reconstruction.py"],
    )
    result = reconstruct(inputs, args.output, progress=lambda row: print(row, flush=True))
    print(
        {
            "publication_numerics_reconstructed": True,
            "upstream_admission_simulated": result["upstream_admission_simulated"],
            "scientific_completion": False,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
