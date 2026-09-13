"""Complete original/functional/exact transport after actual checkpoint-backed authoring."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import primary_v3_exact_bridge as exact
from . import primary_v3_publication as publication
from . import primary_v3_publication_archive as parent
from . import reconstruction_files as files
from .primary_contract import digest, file_identity, read_json, require_same, verify_file
from .primary_v3_contract import PrimaryV3Contract
from .primary_v3_publication_contract import PublicationContract
from .primary_v3_reconstruction_sources import payload_sources, recheck_selection

PROTOCOL = "configs/dense_primary_v3_exact_bridge_protocol.json"
PROTOCOL_SHA = "270a908d4242a831ce239f6d6258a19b3884329646f076c0af43ab968e5c9fdc"
GEOMETRY_ROLE = "provenance/exact-geometry"
BRIDGE_ROLE = "provenance/exact-bridge"
META = "exact_measurement_reconstruction_preparation"
BOUNDARY = {
    "scope": "dense_primary_v3_exact_measurement_reconstruction_selection",
    "population": "All 60 states, 88 hidden matrices, both displacements and all five exact features",
    "original": "Preserve all nine original weight features and the complete functional inference",
    "numerics": "Recompute spectrum-defined metrics, status, aggregates and exact rational sensitivity",
    "retained": "Saved singular spectra, bases, SVD residuals and weight norms are authenticated measurements, not fresh SVD or checkpoint validation",
    "authority": "Preparation only; no primary admission, manuscript installation or release by transport",
}


def extension(contract, plan, evidence):
    return {
        "boundary": BOUNDARY,
        "exact_bridge_protocol_sha256": contract.sha256,
        "exact_geometry_protocol_sha256": contract.exact.sha256,
        "exact_bridge_plan": plan,
        "exact_bridge_evidence_sha256": digest(evidence),
        "manuscript_installed": False,
        "scientific_completion": False,
    }


def add_sources(contract, selected):
    if contract.sha256 != PROTOCOL_SHA:
        raise ValueError("Exact reconstruction requires the frozen five-feature sensitivity")
    root = contract.primary.repository
    for current in (contract, contract.exact):
        files.select(
            selected,
            "source",
            current.path.relative_to(root).as_posix(),
            current.path,
            file_identity(current.path),
        )
        payload_sources(selected, root, current.payload)
    return selected


def prepare(contract, args):
    """Never offer a transport shortcut around either full actual authoring gather."""
    selected, metadata = parent.prepare(contract, args)
    exact_contract = exact.ExactBridgeContract.load(
        contract.primary.repository / PROTOCOL, contract.primary
    )
    plan, evidence, tables = exact.gather(exact_contract, args)
    publication.reject_simulation((metadata, plan, evidence))
    exact_root = Path(args.exact_bridge_root).absolute()
    geometry_root = Path(args.exact_geometry_root).absolute()
    paper = contract.primary.repository / "paper"
    if any(path == paper or paper in path.parents for path in (exact_root, geometry_root)):
        raise ValueError("Exact authoring inputs cannot be manuscript destinations")
    current = exact.inspect_bundle(exact_root, plan, evidence, tables)
    require_same(
        evidence["original_bridge_evidence"],
        read_json(Path(args.inference_root) / "evidence.json")["original_bridge_evidence"],
    )
    add_sources(exact_contract, selected)
    bound = {row["path"]: row for row in evidence["source_bindings"]}
    for name in files.inventory(geometry_root):
        path = geometry_root / name
        record = bound.get(str(path))
        if record is None:
            raise ValueError("Exact geometry selection lacks full authoring provenance")
        files.select(selected, "provenance", "exact-geometry/" + name, path, record)
    for name in ("manifest.json", "evidence.json", *(key + ".csv" for key in exact.TABLE_COUNTS)):
        record = (
            file_identity(exact_root / name)
            if name == "manifest.json"
            else current["outputs"][name]
        )
        files.select(selected, "provenance", "exact-bridge/" + name, exact_root / name, record)
    metadata = {**metadata, META: extension(exact_contract, plan, evidence)}
    publication.reject_simulation(metadata)
    exact.original.verify_snapshot(evidence["source_bindings"])
    require_same(exact.inspect_bundle(exact_root, plan, evidence, tables), current)
    exact_contract.recheck()
    contract.recheck()
    recheck_selection(selected)
    verify_file(
        Path(__file__),
        selected["source/src/embed_optim/primary_v3_exact_reconstruction_archive.py"][1],
    )
    return selected, metadata


def build(contract, args):
    selected, metadata = prepare(contract, args)
    output = Path(args.output).absolute()
    paper = contract.primary.repository / "paper"
    if output == paper or paper in output.parents:
        raise ValueError("Exact archive preparation never writes manuscript destinations")
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
        "exact-geometry-root",
        "exact-bridge-root",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--vector-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Exact archive preparation is CPU only")
    primary = PrimaryV3Contract.load(
        args.repository / publication.PARENTS["primary"][0], args.repository, args.training_root
    )
    contract = PublicationContract.load(args.repository / parent.PROTOCOL, primary)
    print(build(contract, args))


if __name__ == "__main__":
    main()
