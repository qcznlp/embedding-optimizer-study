"""Checkpoint-backed original/exact manuscript evidence in a new preparation namespace."""

import argparse
import csv
import json
import os
from fractions import Fraction
from pathlib import Path

from . import primary_v3_exact_bridge as exact
from . import primary_v3_exact_publication_render as rendering
from . import primary_v3_publication as original
from .primary_contract import canonical, digest, file_identity, read_json, require_same, verify_file
from .primary_v3_contract import PrimaryV3Contract
from .primary_v3_exact_publication_contract import (
    OUTPUTS,
    PARENTS,
    RULES,
    SCOPE,
    ExactPublicationContract,
)
from .primary_v3_outcomes import inspect_bundle
from .primary_v3_validation_io import write_new

DIFFERENCE_FIELD = "original_minus_exact_feature_mse_rational"
ARGUMENT_PATHS = (
    "repository",
    "training-root",
    "protocol",
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
)


def read_exact_tables(root):
    """Extend the frozen decoder only for the new exact-difference field, including '0'."""
    tables = original.read_tables(root, exact.TABLE_COUNTS)
    name = "predictive_sensitivity_summary"
    with (Path(root) / (name + ".csv")).open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    for row, serialized in zip(tables[name], raw, strict=True):
        value = serialized[DIFFERENCE_FIELD]
        if value == "":
            row[DIFFERENCE_FIELD] = None
            continue
        try:
            canonical_value = str(Fraction(value))
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError("Invalid exact sensitivity rational difference") from error
        if canonical_value != value:
            raise ValueError("Noncanonical exact sensitivity rational difference")
        row[DIFFERENCE_FIELD] = value
    return tables


def generate(primary, original_contents, geometry, bridge, evidence):
    """Pure complete reconstruction; a caller must separately admit real primary inputs."""
    require_same(sorted(original_contents), sorted(original.OUTPUTS))
    original_tables = json.loads(original_contents["tables.json"])
    original_evidence = json.loads(original_contents["evidence.json"])
    functional = original_evidence["functional_evidence"]
    source = functional["original_bridge_evidence"]["outcome_evidence"]
    expected = original.generate(
        primary,
        original_tables,
        source["validation_selection"],
        source["grid"]["runs"],
        functional["decisions"],
        original_evidence,
    )
    if original_contents != expected:
        raise ValueError("Original publication differs from its complete fresh reconstruction")
    require_same(
        {name: len(rows) for name, rows in geometry.items()}, exact.exact_geometry.TABLE_COUNTS
    )
    summary = rendering.checked_sensitivity(
        primary,
        original_tables["bridge"],
        geometry["checkpoint_exact_geometry"],
        geometry["run_pair_exact_subspace_overlap"],
        bridge,
    )
    exact_latex = rendering.render(summary).encode()
    contents = dict(original_contents)
    contents["original-optimizer-primary.tex"] = original_contents["optimizer-primary.tex"]
    contents["optimizer-primary.tex"] = rendering.combine(
        original_contents["optimizer-primary.tex"], exact_latex
    )
    values = {
        "evidence.json": evidence,
        "tables.json": {**original_tables, "exact_geometry": geometry, "exact_bridge": bridge},
        "findings.json": {
            **json.loads(original_contents["findings.json"]),
            "exact_geometry_prediction": rendering.finding(summary["rows"]),
        },
        "exact-summary.json": summary,
    }
    contents.update({name: canonical(value) + b"\n" for name, value in values.items()})
    contents["exact-sensitivity.tex"] = exact_latex
    require_same(sorted(contents), sorted(OUTPUTS))
    return contents


def gather(contract, args):
    contract = contract.recheck()
    # Real complete admission comes first; missing/simulated primary evidence is not a fixture mode.
    original_plan, original_contents, original_bound = original.gather(contract.publication, args)
    original_evidence = json.loads(original_contents["evidence.json"])
    original.reject_simulation(original_plan)
    original.reject_simulation(original_evidence)
    original_reader = original.inspect(args.publication_root, original_plan, original_contents)
    exact_plan, exact_evidence, bridge = exact.gather(contract.exact, args)
    original.reject_simulation(exact_plan)
    original.reject_simulation(exact_evidence)
    require_same(
        original_evidence["functional_evidence"]["original_bridge_evidence"],
        exact_evidence["original_bridge_evidence"],
    )
    exact_reader = inspect_bundle(args.exact_bridge_root, exact_plan, exact_evidence, bridge)
    require_same(read_exact_tables(args.exact_bridge_root), bridge)
    bound = list(original_bound)
    bound += exact.original.snapshot(args.publication_root, ["manifest.json", *original.OUTPUTS])
    bound += exact_evidence["source_bindings"]
    bound += exact.original.snapshot(
        args.exact_bridge_root,
        ["manifest.json", "evidence.json", *(name + ".csv" for name in exact.TABLE_COUNTS)],
    )
    geometry = original.read_tables(args.exact_geometry_root, exact.exact_geometry.TABLE_COUNTS)
    evidence = {
        "original_publication_plan": original_plan,
        "original_publication_evidence": original_evidence,
        "original_publication_reader": original_reader,
        "exact_bridge_plan": exact_plan,
        "exact_bridge_evidence": exact_evidence,
        "exact_bridge_reader": exact_reader,
        "source_bindings": bound,
        "publication_rules": RULES,
        "manuscript_installed": False,
        "scientific_completion": False,
    }
    contents = generate(contract.primary, original_contents, geometry, bridge, evidence)
    summary = json.loads(contents["exact-summary.json"])
    require_same(
        summary["diagnostics"],
        {key: exact_evidence[key] for key in ("numerical_diagnostics", "measurement_coverage")},
    )
    exact.original.verify_snapshot(bound)
    contract.recheck()
    return (
        {
            "scope": SCOPE,
            "exact_publication_protocol_sha256": contract.sha256,
            "primary_protocol_sha256": contract.primary.sha256,
            "original_publication_protocol_sha256": contract.publication.sha256,
            "exact_bridge_protocol_sha256": contract.exact.sha256,
            "evidence_sha256": digest(evidence),
            "scientific_completion": False,
            "manuscript_installed": False,
            "final_portable_publication_verified": False,
        },
        contents,
        bound,
    )


def inspect(output, plan, contents):
    """Rehashed output edits still fail an exact fresh-evidence comparison."""
    output = Path(output).absolute()
    if not output.is_dir() or any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError("Require an ordinary exact publication preparation directory")
    require_same(sorted(contents), sorted(OUTPUTS))
    if {path.name for path in output.iterdir()} != {*OUTPUTS, "manifest.json"}:
        raise ValueError("Exact publication inventory differs")
    if not (output / "manifest.json").is_file() or (output / "manifest.json").is_symlink():
        raise ValueError("Require an ordinary exact publication manifest")
    manifest = read_json(output / "manifest.json")
    expected = {"status": "complete_preparation", "plan": plan, "outputs": {}}
    for name, content in contents.items():
        path = output / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("Require ordinary exact publication evidence files")
        verify_file(path, manifest["outputs"][name])
        if path.read_bytes() != content:
            raise ValueError(
                "Exact publication differs from freshly reconstructed evidence: " + name
            )
        expected["outputs"][name] = {"path": name, **file_identity(path)}
    require_same(manifest, expected)
    return {
        "recomputed_bytes_verified": True,
        "manifest": expected,
        "manuscript_installed": False,
        "scientific_completion": False,
    }


def save(output, plan, contents):
    output = Path(output).absolute()
    if (
        output.exists()
        or any(path.is_symlink() for path in (output, *output.parents))
        or any(type(value) is not bytes for value in contents.values())
    ):
        raise ValueError("Require complete bytes and a new ordinary exact preparation directory")
    require_same(sorted(contents), sorted(OUTPUTS))
    output.mkdir(parents=True, exist_ok=False)
    for name, content in contents.items():
        with (output / name).open("xb") as stream:
            stream.write(content)
    write_new(
        output / "manifest.json",
        {
            "status": "complete_preparation",
            "plan": plan,
            "outputs": {name: {"path": name, **file_identity(output / name)} for name in contents},
        },
    )
    return inspect(output, plan, contents)


def operate(contract, args, *, write):
    plan, contents, bound = gather(contract, args)
    output, paper = Path(args.output).absolute(), contract.primary.repository / "paper"
    if output == paper or paper in output.parents:
        raise ValueError("Preparation never writes or admits manuscript destinations")
    result = (save if write else inspect)(output, plan, contents)
    exact.original.verify_snapshot(bound)
    contract.recheck()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "inspect"))
    for name in ARGUMENT_PATHS:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--vector-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Exact publication preparation is CPU only")
    primary = PrimaryV3Contract.load(
        args.repository / PARENTS["primary"][0], args.repository, args.training_root
    )
    contract = ExactPublicationContract.load(args.protocol, primary)
    result = operate(contract, args, write=args.mode == "build")
    print(
        {
            "exact_publication_preparation_verified": result["recomputed_bytes_verified"],
            "manuscript_installed": False,
            "scientific_completion": False,
        }
    )


if __name__ == "__main__":
    main()
