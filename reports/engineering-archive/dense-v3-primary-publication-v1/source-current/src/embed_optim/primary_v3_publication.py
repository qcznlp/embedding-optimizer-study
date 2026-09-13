"""Checkpoint-backed v3 evidence authoring; never install or publish a manuscript."""

import argparse
import csv
import os
from fractions import Fraction
from pathlib import Path

from . import primary_v3_dimension_inference as inference
from . import primary_v3_geometry as geometry
from . import primary_v3_publication_render as rendering
from . import primary_v3_publication_tables as display
from .primary_contract import canonical, digest, file_identity, read_json, require_same, verify_file
from .primary_v3_contract import PrimaryV3Contract
from .primary_v3_exact_bridge import typed_csv
from .primary_v3_outcomes import TABLE_COUNTS, inspect_bundle
from .primary_v3_publication_contract import OUTPUTS, PARENTS, RULES, SCOPE, PublicationContract
from .primary_v3_validation_io import write_new

SIMULATION_KEYS = {
    "upstream_primary_admission_simulated",
    "upstream_admission_simulated",
    "simulated_upstream_admission",
    "simulated_admission",
    "all_upstream_measurements_and_primary_admissions_simulated",
}
RATIONAL_FIELDS = {
    "baseline_mse_exact",
    "feature_mse_exact",
    "mse_reduction_exact",
    "exact_residual_energy",
    "baseline_prediction_exact",
    "feature_prediction_exact",
    "pooled_baseline_mse_exact",
    "pooled_feature_mse_exact",
    "pooled_mse_reduction_exact",
    "feature_residual_energy_exact",
    "outcome_residual_energy_exact",
}


def reject_simulation(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in SIMULATION_KEYS and item is not False:
                raise ValueError("Synthetic admission cannot supply primary publication")
            reject_simulation(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_simulation(item)


def read_tables(root, counts):
    tables = {}
    for name in counts:
        path = Path(root) / (name + ".csv")
        tables[name] = typed_csv(path)
        with path.open(newline="", encoding="utf-8") as stream:
            raw = list(csv.DictReader(stream))
        for row, serialized in zip(tables[name], raw, strict=True):
            for field in RATIONAL_FIELDS.intersection(row):
                value = serialized[field]
                if value == "":
                    row[field] = None
                    continue
                try:
                    exact = str(Fraction(value))
                except (ValueError, ZeroDivisionError) as error:
                    raise ValueError("Invalid exact rational publication field") from error
                if exact != value:
                    raise ValueError("Noncanonical exact rational publication field")
                row[field] = value
    require_same({name: len(rows) for name, rows in tables.items()}, counts)
    return tables


def generate(primary, tables, selection, runs, functional_decisions, evidence):
    """Pure full-population generation; this helper itself supplies no model admission."""
    expected_counts = {
        "outcomes": TABLE_COUNTS,
        "geometry": geometry.TABLE_COUNTS,
        "bridge": inference.original.TABLE_COUNTS,
        "functional": inference.TABLE_COUNTS,
    }
    require_same(
        {kind: {name: len(rows) for name, rows in group.items()} for kind, group in tables.items()},
        expected_counts,
    )
    summary = display.summarize(
        primary,
        tables["outcomes"],
        tables["geometry"]["checkpoint_geometry"],
        tables["geometry"]["run_pair_subspace_overlap"],
        tables["bridge"],
        selection,
        runs,
    )
    values = {
        "evidence.json": evidence,
        "tables.json": tables,
        "primary-summary.json": summary,
        "findings.json": rendering.findings(summary),
    }
    contents = {name: canonical(value) + b"\n" for name, value in values.items()}
    contents["optimizer-primary.tex"] = rendering.render(summary).encode()
    functional = inference.render(tables["functional"], functional_decisions)
    contents["functional-inference.tex"] = functional.encode()
    contents["dimension-utilization.tex"] = rendering.bind_functional_table(functional).encode()
    require_same(sorted(contents), sorted(OUTPUTS))
    return contents


def gather(contract, args):
    contract = contract.recheck()
    # Full actual primary admission precedes every later input and output access.
    plan, evidence, functional = inference.gather(contract.inference, args)
    reject_simulation(plan)
    reject_simulation(evidence)
    current = inspect_bundle(args.inference_root, plan, evidence, functional)
    bound = list(evidence["source_bindings"])
    bound += inference.original.snapshot(
        args.inference_root,
        ["manifest.json", "evidence.json", *(name + ".csv" for name in inference.TABLE_COUNTS)],
    )
    tables = {
        "outcomes": read_tables(args.outcomes_root, TABLE_COUNTS),
        "geometry": read_tables(args.geometry_root, geometry.TABLE_COUNTS),
        "bridge": read_tables(args.bridge_root, inference.original.TABLE_COUNTS),
        "functional": functional,
    }
    source = evidence["original_bridge_evidence"]["outcome_evidence"]
    publication_evidence = {
        "functional_plan": plan,
        "functional_evidence": evidence,
        "functional_reader": current,
        "source_bindings": bound,
        "publication_rules": RULES,
        "manuscript_installed": False,
        "scientific_completion": False,
    }
    contents = generate(
        contract.primary,
        tables,
        source["validation_selection"],
        source["grid"]["runs"],
        evidence["decisions"],
        publication_evidence,
    )
    inference.original.verify_snapshot(bound)
    contract.recheck()
    return (
        {
            "scope": SCOPE,
            "publication_protocol_sha256": contract.sha256,
            "primary_protocol_sha256": contract.primary.sha256,
            "inference_protocol_sha256": contract.inference.sha256,
            "evidence_sha256": digest(publication_evidence),
            "scientific_completion": False,
            "manuscript_installed": False,
            "final_portable_publication_verified": False,
        },
        contents,
        bound,
    )


def inspect(output, plan, contents):
    """Compare every byte with fresh expected evidence, not only mutable output hashes."""
    output = Path(output).absolute()
    if not output.is_dir() or any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError("Require an ordinary publication preparation directory")
    require_same(sorted(contents), sorted(OUTPUTS))
    if {p.name for p in output.iterdir()} != {*OUTPUTS, "manifest.json"}:
        raise ValueError("Publication preparation inventory differs")
    if not (output / "manifest.json").is_file() or (output / "manifest.json").is_symlink():
        raise ValueError("Require an ordinary publication manifest")
    manifest = read_json(output / "manifest.json")
    expected = {"status": "complete_preparation", "plan": plan, "outputs": {}}
    for name, content in contents.items():
        if not (output / name).is_file() or (output / name).is_symlink():
            raise ValueError("Require ordinary publication evidence files")
        verify_file(output / name, manifest["outputs"][name])
        if (output / name).read_bytes() != content:
            raise ValueError("Publication differs from freshly reconstructed evidence: " + name)
        expected["outputs"][name] = {"path": name, **file_identity(output / name)}
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
        raise ValueError("Require complete bytes and a new ordinary preparation directory")
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
    inference.original.verify_snapshot(bound)
    contract.recheck()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "inspect"))
    for name in (
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
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--vector-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Publication preparation is CPU only")
    primary = PrimaryV3Contract.load(
        args.repository / PARENTS["primary"][0], args.repository, args.training_root
    )
    contract = PublicationContract.load(args.protocol, primary)
    result = operate(contract, args, write=args.mode == "build")
    print(
        {
            "publication_preparation_verified": result["recomputed_bytes_verified"],
            "manuscript_installed": False,
            "scientific_completion": False,
        }
    )


if __name__ == "__main__":
    main()
