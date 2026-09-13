"""All-rate/stage exact geometry under a separate prospective measurement amendment."""

from __future__ import annotations

import argparse
import itertools
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from .primary_contract import DRAFT, file_identity, read_json, require_same, verify_file
from .primary_exact_geometry_kernels import KINDS, SETTINGS, optimizer_rows, pair_row
from .primary_v3_contract import PrimaryV3Contract
from .primary_v3_exact_geometry_io import inspect_run, produce_run, read_stage_views
from .primary_v3_geometry import SOURCES as ORIGINAL_SOURCES
from .primary_v3_geometry import GeometryContract, admit_primary
from .primary_v3_outcomes import csv_bytes
from .primary_v3_validation_io import write_new
from .primary_weight_entries import reference_identity

SCOPE = "dense_primary_v3_exact_geometry"
PARENTS = {
    "primary": (
        "configs/dense_primary_v3_protocol.json",
        "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b",
    ),
    "approximate_geometry": (
        "configs/dense_primary_v3_geometry_protocol_v2.json",
        "9b6dad688072c08e8994e45f80c897a7889ab5fbcbccab27b643944c85ae625b",
    ),
    "robustness_acceptance": (
        "reports/engineering-archive/dense-geometry-robustness-v1/validation.json",
        "03165391e649ad37db2cb182a92165999f0682fe30898154da69206a60681dfb",
    ),
    "analysis": (
        "configs/dense_no_packing_analysis_protocol.json",
        "91d7fa09ebb7e609afc5eb584f499b97baf9718d8cc23ff67eb295adfdd9f27a",
    ),
    "unchanged_bridge": (
        "configs/dense_no_packing_bridge_implementation_protocol_v2.json",
        "88fb013be3fc69c7477750314af264df78080ba2573c578c95e886bb0712ec46",
    ),
}
SOURCES = tuple(
    sorted(
        set(ORIGINAL_SOURCES)
        | {
            "src/embed_optim/primary_v3_exact_geometry.py",
            "src/embed_optim/primary_v3_exact_geometry_io.py",
            "src/embed_optim/primary_exact_geometry_kernels.py",
            "src/embed_optim/geometry_robustness.py",
            "scripts/prepare_dense_v3_exact_geometry.py",
        }
    )
)
TABLE_COUNTS = {
    "checkpoint_exact_geometry": 60,
    "run_pair_exact_subspace_overlap": 660,
    "optimizer_pair_exact_subspace_summary": 60,
    "exact_matrix_geometry": 10560,
}
RULES = {
    "population": "All 88 main-optimizer hidden matrices at all five stages for all twelve rates; no score-dependent selection or sampling",
    "anchors": "Stage-one saved segment and every cumulative displacement use the pinned pretrained base; later segments use the preceding retained checkpoint",
    "precision": "Identical FP32 saved-weight differences as the parent; lossless FP64 promotion; complete NumPy SVD, no randomized sketch",
    "spectrum": "Save the entire singular spectrum; stable rank uses full squared energy/leading singular value squared; entropy uses normalized singular values from the full spectrum",
    "rank_definition": "Retain min(16, smaller matrix dimension); numerical signal rank uses float64 epsilon * maximum matrix dimension * leading singular value",
    "boundary": "A nonzero matrix with sufficient numerical rank is unresolved at the boundary when (sigma_r-sigma_(r+1))/sigma_1 <= the declared tolerance; full-dimension rank has no external boundary",
    "undefined": "Zero, insufficient signal rank and unresolved boundary retain complete raw rows/spectra but no arbitrary retained basis; full spectral metrics of nonzero matrices remain defined",
    "checkpoint_means": "Exact rank/entropy means are parameter-weighted over every nonzero matrix with the nonzero denominator exposed; do not impute zero rank for the zero matrix or equate these means with old zero-imputed approximate columns",
    "pair_mean": "Exclude matrices with either displacement exactly zero and report that mass. Separately report unsupported/boundary-unresolved nonzero mass. A complete nonzero-population overlap is undefined if any such unsafe mass exists; resolved-only conditional overlap is separately labeled",
    "optimizer_mean": "Use every unordered rate pair equally. If any rate pair's complete overlap is undefined, report the optimizer-pair mean undefined and retain pair/coverage counts; do not average only favorable available pairs",
    "overlap": "FP64 cross-Gram squared Frobenius norm/rank for exact orthonormal left/right bases, mean sides then parameter-weight matrices; independent explicit-projector controls verify the fast identity",
    "reuse": "Within one matrix/stage, reuse SVD only if the FP32 saved-segment and cumulative arrays are exactly equal; no norm-based or approximate cache matching",
    "replay": "Recompute all spectra, status, scalar rows and retained bases from original admitted checkpoints; exact same-source/runtime CPU replay, not cross-host bitwise equivalence",
    "old_features": "Original approximate outputs and all nine frozen bridge features remain unchanged. New exact fields are a separately declared measurement layer; a later explicit sensitivity bridge must not silently replace old features",
    "scientific_boundary": "Post-diagnosis prospective preparation before any valid full v3 primary checkpoint; short diagnostic and invalid historical states have been observed. Weight rank/subspaces alone do not demonstrate useful embedding dimensions, causality or optimizer retrieval superiority",
}


@dataclass(frozen=True)
class ExactGeometryContract:
    path: Path
    payload: dict
    sha256: str
    primary: PrimaryV3Contract

    @classmethod
    def load(cls, path, primary):
        path = Path(path).resolve()
        value = read_json(path)
        if (
            value.get("scope") != SCOPE
            or value.get("status") != DRAFT
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or value.get("scientific_completion") is not False
            or value.get("formal_execution_authorized") is not False
        ):
            raise ValueError("Not the prepared exact geometry amendment")
        require_same(value["settings"], SETTINGS)
        require_same(value["measurement_rules"], RULES)
        require_same(value["table_counts"], TABLE_COUNTS)
        if set(value["parents"]) != set(PARENTS) or set(value["sources"]) != set(SOURCES):
            raise ValueError("Incomplete exact geometry identity closure")
        for key, (name, sha) in PARENTS.items():
            if value["parents"][key]["path"] != name or value["parents"][key]["sha256"] != sha:
                raise ValueError("Changed exact geometry parent")
            verify_file(primary.repository / name, value["parents"][key])
        if primary.sha256 != PARENTS["primary"][1]:
            raise ValueError("Changed primary geometry population")
        GeometryContract.load(primary.repository / PARENTS["approximate_geometry"][0], primary)
        for name, bound in value["sources"].items():
            verify_file(primary.repository / name, bound)
        verify_file(
            Path(__file__).resolve(),
            value["sources"]["src/embed_optim/primary_v3_exact_geometry.py"],
        )
        return cls(path, value, file_identity(path)["sha256"], primary)


def tables_from_verified_runs(admitted, root, reference_bound, checked_rows, settings):
    steps = admitted[0][2]["steps"]
    for _, _, checked in admitted:
        require_same(checked["steps"], steps)
    pairs, shapes = [], reference_bound["hidden_shapes"]
    for stage, step in enumerate(steps, start=1):
        for kind in KINDS:
            views = {
                expected["recipe"]["run_id"]: read_stage_views(
                    root / expected["recipe"]["run_id"], step, shapes, kind
                )
                for _, expected, _ in admitted
            }
            for first, second in itertools.combinations(admitted, 2):
                a, b = first[1]["recipe"], second[1]["recipe"]
                pairs.append(
                    pair_row(
                        a,
                        b,
                        stage=stage,
                        steps=steps,
                        kind=kind,
                        first_views=views[a["run_id"]],
                        second_views=views[b["run_id"]],
                        shapes=shapes,
                        rank=settings["subspace_rank"],
                    )
                )
    return {
        "checkpoint_exact_geometry": [
            row for result in checked_rows for row in result["checkpoint_rows"]
        ],
        "run_pair_exact_subspace_overlap": pairs,
        "optimizer_pair_exact_subspace_summary": optimizer_rows(pairs),
        "exact_matrix_geometry": [row for result in checked_rows for row in result["matrix_rows"]],
    }


def inspect_outputs(contract, admitted, reference, reference_bound, output):
    root = output / "runs"
    if output.is_symlink() or root.is_symlink() or not root.is_dir():
        raise ValueError("Require ordinary complete exact geometry directories")
    if {path.name for path in root.iterdir()} != {row[1]["recipe"]["run_id"] for row in admitted}:
        raise ValueError("Missing, extra or historical exact geometry outputs")
    checked_rows = [
        inspect_run(
            root / expected["recipe"]["run_id"],
            run,
            expected,
            checked,
            reference,
            reference_bound,
            contract.payload["settings"],
            scope=SCOPE,
            contract_sha=contract.sha256,
        )
        for run, expected, checked in admitted
    ]
    tables = tables_from_verified_runs(
        admitted, root, reference_bound, checked_rows, contract.payload["settings"]
    )
    require_same({name: len(rows) for name, rows in tables.items()}, TABLE_COUNTS)
    bindings = []
    for (_, expected, _), checked in zip(admitted, checked_rows, strict=True):
        path = root / expected["recipe"]["run_id"]
        verify_file(path / "manifest.json", checked["manifest"])
        manifest = read_json(path / "manifest.json")
        for checkpoint in manifest["outputs"]:
            for key in ("records", "arrays"):
                bound = checkpoint[key]
                verify_file(path / bound["path"], bound)
                bindings.append(
                    {
                        "path": (path / bound["path"]).relative_to(output).as_posix(),
                        **file_identity(path / bound["path"]),
                    }
                )
        bindings.append(
            {
                "path": (path / "manifest.json").relative_to(output).as_posix(),
                **file_identity(path / "manifest.json"),
            }
        )
    return tables, bindings


def operate(contract, experiment_root, reference, output, *, produce):
    output, reference = Path(output), Path(reference)
    admitted = admit_primary(contract, experiment_root)  # All twelve before tensor reads/output.
    reference_bound = reference_identity(contract.primary, reference)
    torch.set_num_threads(contract.payload["settings"]["cpu_threads"])
    with threadpool_limits(limits=contract.payload["settings"]["cpu_threads"]):
        if produce:
            if output.exists() or output.is_symlink() or not output.parent.is_dir():
                raise ValueError("Require a new exact geometry output namespace")
            output.mkdir()
            (output / "runs").mkdir()
            for run, expected, checked in admitted:
                produce_run(
                    run,
                    expected,
                    checked,
                    reference,
                    reference_bound,
                    contract.payload["settings"],
                    output / "runs" / expected["recipe"]["run_id"],
                    scope=SCOPE,
                    contract_sha=contract.sha256,
                )
        tables, raw_bindings = inspect_outputs(
            contract, admitted, reference, reference_bound, output
        )
    summary = {
        "scope": SCOPE,
        "primary_protocol_sha256": contract.primary.sha256,
        "exact_geometry_protocol_sha256": contract.sha256,
        "raw_bindings": raw_bindings,
        "table_counts": TABLE_COUNTS,
        "scientific_completion": False,
        "boundary": "Complete exact measurement layer only; valid primary runs, retrieval/dimension/factorial consumers, explicit release transition and scientific acceptance remain separate gates.",
    }
    if produce:
        for name, rows in tables.items():
            with (output / f"{name}.csv").open("xb") as stream:
                stream.write(csv_bytes(rows))
        write_new(output / "summary.json", summary)
    else:
        require_same(read_json(output / "summary.json"), summary)
        for name, rows in tables.items():
            path = output / f"{name}.csv"
            if path.is_symlink() or path.read_bytes() != csv_bytes(rows):
                raise ValueError("Exact geometry table differs from fresh numeric recomputation")
    ExactGeometryContract.load(contract.path, contract.primary)
    if {path.name for path in output.iterdir()} != {
        "runs",
        "summary.json",
        *(f"{name}.csv" for name in tables),
    }:
        raise ValueError("Undeclared exact geometry artifacts")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repository",
        "training-root",
        "experiment-root",
        "primary-protocol",
        "exact-protocol",
        "reference",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--action", choices=("produce", "inspect"), required=True)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Exact geometry is a CPU-only consumer")
    primary = PrimaryV3Contract.load(args.primary_protocol, args.repository, args.training_root)
    contract = ExactGeometryContract.load(args.exact_protocol, primary)
    operate(
        contract,
        args.experiment_root,
        args.reference,
        args.output,
        produce=args.action == "produce",
    )
    print({"exact_geometry_operation_passed": True, "scientific_completion": False})


if __name__ == "__main__":
    main()
