"""Complete v3 geometry consumer: all rates/stages, strict numeric readback, no score selection."""

from __future__ import annotations

import argparse
import itertools
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from .corrected_geometry_summary import OPTIMIZER_ORDER, _optimizer_pair_rows
from .primary_contract import DRAFT, file_identity, read_json, require_same, verify_file
from .primary_geometry_kernels import KINDS, pair_row
from .primary_v3_contract import PrimaryV3Contract
from .primary_v3_geometry_io import inspect_run, produce_run, read_stage_bases
from .primary_v3_outcomes import csv_bytes
from .primary_v3_validation_io import write_new
from .primary_weight_entries import load_amendment, reference_identity

SCOPE = "dense_primary_v3_geometry"
PARENTS = {
    "primary": (
        "configs/dense_primary_v3_protocol.json",
        "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b",
    ),
    "analysis": (
        "configs/dense_no_packing_analysis_protocol.json",
        "91d7fa09ebb7e609afc5eb584f499b97baf9718d8cc23ff67eb295adfdd9f27a",
    ),
    "entry_amendment": (
        "configs/dense_weight_entry_measurement_amendment.json",
        "a4b4f6839b185931d2de7f0771487123b971a1e7eb69ef34bb5f9e4350060ec5",
    ),
    "entry_acceptance": (
        "reports/engineering-archive/dense-v3-geometry-v1/validation.json",
        "3b8575f2818c2b4d9f35f93610b4a0100078bb854941af6e0b61da7f76236df9",
    ),
    "geometry_predecessor": (
        "configs/dense_primary_v3_geometry_protocol.json",
        "9f11cbcdfa8b75593cc3ebd22336e5a34073531426431da88cf904f6e470f358",
    ),
    "reduction_localization": (
        "reports/engineering-archive/dense-v3-geometry-chain-v1/reduction-localization.json",
        "3fd2ebb1c9fbef543c32b214b39733dcac91554b69e29110864aff64c2592ca2",
    ),
}
SOURCES = (
    "src/embed_optim/primary_v3_geometry.py",
    "src/embed_optim/primary_v3_geometry_io.py",
    "src/embed_optim/primary_geometry_kernels.py",
    "src/embed_optim/primary_geometry_reductions.py",
    "scripts/prepare_dense_v3_geometry.py",
    "src/embed_optim/geometry.py",
    "src/embed_optim/corrected_geometry_summary.py",
    "src/embed_optim/optimizers.py",
    "src/embed_optim/config.py",
    "src/embed_optim/primary_contract.py",
    "src/embed_optim/primary_completion.py",
    "src/embed_optim/primary_v3_contract.py",
    "src/embed_optim/primary_weight_entries.py",
    "src/embed_optim/primary_v3_outcomes.py",
    "src/embed_optim/primary_v3_validation_io.py",
)
SETTINGS = {
    "sketch_rank": 64,
    "subspace_rank": 16,
    "oversample": 8,
    "power_iterations": 2,
    "seed": 20260903,
    "cpu_threads": 4,
}
TABLE_COUNTS = {
    "checkpoint_geometry": 60,
    "run_pair_subspace_overlap": 660,
    "optimizer_pair_subspace_summary": 60,
    "subspace_health": 10560,
}
VALIDITY = {
    "zero": "Exactly zero saved displacement has undefined subspace; exclude only those matrices and report defined parameter mass",
    "nonzero_signal_rank": "Require every retained singular value above float32 epsilon * max(matrix dimensions) * projected leading singular value; refuse unsupported rank rather than adding null directions or dropping that matrix",
    "boundary_gap": "Retain the projected relative gap at the retained boundary descriptively; no post-hoc gap threshold, rank choice or optimizer support rule",
    "partial_spectrum": "Entropy effective rank is computed from the top-64 sketch, not the full singular spectrum or useful embedding dimensions; always retain captured energy",
    "readback": "Recompute all raw metrics and singular bases from the same saved weights with the declared CPU thread count; exact same-runtime equality, not a cross-host bitwise guarantee",
    "entry_fraction": "Use the separate exact-entry amendment; preserve the original matrix-mass proxy under its explicit different name",
    "legacy_outputs_adopted": False,
    "formal_training_or_retrieval_rules_changed": False,
}
IMPLEMENTATION_REVISION = {
    "version": 2,
    "change": "Global Frobenius norms and displacement/weight cosine dot/norm reductions accumulate in float64 on the same FP32 kernel inputs",
    "reason": "Three of twelve preselected full-matrix norm controls failed the original 2e-5 relative / 1e-10 absolute check; independent FP64 and scalar compensated summation agree",
    "spectral_kernel_and_subspace_bases_changed": False,
    "row_column_statistics_changed": False,
    "ranks_seeds_stages_pairing_changed": False,
    "measurement_definitions_changed": False,
    "tolerances_changed": False,
    "failed_predecessor_overridden": False,
}


@dataclass(frozen=True)
class GeometryContract:
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
            or primary.sha256 != PARENTS["primary"][1]
        ):
            raise ValueError("Not the prepared complete v3 geometry contract")
        require_same(value["settings"], SETTINGS)
        require_same(value["subspace_validity"], VALIDITY)
        require_same(value["table_counts"], TABLE_COUNTS)
        require_same(value.get("implementation_revision"), IMPLEMENTATION_REVISION)
        if set(value["parents"]) != set(PARENTS) or set(value["sources"]) != set(SOURCES):
            raise ValueError("Incomplete geometry parent/source closure")
        for key, (name, sha) in PARENTS.items():
            if value["parents"][key]["path"] != name or value["parents"][key]["sha256"] != sha:
                raise ValueError("Changed immutable geometry parent")
            verify_file(primary.repository / name, value["parents"][key])
        original = read_json(primary.repository / PARENTS["analysis"][0])
        require_same(value["weight_space_rules"], original["weight_space"])
        for name, binding in value["sources"].items():
            verify_file(primary.repository / name, binding)
        verify_file(
            Path(__file__).resolve(), value["sources"]["src/embed_optim/primary_v3_geometry.py"]
        )
        load_amendment(primary.repository / PARENTS["entry_amendment"][0], primary)
        return cls(path, value, file_identity(path)["sha256"], primary)


def admit_primary(contract, experiment_root):
    primary = contract.primary
    admitted = []
    for source in primary.inputs["runs"]:
        run_id = source["run_id"]
        run = Path(experiment_root) / primary.payload["output_root"] / "dense" / run_id
        checked = primary.complete_run(run, run_id)
        admitted.append((run, primary.expected_identity(run_id), checked))
    if len(admitted) != 12 or sum(len(row[2]["steps"]) for row in admitted) != 60:
        raise ValueError("Require every complete primary run and scheduled checkpoint")
    return sorted(
        admitted,
        key=lambda row: (
            OPTIMIZER_ORDER[row[1]["recipe"]["optimizer"]["name"]],
            row[1]["recipe"]["optimizer"]["lr"],
            row[1]["recipe"]["run_id"],
        ),
    )


def tables_from_verified_runs(admitted, root, reference_bound, checked_rows):
    """Pure summaries of numerically admitted files; no validation/BEIR input exists here."""
    steps = admitted[0][2]["steps"]
    for _, _, checked in admitted:
        require_same(checked["steps"], steps)
    pairs = []
    shapes = reference_bound["hidden_shapes"]
    for stage, step in enumerate(steps, start=1):
        for kind in KINDS:
            bases = {
                expected["recipe"]["run_id"]: read_stage_bases(
                    root / expected["recipe"]["run_id"], step, shapes, kind
                )
                for _, expected, _ in admitted
            }
            for first, second in itertools.combinations(admitted, 2):
                left, right = first[1]["recipe"], second[1]["recipe"]
                pairs.append(
                    pair_row(
                        left,
                        right,
                        stage=stage,
                        steps=steps,
                        kind=kind,
                        left_bases=bases[left["run_id"]],
                        right_bases=bases[right["run_id"]],
                        shapes=shapes,
                    )
                )
    return {
        "checkpoint_geometry": [
            row for result in checked_rows for row in result["checkpoint_rows"]
        ],
        "run_pair_subspace_overlap": pairs,
        "optimizer_pair_subspace_summary": _optimizer_pair_rows(pairs),
        "subspace_health": [row for result in checked_rows for row in result["subspace_health"]],
    }


def inspect_outputs(contract, admitted, reference, reference_bound, output):
    root = output / "runs"
    if output.is_symlink() or root.is_symlink() or not root.is_dir():
        raise ValueError("Require ordinary complete geometry directories")
    expected_names = {row[1]["recipe"]["run_id"] for row in admitted}
    if {path.name for path in root.iterdir()} != expected_names:
        raise ValueError("Missing, extra or historical geometry run outputs")
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
    tables = tables_from_verified_runs(admitted, root, reference_bound, checked_rows)
    require_same({name: len(rows) for name, rows in tables.items()}, TABLE_COUNTS)
    bindings = []
    for (_, expected, _), checked_output in zip(admitted, checked_rows, strict=True):
        path = root / expected["recipe"]["run_id"]
        verify_file(path / "manifest.json", checked_output["manifest"])
        manifest = read_json(path / "manifest.json")
        for checkpoint in manifest["outputs"]:
            for key in ("records", "bases"):
                binding = checkpoint[key]
                verify_file(path / binding["path"], binding)
                bindings.append(
                    {
                        "path": (path / binding["path"]).relative_to(output).as_posix(),
                        **file_identity(path / binding["path"]),
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
    admitted = admit_primary(contract, experiment_root)  # All twelve before any tensor read/output.
    reference_bound = reference_identity(contract.primary, reference)
    torch.set_num_threads(contract.payload["settings"]["cpu_threads"])
    if produce:
        if output.exists() or output.is_symlink() or not output.parent.is_dir():
            raise ValueError("Require a new complete geometry output namespace")
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
    tables, raw_bindings = inspect_outputs(contract, admitted, reference, reference_bound, output)
    summary = {
        "scope": SCOPE,
        "primary_protocol_sha256": contract.primary.sha256,
        "geometry_protocol_sha256": contract.sha256,
        "raw_bindings": raw_bindings,
        "table_counts": TABLE_COUNTS,
        "scientific_completion": False,
        "boundary": "Complete geometry only; retrieval bridge, useful dimensions, factorial, publication and formal release remain separate acceptance gates.",
    }
    if produce:
        for name, rows in tables.items():
            with (output / f"{name}.csv").open("xb") as stream:
                stream.write(csv_bytes(rows))
        write_new(output / "summary.json", summary)
    else:
        require_same(read_json(output / "summary.json"), summary)
        for name, rows in tables.items():
            if (output / f"{name}.csv").is_symlink() or (
                output / f"{name}.csv"
            ).read_bytes() != csv_bytes(rows):
                raise ValueError("Stored geometry table differs from fresh numeric recomputation")
    GeometryContract.load(contract.path, contract.primary)
    if {p.name for p in output.iterdir()} != {
        "runs",
        "summary.json",
        *(f"{name}.csv" for name in tables),
    }:
        raise ValueError("Undeclared complete geometry outputs")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repository",
        "training-root",
        "experiment-root",
        "primary-protocol",
        "geometry-protocol",
        "reference",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--action", choices=("produce", "inspect"), required=True)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Geometry is a CPU-only consumer")
    primary = PrimaryV3Contract.load(args.primary_protocol, args.repository, args.training_root)
    contract = GeometryContract.load(args.geometry_protocol, primary)
    operate(
        contract,
        args.experiment_root,
        args.reference,
        args.output,
        produce=args.action == "produce",
    )
    print({"geometry_operation_passed": True, "scientific_completion": False})


if __name__ == "__main__":
    main()
