"""Exact changed-entry counts for source-admitted Dense checkpoint trajectories.

This measurement corrects a matrix-level proxy, not an optimizer, and does not
supply the remaining spectral/subspace/dimension/retrieval mechanism analyses.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import torch

from .geometry import TensorStore, _partition_summary
from .primary_completion import DENSE_PARTITION, integer
from .primary_contract import DRAFT, digest, file_identity, read_json, require_same, verify_file
from .primary_v3_contract import PrimaryV3Contract
from .primary_v3_validation_io import write_new

SCOPE = "dense_primary_v3_weight_entry_census"
PRIMARY_SHA = "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b"
PARENTS = {
    "analysis": (
        "configs/dense_no_packing_analysis_protocol.json",
        "91d7fa09ebb7e609afc5eb584f499b97baf9718d8cc23ff67eb295adfdd9f27a",
    ),
    "outcome_acceptance": (
        "reports/engineering-archive/dense-v3-outcomes-v1/validation.json",
        "58d328af52a6fc94e06c5ef514c9f53c2fb28e71dbb833f569cea7861c5ee5db",
    ),
}
SOURCES = (
    "src/embed_optim/primary_weight_entries.py",
    "scripts/prepare_dense_weight_entries.py",
    "src/embed_optim/geometry.py",
    "src/embed_optim/optimizers.py",
    "src/embed_optim/corrected_geometry_summary.py",
    "src/embed_optim/primary_contract.py",
    "src/embed_optim/primary_completion.py",
    "src/embed_optim/primary_v3_contract.py",
    "src/embed_optim/primary_v3_validation_io.py",
)
DEFINITION = {
    "numerator": "Number of individual hidden-matrix parameters whose saved values differ from the declared anchor",
    "denominator": "All individual parameters in all 88 hidden matrices, including unchanged entries and zero-displacement matrices",
    "comparison": "Exact saved-value inequality after lossless conversion to float64; no tolerance, threshold, norm or gradient",
    "saved_segment_anchor": "Initialization at the first retained stage, otherwise the immediately preceding retained checkpoint",
    "cumulative_anchor": "The same immutable untrained initialization for every stage",
    "old_proxy": "Parameter mass of matrices having any nonzero displacement; not the fraction of individual parameters changed",
    "legacy_artifacts_rewritten": False,
    "training_or_primary_retrieval_estimand_changed": False,
}


def changed_entries(current, anchor):
    if (
        not isinstance(current, torch.Tensor)
        or not isinstance(anchor, torch.Tensor)
        or current.ndim != 2
        or current.shape != anchor.shape
        or not current.numel()
        or not current.is_floating_point()
        or not anchor.is_floating_point()
    ):
        raise ValueError("Require nonempty matching floating-point matrices")
    # Count differing values, not a norm indicator or a thresholded/rounded update.
    left, right = (t.detach().to(device="cpu", dtype=torch.float64) for t in (current, anchor))
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise ValueError("Changed-entry inputs must be finite")
    return int(torch.count_nonzero(left != right).item())


def summarize_entries(records, expected_shapes):
    if Counter(r["tensor"] for r in records) != Counter(expected_shapes.keys()):
        raise ValueError("Entry census lacks the exact distinct hidden tensor population")
    total = 0
    sums = {kind: 0 for kind in ("saved_segment", "cumulative")}
    proxy = dict(sums)
    for row in records:
        shape = expected_shapes[row["tensor"]]
        require_same(row["shape"], list(shape))
        count = math.prod(shape)
        if (
            len(shape) != 2
            or any(type(x) is not int or x <= 0 for x in shape)
            or integer(row["parameters"], minimum=1) != count
        ):
            raise ValueError("Invalid matrix shape or parameter denominator")
        total += count
        for kind in sums:
            value = integer(row[f"{kind}_nonzero_parameters"])
            if value > count:
                raise ValueError("Changed count exceeds its actual parameter population")
            sums[kind] += value
            proxy[kind] += count if value else 0
    if not total:
        raise ValueError("Cannot aggregate an empty parameter population")
    return {
        "hidden_tensors": len(records),
        "hidden_parameters": total,
        **{f"{kind}_nonzero_parameters": value for kind, value in sums.items()},
        **{f"{kind}_nonzero_parameter_fraction": value / total for kind, value in sums.items()},
        **{
            f"{kind}_parameter_mass_in_nonzero_matrices": value / total
            for kind, value in proxy.items()
        },
    }


def reference_identity(primary, reference):
    reference = Path(reference).resolve()
    expected = primary.inputs["common_identity"]["model"]["files"]
    for row in expected:
        verify_file(reference / row["path"], row)
    with TensorStore(reference) as store:
        require_same(_partition_summary(store), DENSE_PARTITION)
        all_shapes = {name: list(store.shape(name)) for name in store.keys()}
        from .optimizers import parameter_partition_name

        hidden = {
            name: shape
            for name, shape in all_shapes.items()
            if parameter_partition_name(name, len(shape)) == "hidden"
        }
    if len(hidden) != 88 or sum(math.prod(s) for s in hidden.values()) != 110297088:
        raise ValueError("Reference does not have the actual Dense hidden parameter population")
    return {"files": expected, "all_shapes": all_shapes, "hidden_shapes": hidden}


def stream_run_census(run_root, expected, checked, reference, reference_bound, *, scope):
    """Pure CPU trajectory reader; caller separately authenticates expected identity."""
    run_root, reference = Path(run_root), Path(reference)
    if (
        checked.get("whole_run_artifacts_verified") is not True
        or checked.get("run_identity_sha256") != digest(expected)
        or [r["step"] for r in checked["checkpoints"]] != checked["steps"]
    ):
        raise ValueError("Entry census needs complete identity-bound run admission")
    for record in checked["checkpoints"]:
        for row in record["files"]:
            verify_file(run_root / f"checkpoint-{record['step']}" / row["path"], row)
    for row in reference_bound["files"]:
        verify_file(reference / row["path"], row)
    checkpoints = []
    for index, step in enumerate(checked["steps"]):
        current_path = run_root / f"checkpoint-{step}"
        previous_step = checked["steps"][index - 1] if index else 0
        previous_path = run_root / f"checkpoint-{previous_step}" if index else reference
        with ExitStack() as stack:
            initial = stack.enter_context(TensorStore(reference))
            current = stack.enter_context(TensorStore(current_path))
            previous = stack.enter_context(TensorStore(previous_path)) if index else initial
            for store in (initial, current, previous):
                require_same(
                    {name: list(store.shape(name)) for name in store.keys()},
                    reference_bound["all_shapes"],
                )
            rows = []
            for name, shape in reference_bound["hidden_shapes"].items():
                weight = current.tensor(name)
                rows.append(
                    {
                        "tensor": name,
                        "shape": shape,
                        "parameters": math.prod(shape),
                        "saved_segment_nonzero_parameters": changed_entries(
                            weight, previous.tensor(name)
                        ),
                        "cumulative_nonzero_parameters": changed_entries(
                            weight, initial.tensor(name)
                        ),
                    }
                )
        checkpoints.append(
            {
                "step": step,
                "previous_step": previous_step,
                "cumulative_anchor_step": 0,
                "records": rows,
                "summary": summarize_entries(rows, reference_bound["hidden_shapes"]),
            }
        )
    for record in checked["checkpoints"]:
        for row in record["files"]:
            verify_file(run_root / f"checkpoint-{record['step']}" / row["path"], row)
    for row in reference_bound["files"]:
        verify_file(reference / row["path"], row)
    return {
        "scope": scope,
        "run_id": expected["recipe"]["run_id"],
        "run_identity_sha256": digest(expected),
        "reference": reference_bound,
        "definition": DEFINITION,
        "checkpoints": checkpoints,
        "scientific_completion": False,
    }


def load_amendment(path, primary):
    path = Path(path)
    value = read_json(path)
    if (
        value.get("scope") != SCOPE
        or value.get("status") != DRAFT
        or value.get("schema_version") != 1
        or isinstance(value["schema_version"], bool)
        or value.get("scientific_completion") is not False
        or value.get("formal_execution_authorized") is not False
        or primary.sha256 != PRIMARY_SHA
        or value["primary"]["sha256"] != PRIMARY_SHA
        or value["primary"]["path"] != primary.path.relative_to(primary.repository).as_posix()
    ):
        raise ValueError("Not the prepared v3 changed-entry measurement amendment")
    verify_file(primary.path, value["primary"])
    require_same(value["measurement"], DEFINITION)
    if set(value["parents"]) != set(PARENTS) or set(value["sources"]) != set(SOURCES):
        raise ValueError("Incomplete changed-entry source/parent closure")
    for key, (name, sha) in PARENTS.items():
        binding = value["parents"][key]
        if binding["path"] != name or binding["sha256"] != sha:
            raise ValueError("Changed immutable census parent")
        verify_file(primary.repository / name, binding)
    for name, binding in value["sources"].items():
        verify_file(primary.repository / name, binding)
    verify_file(
        Path(__file__).resolve(), value["sources"]["src/embed_optim/primary_weight_entries.py"]
    )
    return value


def collect_primary(primary, amendment, experiment_root, reference):
    checked = []
    for row in primary.inputs["runs"]:
        path = Path(experiment_root) / primary.payload["output_root"] / "dense" / row["run_id"]
        checked.append((path, primary.complete_run(path, row["run_id"])))
    # No partial optimizer family or diagnostic population is promoted.
    if len(checked) != 12:
        raise ValueError("Require all twelve complete primary runs before any census")
    reference_bound = reference_identity(primary, reference)
    results = [
        stream_run_census(
            path,
            primary.expected_identity(run["run_id"]),
            run,
            reference,
            reference_bound,
            scope=SCOPE,
        )
        for path, run in checked
    ]
    if sum(len(r["checkpoints"]) for r in results) != 60:
        raise ValueError("Require every scheduled primary stage")
    return {
        "scope": SCOPE,
        "primary_protocol_sha256": primary.sha256,
        "measurement_amendment": amendment,
        "runs": results,
        "scientific_completion": False,
        "boundary": "Changed-entry census only, not full spectral/subspace analysis, dimension utility, retrieval inference or paper acceptance.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "primary-protocol",
        "amendment",
        "repository",
        "training-root",
        "experiment-root",
        "reference",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a new census output; preserve old artifacts")
    primary = PrimaryV3Contract.load(args.primary_protocol, args.repository, args.training_root)
    amendment = load_amendment(args.amendment, primary)
    result = collect_primary(primary, amendment, args.experiment_root, args.reference)
    load_amendment(args.amendment, primary)
    if not args.output.parent.is_dir():
        raise ValueError("Require an existing explicit output parent")
    write_new(args.output, result)
    print(
        {
            "entry_census_produced": True,
            "scientific_completion": False,
            **file_identity(args.output),
        }
    )


if __name__ == "__main__":
    main()
