"""Complete spectra and identifiable fixed-rank subspaces, without random probes."""

from __future__ import annotations

import itertools
import math

import numpy as np

from .geometry_robustness import exact_reference, finite_matrix
from .primary_completion import integer
from .primary_contract import require_same

KINDS = ("saved_segment", "cumulative")
STATUSES = ("resolved", "zero", "insufficient_signal_rank", "unresolved_boundary")
OPTIMIZERS = ("adamw", "muon", "normuon")
SETTINGS = {"subspace_rank": 16, "cpu_threads": 4, "boundary_gap_to_leading_tolerance": 2e-8}


def exact_matrix(matrix, settings):
    matrix = finite_matrix(matrix)
    rank = min(integer(settings["subspace_rank"], minimum=1), min(matrix.shape))
    tolerance = settings["boundary_gap_to_leading_tolerance"]
    if (
        type(tolerance) not in (int, float)
        or not math.isfinite(tolerance)
        or not 0 <= tolerance < 1
    ):
        raise ValueError("Invalid exact boundary resolution tolerance")
    if not np.any(matrix != 0):
        return (
            {
                "basis_status": "zero",
                "retained_rank": rank,
                "numerical_rank": 0,
                "numerical_threshold": 0.0,
                "frobenius_norm": 0.0,
                "spectral_norm": 0.0,
                "stable_rank": None,
                "full_entropy_effective_rank": None,
                "top_rank_energy": None,
                "boundary_gap_to_leading": None,
                "relative_boundary_gap": None,
                "relative_reconstruction_residual": None,
            },
            np.zeros(min(matrix.shape), dtype=np.float64),
            None,
        )
    (left, singular, right), metrics = exact_reference(matrix)
    threshold = float(np.finfo(np.float64).eps * max(matrix.shape) * singular[0])
    numerical_rank = int(np.sum(singular > threshold))
    difference = float(singular[rank - 1] - singular[rank]) if rank < len(singular) else None
    gap = difference / float(singular[0]) if difference is not None else None
    status = "resolved"
    if numerical_rank < rank:
        status = "insufficient_signal_rank"
    elif gap is not None and gap <= settings["boundary_gap_to_leading_tolerance"]:
        status = "unresolved_boundary"
    relative = singular / singular[0]
    record = {
        "basis_status": status,
        "retained_rank": rank,
        "numerical_rank": numerical_rank,
        "numerical_threshold": threshold,
        **{key: value for key, value in metrics.items() if key != "zero"},
        "top_rank_energy": float(np.sum(relative[:rank] ** 2) / np.sum(relative**2)),
        "boundary_gap_to_leading": gap,
        "relative_boundary_gap": difference / float(singular[rank - 1])
        if difference is not None and singular[rank - 1] > 0
        else None,
    }
    bases = (left[:, :rank].copy(), right[:, :rank].copy()) if status == "resolved" else None
    return record, singular.copy(), bases


def exact_overlap(first, second):
    """O(m r²), sign/rotation-invariant overlap; exact FP64 bases only.

    Cross-Gram evaluation avoids materializing ambient-dimension projectors.
    Independent explicit-projector controls verify this identity in tests/audits.
    """
    values = []
    for left, right in zip(first, second, strict=True):
        left, right = finite_matrix(left), finite_matrix(right)
        if left.shape != right.shape or left.shape[1] > left.shape[0]:
            raise ValueError("Exact subspaces need matching shapes and ranks")
        rank = left.shape[1]
        for basis in (left, right):
            if np.max(np.abs(basis.T @ basis - np.eye(rank))) > 1e-10:
                raise ValueError("An exact basis is not FP64 orthonormal")
        value = float(np.sum((left.T @ right) ** 2) / rank)
        if not -1e-10 <= value <= 1 + 1e-10:
            raise ValueError("Exact overlap lies outside projector bounds")
        values.append(float(np.clip(value, 0, 1)))
    if len(values) != 2:
        raise ValueError("Require both left and right exact subspaces")
    return values[0], values[1], sum(values) / 2


def checkpoint_row(recipe, stage, steps, records, shapes, weight_norms):
    total = sum(math.prod(shape) for shape in shapes.values())
    require_same(sorted(weight_norms), sorted(shapes))
    if not total or any(not math.isfinite(value) or value < 0 for value in weight_norms.values()):
        raise ValueError("Invalid hidden population or weight norm")
    weight_norm = math.sqrt(math.fsum(value**2 for value in weight_norms.values()))
    row = {
        "run_id": recipe["run_id"],
        "optimizer": recipe["optimizer"]["name"],
        "learning_rate": recipe["optimizer"]["lr"],
        "stage": stage,
        "step": steps[stage - 1],
        "progress_fraction": stage / len(steps),
        "hidden_parameters": total,
        "weight_frobenius_norm": weight_norm,
    }
    for kind in KINDS:
        selected = [record for record in records if record["displacement_kind"] == kind]
        require_same([record["tensor"] for record in selected], sorted(shapes))
        for record in selected:
            if record["basis_status"] not in STATUSES or record["parameters"] != math.prod(
                shapes[record["tensor"]]
            ):
                raise ValueError("Changed exact matrix population")
        nonzero = [record for record in selected if record["basis_status"] != "zero"]
        denominator = sum(record["parameters"] for record in nonzero)
        prefix = f"exact_{kind}"
        norm = math.sqrt(math.fsum(record["frobenius_norm"] ** 2 for record in selected))
        row[f"{prefix}_frobenius_norm"] = norm
        row[f"{prefix}_to_weight_ratio"] = norm / weight_norm if weight_norm else None
        row[f"{prefix}_nonzero_parameters"] = denominator
        row[f"{prefix}_nonzero_parameter_fraction"] = denominator / total
        for status in STATUSES:
            row[f"{prefix}_{status}_parameters"] = sum(
                record["parameters"] for record in selected if record["basis_status"] == status
            )
        for field in ("stable_rank", "full_entropy_effective_rank", "top_rank_energy"):
            row[f"{prefix}_{field}_parameter_weighted_nonzero"] = (
                math.fsum(record["parameters"] * record[field] for record in nonzero) / denominator
                if denominator
                else None
            )
            if field != "top_rank_energy":
                row[f"{prefix}_{field}_fraction_parameter_weighted_nonzero"] = (
                    math.fsum(
                        record["parameters"] * record[field] / min(shapes[record["tensor"]])
                        for record in nonzero
                    )
                    / denominator
                    if denominator
                    else None
                )
    return row


def pair_row(first, second, *, stage, steps, kind, first_views, second_views, shapes, rank):
    require_same(sorted(first_views), sorted(shapes))
    require_same(sorted(second_views), sorted(shapes))
    integer(rank, minimum=1)
    if not shapes:
        raise ValueError("Require a nonempty hidden population")
    masses = {status: 0 for status in STATUSES}
    sums = [[], [], []]
    tensors = 0
    for name, shape in sorted(shapes.items()):
        a, b = first_views[name], second_views[name]
        if a["status"] not in STATUSES or b["status"] not in STATUSES:
            raise ValueError("Unknown exact subspace status")
        for view in (a, b):
            if (view["bases"] is not None) != (view["status"] == "resolved"):
                raise ValueError("Undefined exact subspace must not supply an arbitrary basis")
            if view["bases"] is not None:
                retained = min(rank, min(shape))
                if len(view["bases"]) != 2 or any(
                    value.shape != (size, retained)
                    for value, size in zip(view["bases"], shape, strict=True)
                ):
                    raise ValueError("Exact basis differs from the declared tensor shape/rank")
        status = next(
            (
                value
                for value in ("zero", "insufficient_signal_rank", "unresolved_boundary")
                if value in (a["status"], b["status"])
            ),
            "resolved",
        )
        count = math.prod(shape)
        masses[status] += count
        if status == "resolved":
            tensors += 1
            for accumulator, value in zip(sums, exact_overlap(a["bases"], b["bases"]), strict=True):
                accumulator.append(count * value)
    resolved = masses["resolved"]
    unsafe = masses["insufficient_signal_rank"] + masses["unresolved_boundary"]
    conditional = [math.fsum(values) / resolved if resolved else None for values in sums]
    return {
        "stage": stage,
        "step": steps[stage - 1],
        "progress_fraction": stage / len(steps),
        "displacement_kind": kind,
        **{
            f"{prefix}_{key}": recipe["run_id"]
            if key == "run_id"
            else recipe["optimizer"]["name" if key == "optimizer" else "lr"]
            for prefix, recipe in (("first", first), ("second", second))
            for key in ("run_id", "optimizer", "learning_rate")
        },
        "defined_tensors": tensors,
        "defined_parameters": resolved,
        "defined_parameter_fraction": resolved / sum(masses.values()),
        "undefined_zero_parameters": masses["zero"],
        "undefined_signal_rank_parameters": masses["insufficient_signal_rank"],
        "undefined_boundary_parameters": masses["unresolved_boundary"],
        "all_nonzero_pair_subspaces_resolved": unsafe == 0,
        "resolved_only_mean_subspace_overlap": conditional[2],
        **{
            key: value if not unsafe else None
            for key, value in zip(
                ("left_subspace_overlap", "right_subspace_overlap", "mean_subspace_overlap"),
                conditional,
                strict=True,
            )
        },
    }


def optimizer_rows(pairs):
    rows = []
    for stage in sorted({row["stage"] for row in pairs}):
        for kind in KINDS:
            for first, second in itertools.combinations_with_replacement(OPTIMIZERS, 2):
                selected = [
                    row
                    for row in pairs
                    if row["stage"] == stage
                    and row["displacement_kind"] == kind
                    and sorted((row["first_optimizer"], row["second_optimizer"]))
                    == sorted((first, second))
                ]
                if not selected:
                    continue
                values = [
                    row["mean_subspace_overlap"]
                    for row in selected
                    if row["mean_subspace_overlap"] is not None
                ]
                rows.append(
                    {
                        "stage": stage,
                        "displacement_kind": kind,
                        "first_optimizer": first,
                        "second_optimizer": second,
                        "run_pairs": len(selected),
                        "defined_run_pairs": len(values),
                        "mean_subspace_overlap": math.fsum(values) / len(values)
                        if len(values) == len(selected)
                        else None,
                        "mean_defined_parameter_fraction": math.fsum(
                            row["defined_parameter_fraction"] for row in selected
                        )
                        / len(selected),
                        "unsafe_nonzero_run_pairs": sum(
                            not row["all_nonzero_pair_subspaces_resolved"] for row in selected
                        ),
                    }
                )
    return rows
