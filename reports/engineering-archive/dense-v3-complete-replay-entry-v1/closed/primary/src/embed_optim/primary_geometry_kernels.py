"""Audited geometry projections; preserved FP32 metrics, explicit subspace validity."""

from __future__ import annotations

import math

import torch

from .corrected_geometry_summary import _basis_overlap, _rss, _weighted_metric
from .primary_completion import integer
from .primary_weight_entries import summarize_entries

KINDS = ("saved_segment", "cumulative")


def audited_top_bases(matrix, *, rank, oversample, power_iterations, seed):
    """The retained algorithm with its discarded singular values exposed for validity checks.

    Bases, RNG, precision and power steps match the old `_top_bases`. A nonzero
    matrix with fewer numerically supported retained directions is refused, not
    completed with null directions, filtered from a pair or assigned overlap zero.
    """
    matrix = matrix.detach().to(device="cpu", dtype=torch.float32)
    if matrix.ndim != 2 or not matrix.numel() or not bool(torch.isfinite(matrix).all()):
        raise ValueError("Subspace input must be a nonempty finite matrix")
    retained = min(integer(rank, minimum=1), min(matrix.shape))
    integer(oversample)
    integer(power_iterations)
    squared_norm = matrix.square().sum()
    if not bool(torch.isfinite(squared_norm)):
        raise ValueError("Subspace Frobenius energy overflowed")
    if not bool((matrix != 0).any()):
        return None, {
            "zero_displacement": True,
            "retained_rank": 0,
            "projected_numerical_rank": 0,
            "retained_signal_supported": None,
            "numerical_threshold": 0.0,
            "smallest_retained_singular_value": None,
            "next_singular_value": None,
            "relative_boundary_gap": None,
            "retained_frobenius_energy": 0.0,
        }
    if squared_norm <= 0:
        raise ValueError("Nonzero saved displacement underflowed in the frozen FP32 kernel")
    limit = min(matrix.shape)
    if retained == limit:
        left, singular, right_h = torch.linalg.svd(matrix, full_matrices=False)
        left = left[:, :retained]
    else:
        width = min(limit, retained + oversample)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        omega = torch.randn(matrix.shape[1], width, generator=generator, dtype=matrix.dtype)
        sample = matrix @ omega
        for _ in range(power_iterations):
            basis = torch.linalg.qr(sample, mode="reduced").Q
            sample = matrix @ (matrix.T @ basis)
        basis = torch.linalg.qr(sample, mode="reduced").Q
        projected_left, singular, right_h = torch.linalg.svd(basis.T @ matrix, full_matrices=False)
        left = basis @ projected_left[:, :retained]
    threshold = torch.finfo(torch.float32).eps * max(matrix.shape) * singular[0]
    numerical_rank = int((singular > threshold).sum())
    if numerical_rank < retained:
        raise ValueError("Nonzero displacement cannot support the declared retained signal rank")
    next_value = float(singular[retained]) if len(singular) > retained else None
    smallest = float(singular[retained - 1])
    health = {
        "zero_displacement": False,
        "retained_rank": retained,
        "projected_numerical_rank": numerical_rank,
        "retained_signal_supported": True,
        "numerical_threshold": float(threshold),
        "smallest_retained_singular_value": smallest,
        "next_singular_value": next_value,
        "relative_boundary_gap": (smallest - next_value) / smallest
        if next_value is not None
        else None,
        "retained_frobenius_energy": float(
            (singular[:retained].square().sum() / squared_norm).clamp(max=1)
        ),
    }
    return (left, right_h[:retained].T), health


def checkpoint_row(recipe, stage, steps, records, entry_records, expected_shapes):
    segment = "delta_from_reference" if stage == 1 else "delta_from_previous"
    weight_norm = _rss(records, "weight")
    if weight_norm <= 0:
        raise ValueError("Cannot normalize a displacement by a zero weight norm")
    segment_norm = _rss(records, segment)
    cumulative_norm = _rss(records, "delta_from_reference")
    row = {
        "run_id": recipe["run_id"],
        "optimizer": recipe["optimizer"]["name"],
        "learning_rate": recipe["optimizer"]["lr"],
        "stage": stage,
        "progress_fraction": stage / len(steps),
        "step": steps[stage - 1],
        **summarize_entries(entry_records, expected_shapes),
        "weight_frobenius_norm": weight_norm,
        "saved_segment_frobenius_norm": segment_norm,
        "saved_segment_to_weight_ratio": segment_norm / weight_norm,
        "cumulative_displacement_frobenius_norm": cumulative_norm,
        "cumulative_displacement_to_weight_ratio": cumulative_norm / weight_norm,
        "saved_segment_row_cv_parameter_weighted": _weighted_metric(
            records, segment, "row_norms", "cv"
        ),
        "saved_segment_top_1pct_row_energy_parameter_weighted": _weighted_metric(
            records, segment, "top_1pct_row_energy"
        ),
    }
    for kind, metric in (("saved_segment", segment), ("cumulative", "delta_from_reference")):
        for label, key in (
            ("stable_rank", "approx_stable_rank"),
            ("sketch_effective_rank", "sketched_entropy_effective_rank"),
        ):
            row[f"{kind}_{label}_parameter_weighted"] = _weighted_metric(records, metric, key)
            row[f"{kind}_{label}_fraction_parameter_weighted"] = _weighted_metric(
                records, metric, key, normalize_rank=True
            )
        row[f"{kind}_sketch_captured_energy_parameter_weighted"] = _weighted_metric(
            records, metric, "captured_frobenius_energy"
        )
    return row


def pair_row(first, second, *, stage, steps, kind, left_bases, right_bases, shapes):
    if set(left_bases) != set(shapes) or set(right_bases) != set(shapes):
        raise ValueError("Subspace pair lacks the exact hidden tensor population")
    sums, defined, undefined, tensors = [0.0, 0.0, 0.0], 0, 0, 0
    for name in sorted(shapes):
        count = math.prod(shapes[name])
        left, right = left_bases[name], right_bases[name]
        if left is None or right is None:
            undefined += count
            continue
        for index, value in enumerate(_basis_overlap(left, right)):
            sums[index] += count * value
        defined += count
        tensors += 1
    total = defined + undefined
    if not total:
        raise ValueError("Subspace pair has no parameter population")
    return {
        "stage": stage,
        "step": steps[stage - 1],
        "progress_fraction": stage / len(steps),
        "displacement_kind": kind,
        "first_run_id": first["run_id"],
        "first_optimizer": first["optimizer"]["name"],
        "first_learning_rate": first["optimizer"]["lr"],
        "second_run_id": second["run_id"],
        "second_optimizer": second["optimizer"]["name"],
        "second_learning_rate": second["optimizer"]["lr"],
        "defined_tensors": tensors,
        "defined_parameters": defined,
        "undefined_zero_parameters": undefined,
        "defined_parameter_fraction": defined / total,
        **{
            name: value / defined if defined else None
            for name, value in zip(
                ("left_subspace_overlap", "right_subspace_overlap", "mean_subspace_overlap"),
                sums,
                strict=True,
            )
        },
    }
