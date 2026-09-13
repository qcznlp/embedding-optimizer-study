"""Exact-spectrum controls, separate from the frozen approximate primary features.

All measurements are on the identical FP32 saved displacement promoted losslessly
to FP64. QR only changes a basis representation during projector comparisons;
it never changes the saved primary bases or the column space being measured.
"""

from __future__ import annotations

import math

import numpy as np

ORTHOGONAL_ATOL = 2e-5
REFERENCE_RTOL = 2e-8
REFERENCE_ATOL = 1e-10


def finite_matrix(value):
    value = np.asarray(value)
    if (
        value.ndim != 2
        or not value.size
        or np.iscomplexobj(value)
        or not np.issubdtype(value.dtype, np.number)
        or not np.isfinite(value).all()
    ):
        raise ValueError("Require a nonempty finite real matrix")
    return value.astype(np.float64, copy=False)


def orthogonal_basis(value):
    value = finite_matrix(value)
    if value.shape[1] > value.shape[0]:
        raise ValueError("Basis has more columns than ambient dimensions")
    error = float(np.max(np.abs(value.T @ value - np.eye(value.shape[1]))))
    if error > ORTHOGONAL_ATOL:
        raise ValueError("Input columns are not an approximately orthonormal basis")
    return np.linalg.qr(value, mode="reduced")[0], error


def projector_comparison(first, second):
    """Rotation/sign-invariant overlap and RMS sine of principal angles.

    RMS sine is ||P-Q||_F / sqrt(2r), computed from the actual projectors
    to avoid subtractive loss when the spaces agree nearly exactly.
    """
    first, error_first = orthogonal_basis(first)
    second, error_second = orthogonal_basis(second)
    if first.shape != second.shape:
        raise ValueError("Compare bases of the same dimension and rank")
    rank = first.shape[1]
    cross = first.T @ second
    cosines = np.linalg.svd(cross, compute_uv=False)
    if cosines.max() > 1 + REFERENCE_RTOL:
        raise ValueError("Principal-angle cosines exceed one")
    cosines = np.clip(cosines, 0, 1)
    difference = first @ first.T - second @ second.T
    distance = float(np.linalg.norm(difference) / math.sqrt(2 * rank))
    overlap = float(np.sum(cross**2) / rank)
    if not math.isclose(distance**2, 1 - overlap, abs_tol=REFERENCE_ATOL):
        raise ValueError("Projector and cross-Gram definitions disagree")
    return {
        "overlap": float(np.clip(overlap, 0, 1)),
        "projector_rms_sine": distance,
        "largest_principal_angle_degrees": float(np.degrees(np.arccos(cosines.min()))),
        "first_original_gram_max_error": error_first,
        "second_original_gram_max_error": error_second,
    }


def spectrum_summary(singular):
    singular = np.asarray(singular, dtype=np.float64)
    if (
        singular.ndim != 1
        or not singular.size
        or not np.isfinite(singular).all()
        or np.any(singular < 0)
        or np.any(np.diff(singular) > 0)
    ):
        raise ValueError("Require the complete sorted nonnegative singular spectrum")
    if singular[0] == 0:
        return {
            "zero": True,
            "spectral_norm": 0.0,
            "frobenius_norm": 0.0,
            "stable_rank": None,
            "full_entropy_effective_rank": None,
        }
    # Scaling prevents avoidable overflow/underflow of the spectrum's ratios.
    relative = singular / singular[0]
    probabilities = relative[relative > 0] / relative.sum()
    frobenius = float(singular[0] * np.linalg.norm(relative))
    if not math.isfinite(frobenius):
        raise ValueError("Spectrum Frobenius norm overflowed")
    return {
        "zero": False,
        "spectral_norm": float(singular[0]),
        "frobenius_norm": frobenius,
        "stable_rank": float(np.sum(relative**2)),
        "full_entropy_effective_rank": float(
            np.exp(-np.sum(probabilities * np.log(probabilities)))
        ),
    }


def exact_reference(matrix, *, backend="numpy"):
    matrix = finite_matrix(matrix)
    norm = float(np.linalg.norm(matrix))
    if not math.isfinite(norm) or (np.any(matrix != 0) and norm == 0):
        raise ValueError("Matrix norm lies outside this FP64 reference's supported scale")
    if backend == "numpy":
        left, singular, right_h = np.linalg.svd(matrix, full_matrices=False)
    elif backend == "torch":
        import torch

        left, singular, right_h = (
            value.numpy()
            for value in torch.linalg.svd(torch.from_numpy(matrix.copy()), full_matrices=False)
        )
    else:
        raise ValueError("Undeclared exact SVD backend")
    summary = spectrum_summary(singular)
    error = float(np.linalg.norm((left * singular) @ right_h - matrix))
    if error > REFERENCE_ATOL + REFERENCE_RTOL * norm:
        raise ValueError("Full SVD does not reconstruct its input")
    summary["relative_reconstruction_residual"] = error / norm if norm else None
    return (left, singular, right_h.T), summary


def subspace_quality(matrix, exact, candidate, *, rank):
    matrix = finite_matrix(matrix)
    left, singular, right = exact
    if type(rank) is not int or not 0 < rank <= min(matrix.shape):
        raise ValueError("Invalid retained rank")
    if singular[0] <= 0:
        raise ValueError("Zero displacement has no defined signal subspace")
    threshold = np.finfo(np.float64).eps * max(matrix.shape) * singular[0]
    if singular[rank - 1] <= threshold:
        raise ValueError("Exact spectrum does not support the declared rank")
    candidate_left, _ = orthogonal_basis(candidate[0])
    candidate_right, _ = orthogonal_basis(candidate[1])
    if candidate_left.shape != (matrix.shape[0], rank) or candidate_right.shape != (
        matrix.shape[1],
        rank,
    ):
        raise ValueError("Candidate does not have the declared shape and rank")
    norm_squared = float(np.sum(matrix**2))
    if not math.isfinite(norm_squared) or norm_squared <= 0:
        raise ValueError("Projection energy lies outside the supported FP64 scale")
    optimum = float(np.sum(singular[:rank] ** 2) / norm_squared)
    left_energy = float(np.sum((candidate_left.T @ matrix) ** 2) / norm_squared)
    right_energy = float(np.sum((matrix @ candidate_right) ** 2) / norm_squared)
    if max(left_energy, right_energy) > optimum + REFERENCE_ATOL:
        raise ValueError("Projected energy exceeds the exact variational optimum")
    next_value = float(singular[rank]) if rank < len(singular) else None
    return {
        "rank": rank,
        "exact_relative_boundary_gap": (float(singular[rank - 1]) - next_value)
        / float(singular[rank - 1])
        if next_value is not None
        else None,
        "exact_boundary_gap_to_leading": (float(singular[rank - 1]) - next_value)
        / float(singular[0])
        if next_value is not None
        else None,
        "exact_top_rank_energy": optimum,
        "left_captured_energy": left_energy,
        "right_captured_energy": right_energy,
        "left_energy_relative_shortfall": (optimum - left_energy) / optimum,
        "right_energy_relative_shortfall": (optimum - right_energy) / optimum,
        "left": projector_comparison(left[:, :rank], candidate[0]),
        "right": projector_comparison(right[:, :rank], candidate[1]),
    }
