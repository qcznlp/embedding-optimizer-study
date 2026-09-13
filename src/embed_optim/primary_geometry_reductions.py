"""Explicit global-reduction precision correction; the old spectral kernel is unchanged."""

import torch

from .geometry import matrix_metrics as legacy_matrix_metrics


def matrix_metrics(tensor, **settings):
    matrix = tensor.detach().to(device="cpu", dtype=torch.float32)
    result = legacy_matrix_metrics(matrix, **settings)
    # A global FP32 vector reduction over millions of entries failed its fixed
    # independent norm check. Promote the reduction, not the stored inputs/SVD.
    result["frobenius_norm"] = float(torch.linalg.vector_norm(matrix, dtype=torch.float64))
    return result


def global_cosine(left, right):
    left, right = (
        value.detach().to(device="cpu", dtype=torch.float32).double().flatten()
        for value in (left, right)
    )
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if denominator <= 0:
        return None
    return float(torch.dot(left, right) / denominator)
