import math

import numpy as np
import pytest
import torch

from embed_optim.geometry import matrix_metrics as legacy_metrics
from embed_optim.primary_geometry_reductions import global_cosine, matrix_metrics


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (64, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_global_norm_matches_independent_fp64_on_same_kernel_inputs(shape, dtype):
    matrix = torch.randn(shape, generator=torch.Generator().manual_seed(51)).to(dtype)
    expected = float(np.linalg.norm(matrix.float().double().numpy()))
    result = matrix_metrics(matrix, sketch_rank=0)
    assert result["frobenius_norm"] == pytest.approx(expected, rel=1e-12, abs=1e-14)


def test_million_entry_norm_uses_fp64_reduction():
    matrix = torch.full((1024, 1024), 1e-5)
    expected = math.sqrt(matrix.numel()) * float(matrix[0, 0])
    assert matrix_metrics(matrix, sketch_rank=0)["frobenius_norm"] == pytest.approx(
        expected, rel=1e-12
    )


def test_original_spectral_and_row_column_outputs_are_bitwise_unchanged():
    matrix = torch.randn((17, 9), generator=torch.Generator().manual_seed(51))
    settings = dict(sketch_rank=4, oversample=2, power_iterations=2, seed=19)
    old, new = legacy_metrics(matrix, **settings), matrix_metrics(matrix, **settings)
    old.pop("frobenius_norm")
    new.pop("frobenius_norm")
    assert old == new


def test_global_cosine_matches_independent_fp64_without_input_changes():
    left = torch.randn((512, 768), generator=torch.Generator().manual_seed(51))
    right = torch.randn(left.shape, generator=torch.Generator().manual_seed(52))
    before = left.clone(), right.clone()
    a, b = left.double().numpy().ravel(), right.double().numpy().ravel()
    expected = float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
    assert global_cosine(left, right) == pytest.approx(expected, rel=1e-12, abs=1e-14)
    assert torch.equal(left, before[0]) and torch.equal(right, before[1])


def test_zero_cosine_stays_undefined():
    assert global_cosine(torch.eye(2), torch.zeros(2, 2)) is None
