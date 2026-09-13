"""Independent oracle controls; these do not change production bridge arithmetic."""

import pytest
import sympy as sp

from scripts.dimension_inference_reference import correlation, ranks


@pytest.mark.parametrize(
    "values,expected",
    [
        ([1, 1, 3], [sp.Rational(3, 2), sp.Rational(3, 2), 3]),
        ([3, -1, 0, -1], [4, sp.Rational(3, 2), 3, sp.Rational(3, 2)]),
        ([sp.Rational(1, 7), sp.Rational(2, 7), sp.Rational(3, 7)], [1, 2, 3]),
    ],
)
def test_sympy_rank_counts_and_exact_ties(values, expected):
    assert ranks(sp.Matrix(values)) == sp.Matrix(expected)


@pytest.mark.parametrize("sign", [-1, 1])
def test_independent_correlation_keeps_both_directions(sign):
    vector = sp.Matrix([1, 2, 2, 4])
    assert correlation(vector, sign * vector) == sign
    assert correlation(ranks(vector), ranks(sign * vector)) == sign
