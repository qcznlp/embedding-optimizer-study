"""Small exact OLS systems on the unchanged binary64 bridge inputs.

No rounded decimal inputs, denominator truncation, ridge term or fitted error
epsilon is used. Square roots are display-only; inference compares rational MSE.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction as Q

import numpy as np


def rational(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError("Require a finite real numeric bridge input")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Require a finite real numeric bridge input")
    return Q.from_float(value)


def dot(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), Q(0))


def inverse(matrix):
    """Exact Gauss–Jordan inverse; a singular baseline is an error, not a fit."""
    n = len(matrix)
    if not n or any(len(row) != n for row in matrix):
        raise ValueError("Require a nonempty square rational matrix")
    work = [list(row) + [Q(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = next((i for i in range(column, n) if work[i][column]), None)
        if pivot is None:
            raise ValueError("Singular exact baseline design")
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [x / scale for x in work[column]]
        for i in range(n):
            if i != column and work[i][column]:
                scale = work[i][column]
                work[i] = [x - scale * y for x, y in zip(work[i], work[column], strict=True)]
    return [row[n:] for row in work]


def decimal(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def displayed(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Exact result cannot be represented as a finite display value")
    return result


def root(value):
    if value < 0:
        raise ValueError("Negative squared quantity")
    with localcontext() as context:
        context.prec = 80
        return displayed(decimal(value).sqrt())


def rmse_reduction(base_mse, feature_mse):
    """Cancellation-safe display; the support decision never uses this float."""
    if base_mse == feature_mse:
        return 0.0
    with localcontext() as context:
        context.prec = 80
        denominator = decimal(base_mse).sqrt() + decimal(feature_mse).sqrt()
        return displayed(decimal(base_mse - feature_mse) / denominator)


def mse(observed, predicted):
    if not observed or len(observed) != len(predicted):
        raise ValueError("Require equal nonempty prediction vectors")
    return sum(((x - y) ** 2 for x, y in zip(observed, predicted, strict=True)), Q(0)) / len(
        observed
    )


def average_ranks(values):
    ordered = sorted(set(values))
    counts = {x: values.count(x) for x in ordered}
    ranks, offset = {}, 0
    for value in ordered:
        count = counts[value]
        ranks[value] = Q(2 * offset + count + 1, 2)
        offset += count
    return [ranks[x] for x in values]


def correlation(a, b):
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Require equal non-trivial association vectors")
    am, bm = sum(a) / len(a), sum(b) / len(b)
    ac, bc = [x - am for x in a], [x - bm for x in b]
    aa, bb = dot(ac, ac), dot(bc, bc)
    if aa == 0 or bb == 0:
        return None
    cross = dot(ac, bc)
    if cross == 0:
        return 0.0
    square = cross * cross / (aa * bb)
    if not 0 <= square <= 1:
        raise ValueError("Invalid exact correlation")
    return (1 if cross > 0 else -1) * root(square)


def design_health(baseline, values):
    """Training-only standardization and the documented NumPy default rank cutoff.

    Scaling before float conversion avoids under/overflow in the diagnostic std.
    These singular values do not replace the exact residual-zero check.
    """
    mean = sum(values) / len(values)
    centered = [x - mean for x in values]
    variance = dot(centered, centered) / len(centered)
    if variance:
        amplitude = max(abs(x) for x in centered)
        bounded = np.array([float(x / amplitude) for x in centered], dtype=np.float64)
        standardized = bounded / np.sqrt(np.mean(bounded * bounded))
    else:
        standardized = np.zeros(len(values), dtype=np.float64)
    augmented = np.column_stack((baseline, standardized))
    singular = np.linalg.svd(augmented, compute_uv=False)
    cutoff = float(np.finfo(np.float64).eps * max(augmented.shape) * singular[0])
    return {
        "training_feature_mean": displayed(mean),
        "training_feature_scale": root(variance),
        "training_feature_mean_exact": str(mean),
        "training_feature_variance_exact": str(variance),
        "augmented_numeric_rank": int(np.count_nonzero(singular > cutoff)),
        "rank_cutoff": cutoff,
        "singular_values": singular.tolist(),
    }


class Baseline:
    """One full-rank baseline fit reused by all single-feature comparisons."""

    def __init__(self, design, indices):
        self.design = design
        self.indices = tuple(indices)
        self.train = [design[i] for i in self.indices]
        self.numeric = np.array(self.train, dtype=np.float64)
        singular = np.linalg.svd(self.numeric, compute_uv=False)
        cutoff = np.finfo(np.float64).eps * max(self.numeric.shape) * singular[0]
        if np.count_nonzero(singular > cutoff) != 8:
            raise ValueError("Baseline must have numerical rank eight in every fold")
        self.columns = list(map(list, zip(*self.train, strict=True)))
        self.inverse_gram = inverse([[dot(a, b) for b in self.columns] for a in self.columns])

    def project(self, values):
        train = [values[i] for i in self.indices]
        cross = [dot(column, train) for column in self.columns]
        beta = [dot(row, cross) for row in self.inverse_gram]
        prediction = [dot(row, beta) for row in self.design]
        residual = [values[i] - prediction[i] for i in self.indices]
        if any(dot(column, residual) for column in self.columns):
            raise ValueError("Exact baseline normal equations failed")
        return prediction, residual

    def feature(self, values, outcome):
        base, outcome_residual = self.project(outcome)
        feature_prediction, residual = self.project(values)
        health = design_health(self.numeric, [values[i] for i in self.indices])
        energy = dot(residual, residual)
        extension = [x - p for x, p in zip(values, feature_prediction, strict=True)]
        if energy == 0:
            status = (
                "baseline_equivalent"
                if all(x == 0 for x in extension)
                else "unidentified_extension"
            )
            predictions = base if status == "baseline_equivalent" else None
        elif health["augmented_numeric_rank"] != 9:
            status, predictions = "unresolved_nonzero_direction", None
        else:
            coefficient = dot(residual, outcome_residual) / energy
            predictions = [b + coefficient * x for b, x in zip(base, extension, strict=True)]
            status = "resolved"
        return {
            "status": status,
            "health": health,
            "exact_residual_energy": str(energy),
            "baseline_prediction": base,
            "feature_prediction": predictions,
            "feature_residual": residual,
        }
