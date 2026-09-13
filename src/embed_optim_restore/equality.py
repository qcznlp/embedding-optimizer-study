"""Original recursive state equality, retained without a numerical tolerance."""


def need(condition, message):
    if not condition:
        raise ValueError(message)


def recursive_equal(left, right, label="state"):
    """Every tensor/array bit pattern and every named scalar/container; no tolerance."""
    import numpy as np
    import torch

    need(type(left) is type(right), "Type mismatch: " + label)
    if isinstance(left, torch.Tensor):
        need(
            left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(
                left.contiguous().reshape(-1).view(torch.uint8),
                right.contiguous().reshape(-1).view(torch.uint8),
            ),
            "Tensor mismatch: " + label,
        )
        return {"tensors": 1, "arrays": 0, "scalars": 0}
    if isinstance(left, np.ndarray):
        need(
            left.dtype == right.dtype
            and left.shape == right.shape
            and left.tobytes() == right.tobytes(),
            "Array mismatch: " + label,
        )
        return {"tensors": 0, "arrays": 1, "scalars": 0}
    if isinstance(left, dict):
        need(left.keys() == right.keys(), "Key mismatch: " + label)
        pairs = [(k, left[k], right[k]) for k in left]
    elif isinstance(left, (list, tuple)):
        need(len(left) == len(right), "Length mismatch: " + label)
        pairs = [(i, a, b) for i, (a, b) in enumerate(zip(left, right))]
    else:
        if isinstance(left, float):
            import math
            import struct

            need(
                math.isfinite(left)
                and math.isfinite(right)
                and struct.pack("!d", left) == struct.pack("!d", right),
                "Scalar mismatch: " + label,
            )
        else:
            need(left == right, "Scalar mismatch: " + label)
        return {"tensors": 0, "arrays": 0, "scalars": 1}
    count = {"tensors": 0, "arrays": 0, "scalars": 0}
    for name, a, b in pairs:
        for key, value in recursive_equal(a, b, label + "/" + str(name)).items():
            count[key] += value
    return count
