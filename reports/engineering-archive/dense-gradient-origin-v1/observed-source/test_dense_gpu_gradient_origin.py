import unittest

import torch

from scripts.audit_dense_gpu_gradient_origin import (
    all_exact,
    compare_named,
    local_sum,
    validate_layout,
)


class GradientOriginGuards(unittest.TestCase):
    def test_local_sum_retains_order_and_inputs(self):
        parts = [
            torch.tensor([1e8]),
            torch.tensor([1.0]),
            torch.tensor([-1e8]),
            torch.tensor([2.0]),
        ]
        result = local_sum({"a": parts}, ["a"])
        self.assertEqual(result["a"].item(), 2.0)
        self.assertEqual(parts[0].item(), 1e8)

    def test_partial_accumulation_rejected(self):
        with self.assertRaises(ValueError):
            local_sum({"a": [torch.ones(1)]}, ["a"])

    def test_layout_accepts_reordering_but_not_coverage_change(self):
        validate_layout([{"index": 0, "names": ["b"]}, {"index": 1, "names": ["a"]}], ["a", "b"])
        with self.assertRaises(ValueError):
            validate_layout([{"index": 0, "names": ["a", "a"]}], ["a", "b"])

    def test_bucket_indices_not_reordered_silently(self):
        with self.assertRaises(ValueError):
            validate_layout([{"index": 1, "names": ["a"]}], ["a"])

    def test_numerical_difference_not_rounded_to_exact(self):
        comparison = compare_named(
            {"a": torch.tensor([1.0])}, {"a": torch.tensor([1.0 + 1e-7])}, ["a"]
        )
        self.assertFalse(all_exact(comparison))

    def test_missing_named_comparison_rejected(self):
        with self.assertRaises(ValueError):
            compare_named({"a": torch.ones(1)}, {}, ["a"])

    def test_empty_comparison_not_a_pass(self):
        self.assertFalse(all_exact({}))
