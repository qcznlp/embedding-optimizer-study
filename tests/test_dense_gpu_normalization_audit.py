import unittest
from types import SimpleNamespace

import torch

from scripts.audit_dense_gpu_normalization import batch_hash, measure_gradients, tensor_hash


class GpuNormalizationDiagnosticGuards(unittest.TestCase):
    def gradients(self, scale=1.0):
        return [("matrix", SimpleNamespace(grad=torch.tensor([[1.0, 2.0], [3.0, 4.0]]) * scale))]

    def test_correct_gradient(self):
        result = measure_gradients(self.gradients(), self.gradients())
        self.assertTrue(result["all_parameters_match_prior_tolerance"])
        self.assertEqual(result["norm_ratio"], 1.0)

    def test_quarter_scale_is_failure(self):
        result = measure_gradients(self.gradients(0.25), self.gradients())
        self.assertFalse(result["all_parameters_match_prior_tolerance"])
        self.assertEqual(result["norm_ratio"], 0.25)

    def test_compensation_is_separate_diagnostic(self):
        result = measure_gradients(self.gradients(0.25), self.gradients(), 4.0)
        self.assertTrue(result["all_parameters_match_prior_tolerance"])
        self.assertEqual(result["diagnostic_multiplier"], 4.0)

    def test_missing_gradient_rejected(self):
        with self.assertRaises(ValueError):
            measure_gradients([("matrix", SimpleNamespace(grad=None))], self.gradients())

    def test_reordered_gradient_rejected(self):
        with self.assertRaises(ValueError):
            measure_gradients([("different", self.gradients()[0][1])], self.gradients())

    def test_nonfinite_rejected(self):
        x = self.gradients()
        x[0][1].grad[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            measure_gradients(x, self.gradients())

    def test_zero_gradient_not_normalization_evidence(self):
        with self.assertRaises(ValueError):
            measure_gradients(self.gradients(0), self.gradients(0))

    def test_token_hash_retains_shape_dtype_and_values(self):
        x = torch.tensor([[1, 2]])
        self.assertNotEqual(tensor_hash(x), tensor_hash(x.float()))
        self.assertNotEqual(tensor_hash(x), tensor_hash(x.flatten()))
        self.assertNotEqual(tensor_hash(x), tensor_hash(x + 1))

    def test_audit_row_metadata_not_reference_tokens(self):
        batch = {"query_input_ids": torch.tensor([[1, 2]]), "_audit_row_ids": torch.tensor([0])}
        self.assertEqual(
            batch_hash(batch), batch_hash({"query_input_ids": batch["query_input_ids"]})
        )
