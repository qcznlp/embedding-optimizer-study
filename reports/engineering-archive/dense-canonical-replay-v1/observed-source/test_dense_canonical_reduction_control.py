import unittest

import torch

from scripts.dense_canonical_reduction_control import CanonicalReducer


class Bucket:
    def __init__(self, index, named, values, last):
        self.i, self.named, self.last = index, named, last
        self.flat = torch.cat([values[name].reshape(-1) for name, _ in named])
        self.parts = []
        offset = 0
        for _, p in named:
            self.parts.append(self.flat[offset : offset + p.numel()].view_as(p))
            offset += p.numel()

    def index(self):
        return self.i

    def parameters(self):
        return [p for _, p in self.named]

    def gradients(self):
        return self.parts

    def buffer(self):
        return self.flat

    def is_last(self):
        return self.last


class CanonicalReductionGuards(unittest.TestCase):
    def fixture(self, collective=None):
        named = [
            ("z", torch.nn.Parameter(torch.zeros(2, 2))),
            ("a", torch.nn.Parameter(torch.zeros(3))),
        ]
        captured = []

        def sum_four_identical_ranks(flat):
            captured.append(flat.clone())
            future = torch.futures.Future()
            future.set_result([flat * 4])
            return future

        state = CanonicalReducer(named, 4, collective or sum_four_identical_ranks)
        values = {"z": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "a": torch.tensor([5.0, 6.0, 7.0])}
        return named, state, values, captured

    def test_partition_and_order_do_not_change_collective_input(self):
        named, state, values, captured = self.fixture()
        first = Bucket(0, named, values, True)
        state.accept(first).wait()
        self.assertTrue(
            torch.equal(first.buffer(), torch.cat([values["z"].flatten(), values["a"]]))
        )
        a, z = Bucket(0, [named[1]], values, False), Bucket(1, [named[0]], values, True)
        pending = state.accept(a)
        self.assertFalse(pending.done())
        state.accept(z).wait()
        self.assertTrue(pending.done())
        self.assertTrue(torch.equal(a.buffer(), values["a"]))
        self.assertTrue(torch.equal(z.buffer(), values["z"].flatten()))
        self.assertTrue(torch.equal(captured[0], captured[1]))
        self.assertTrue(torch.equal(captured[0], torch.tensor([5, 6, 7, 1, 2, 3, 4]) / 4))
        self.assertEqual([x["incoming_ddp_buckets"] for x in state.snapshot()["records"]], [1, 2])

    def test_missing_parameter_fails_and_resolves_prior_future(self):
        named, state, values, _ = self.fixture()
        first = state.accept(Bucket(0, [named[0]], values, False))
        with self.assertRaises(ValueError):
            state.accept(Bucket(1, [named[0]], values, True))
        with self.assertRaises(ValueError):
            first.wait()

    def test_incomplete_reduction_cannot_be_reported(self):
        named, state, values, _ = self.fixture()
        state.accept(Bucket(0, [named[0]], values, False))
        with self.assertRaises(ValueError):
            state.snapshot()

    def test_collective_error_resolves_all_bucket_futures(self):
        def fail(flat):
            future = torch.futures.Future()
            future.set_exception(RuntimeError("synthetic collective failure"))
            return future

        named, state, values, _ = self.fixture(fail)
        first = state.accept(Bucket(0, [named[0]], values, False))
        second = state.accept(Bucket(1, [named[1]], values, True))
        for future in (first, second):
            with self.assertRaises(RuntimeError):
                future.wait()

    def test_duplicate_names_rejected(self):
        p = torch.nn.Parameter(torch.ones(1))
        with self.assertRaises(ValueError):
            CanonicalReducer([("a", p), ("a", torch.nn.Parameter(torch.ones(1)))], 4, None)

    def test_aliased_parameters_rejected(self):
        p = torch.nn.Parameter(torch.ones(1))
        with self.assertRaises(ValueError):
            CanonicalReducer([("a", p), ("b", p)], 4, None)

    def test_non_fp32_rejected(self):
        with self.assertRaises(ValueError):
            CanonicalReducer(
                [("a", torch.nn.Parameter(torch.ones(1, dtype=torch.float64)))], 4, None
            )

    def test_wrong_world_size_rejected(self):
        with self.assertRaises(ValueError):
            CanonicalReducer([("a", torch.nn.Parameter(torch.ones(1)))], 1, None)

    def test_wrong_bucket_order_rejected(self):
        named, state, values, _ = self.fixture()
        with self.assertRaises(ValueError):
            state.accept(Bucket(1, named, values, True))
