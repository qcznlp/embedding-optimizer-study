"""Bounded CPU controls. Synthetic numerical cases are not study outcomes."""
import copy
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('new_factorial_probe_under_test', HERE / 'probe.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
sys.path.insert(0, str(entry.SOURCE / 'src'))
import numpy as np
import torch


def arrays():
    generator = np.random.default_rng(20260912)
    groups = np.repeat([f'synthetic_task_{i:02d}' for i in range(14)], 16)
    ids = [dict(sample_id=i, source=str(groups[i])) for i in range(224)]
    value = dict(sample_ids=np.arange(224, dtype=np.int64), sample_groups=groups,
        query_embeddings=generator.normal(size=(224, 768)).astype(np.float32),
        document_embeddings=generator.normal(size=(224, 8, 768)).astype(np.float32))
    return value, ids


class Controls(unittest.TestCase):
    def test_original_tie_semantics(self):
        _, summary = entry.functions()
        got = summary(torch.zeros((224, 8)), torch.zeros((224, 8)))
        self.assertEqual(got['mean_reciprocal_rank'], 0.125)
        self.assertEqual(got['top1_accuracy'], 0)
        self.assertEqual(got['pretrained_top1_agreement'], 1)
        self.assertEqual(got['positive_margin_mean'], 0)
        self.assertAlmostEqual(got['contrastive_loss_mean'], math.log(8), places=6)

    def test_original_summary_bad_shape(self):
        _, summary = entry.functions()
        with self.assertRaises(ValueError):
            summary(torch.zeros((3, 7)), torch.zeros((3, 7)))

    def test_original_summary_nonfinite(self):
        _, summary = entry.functions()
        with self.assertRaises(ValueError):
            summary(torch.full((3, 8), float('nan')), torch.zeros((3, 8)))

    def test_full_shapes_and_stored_rounding(self):
        a, ids = arrays()
        stored, scores, result = entry.metrics(a, a, ids)
        self.assertEqual(scores.shape, (224, 8))
        self.assertEqual(scores.dtype, np.float32)
        self.assertEqual(stored['query_embeddings'].dtype, np.float16)
        self.assertEqual(result['overall']['pretrained_top1_agreement'], 1)
        self.assertEqual(len(result['by_task']), 14)
        self.assertTrue(all(row['samples'] == 16 for row in result['by_task'].values()))
        self.assertTrue(np.array_equal(stored['query_embeddings'], a['query_embeddings'].astype(np.float16)))

    def test_independent_numpy_six_metrics(self):
        a, ids = arrays()
        _, scores, result = entry.metrics(a, a, ids)
        q = a['query_embeddings'].astype(np.float16).astype(np.float64)
        d = a['document_embeddings'].astype(np.float16).astype(np.float64)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        independent = np.einsum('bd,bcd->bc', q, d)
        np.testing.assert_allclose(scores, independent, atol=4e-8, rtol=4e-5)
        margin = independent[:, 0] - independent[:, 1:].max(1)
        rank = 1 + (independent[:, 1:] >= independent[:, :1]).sum(1)
        logits = independent / 0.02
        maximum = logits.max(1)
        ce = maximum + np.log(np.exp(logits - maximum[:, None]).sum(1)) - logits[:, 0]
        wanted = dict(positive_margin_mean=margin.mean(), positive_margin_p05=np.quantile(margin, .05),
            mean_reciprocal_rank=(1 / rank).mean(), top1_accuracy=(rank == 1).mean(),
            pretrained_top1_agreement=1, contrastive_loss_mean=ce.mean())
        for key, value in wanted.items():
            self.assertAlmostEqual(result['overall'][key], value, delta=6e-7)

    def test_array_missing_identity(self):
        a, ids = arrays()
        a.pop('sample_ids')
        with self.assertRaises(ValueError):
            entry.metrics(a, a, ids)

    def test_array_bad_width(self):
        a, ids = arrays()
        a['query_embeddings'] = a['query_embeddings'][:, :-1]
        with self.assertRaises(ValueError):
            entry.metrics(a, a, ids)

    def test_array_nonfinite(self):
        a, ids = arrays()
        a['query_embeddings'][0, 0] = float('nan')
        with self.assertRaises(ValueError):
            entry.metrics(a, a, ids)

    def test_array_wrong_row_order(self):
        a, ids = arrays()
        ids.reverse()
        with self.assertRaises(ValueError):
            entry.metrics(a, a, ids)

    def test_array_duplicate_sample(self):
        a, ids = arrays()
        a['sample_ids'][1] = a['sample_ids'][0]
        ids[1] = ids[0]
        with self.assertRaises(ValueError):
            entry.metrics(a, a, ids)

    def test_checkpoint_wrong_stage(self):
        with self.assertRaises(ValueError):
            entry.checkpoint({}, 'a', entry.e.queues()['a'][0], 78)

    def test_checkpoint_wrong_pool(self):
        with self.assertRaises(ValueError):
            entry.checkpoint({}, 'b', entry.e.queues()['a'][0], 79)

    def test_twelve_by_five_population(self):
        actual = [(r, s) for runs in entry.e.queues().values() for r in runs for s in entry.STEPS]
        self.assertEqual(len(actual), 60)
        self.assertEqual(len(set(actual)), 60)
        self.assertEqual(entry.STEPS, [79, 157, 235, 313, 391])

    def test_no_lease_before_whole_training_pool(self):
        class StopWaiting(Exception):
            pass
        with tempfile.TemporaryDirectory(prefix='factorial-probe-wait-control.') as tmp:
            parent = SimpleNamespace(handoff=Mock(), leases=Mock(side_effect=AssertionError('Early GPU request')))
            old = SimpleNamespace(process_start_ticks=lambda _: 0)
            with patch.object(entry, 'HERE', Path(tmp) / 'entry'), patch.object(entry, 'TRAIN', Path(tmp) / 'train'), \
                 patch.object(entry, 'authenticate', return_value=({}, dict(parent=parent, old=old))), \
                 patch.object(entry.time, 'sleep', side_effect=StopWaiting):
                with self.assertRaises(StopWaiting):
                    entry.coordinate(SimpleNamespace(pool='a', source_sha256='synthetic', authorization_sha256='synthetic'))
            parent.leases.assert_not_called()

    def test_wrong_digest_refused(self):
        with tempfile.TemporaryDirectory(prefix='factorial-probe-hash-control.') as tmp:
            p = Path(tmp) / 'value.json'
            p.write_text('{}')
            with self.assertRaises(ValueError):
                entry.read(p, '0' * 64)

    def test_duplicate_json_refused(self):
        with tempfile.TemporaryDirectory(prefix='factorial-probe-json-control.') as tmp:
            p = Path(tmp) / 'value.json'
            p.write_text('{"a":1,"a":2}')
            with self.assertRaises(ValueError):
                entry.read(p)

    def test_lease_wrong_parent_refused(self):
        args = SimpleNamespace(token='7', lease_fds='90,91')
        request = dict(pool='a', coordinator_pid=-1, coordinator_start_ticks=0)
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='7'):
            with self.assertRaises(ValueError):
                entry.check_worker_lease(args, request, {})

    def test_lease_wrong_token_refused(self):
        args = SimpleNamespace(token='7', lease_fds='90,91')
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='6'):
            with self.assertRaises(ValueError):
                entry.check_worker_lease(args, dict(pool='a'), {})


if __name__ == '__main__':
    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    record = dict(scope='bounded-factorial-probe-operational-and-synthetic-numerical-controls',
        source=entry.identity(HERE / 'probe.py'), test_source=entry.identity(Path(__file__)),
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        actual_gpu_execution=False, actual_scientific_results=False)
    entry.write(HERE / 'tests.json', record)
    print(json.dumps(record))
    raise SystemExit(0 if result.wasSuccessful() else 1)
