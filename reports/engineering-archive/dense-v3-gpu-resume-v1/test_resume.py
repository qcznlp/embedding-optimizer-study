"""Synthetic operational/comparison controls, not model or experiment evidence."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import resume as r


def complete_fixture(pool='a'):
    runs = r.e.queues()[pool]
    train = dict(pool=pool, source_sha256=r.TRAIN_SHA, authorization_sha256=r.TRAIN_AUTH,
        full_branches=6, runs=[dict(run_id=run, source_sha256=r.TRAIN_SHA, authorization_sha256=r.TRAIN_AUTH,
        actual_rank_exits=[0]*4, actual_fresh_reader_exit=0, all_five_checkpoints_verified=True,
        full_horizon_391_steps_verified=True) for run in runs])
    evaluation = dict(source_sha256=r.EVAL_SHA, authorization_sha256=r.EVAL_AUTH, verified_tasks=84,
                      runs=dict.fromkeys(runs, {}))
    probe = dict(source_sha256=r.PROBE_SHA, authorization_sha256=r.PROBE_AUTH,
                 checkpoints=[dict(run_id=run, step=s) for run in runs for s in r.STEPS])
    return [train, evaluation, probe]


class OperationalControls(unittest.TestCase):
    def test_complete_pool_a(self):
        r.require_finished('a', complete_fixture())

    def test_complete_pool_b(self):
        r.require_finished('b', complete_fixture('b'))

    def test_training_gap_rejected(self):
        v = complete_fixture(); v[0]['full_branches'] = 5
        with self.assertRaises(ValueError): r.require_finished('a', v)

    def test_failed_rank_rejected(self):
        v = complete_fixture(); v[0]['runs'][0]['actual_rank_exits'][3] = 1
        with self.assertRaises(ValueError): r.require_finished('a', v)

    def test_unread_training_rejected(self):
        v = complete_fixture(); v[0]['runs'][0]['actual_fresh_reader_exit'] = 1
        with self.assertRaises(ValueError): r.require_finished('a', v)

    def test_partial_beir_rejected(self):
        v = complete_fixture(); v[1]['verified_tasks'] = 83
        with self.assertRaises(ValueError): r.require_finished('a', v)

    def test_wrong_beir_run_rejected(self):
        v = complete_fixture(); v[1]['runs'].pop(next(iter(v[1]['runs'])))
        with self.assertRaises(ValueError): r.require_finished('a', v)

    def test_wrong_beir_source_rejected(self):
        v = complete_fixture(); v[1]['source_sha256'] = '0'*64
        with self.assertRaises(ValueError): r.require_finished('a', v)

    def test_partial_probe_rejected(self):
        v = complete_fixture(); v[2]['checkpoints'].pop()
        with self.assertRaises(ValueError): r.require_finished('a', v)

    def test_duplicate_probe_rejected(self):
        v = complete_fixture(); v[2]['checkpoints'][-1] = v[2]['checkpoints'][0]
        with self.assertRaises(ValueError): r.require_finished('a', v)

    def test_wrong_probe_step_rejected(self):
        v = complete_fixture(); v[2]['checkpoints'][-1]['step'] = 390
        with self.assertRaises(ValueError): r.require_finished('a', v)

    def test_wrong_probe_authority_rejected(self):
        v = complete_fixture(); v[2]['authorization_sha256'] = '0'*64
        with self.assertRaises(ValueError): r.require_finished('a', v)

    def test_existing_output_rejected(self):
        with tempfile.TemporaryDirectory(prefix='resume-control-') as d:
            with self.assertRaises(ValueError): r.no_existing(Path(d))
            r.no_existing(Path(d) / 'new')

    def test_relative_output_rejected(self):
        with self.assertRaises(ValueError): r.no_existing(Path('relative'))

    def test_symlink_output_rejected(self):
        with tempfile.TemporaryDirectory(prefix='resume-control-') as d:
            p = Path(d) / 'link'; p.symlink_to(Path(d) / 'missing')
            with self.assertRaises(ValueError): r.no_existing(p / 'new')

    def test_full_named_tree_comparison(self):
        x = {'name': [torch.ones(2), torch.tensor(3)], 'rng': (np.arange(3), 1, 2.0, None)}
        self.assertEqual(r.recursive_equal(x, copy.deepcopy(x)), dict(tensors=2, arrays=1, scalars=3))

    def test_model_tensor_mismatch(self):
        with self.assertRaises(ValueError): r.recursive_equal(torch.ones(2), torch.zeros(2))

    def test_tensor_dtype_mismatch(self):
        with self.assertRaises(ValueError): r.recursive_equal(torch.ones(2), torch.ones(2, dtype=torch.float64))

    def test_signed_tensor_zero_mismatch(self):
        with self.assertRaises(ValueError): r.recursive_equal(torch.tensor(0.), torch.tensor(-0.))

    def test_signed_scalar_zero_mismatch(self):
        with self.assertRaises(ValueError): r.recursive_equal(0., -0.)

    def test_array_byte_mismatch(self):
        with self.assertRaises(ValueError): r.recursive_equal(np.array([0.]), np.array([-0.]))

    def test_optimizer_key_mismatch(self):
        with self.assertRaises(ValueError): r.recursive_equal({'momentum': 1}, {'variance': 1})

    def test_container_type_mismatch(self):
        with self.assertRaises(ValueError): r.recursive_equal([1], (1,))

    def test_length_mismatch(self):
        with self.assertRaises(ValueError): r.recursive_equal([1], [1, 2])

    def test_nonfinite_scalar_rejected(self):
        with self.assertRaises(ValueError): r.recursive_equal(float('nan'), float('nan'))

    def test_bf16_scalar_and_noncontiguous(self):
        for x in [torch.tensor(2., dtype=torch.bfloat16), torch.arange(12).reshape(3,4).T]:
            self.assertEqual(r.recursive_equal(x, x.clone())['tensors'], 1)

    def test_native_resume_not_fresh_worker(self):
        tree = ast.parse(Path(r.__file__).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        names = [n.func.attr for n in calls if isinstance(n.func, ast.Attribute)]
        self.assertIn('train', names)
        self.assertNotIn('prepare_trainer', names)
        self.assertNotIn('save_run_completion', names)
        self.assertNotIn('run_branch', names)
        train = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'train']
        self.assertEqual(len(train), 1)
        self.assertEqual([k.arg for k in train[0].keywords], ['resume_from_checkpoint'])

    def test_no_duplicate_dynamic_dict_keywords(self):
        tree = ast.parse(Path(r.__file__).read_text())
        # process_identity already owns 'command'; never expand it into dict(**x, command=...).
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'dict':
                if any(k.arg is None for k in n.keywords):
                    self.assertNotIn('command', [k.arg for k in n.keywords])

    def test_explicit_command_source_and_authority(self):
        a = SimpleNamespace(source_sha='a'*64, authorization_sha='b'*64, pool='a')
        cmd = r.command(a, 'compare')
        self.assertEqual(cmd[:4], ['/usr/bin/python', '-B', str(r.HERE / 'resume.py'), 'compare'])
        self.assertEqual(cmd[-2:], ['--pool', 'a'])


if __name__ == '__main__':
    unittest.main()
