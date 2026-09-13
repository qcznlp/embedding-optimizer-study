"""Synthetic portability controls; never native scientific/model admission."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import combined as c


class CompletePortabilityControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='combined-portability-controls-')
        cls.root = Path(cls.temp.name)
        for relative, source, sha in (
            ('source/original_summary.py', c.STORY / 'src/embed_optim/state_operator_factorial_summary.py', c.ORIGINAL_SHA),
            ('source/native_summary.py', c.SUMMARY / 'summarize.py', c.SUMMARY_SHA)):
            c.copy(source, cls.root / relative, sha)
        cls.tasks = ['fixture-task-' + str(i).zfill(2) for i in range(14)]
        cls.training = {'queues': {'a': [], 'b': []}, 'requests': {}}
        cls.beir = {'rows': []}
        cls.probe = {'overall_rows': [], 'task_rows': []}
        for seed in (314159, 271828, 161803):
            for si, state in enumerate(('adamw_state', 'muon_state')):
                for oi, operator in enumerate(('adamw', 'muon')):
                    run = f'fixture-{state}-{operator}-{seed}'
                    cls.training['queues'][('a', 'b')[oi]].append(run)
                    cls.training['requests'][run] = dict(state=state, operator=operator, seed=seed)
                    for ti, task in enumerate(cls.tasks):
                        cls.beir['rows'].append(dict(run_id=run, state=state, operator=operator,
                            seed=seed, task=task, ndcg_at_10=.2 + .01 * si + .02 * oi + ti / 1000))
                    for stage, step in enumerate((79, 157, 235, 313, 391), 1):
                        row = dict(state=state, operator=operator, seed=seed, stage=stage, step=step,
                            label=run + '/checkpoint-' + str(step), fraction=stage / 5, samples=224,
                            contrastive_loss_mean=.2, positive_margin_mean=.1, positive_margin_p05=.05,
                            mean_reciprocal_rank=.8, top1_accuracy=.7, pretrained_top1_agreement=.6)
                        cls.probe['overall_rows'].append(row)
                        for task in cls.tasks:
                            cls.probe['task_rows'].append(dict(row, task=task, samples=16))
        cls.functions = c.factorial_functions(cls.root, cls.training)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_connected_original_numerics_and_independent_check(self):
        tables = self.functions['infer'](self.beir, self.probe, self.tasks, self.training)
        self.assertEqual({k: len(v) for k, v in tables.items()}, c.COUNTS)
        check = self.functions['independent'](tables)
        self.assertEqual(check['independent_rational_seed_task_effects'], 126)
        self.assertEqual(check['tolerance'], 2e-15)
        self.assertFalse(check['scientific_completion'])

    def test_real_assertion_not_an_admission_callback(self):
        metadata = c.MetadataAccess(self.training['queues'])
        with self.assertRaises(ValueError):
            metadata.need(False, 'genuine refusal')
        self.assertFalse(hasattr(metadata, 'admit'))
        self.assertFalse(hasattr(metadata, 'recheck'))

    def test_missing_or_duplicate_beir_refused(self):
        for rows in (self.beir['rows'][:-1], self.beir['rows'][:-1] + [self.beir['rows'][0]]):
            with self.assertRaises(ValueError):
                self.functions['validate_scores'](rows, self.tasks, self.training)

    def test_changed_seed_or_metric_refused(self):
        for key, value in (('seed', 42), ('seed', 314159.0), ('ndcg_at_10', float('nan')),
                           ('ndcg_at_10', 1.01), ('task', 'unregistered')):
            rows = copy.deepcopy(self.beir['rows'])
            rows[0][key] = value
            with self.assertRaises(ValueError):
                self.functions['validate_scores'](rows, self.tasks, self.training)

    def test_incomplete_probe_refused(self):
        for family in ('overall_rows', 'task_rows'):
            probe = copy.deepcopy(self.probe)
            probe[family].pop()
            with self.assertRaises(ValueError):
                self.functions['validate_probe'](probe, self.tasks, self.training)

    def test_invalid_probe_values_refused(self):
        for key, value in (('fraction', .3), ('samples', 223), ('positive_margin_mean', float('inf'))):
            probe = copy.deepcopy(self.probe)
            probe['overall_rows'][0][key] = value
            with self.assertRaises(ValueError):
                self.functions['validate_probe'](probe, self.tasks, self.training)

    def test_original_bootstrap_settings_remain_fixed(self):
        import numpy as np
        numerical = self.functions['numerical_functions'](self.tasks)
        for args in ({'samples': 10}, {'seed': 42}):
            with self.assertRaises(ValueError):
                numerical['two_way_cluster_bootstrap'](np.zeros((3, 14)), **args)

    def test_nonlocal_roles_refused(self):
        for value in ('', '../outside', '/outside', 'x/../outside', 'x//y'):
            with self.assertRaises(ValueError):
                c.local_name(value)

    def make_inventory(self):
        root = Path(tempfile.mkdtemp(prefix='inventory-', dir=self.root))
        c.write(root / 'data.json', {'synthetic_only': True})
        c.write(root / 'manifest.json', {'scope': 'explicitly-synthetic-integrity-only',
                'files': {'data.json': c.identity(root / 'data.json')}})
        return root, c.identity(root / 'manifest.json')['sha256']

    def test_inventory_integrity_only_positive(self):
        root, sha = self.make_inventory()
        result = c.inventory(root, sha, 'explicitly-synthetic-integrity-only')
        self.assertNotEqual(result['scope'], c.SCOPE)

    def test_changed_external_anchor_refused(self):
        root, sha = self.make_inventory()
        with self.assertRaises(ValueError):
            c.inventory(root, '0' * 64, 'explicitly-synthetic-integrity-only')

    def test_extra_and_missing_inventory_refused(self):
        for missing in (False, True):
            root, sha = self.make_inventory()
            if missing:
                (root / 'data.json').rename(root / 'unexpected.json')
            else:
                c.write(root / 'unexpected.json', {'synthetic_only': True})
            with self.assertRaises(ValueError):
                c.inventory(root, sha, 'explicitly-synthetic-integrity-only')

    def test_symlink_refused(self):
        root, sha = self.make_inventory()
        (root / 'unexpected-link').symlink_to(root / 'data.json')
        with self.assertRaises(ValueError):
            c.inventory(root, sha, 'explicitly-synthetic-integrity-only')

    def test_prior_output_preserved(self):
        root, _ = self.make_inventory()
        old = c.identity(root / 'data.json')
        with self.assertRaises(FileExistsError):
            c.write(root / 'data.json', {'overwrite': True})
        self.assertEqual(c.identity(root / 'data.json'), old)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CompletePortabilityControls))
    c.write(c.HERE / 'tests-final.json', {'scope': 'synthetic-combined-portability-controls',
        'source': c.identity(c.HERE / 'combined.py'), 'test_source': c.identity(Path(__file__)),
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'skips': len(result.skipped), 'actual_factorial_or_paper_admission': False})
    raise SystemExit(0 if result.wasSuccessful() else 1)
