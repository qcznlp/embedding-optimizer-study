"""Synthetic estimator and complete-population controls, not experiment results."""
import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('new_factorial_inference_under_test', HERE / 'summarize.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def fixtures():
    _, tasks, training = entry.context()
    rng = np.random.default_rng(904)
    scores, overall, task_rows = [], [], []
    for runs in entry.e.queues().values():
        for run in runs:
            req = training['requests'][run]
            state, operator, seed = req['state'], req['operator'], req['seed']
            s, o = int(state == 'muon_state'), int(operator == 'muon')
            for task in tasks:
                scores.append(dict(state=state, operator=operator, seed=seed, run_id=run, task=task,
                    ndcg_at_10=float(.5 + s * .03 + o * .02 + s * o * .012 + rng.normal(0, .002)),
                    result_path='synthetic-only-not-an-actual-result'))
            for stage, step in enumerate((79, 157, 235, 313, 391), 1):
                meta = dict(state=state, operator=operator, seed=seed, stage=stage, fraction=stage / 5,
                            step=step, label=run + '/checkpoint-' + str(step))
                metric = dict(contrastive_loss_mean=.2, positive_margin_mean=.1, positive_margin_p05=-.1,
                    mean_reciprocal_rank=.8, top1_accuracy=.7, pretrained_top1_agreement=.6)
                overall.append({**meta, 'samples': 224, **metric})
                task_rows.extend({**meta, 'task': task, 'samples': 16, **metric} for task in tasks)
    return dict(rows=scores), dict(overall_rows=overall, task_rows=task_rows), tasks, training


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.beir, cls.probe, cls.tasks, cls.training = fixtures()
        cls.functions = entry.numerical_functions(cls.tasks)

    def test_all_original_tables_and_independent_bootstrap(self):
        tables = entry.infer(self.beir, self.probe, self.tasks, self.training)
        self.assertEqual({k: len(v) for k, v in tables.items()}, dict(beir_seed_task_scores=168,
            factorial_cell_summary=4, estimand_seed_task_contrasts=126, estimand_summary=3,
            probe_checkpoint_metrics=60, probe_task_metrics=840))
        independent = entry.independent(tables)
        self.assertEqual(independent['independent_rational_seed_task_effects'], 126)
        self.assertEqual(len(independent['independent_multiplicity_bootstrap']), 3)
        self.assertTrue(all(row['decision'] == 'supported_positive' for row in tables['estimand_summary']))

    def test_zero_effect_is_inconclusive(self):
        out = self.functions['two_way_cluster_bootstrap'](np.zeros((3, 14)))
        self.assertEqual(out['decision'], 'inconclusive')
        self.assertEqual(out['point_estimate'], 0)
        self.assertEqual(out['bootstrap_ci_95_lower'], 0)
        self.assertEqual(out['bootstrap_ci_95_upper'], 0)

    def test_negative_effect_is_retained(self):
        out = self.functions['two_way_cluster_bootstrap'](-np.ones((3, 14)))
        self.assertEqual(out['decision'], 'supported_negative')

    def test_wrong_bootstrap_seed(self):
        with self.assertRaises(ValueError):
            self.functions['two_way_cluster_bootstrap'](np.zeros((3, 14)), seed=1)

    def test_wrong_bootstrap_draws(self):
        with self.assertRaises(ValueError):
            self.functions['two_way_cluster_bootstrap'](np.zeros((3, 14)), samples=99999)

    def test_wrong_bootstrap_shape(self):
        with self.assertRaises(ValueError):
            self.functions['two_way_cluster_bootstrap'](np.zeros((2, 14)))

    def test_nonfinite_bootstrap(self):
        with self.assertRaises(ValueError):
            self.functions['two_way_cluster_bootstrap'](np.full((3, 14), float('nan')))

    def test_reordered_complete_panel_is_not_selected(self):
        rows = list(reversed(self.beir['rows']))
        entry.validate_scores(rows, self.tasks, self.training)
        self.assertEqual(self.functions['_effect_rows'](rows), self.functions['_effect_rows'](self.beir['rows']))

    def test_actual_collectors_refuse_missing_beir(self):
        with tempfile.TemporaryDirectory(prefix='factorial-collector-no-beir.') as tmp:
            with patch.object(entry.c, 'EVAL_ROOT', Path(tmp)):
                with self.assertRaises(FileNotFoundError):
                    entry.c.require_complete_pair('beir')

    def test_actual_collectors_refuse_missing_probe(self):
        with tempfile.TemporaryDirectory(prefix='factorial-collector-no-probe.') as tmp:
            with patch.object(entry.c, 'PROBE_ROOT', Path(tmp)):
                with self.assertRaises(FileNotFoundError):
                    entry.c.require_complete_pair('probe')

    def test_waiting_does_not_start_collectors(self):
        class StopWaiting(Exception):
            pass
        with tempfile.TemporaryDirectory(prefix='factorial-summary-wait-control.') as tmp:
            helper = SimpleNamespace(process_start_ticks=lambda _: 0, environment=Mock())
            with patch.object(entry, 'HERE', Path(tmp)), patch.object(entry, 'authenticate', return_value={}), \
                 patch.object(entry.e, 'load', return_value=helper), patch.object(entry, 'ready', return_value=False), \
                 patch.object(entry.time, 'sleep', side_effect=StopWaiting), patch.object(entry.subprocess, 'Popen') as child:
                with self.assertRaises(StopWaiting):
                    entry.coordinate(SimpleNamespace(source_sha256='synthetic', authorization_sha256='synthetic'))
            child.assert_not_called()


SCORE_MUTATIONS = {
    'missing_score': lambda rows: rows.pop(),
    'duplicate_score': lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
    'unexpected_run': lambda rows: rows[0].update(run_id='historical-or-unregistered'),
    'unexpected_seed': lambda rows: rows[0].update(seed=42),
    'unexpected_operator': lambda rows: rows[0].update(operator='normuon'),
    'unexpected_task': lambda rows: rows[0].update(task='unknown-task'),
    'bool_score': lambda rows: rows[0].update(ndcg_at_10=True),
    'nonfinite_score': lambda rows: rows[0].update(ndcg_at_10=float('inf')),
    'unbounded_score': lambda rows: rows[0].update(ndcg_at_10=1.001),
}
for name, mutation in SCORE_MUTATIONS.items():
    def test(self, mutate=mutation):
        rows = copy.deepcopy(self.beir['rows'])
        mutate(rows)
        with self.assertRaises(ValueError):
            entry.validate_scores(rows, self.tasks, self.training)
    setattr(Controls, 'test_refuse_' + name, test)

PROBE_MUTATIONS = {
    'missing_probe_state': lambda value: value['overall_rows'].pop(),
    'duplicate_probe_state': lambda value: value['overall_rows'].__setitem__(1, copy.deepcopy(value['overall_rows'][0])),
    'wrong_probe_stage': lambda value: value['overall_rows'][0].update(step=78),
    'wrong_probe_fraction': lambda value: value['overall_rows'][0].update(fraction=.1),
    'missing_probe_task': lambda value: value['task_rows'].pop(),
    'duplicate_probe_task': lambda value: value['task_rows'].__setitem__(1, copy.deepcopy(value['task_rows'][0])),
    'wrong_probe_sample_count': lambda value: value['task_rows'][0].update(samples=15),
    'nonfinite_probe_metric': lambda value: value['task_rows'][0].update(positive_margin_mean=float('nan')),
}
for name, mutation in PROBE_MUTATIONS.items():
    def test(self, mutate=mutation):
        value = copy.deepcopy(self.probe)
        mutate(value)
        with self.assertRaises(ValueError):
            entry.validate_probe(value, self.tasks, self.training)
    setattr(Controls, 'test_refuse_' + name, test)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    record = dict(scope='synthetic-full-population-estimator-and-waiting-controls-not-experiments',
        source=entry.e.identity(HERE / 'summarize.py'), collector_source=entry.e.identity(HERE / 'collect.py'),
        test_source=entry.e.identity(Path(__file__)), tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        actual_model_or_outcome_collector_executed=False, actual_scientific_results=False)
    entry.e.write(HERE / 'tests.json', record)
    print(json.dumps(record))
    raise SystemExit(0 if result.wasSuccessful() else 1)
