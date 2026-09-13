"""Synthetic transport/population controls; never native experimental admission."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import outcomes as o


def population_fixture():
    tasks = ['synthetic-task-' + str(i) for i in range(14)]
    training = {'queues': {'a': [], 'b': []}, 'requests': {}}
    rows = []
    for state in ('adamw_state', 'muon_state'):
        for op in ('adamw', 'muon'):
            for seed in (314159, 271828, 161803):
                run = state + '-' + op + '-' + str(seed)
                training['queues']['a' if state == 'adamw_state' else 'b'].append(run)
                training['requests'][run] = {'state': state, 'operator': op, 'seed': seed}
                rows.extend({'run_id': run, 'task': task, **training['requests'][run], 'ndcg_at_10': .5} for task in tasks)
    tables = {k: [{} for _ in range(n)] for k, n in o.COUNTS.items()}
    tables['beir_seed_task_scores'] = rows
    return tables, training, tasks


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = o.transport()

    def test_full_population_shape_only(self):
        o.population(*population_fixture())

    def test_missing_table(self):
        tables, training, tasks = population_fixture()
        del tables['probe_checkpoint_metrics']
        with self.assertRaises(ValueError):
            o.population(tables, training, tasks)

    def test_missing_score(self):
        tables, training, tasks = population_fixture()
        tables['beir_seed_task_scores'].pop()
        with self.assertRaises(ValueError):
            o.population(tables, training, tasks)

    def test_duplicate_score(self):
        tables, training, tasks = population_fixture()
        tables['beir_seed_task_scores'][-1] = tables['beir_seed_task_scores'][0].copy()
        with self.assertRaises(ValueError):
            o.population(tables, training, tasks)

    def test_operator_mismatch(self):
        tables, training, tasks = population_fixture()
        tables['beir_seed_task_scores'][0]['operator'] = 'normuon'
        with self.assertRaises(ValueError):
            o.population(tables, training, tasks)

    def test_nonfinite_score(self):
        tables, training, tasks = population_fixture()
        tables['beir_seed_task_scores'][0]['ndcg_at_10'] = float('nan')
        with self.assertRaises(ValueError):
            o.population(tables, training, tasks)

    def test_boolean_score(self):
        tables, training, tasks = population_fixture()
        tables['beir_seed_task_scores'][0]['ndcg_at_10'] = True
        with self.assertRaises(ValueError):
            o.population(tables, training, tasks)

    def test_only_new_remote_subtree(self):
        before = {'README.md': 'old', o.NAMESPACE: 'old-tree'}
        after = {'README.md': 'old', o.NAMESPACE: 'new-tree'}
        sub = {o.NAMESPACE + '/older': 'old-id'}
        o.preserved(before, after, sub, {**sub, o.ADDITION: 'new-id'})

    def test_old_root_change(self):
        with self.assertRaises(ValueError):
            o.preserved({'README.md': 'old'}, {'README.md': 'changed'}, {}, {o.ADDITION: 'new'})

    def test_old_subtree_change(self):
        with self.assertRaises(ValueError):
            o.preserved({o.NAMESPACE: 'old'}, {o.NAMESPACE: 'new'}, {'old': 'a'}, {'old': 'b', o.ADDITION: 'new'})

    def test_existing_target(self):
        with self.assertRaises(ValueError):
            o.preserved({o.NAMESPACE: 'old'}, {o.NAMESPACE: 'new'}, {o.ADDITION: 'old'}, {o.ADDITION: 'new'})

    def test_path_traversal(self):
        with self.assertRaises(ValueError):
            self.t.safe_name('../outside.json')

    def test_source_symlink(self):
        with tempfile.TemporaryDirectory(prefix='dense-outcomes-synthetic-') as temp:
            p = Path(temp)
            (p / 'a.json').write_text('{}')
            (p / 'b.json').symlink_to(p / 'a.json')
            with self.assertRaises(ValueError):
                o.sha(p / 'b.json')

    def test_raw_example_exclusion(self):
        with tempfile.TemporaryDirectory(prefix='dense-outcomes-synthetic-') as temp:
            p = Path(temp) / 'fixture.json'
            p.write_text(json.dumps({'query_text': 'synthetic, not a real example'}))
            with self.assertRaises(ValueError):
                self.t.scan_text(p)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Controls)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    t = o.transport()
    t.write_new(o.HERE / 'tests.json', {'scope': 'synthetic-transport-controls-not-experimental-admission',
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'skipped': len(result.skipped), 'source': t.file_identity(o.HERE / 'outcomes.py'),
        'test_source': t.file_identity(Path(__file__)), 'scientific_completion': False})
    raise SystemExit(0 if result.wasSuccessful() else 1)
