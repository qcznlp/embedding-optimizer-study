"""Bounded operational controls only; no actual GPU or scientific result fixtures."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('new_factorial_evaluator_under_test', HERE / 'evaluate.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
TASKS = ['SciFact'] + [f'synthetic_task_{i}' for i in range(13)]
SIZES = {k: i + 1 for i, k in enumerate(TASKS)}


def complete():
    return dict(pool='a', source_sha256=entry.TRAIN_SHA, authorization_sha256=entry.AUTH_SHA, full_branches=6,
        runs=[dict(run_id=r, source_sha256=entry.TRAIN_SHA, authorization_sha256=entry.AUTH_SHA,
            actual_rank_exits=[0, 0, 0, 0], actual_fresh_reader_exit=0,
            all_five_checkpoints_verified=True, full_horizon_391_steps_verified=True)
            for r in entry.queues()['a']])


class OperationalControls(unittest.TestCase):
    def test_complete_pool_shape(self):
        entry.require_pool_completion(complete(), 'a')

    def test_exact_twelve_cells(self):
        actual = [r for values in entry.queues().values() for r in values]
        expected = {f'factorial-v3-{s}-{o}-seed{n}' for s in ('adamw_state', 'muon_state')
                    for o in ('adamw', 'muon') for n in (314159, 271828, 161803)}
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), 12)

    def test_all_final_jobs_and_real_pilot_position(self):
        result = entry.jobs(entry.queues()['a'], TASKS, SIZES)
        self.assertEqual(len(result), 84)
        self.assertEqual(result[0], (entry.queues()['a'][0], 391, 'SciFact'))
        self.assertEqual({j[1] for j in result}, {391})
        self.assertEqual(set(result), {(r, 391, t) for r in entry.queues()['a'] for t in TASKS})
        self.assertEqual(result[1][2], TASKS[-1])

    def test_duplicate_run_refused(self):
        runs = entry.queues()['a']
        runs[1] = runs[0]
        with self.assertRaises(ValueError):
            entry.jobs(runs, TASKS, SIZES)

    def test_missing_task_refused(self):
        with self.assertRaises(ValueError):
            entry.jobs(entry.queues()['a'], TASKS[:-1], SIZES)

    def test_missing_pilot_refused(self):
        tasks = ['not_scifact'] + TASKS[1:]
        with self.assertRaises(ValueError):
            entry.jobs(entry.queues()['a'], tasks, dict(zip(tasks, range(14))))

    def test_mismatched_external_digest_refused(self):
        with tempfile.TemporaryDirectory(prefix='factorial-eval-identity-control.') as tmp:
            path = Path(tmp) / 'value.json'
            path.write_text('{}')
            with self.assertRaises(ValueError):
                entry.bound(path, '0' * 64)

    def test_duplicate_json_refused(self):
        with tempfile.TemporaryDirectory(prefix='factorial-eval-json-control.') as tmp:
            path = Path(tmp) / 'value.json'
            path.write_text('{"a":1,"a":2}')
            with self.assertRaises(ValueError):
                entry.read(path)

    def test_waiting_cannot_take_gpu_lease(self):
        class StopWaiting(Exception):
            pass
        with tempfile.TemporaryDirectory(prefix='factorial-eval-wait-control.') as tmp:
            root = Path(tmp)
            parent = SimpleNamespace(handoff=Mock(), leases=Mock(side_effect=AssertionError('Early GPU request')))
            old = SimpleNamespace(process_start_ticks=lambda _: 0)
            with patch.object(entry, 'HERE', root / 'eval'), patch.object(entry, 'TRAIN', root / 'train'), \
                 patch.object(entry, 'authorized', return_value=({}, parent, old)), \
                 patch.object(entry.time, 'sleep', side_effect=StopWaiting):
                with self.assertRaises(StopWaiting):
                    entry.coordinate(SimpleNamespace(pool='a', source_sha256='synthetic', authorization_sha256='synthetic'))
            parent.leases.assert_not_called()


MUTATIONS = {
    'wrong_pool': lambda p: p.update(pool='b'),
    'wrong_training_source': lambda p: p.update(source_sha256='0' * 64),
    'wrong_authorization': lambda p: p.update(authorization_sha256='0' * 64),
    'short_pool': lambda p: p.update(full_branches=5),
    'missing_run': lambda p: p['runs'].pop(),
    'reordered_run': lambda p: p['runs'].reverse(),
    'rank_failure': lambda p: p['runs'][0].update(actual_rank_exits=[0, 0, 1, 0]),
    'unobserved_rank': lambda p: p['runs'][0].update(actual_rank_exits=[0, 0, None, 0]),
    'reader_failure': lambda p: p['runs'][0].update(actual_fresh_reader_exit=1),
    'missing_stage': lambda p: p['runs'][0].update(all_five_checkpoints_verified=False),
    'short_horizon': lambda p: p['runs'][0].update(full_horizon_391_steps_verified=False),
    'wrong_branch_authority': lambda p: p['runs'][0].update(authorization_sha256='0' * 64),
}
for name, mutation in MUTATIONS.items():
    def test(self, mutate=mutation):
        value = copy.deepcopy(complete())
        mutate(value)
        with self.assertRaises(ValueError):
            entry.require_pool_completion(value, 'a')
    setattr(OperationalControls, 'test_refuse_' + name, test)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OperationalControls)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = dict(scope='bounded-factorial-evaluation-operational-controls-not-real-experiments',
        source=entry.identity(HERE / 'evaluate.py'), reader_source=entry.identity(HERE / 'read_native.py'),
        test_source=entry.identity(Path(__file__)), tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        actual_gpu_execution=False, actual_scientific_results=False)
    entry.write(HERE / 'tests.json', record)
    print(json.dumps(record))
    raise SystemExit(0 if result.wasSuccessful() else 1)
