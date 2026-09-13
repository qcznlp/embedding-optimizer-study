"""Bounded outer-execution tests. All positive process/GPU/native calls are mocked."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('factorial_training_candidate', Path(__file__).with_name('factorial_dispatch.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Child:
    def __init__(self, pid, code=0):
        self.pid, self.returncode = pid, code
        self.terminated = self.killed = False
    def poll(self): return self.returncode
    def wait(self, timeout=None): return self.returncode
    def terminate(self): self.terminated = True; self.returncode = -15
    def kill(self): self.killed = True; self.returncode = -9


class Tests(unittest.TestCase):
    def test_real_unchanged_source_closure(self):
        self.assertEqual(len(m.source_files()), 70)

    def test_exact_design_balanced_disjoint_pools(self):
        queues = m.queues()
        self.assertEqual([len(x) for x in queues.values()], [6, 6])
        self.assertEqual(len(set(sum(queues.values(), []))), 12)
        self.assertFalse(set(m.POOLS['a']) & set(m.POOLS['b']))
        for queue in queues.values():
            self.assertEqual(sum('-adamw-seed' in x for x in queue), 3)
            self.assertEqual(sum('-muon-seed' in x for x in queue), 3)
            self.assertEqual(sum('adamw_state' in x for x in queue), 3)
        self.assertEqual(m.cells()[0], ('adamw_state', 'adamw', 314159))

    def test_namespace_does_not_admit_temporary_source(self):
        with self.assertRaises(ValueError): m.namespace()

    def test_duplicate_json_and_nonfinite_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'bad.json'
            p.write_text('{"a":1,"a":2}')
            with self.assertRaises(ValueError): m.read(p)
            p.write_text('{"a":NaN}')
            with self.assertRaises(ValueError): m.read(p)

    def test_authority_cannot_be_automatic_continuation(self):
        with self.assertRaises(ValueError):
            m.owner({'automatic_continuation': True}, 'a' * 64)

    def test_create_new_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'record.json'
            m.write_new(p, {'status': 'mock'})
            with self.assertRaises(FileExistsError): m.write_new(p, {'status': 'changed'})
            self.assertEqual(m.read(p), {'status': 'mock'})

    def test_four_rank_argv_carries_all_original_descriptors(self):
        args = NS(pool='a', source_sha='s', authorization_sha='a')
        for rank in range(4):
            argv = m.command(args, m.queues()['a'][0], rank=rank, descriptors=list(range(3, 11)))
            parsed = m.parse(argv[3:])
            self.assertEqual(parsed.rank, rank)
            self.assertEqual(parsed.lease_fd, list(range(3, 11)))

    def test_cpu_read_argv_has_no_rank_or_lease(self):
        args = NS(pool='a', source_sha='s', authorization_sha='a')
        argv = m.command(args, m.queues()['a'][0], worker_sha='w')
        parsed = m.parse(argv[3:])
        self.assertEqual(parsed.worker_complete_sha, 'w')
        self.assertEqual(parsed.lease_fd, [])
        self.assertIsNone(parsed.rank)

    def test_bad_worker_topology_refused(self):
        args = NS(pool='a', source_sha='s', authorization_sha='a')
        for rank, fds in [(4, list(range(8))), (True, list(range(8))), (0, [3] * 8), (0, list(range(7))), (0, [-1] + list(range(7)))]:
            with self.assertRaises(ValueError): m.command(args, m.queues()['a'][0], rank=rank, descriptors=fds)
        with self.assertRaises(ValueError):
            m.command(args, m.queues()['b'][0], rank=0, descriptors=list(range(8)))

    def test_parse_cannot_turn_cpu_mode_into_gpu(self):
        base = ['--source-sha', 's', '--authorization-sha', 'a', '--pool', 'a', '--coordinate']
        for extra in (['--rank', '0'], ['--lease-fd', '3'], ['--worker-complete-sha', 'x'], ['--tests-sha', 'x']):
            with self.assertRaises(ValueError): m.parse(base + extra)

    def test_gpu_environment_has_exact_reset_logging_and_all_four_devices(self):
        parent = NS(environment=lambda tokens: {'CUDA_VISIBLE_DEVICES': ','.join(tokens or []),
            'RANK': '9', 'TORCHELASTIC_RUN_ID': 'old', 'WANDB_RUN_ID': 'old'})
        env = m.environment(parent, pool='a', rank=2, run_id=m.queues()['a'][0], port=19887)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '4,5,6,7')
        self.assertEqual(env['RANK'], env['LOCAL_RANK'])
        self.assertEqual(env['RANK'], '2')
        self.assertEqual(env['WORLD_SIZE'], '4')
        self.assertEqual(env['WANDB_RUN_ID'], m.queues()['a'][0])
        self.assertEqual(env['WANDB_RESUME'], 'never')
        self.assertNotIn('TORCHELASTIC_RUN_ID', env)

    def test_cpu_environment_clears_distributed_context(self):
        parent = NS(environment=lambda tokens: {'CUDA_VISIBLE_DEVICES': '', 'RANK': '1', 'MASTER_PORT': '111'})
        env = m.environment(parent)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        self.assertNotIn('RANK', env)
        self.assertNotIn('MASTER_PORT', env)

    def test_lease_pairing_binds_every_token_in_both_namespaces(self):
        check = Mock()
        parent = NS(LEASE_ROOTS=[Path('/mock/one'), Path('/mock/two')])
        with patch.object(m.c, 'imported', return_value=NS(require_inherited_leases=check)):
            m.verify_leases(parent, 'a', list(range(10, 18)))
        self.assertEqual(check.call_count, 4)
        for i, call in enumerate(check.call_args_list):
            self.assertEqual(call.args[0], [10 + i, 14 + i])
            self.assertEqual(call.args[1], [p / f'gpu-{4+i}.lock' for p in parent.LEASE_ROOTS])
        with self.assertRaises(ValueError): m.verify_leases(parent, 'a', [3] * 8)

    def test_supervision_waits_for_all_four_real_handle_results(self):
        children = [Child(100 + i) for i in range(4)]
        self.assertEqual(m.supervise(children), [0] * 4)
        self.assertFalse(any(c.terminated for c in children))

    def test_failed_rank_stops_only_direct_owned_siblings(self):
        children = [Child(100, 1), Child(101, None), Child(102, 0), Child(103, None)]
        with self.assertRaises(RuntimeError): m.supervise(children)
        self.assertEqual([c.terminated for c in children], [False, True, False, True])
        self.assertEqual([c.returncode for c in children], [1, -15, 0, -15])

    def test_native_call_is_same_full_worker_without_numerical_override(self):
        native = NS(run_branch=Mock(return_value={'mock': True}))
        declared = dict(calibrations='C', state='S', operator='O', seed=314159,
            source_root='SR', output_root='OR', record_root='RR', project='P', entity='E', worker_source='W')
        self.assertEqual(m.call_native(native, 'L', declared), {'mock': True})
        native.run_branch.assert_called_once_with('L', 'C', 'S', 'O', 314159, 'SR', 'OR', 'RR',
            project='P', entity='E', expected_worker_source='W')

    def exercise_launch(self, failed=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = NS(pool='a', source_sha='s', authorization_sha='a')
            run_id = m.queues()['a'][0]
            record = root / 'records'
            declared = {'run_root': str(root / 'output'), 'record_root': str(record)}
            auth = {'requests': {run_id: declared}}
            held = []
            @contextmanager
            def leases(tokens):
                self.assertEqual(tokens, m.POOLS['a'])
                held.append(True)
                try: yield tuple(range(3, 11))
                finally: held.pop()
            parent = NS(leases=leases, handoff=lambda: [{'mock': True}],
                environment=lambda tokens: {'CUDA_VISIBLE_DEVICES': ','.join(tokens or [])})
            children = []
            binding = {'bytes': 17, 'sha256': 'b' * 64}
            def popen(argv, **kwargs):
                cpu = '--verify' in argv
                self.assertEqual(bool(held), not cpu)
                child = Child(1100 + len(children), 1 if failed and len(children) == 0 else 0)
                children.append(child)
                job = root / run_id
                if not cpu:
                    self.assertEqual(kwargs['pass_fds'], tuple(range(3, 11)))
                    rank = int(argv[argv.index('--rank') + 1])
                    m.write_new(job / f'rank-{rank}.returned.json', {
                        'rank': rank, 'run_id': run_id, 'source_sha256': 's',
                        'authorization_sha256': 'a', 'worker_completion': binding})
                else:
                    self.assertNotIn('pass_fds', kwargs)
                    self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                    m.write_new(job / 'fresh-native-readback.json', {
                        'worker_completion': binding, 'fresh_process_native_readback': True,
                        'fixture_only': True})
                return child
            dispatch = NS(process_identity=lambda pid: {'pid': pid, 'ppid': 1, 'start_ticks': pid * 10})
            with patch.object(m.c, 'imported', return_value=dispatch), patch.object(m.subprocess, 'Popen', side_effect=popen):
                if failed:
                    with self.assertRaises(RuntimeError): m.launch_one(args, auth, parent, run_id, root)
                    self.assertEqual(len(children), 4)
                    self.assertTrue((root / run_id / 'failed.json').is_file())
                    self.assertFalse((root / run_id / 'completed.json').exists())
                else:
                    complete = m.launch_one(args, auth, parent, run_id, root)
                    self.assertEqual(complete['actual_rank_exits'], [0] * 4)
                    self.assertEqual(len(children), 5)
                    self.assertTrue((root / run_id / 'reader.exited.json').is_file())
                    with self.assertRaises(ValueError): m.launch_one(args, auth, parent, run_id, root)
                self.assertFalse(held)

    def test_complete_production_launch_sequence_with_explicit_mock_workers(self):
        self.exercise_launch()

    def test_failed_production_launch_preserves_attempt_and_no_retry(self):
        self.exercise_launch(failed=True)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    summary = {'source_sha256': m.identity(Path(m.__file__))['sha256'],
        'test_source': m.identity(Path(__file__)), 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
        'positive_gpu_model_native_process_calls': 'explicit mocks, not genuine experiments',
        'finished_at_utc': m.now()}
    m.write_new(Path(__file__).with_name('tests.json'), summary)
    raise SystemExit(not result.wasSuccessful())
