"""Bounded operational tests using actual admitted inputs, no primary GPU work."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('functional_dispatch_under_test', HERE / 'dispatch.py')
dispatch = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dispatch
spec.loader.exec_module(dispatch)


class OperationalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entry = dispatch.load_entry()
        cls.context, cls.inputs = cls.entry.authenticate(dispatch.FUNCTIONAL_SHA, dispatch.INPUTS_SHA)
        cls.runs = {run_id: {'root': cls.entry.EXPERIMENT / cls.context.primary.payload['output_root'] / 'dense' / run_id,
                            'checked': checked}
                    for run_id, checked in cls.context.admitted['complete_runs'].items()}

    def plans(self, runs):
        return self.entry.build_plans(self.context.primary, self.context.contract, runs,
                                      self.context.reference, self.context.probe)

    def changed_runs(self):
        return copy.deepcopy(self.runs)

    def test_01_actual_complete_population(self):
        jobs, admitted = self.plans(self.runs)
        self.assertEqual(admitted, self.context.admitted)
        self.assertEqual(len(jobs), 61)
        self.assertEqual(len(admitted['complete_runs']), 12)
        self.assertEqual(jobs[0]['checkpoint'], self.entry.REFERENCE)
        self.assertEqual(sum(len(v['checkpoints']) for v in admitted['complete_runs'].values()), 60)
        self.assertEqual(len(self.context.identities), 224)
        self.assertEqual(len(set(v['source'] for v in self.context.identities)), 14)

    def test_02_every_actual_checkpoint_mapping(self):
        jobs, _ = self.plans(self.runs)
        for job in jobs[1:]:
            state = job['plan']['state']
            checked = self.runs[state['meta']['run_id']]['checked']
            self.assertEqual(job['plan']['model']['checkpoint'], checked['checkpoints'][state['stage'] - 1])
            self.assertEqual(job['checkpoint'].name, f"checkpoint-{state['meta']['step']}")
            self.assertEqual(job['plan']['model']['complete_run_sha256'], self.context.digest(checked))

    def test_03_missing_run_refused(self):
        runs = self.changed_runs(); runs.pop(next(iter(runs)))
        with self.assertRaisesRegex(ValueError, 'complete twelve-run'):
            self.plans(runs)

    def test_04_extra_historical_run_refused(self):
        runs = self.changed_runs(); runs['historical'] = next(iter(runs.values()))
        with self.assertRaisesRegex(ValueError, 'complete twelve-run'):
            self.plans(runs)

    def test_05_changed_run_identity_refused(self):
        runs = self.changed_runs(); next(iter(runs.values()))['checked']['run_identity_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'mismatched actual'):
            self.plans(runs)

    def test_06_incomplete_run_proof_refused(self):
        runs = self.changed_runs(); next(iter(runs.values()))['checked']['whole_run_artifacts_verified'] = False
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.plans(runs)

    def test_07_missing_checkpoint_refused(self):
        runs = self.changed_runs(); next(iter(runs.values()))['checked']['checkpoints'].pop()
        with self.assertRaises(ValueError):
            self.plans(runs)

    def test_08_wrong_checkpoint_stage_refused(self):
        runs = self.changed_runs(); next(iter(runs.values()))['checked']['checkpoints'][0]['step'] = 3907
        with self.assertRaises(ValueError):
            self.plans(runs)

    def test_09_changed_model_root_refused(self):
        runs = self.changed_runs(); next(iter(runs.values()))['root'] = Path('/tmp/not-a-primary-model')
        with self.assertRaisesRegex(ValueError, 'Changed model root'):
            self.plans(runs)

    def test_10_old_guards_remain_unpassed(self):
        actual = self.context.actual_completion
        self.assertIs(actual['original_single_selection_guard_passed'], False)
        self.assertIs(actual['committed_source_release'], False)
        self.assertIs(self.context.contract.payload['formal_execution_authorized'], False)
        with self.assertRaises(ValueError):
            self.context.contract.require_execution()

    def test_11_native_encoding_and_kernel_routing(self):
        self.assertIs(self.context.exports.encode_state, self.context.vectors.encode_state)
        self.assertIs(self.context.features.compute_state, self.context.kernels.compute_state)
        self.assertIs(self.context.features.save_state, self.context.feature_io.save_state)
        self.assertIs(self.context.features.inspect_state, self.context.feature_io.inspect_state)
        for job in self.context.jobs:
            encoding = job['plan']['encoding']
            self.assertEqual((encoding['batch_size'], encoding['max_length'], encoding['device']), (8, 8192, 'cuda:0'))
            self.assertEqual((encoding['model_dtype'], encoding['storage_dtype']), ('bfloat16', 'float32'))
            self.assertIs(encoding['flash_attention'], True)

    def test_12_unknown_state_refused(self):
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            dispatch.job_for(self.context, 'historical', '0' * 64)

    def test_13_changed_plan_digest_refused(self):
        with self.assertRaisesRegex(ValueError, 'changed functional'):
            dispatch.job_for(self.context, 'pretrained', '0' * 64)

    def test_14_worker_command_retains_two_fds(self):
        args = SimpleNamespace(source_sha='a' * 64, authorization_sha='b' * 64)
        command = dispatch.worker_command(args, self.context.jobs[0], self.context, '7', (31, 32))
        self.assertEqual(command[:2], ['/usr/bin/python', '-B'])
        self.assertEqual(command[-4:], ['--lease-fd', '31', '--lease-fd', '32'])
        self.assertEqual(command[command.index('--gpu-token') + 1], '7')
        self.assertEqual(command[command.index('--plan-sha') + 1], self.context.digest(self.context.jobs[0]['plan']))

    def test_15_duplicate_fds_refused(self):
        args = SimpleNamespace(source_sha='a' * 64, authorization_sha='b' * 64)
        with self.assertRaisesRegex(ValueError, 'distinct inherited'):
            dispatch.worker_command(args, self.context.jobs[0], self.context, '0', (31, 31))

    def test_16_undeclared_gpu_refused(self):
        args = SimpleNamespace(source_sha='a' * 64, authorization_sha='b' * 64)
        with self.assertRaises(ValueError):
            dispatch.worker_command(args, self.context.jobs[0], self.context, '8', (31, 32))

    def test_17_cpu_mode_gpu_arguments_refused(self):
        with self.assertRaisesRegex(ValueError, 'CPU modes'):
            dispatch.parse_args(['--source-sha', 'a' * 64, '--authorization-sha', 'b' * 64,
                                 '--coordinate', '--gpu-token', '0'])

    def test_18_single_fd_worker_refused(self):
        with self.assertRaisesRegex(ValueError, 'both distinct descriptors'):
            dispatch.parse_args(['--source-sha', 'a' * 64, '--authorization-sha', 'b' * 64,
                '--worker', 'pretrained', '--plan-sha', 'c' * 64, '--gpu-token', '0', '--lease-fd', '31'])

    def priority_fixture(self, root, count=12):
        root = Path(root); (root / 'jobs').mkdir()
        names = sorted(self.runs)
        for run_id in names[:count]:
            self.context.geometry.write_new(root / 'jobs' / f'{run_id}.scored.json', {'fixture': True})
        self.context.geometry.write_new(root / 'completed.json', {
            'all_twelve_validations_verified': True, 'scientific_completion': False,
            'selection': {'deliberately_unused': 'no scores may choose functional states'}})

    def test_19_validation_waits_without_completion(self):
        with tempfile.TemporaryDirectory(prefix='functional-priority-test.') as directory:
            self.assertIsNone(dispatch.validation_priority(directory, self.runs, self.context.geometry.identity))

    def test_20_validation_requires_all_twelve(self):
        with tempfile.TemporaryDirectory(prefix='functional-priority-test.') as directory:
            self.priority_fixture(directory, 11)
            with self.assertRaisesRegex(ValueError, 'every declared'):
                dispatch.validation_priority(directory, self.runs, self.context.geometry.identity)

    def test_21_priority_does_not_select_using_scores(self):
        with tempfile.TemporaryDirectory(prefix='functional-priority-test.') as directory:
            self.priority_fixture(directory)
            result = dispatch.validation_priority(directory, self.runs, self.context.geometry.identity)
            self.assertEqual(result['scored_run_ids'], sorted(self.runs))
            self.assertIs(result['scores_used_for_state_selection'], False)
            self.assertNotIn('selection', result)

    def test_22_original_validation_failure_refused(self):
        with tempfile.TemporaryDirectory(prefix='functional-priority-test.') as directory:
            self.priority_fixture(directory)
            self.context.geometry.write_new(Path(directory) / 'failed.json', {'fixture': True})
            with self.assertRaisesRegex(ValueError, 'reported failure'):
                dispatch.validation_priority(directory, self.runs, self.context.geometry.identity)

    def test_23_actual_inherited_descriptor_lifetime(self):
        # Only fresh diagnostic files and this explicit child; no project GPU lease is inspected.
        with tempfile.TemporaryDirectory(prefix='functional-fd-test.') as directory:
            paths = [Path(directory) / name for name in ('first.lock', 'second.lock')]
            descriptors = [os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600) for path in paths]
            child = None
            try:
                for fd in descriptors:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.context.validation.require_inherited_leases(descriptors, paths)
                code = (
                    'import importlib.util,json,sys; from pathlib import Path; '
                    's=importlib.util.spec_from_file_location("original_lease_checker",sys.argv[1]); '
                    'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
                    'sys.stdin.readline(); '
                    'm.require_inherited_leases(json.loads(sys.argv[2]),[Path(p) for p in json.loads(sys.argv[3])]); '
                    'print("both_original_fds_verified_after_parent_close",flush=True); sys.stdin.readline()')
                child = subprocess.Popen(['/usr/bin/python', '-B', '-c', code, str(self.entry.VALIDATION),
                    json.dumps(descriptors), json.dumps([str(path) for path in paths])],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, pass_fds=tuple(descriptors), env=dispatch.environment(self.context, self.entry))
                for fd in descriptors:
                    os.close(fd)
                descriptors = []
                child.stdin.write('parent_closed\n'); child.stdin.flush()
                self.assertEqual(child.stdout.readline().strip(), 'both_original_fds_verified_after_parent_close')
                self.assertIsNone(child.poll())
                for path in paths:
                    fd = os.open(path, os.O_RDWR)
                    try:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    finally:
                        os.close(fd)
                stdout, stderr = child.communicate('finish\n', timeout=15)
                self.assertEqual(child.returncode, 0, stderr)
            finally:
                if child is not None and child.poll() is None:
                    child.communicate('finish\n', timeout=15)
                for fd in descriptors:
                    os.close(fd)

    def test_24_aliased_lease_inodes_refused(self):
        with tempfile.TemporaryDirectory(prefix='functional-fd-test.') as directory:
            path = Path(directory) / 'one.lock'
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            duplicate = os.dup(fd)
            try:
                with self.assertRaisesRegex(ValueError, 'distinct files'):
                    self.context.validation.require_inherited_leases([fd, duplicate], [path, path])
            finally:
                os.close(fd); os.close(duplicate)

    def test_25_source_guard_refuses_changed_dispatcher(self):
        with self.assertRaisesRegex(ValueError, 'dispatcher source changed'):
            dispatch.source_guard(self.entry, '0' * 64)


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OperationalTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    geometry = dispatch.load_entry().geometry_parent()
    record = {
        'scope': 'bounded_functional_operational_admission_and_lease_tests',
        'observed_at_utc': geometry.now(), 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
        'dispatcher_source_sha256': geometry.identity(HERE / 'dispatch.py')['sha256'],
        'functional_source_sha256': dispatch.FUNCTIONAL_SHA, 'inputs_sha256': dispatch.INPUTS_SHA,
        'test_source_sha256': geometry.identity(__file__)['sha256'],
        'actual_input_admission_passed': hasattr(OperationalTests, 'context'),
        'actual_primary_gpu_encoding_tested': False,
        'native_numerical_functions_changed': False, 'scientific_completion': False,
        'boundary': 'Actual 61-state primary completion/probe admission and bounded new dispatch controls. Fresh temporary-file child verifies both inherited FD lifetimes. No original GPU lease, encoder, primary scoring, feature computation or release is exercised by these tests.'}
    geometry.write_new(HERE / 'tests-first.json', record)
    print(json.dumps(record, sort_keys=True), flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
