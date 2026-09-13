"""Recovery-boundary tests. All processes, GPU leases and model execution are explicit fixtures."""
from __future__ import annotations
import argparse
import ast
import copy
from datetime import datetime, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import FunctionType, SimpleNamespace
import unittest
from unittest.mock import patch

import recovery as dispatch
import recovery_support as support
import record_layout as layout
import flow_candidate as flow
import observe_recovery as observer

ROOT = Path(__file__).resolve().parent
FIXTURE_SOURCE = support.STORY / 'reports/engineering-archive/dense-v3-functional-flow-candidate-v1/source/test_flow.py'
support.need(support.identity(FIXTURE_SOURCE)['sha256'] ==
             '98dc98b3ed80f602837f06bcb4977950a34651a82111c3ea08d822df8ce15349',
             'Original explicit fixture source differs')
spec = importlib.util.spec_from_file_location('previous_explicit_fixture', FIXTURE_SOURCE)
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)
OBSERVATIONS = []


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def actual_module_fixture(root, failure=None):
    f = fixture_module.Fixture(root, failure)
    runtime = f.ns
    runtime.update(HERE=root / 'launch', __file__=str(root / 'launch/recovery.py'),
                   SETTINGS=dispatch.SETTINGS,
                   recovery=SimpleNamespace(modules=lambda _: (layout, flow)))
    def authenticate(args, **_):
        args.run_root = root / 'run'
        return f.entry, f.context, {'pretrained_origin': f.origin}
    runtime['authenticate'] = authenticate
    # Positive fixture handles are NEVER looked up; every process dependency is mocked.
    runtime['process_identity'] = lambda pid, command=None: {
        'pid': 1000000 + abs(pid), 'ppid': 1000100, 'start_ticks': 2000000 + abs(pid),
        'command': command or ['/usr/bin/python', '-B', str(root / 'launch/recovery.py'),
            '--source-sha', f.args.source_sha, '--authorization-sha', f.args.authorization_sha, '--coordinate']}
    for name in flow.FUNCTIONS:
        function = getattr(dispatch, name)
        runtime[name] = FunctionType(function.__code__, runtime, name,
                                      function.__defaults__, function.__closure__)
    f.ns = runtime
    return f


class Tests(unittest.TestCase):
    def root(self):
        return Path(tempfile.mkdtemp(prefix='synthetic-case-', dir=FIXTURES))

    def proposal(self):
        return support.proposal_data(ROOT, dispatch.SETTINGS)

    def approval(self, sha='a' * 64):
        return {'scope': support.SCOPE, 'approved': True, 'source': 'direct_user_message',
                'owner_message': support.OWNER_MESSAGE, 'automatic_continuation': False,
                'proposal_sha256': sha, 'preserve_original_attempt': True,
                'protected_helper_access': False, 'synthetic_fixture_only': True}

    def test_unchanged_operational_bodies(self):
        def nodes(path):
            return {n.name: ast.dump(n) for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef)}
        old, new = nodes(support.ORIGINAL / 'dispatch.py'), nodes(ROOT / 'recovery.py')
        names = ('require', 'validation_priority', 'priority', 'job_for', 'environment',
                 'worker_command', 'process_identity', 'worker', 'encode_one',
                 'finalize_vectors', 'feature_worker')
        self.assertEqual({n: old[n] for n in names}, {n: new[n] for n in names})
        OBSERVATIONS.append({'unchanged_production_function_bodies': list(names)})

    def test_source_order_original_proof_and_new_namespace_proposal(self):
        p = self.proposal()
        self.assertEqual(p['state_order'], list(layout.CELLS))
        self.assertEqual(p['remaining_new_encodings'], 60)
        self.assertFalse(p['execution_authorized'])
        self.assertNotEqual(p['output_root'], str(support.OLD_OUTPUT))
        self.assertEqual(p['source_files'], support.source_files(ROOT))
        self.assertEqual(len(p['pretrained_origin']['files']), 6)
        OBSERVATIONS.append({'actual_metadata_only_proposal_checked': True,
                             'original_sources': len(p['sources']), 'states': 61})

    def test_original_or_temporary_path_refused(self):
        for path in (support.ORIGINAL, ROOT, ROOT / 'run'):
            with self.subTest(path=str(path)), self.assertRaisesRegex(ValueError, 'namespace'):
                support.namespace(path)

    def test_output_symlink_refused(self):
        root = self.root(); launch = root / 'launch'; launch.mkdir()
        output = root / 'output'; output.symlink_to(root / 'elsewhere')
        with patch.object(support, 'DEPLOY', launch), patch.object(support, 'OUTPUT', output):
            with self.assertRaisesRegex(ValueError, 'ancestry'):
                support.namespace(launch)

    def test_approval_missing_or_automatic_refused(self):
        for field, value in (('approved', False), ('source', 'automatic_goal'),
                             ('automatic_continuation', True), ('owner_message', ''),
                             ('preserve_original_attempt', False), ('protected_helper_access', True)):
            a = self.approval(); a[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                support.check_approval(a, 'a' * 64)
        with self.assertRaises(ValueError):
            support.check_approval({}, 'a' * 64)

    def test_approval_wrong_proposal_refused(self):
        with self.assertRaises(ValueError):
            support.check_approval(self.approval(), 'b' * 64)
        support.check_approval(self.approval(), 'a' * 64)

    def test_duplicate_nonfinite_and_wrong_external_json_refused(self):
        root = self.root()
        for name, payload in (('duplicate', '{"x":1,"x":2}'), ('nan', '{"x":NaN}')):
            path = root / name
            with path.open('x') as stream: stream.write(payload)
            with self.assertRaises(ValueError): support.read(path)
        path = root / 'ordinary'; write(path, {'x': 1})
        with self.assertRaises(ValueError): support.read(path, 'a' * 64)

    def test_preparation_needs_explicit_approval_arguments(self):
        with self.assertRaises(ValueError):
            dispatch.parse_args(['--prepare', '--source-sha', 'a' * 64, '--tests-sha', 'b' * 64])
        args = dispatch.parse_args(['--prepare', '--source-sha', 'a' * 64, '--tests-sha', 'b' * 64,
                                    '--proposal-sha', 'c' * 64, '--approval-sha', 'd' * 64])
        self.assertTrue(args.prepare)

    def test_worker_cannot_carry_prepare_flags_or_duplicate_leases(self):
        base = ['--worker', layout.CELLS[1], '--source-sha', 'a' * 64,
                '--authorization-sha', 'b' * 64, '--plan-sha', 'c' * 64, '--gpu-token', '0']
        for tail in (['--lease-fd', '3', '--lease-fd', '3'],
                     ['--lease-fd', '3', '--lease-fd', '4', '--approval-sha', 'd' * 64]):
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                dispatch.parse_args(base + tail)

    def test_facade_does_not_mutate_native_entry_or_functions(self):
        original = SimpleNamespace(OUTPUT=support.OLD_OUTPUT)
        original.authenticate = lambda *_: original.OUTPUT
        facade = support.output_facade(original)
        self.assertEqual(facade.OUTPUT, support.OUTPUT)
        self.assertEqual(original.OUTPUT, support.OLD_OUTPUT)
        self.assertIs(facade.authenticate, original.authenticate)
        self.assertEqual(facade.authenticate(), support.OLD_OUTPUT)

    def test_failed_or_source_changed_tests_refused(self):
        proposal = {'source_files': {'synthetic': 'source'}}
        good = {'source_files': proposal['source_files'], 'tests_run': 12, 'errors': 0,
                'failures': 0, 'skipped': 0, 'production_module_composition_passed': True,
                'original_operational_bodies_unchanged': True,
                'synthetic_process_model_lease_fixtures': True}
        support.check_tests(good, proposal)
        for field, bad in (('source_files', {}), ('failures', 1), ('skipped', 1),
                           ('production_module_composition_passed', False)):
            altered = {**good, field: bad}
            with self.subTest(field=field), self.assertRaises(ValueError):
                support.check_tests(altered, proposal)

    def test_preparation_rejects_existing_output_before_native_admission(self):
        root = self.root(); launch = root / 'launch'; launch.mkdir()
        output = root / 'output'; output.mkdir()
        entry = SimpleNamespace(authenticate=lambda *_: self.fail('Unexpected model admission'))
        with patch.object(support, 'DEPLOY', launch), patch.object(support, 'OUTPUT', output), \
                patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            with self.assertRaisesRegex(ValueError, 'fresh'):
                support.prepare(SimpleNamespace(), entry, None, launch, {}, 'a', 'b')

    def test_complete_actual_production_module_with_explicit_stubs(self):
        f = actual_module_fixture(self.root())
        with patch('sys.stdout', new=io.StringIO()): result = f.run()
        states = f.observe()
        self.assertEqual(result['feature_states'], 61)
        self.assertEqual(f.encoded, list(layout.CELLS[1:]))
        self.assertEqual((len(f.children), len(f.feature_calls)), (61, 122))
        self.assertEqual((f.events.count('lease-enter'), f.events.count('lease-close')), (60, 60))
        self.assertEqual(len(states['features_verified']), 61)
        self.assertFalse((f.root / 'run/jobs/pretrained.started.json').exists())
        self.assertTrue(states['reused_pretrained'])
        # Run the new production observer's terminal composition with no process reads.
        proposal = {'state_order': list(layout.CELLS),
                    'plans': {j['plan']['state']['cell']: f.context.digest(j['plan']) for j in f.context.jobs},
                    'pretrained_origin': f.origin}
        args = SimpleNamespace(source_sha=f.args.source_sha, authorization_sha=f.args.authorization_sha,
            started_sha=support.identity(f.root / 'run/coordinator.started.json')['sha256'])
        with patch.object(dispatch, 'HERE', f.root / 'launch'), patch.object(dispatch, 'RUN', f.root / 'run'), \
                patch.object(support, 'OUTPUT', f.entry.OUTPUT), \
                patch.object(support, 'authority', return_value=({}, proposal)), \
                patch.object(support, 'modules', return_value=(layout, flow)):
            observed = observer.observe(args, exact=lambda _: self.fail('Unexpected actual process read'))
            self.assertEqual(observed['coordinator'], {'terminal_record': 'completed'})
            self.assertEqual(observed['feature_worker'], {'terminal': True, 'exit_code': 0})
        OBSERVATIONS.append({'production_fixture_root': str(f.root), 'new_mock_workers': 60,
                             'reused_synthetic_pretrained': 1, 'feature_stub_calls': 122,
                             'production_observer_terminal_chain': True,
                             'real_model_or_process_or_lease': False})

    def test_worker_failure_preserves_partial_and_does_not_retry(self):
        f = actual_module_fixture(self.root(), 'worker-exit')
        with self.assertRaisesRegex(ValueError, 'failed'): f.run()
        self.assertEqual(len(f.children), 1)
        self.assertTrue((f.root / 'run/failed.json').exists())
        self.assertTrue((f.root / 'run/pretrained.reused.json').exists())
        self.assertFalse((f.root / 'run/completed.json').exists())
        self.assertEqual(f.events.count('lease-close'), 1)

    def test_native_readback_failure_precedes_any_new_model(self):
        f = actual_module_fixture(self.root(), 'copied-native-readback')
        with self.assertRaisesRegex(ValueError, 'readback'): f.run()
        self.assertEqual(f.children, [])
        self.assertTrue((f.root / 'run/failed.json').exists())

    def test_feature_failure_is_not_success_or_retry(self):
        f = actual_module_fixture(self.root(), 'feature-exit')
        with self.assertRaisesRegex(ValueError, 'failed'): f.run()
        self.assertEqual(len(f.children), 61)
        self.assertTrue((f.root / 'run/vectors.completed.json').exists())
        self.assertFalse((f.root / 'run/completed.json').exists())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    args = parser.parse_args()
    FIXTURES = args.fixtures; FIXTURES.mkdir(parents=True, exist_ok=False)
    before = support.source_files(ROOT)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    support.need(support.source_files(ROOT) == before, 'Source changed during tests')
    write(args.output, {'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'tests_run': result.testsRun, 'errors': len(result.errors), 'failures': len(result.failures),
        'skipped': len(result.skipped), 'source_files': before,
        'production_module_composition_passed': result.wasSuccessful(),
        'original_operational_bodies_unchanged': result.wasSuccessful(),
        'synthetic_process_model_lease_fixtures': True,
        'actual_gpu_or_model_execution': False, 'observations': OBSERVATIONS})
    raise SystemExit(0 if result.wasSuccessful() else 1)
