"""New calibration dispatch boundaries; all positive process/model/lease cases are synthetic."""
import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import calibration_dispatch as entry

WORK = Path(__file__).resolve().parent
OBSERVED = []


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2) + '\n')


class Fixture:
    def __init__(self, root, fail=None):
        self.root = root; self.fail = fail; self.children = []; self.events = []
        self.auth = {'source_files': {}, 'inputs_sha256': entry.digest({'synthetic': True}),
                     'requests': {state: {'state': state} for state in entry.STATES}}
        self.inputs = {'synthetic': True}
        self.args = SimpleNamespace(source_sha='a'*64, authorization_sha='b'*64)
        self.output = root / 'output'; self.run = root / 'run'
        self.geometry = SimpleNamespace(identity=entry.identity, now=lambda:datetime.now(timezone.utc).isoformat(), write_new=write)
        @contextmanager
        def leases(tokens):
            self.events.append(('lease-open', tokens[0]))
            try:
                yield (31, 32) if tokens == ['1'] else (41, 42)
            finally:
                self.events.append(('lease-close', tokens[0]))
        self.parent = SimpleNamespace(leases=leases, handoff=lambda:self.events.append(('handoff',)),
            environment=lambda tokens: {'CUDA_VISIBLE_DEVICES': ','.join(tokens or [])})
        self.handles = SimpleNamespace(process_identity=lambda pid, command=None: {
            'pid': pid, 'ppid': 999999, 'start_ticks': pid + 100,
            'command': command or ['SYNTHETIC_COORDINATOR_NO_PROC_READ']})
        def gradients(locations, state, root):
            root.mkdir()
            write(root / 'gradient-receipt.json', {'synthetic': True, 'state': state})
        def replay(locations, state, root, binding):
            write(root / 'directions/calibration.json', {'synthetic': True, 'state': state})
        def read_calibration(inputs, state, root, binding):
            if self.fail == 'readback' and state == 'adamw_state':
                raise ValueError('Synthetic native readback refusal')
            return {'calibration': {'synthetic_fixture_only': True, 'state': state}}
        self.calibration = SimpleNamespace(export_gradients=gradients, replay_calibration=replay,
            read_calibration=read_calibration, calibration_request=lambda inputs,state:{'state':state})
        owner = self
        class Child:
            def __init__(self, argv, **kwargs):
                self.argv = argv; self.pid = 900000 + len(owner.children); self.done = None
                owner.children.append(self)
                if '--worker' in argv:
                    state = argv[argv.index('--worker')+1]
                    expected = (31,32) if state == 'adamw_state' else (41,42)
                    if kwargs['pass_fds'] != expected: raise AssertionError('Missing second lease')
                    if kwargs['env']['CUDA_VISIBLE_DEVICES'] != entry.GPU[state]: raise AssertionError('Wrong GPU')
                else:
                    if not kwargs['close_fds'] or kwargs['env']['CUDA_VISIBLE_DEVICES'] != '':
                        raise AssertionError('CPU verifier inherited GPU access')
            def wait(self):
                if self.done is not None: return self.done
                args = entry.parse(self.argv[3:])
                if owner.fail == 'first-worker' and args.worker == 'adamw_state':
                    self.done = 7; return self.done
                try:
                    (entry.worker if args.worker else entry.verify)(args)
                    self.done = 0
                except Exception:
                    self.done = 8
                return self.done
        self.process = SimpleNamespace(Popen=Child, DEVNULL=None, STDOUT=None)

    def run_all(self):
        with patch.object(entry, 'HERE', self.root), patch.object(entry, 'RUN', self.run), \
             patch.object(entry, 'OUTPUT', self.output), \
             patch.object(entry, 'authenticate', return_value=(self.auth, self.parent)), \
             patch.object(entry, 'context', return_value=(self.calibration, None, self.inputs)), \
             patch.object(entry, 'check_imports'), patch.object(entry, 'sources', return_value=({}, {})), \
             patch.object(entry, 'configured_calibration_loader', side_effect=lambda _:nullcontext()), \
             patch.object(entry, 'imported', side_effect=lambda name,path:self.handles if path == entry.DISPATCH else self.geometry), \
             patch.object(entry, 'subprocess', self.process), patch('sys.stdout',new=io.StringIO()):
            return entry.coordinate(self.args)


class Tests(unittest.TestCase):
    def root(self):
        return Path(tempfile.mkdtemp(prefix='explicit-synthetic-', dir=FIXTURES))

    def test_original_sixty_six_numerical_files_unchanged(self):
        files, inputs = entry.sources()
        self.assertEqual(len(files), 66)
        self.assertEqual(inputs['branch']['rows'], 50000)
        self.assertEqual(inputs['calibration']['rows'], 32)

    def test_primary_configuration_runs_before_native_guard_and_restores_loader(self):
        model=SimpleNamespace(can_flatten_inputs=True)
        original=lambda *args,**kwargs:model
        calibration=SimpleNamespace(gradient_probe=SimpleNamespace(_load_model=original,_hidden_parameter_mapping=lambda *a:None))
        calls=[]
        def configure(loaded, config):
            calls.append(config.dense_can_flatten_inputs)
            loaded.can_flatten_inputs=config.dense_can_flatten_inputs
        module=SimpleNamespace(probe_export=SimpleNamespace(_load_model=original),
                               train=SimpleNamespace(_configure_dense_input_execution=configure))
        with patch.dict(sys.modules,{'embed_optim':module}):
            with entry.configured_calibration_loader(calibration):
                loaded=calibration.gradient_probe._load_model('dense','synthetic-source',device='cuda:0')
                self.assertIs(loaded,model);self.assertFalse(loaded.can_flatten_inputs)
            self.assertIs(calibration.gradient_probe._load_model,original)
        self.assertEqual(calls,[False])

    def test_loader_restore_on_failure_and_unknown_override_refusal(self):
        original=lambda *args,**kwargs:None
        calibration=SimpleNamespace(gradient_probe=SimpleNamespace(_load_model=original,_hidden_parameter_mapping=lambda *a:None))
        module=SimpleNamespace(probe_export=SimpleNamespace(_load_model=original),train=SimpleNamespace())
        with patch.dict(sys.modules,{'embed_optim':module}):
            with self.assertRaisesRegex(ValueError,'synthetic'):
                with entry.configured_calibration_loader(calibration):raise ValueError('synthetic')
            self.assertIs(calibration.gradient_probe._load_model,original)
            calibration.gradient_probe._load_model=lambda:None
            with self.assertRaisesRegex(ValueError,'override'):
                with entry.configured_calibration_loader(calibration):pass

    def test_raw_checkpoint_name_adapter_preserves_all_parameter_objects(self):
        from contextlib import nullcontext
        keys=[f'layers.{i}.weight' for i in range(88)]
        parameters=[SimpleNamespace(shape=(2,3)) for _ in keys]
        mapped=[('0.model.'+name,'0.'+name,p) for name,p in zip(keys,parameters)]
        store=SimpleNamespace(keys=lambda:keys,get_slice=lambda name:SimpleNamespace(get_shape=lambda:(2,3)))
        modules={'safetensors':SimpleNamespace(safe_open=lambda *a,**k:nullcontext(store)),
                 'embed_optim.optimizers':SimpleNamespace(parameter_partition_name=lambda *a:'hidden')}
        with patch.dict(sys.modules,modules):
            actual=entry.raw_checkpoint_mapping(mapped,Path('/synthetic-checkpoint'))
            self.assertEqual([r[1] for r in actual],keys)
            self.assertTrue(all(a[2] is b for a,b in zip(actual,parameters)))
            bad=mapped.copy();bad[0]=(bad[0][0],'wrong.alias',bad[0][2])
            with self.assertRaisesRegex(ValueError,'alias'):entry.raw_checkpoint_mapping(bad,Path('/synthetic-checkpoint'))
            with self.assertRaisesRegex(ValueError,'88'):entry.raw_checkpoint_mapping(mapped[:-1],Path('/synthetic-checkpoint'))

    def test_new_namespace_only(self):
        with self.assertRaisesRegex(ValueError, 'namespace'): entry.namespace()

    def test_approval_not_automatic_or_implicit(self):
        value = {'scope': entry.SCOPE, 'approved': True, 'source': 'direct_user_message',
            'owner_message': entry.OWNER_MESSAGE, 'automatic_continuation': False,
            'source_sha256': 'a', 'protected_helper_access': False, 'old_controller_transition': False}
        entry.approval(value, 'a')
        for key, bad in (('automatic_continuation', True), ('approved', False), ('source_sha256', 'b'),
                         ('protected_helper_access', True), ('old_controller_transition', True)):
            with self.subTest(key=key), self.assertRaises(ValueError): entry.approval({**value,key:bad}, 'a')

    def test_prepare_requires_approval_and_tests(self):
        with self.assertRaises(ValueError): entry.parse(['--source-sha', 'a', '--prepare'])
        self.assertTrue(entry.parse(['--source-sha', 'a', '--prepare', '--approval-sha', 'b', '--tests-sha', 'c']).prepare)

    def test_worker_exact_disjoint_tokens(self):
        for state in entry.STATES:
            argv = entry.command(SimpleNamespace(source_sha='a', authorization_sha='b'), state, (31,32))
            args = entry.parse(argv[3:])
            self.assertEqual(args.gpu_token, entry.GPU[state])
            self.assertNotEqual(args.gpu_token, '0')
            self.assertEqual(args.lease_fd, [31,32])

    def test_wrong_token_and_bad_leases_refused(self):
        argv = ['--source-sha','a','--authorization-sha','b','--worker','adamw_state']
        for tail in (['--gpu-token','2','--lease-fd','3','--lease-fd','4'],
                     ['--gpu-token','1','--lease-fd','3','--lease-fd','3'],
                     ['--gpu-token','1','--lease-fd','-1','--lease-fd','4']):
            with self.subTest(tail=tail), self.assertRaises(ValueError): entry.parse(argv+tail)

    def test_cpu_native_readback_no_gpu_arguments(self):
        argv = entry.command(SimpleNamespace(source_sha='a', authorization_sha='b'), 'muon_state', calibration_sha='c')
        args = entry.parse(argv[3:])
        self.assertEqual(args.verify, 'muon_state'); self.assertEqual(args.lease_fd, [])
        with self.assertRaises(ValueError): entry.parse(argv[3:]+['--gpu-token','2'])

    def test_existing_run_cannot_be_restarted(self):
        f = Fixture(self.root()); f.run.mkdir()
        with self.assertRaisesRegex(ValueError, 'restart'): f.run_all()
        self.assertEqual(f.children, [])

    def test_complete_two_gpu_and_two_fresh_cpu_bodies(self):
        f = Fixture(self.root()); result = f.run_all()
        self.assertEqual(result['actual_gpu_worker_exits'], {state:0 for state in entry.STATES})
        self.assertEqual(result['fresh_process_native_readbacks'], 2)
        self.assertEqual(result['formal_branches_trained'], 0)
        self.assertEqual(len(f.children), 4)
        self.assertEqual(sum(e[0]=='lease-close' for e in f.events), 2)
        self.assertEqual([c.done for c in f.children], [0,0,0,0])
        OBSERVED.append({'explicit_fixture_root': str(f.root), 'mock_gpu_children': 2,
                         'mock_fresh_cpu_readers': 2, 'actual_gpu_model_or_lease': False})

    def test_failed_worker_does_not_orphan_sibling_or_retry(self):
        f = Fixture(self.root(), 'first-worker')
        with self.assertRaisesRegex(ValueError, 'failed'): f.run_all()
        self.assertEqual(len(f.children), 2)
        self.assertEqual([c.done for c in f.children], [7,0])
        self.assertEqual(entry.read(f.run/'muon_state.exited.json')['exit_code'],0)
        self.assertFalse((f.run/'completed.json').exists())
        self.assertTrue((f.run/'failed.json').exists())

    def test_native_refusal_not_counted_as_calibration(self):
        f = Fixture(self.root(), 'readback')
        with self.assertRaisesRegex(ValueError, 'failed'): f.run_all()
        self.assertFalse((f.run/'completed.json').exists())
        self.assertEqual(len(f.children),2)

    def test_duplicate_json_and_changed_anchor_refused(self):
        root=self.root(); path=root/'duplicate.json'
        with path.open('x') as stream:stream.write('{"x":1,"x":2}')
        with self.assertRaises(ValueError):entry.read(path)
        with self.assertRaises(ValueError):entry.read(path,'a'*64)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--fixtures',type=Path,required=True)
    args=p.parse_args();FIXTURES=args.fixtures;FIXTURES.mkdir()
    source=entry.identity(entry.__file__);test_source=entry.identity(__file__)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    entry.need(entry.identity(entry.__file__)==source and entry.identity(__file__)==test_source,'Test source raced')
    write(args.output,{'observed_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':source['sha256'],'test_source':test_source,'tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
        'positive_process_model_lease_cases_are_explicit_fixtures':True,'observations':OBSERVED})
    raise SystemExit(0 if result.wasSuccessful() else 1)
