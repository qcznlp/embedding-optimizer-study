"""New CPU-only filesystem tests, with explicitly mocked model/process work.

All generated fixtures remain below this candidate directory, including failures.
No original dispatcher mode, process reader, GPU lease or model loader is run.
"""

from __future__ import annotations

import ast
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import record_layout as layout

WORK = Path(__file__).resolve().parent
ORIGINAL = Path('/root/embedding-optimizer-v3-experiment/launch/functional-dimensions')
STORY = Path('/root/embedding-optimizer-story-refactor')
GEOMETRY = ORIGINAL.parent / 'weight-geometry/run_geometry.py'
BOUND = {
    ORIGINAL / 'dispatch.py': '3b02f1d8486f3c2b41d0403cbf291580d0ca32868b8b29cb05514a22770257f5',
    ORIGINAL / 'functional.py': 'c55d3fa101fa681070509a95d987023b7ed442d89ce385a8885f940ea3cc05b8',
    ORIGINAL / 'inputs.json': 'be6d92fda2f7f0ba3e86f2c2866fa9dca530f066d29ce92d0887fdad91684067',
    ORIGINAL / 'authorization.json': 'd72527595e2c963eae7fd46b0a1de2bd83d1ea15a863ae467a1065cfc5b2f336',
    ORIGINAL / 'run/failed.json': 'c84f93bc431612be2b885d7a467828d08a18d5f259829c4b925a4f23ea7def2e',
    ORIGINAL / 'run/jobs/pretrained.started.json': '2da0ecee9ab23c2dcd87f77e41587c6c24d3285aafdc52b95f8c289cda4c3722',
    ORIGINAL / 'run/jobs/pretrained.exited.json': '8a82b1f85a96e67f07e7e03ef9ae457bebed7b3c9d5e463d798bd504ae9b4e76',
    ORIGINAL / 'run/jobs/pretrained.encoded.json': 'd668916f56dfd7c4f8ee35f6f5f189a4f397e7c08f09456e0bca4e4d477d32c5',
    ORIGINAL / 'run/jobs/pretrained.verified.json': '8867643a9309af31b73e9b4f6bd9b3b8329389bb687b1e967d5043482cdd1dcb',
    ORIGINAL.parent.parent / 'analyses/dense-primary-v3-functional-dimensions/vectors/states/pretrained/vectors.npz':
        'e57057107312363619ebb1ba19b55892fe539568d9545726f3c9b9277325112f',
    ORIGINAL.parent.parent / 'analyses/dense-primary-v3-functional-dimensions/vectors/states/pretrained/manifest.json':
        '90d9f044b53c05065b00d16a53a31a5c82a37d86e33552d797f6c38364e64fac',
    GEOMETRY: '5ab87568a1a9050e423c2e778a4fda8b53f906eec2ee2f5a824e2522c09c0ec6',
    STORY / 'configs/dense_primary_v3_dimensions_protocol.json': '9246f7d15f141ba270eb4c839948c852ccc503de282f09090f58a0efdafe09f2',
}


def identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Require an ordinary bound source or fixture file')
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def protected_identities():
    for path, expected in BOUND.items():
        if identity(path)['sha256'] != expected:
            raise ValueError(f'Original source or failure record changed: {path.name}')
    authority = json.loads((ORIGINAL / 'authorization.json').read_bytes())
    result = {str(path): identity(path) for path in BOUND}
    for name, expected in authority['sources'].items():
        if Path(name).name == 'gpu.py':
            raise ValueError('Protected helper is never a permitted test input')
        actual = identity(name)
        if actual != expected:
            raise ValueError('Frozen functional dependency changed')
        result[name] = actual
    for name in ('observe.py', 'test_dispatch.py', 'tests-first.json'):
        result[str(ORIGINAL / name)] = identity(ORIGINAL / name)
    return result


PROTECTED_BEFORE = protected_identities()
INPUTS = json.loads((ORIGINAL / 'inputs.json').read_bytes())
AUTHORITY = json.loads((ORIGINAL / 'authorization.json').read_bytes())
PROTOCOL = json.loads((STORY / 'configs/dense_primary_v3_dimensions_protocol.json').read_bytes())
ACTUAL_CELLS = tuple(job['plan']['state']['cell'] for job in INPUTS['jobs'])

# Import only the original stdlib dispatcher module, not its entry/main/authenticate.
spec = importlib.util.spec_from_file_location('original_dispatch_lifecycle_fixture', ORIGINAL / 'dispatch.py')
dispatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatch)

# Execute the exact original exclusive JSON writer AST; no geometry-module import.
tree = ast.parse(GEOMETRY.read_text())
writer_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'write_new']
if len(writer_nodes) != 1:
    raise ValueError('Original writer definition differs')
writer_namespace = {'json': json, 'Path': Path, 'os': os}
exec(compile(ast.Module(body=writer_nodes, type_ignores=[]), str(GEOMETRY), 'exec'), writer_namespace)
write_new = writer_namespace['write_new']
FIXTURES = WORK / 'fixtures'
FIXTURES.mkdir(exist_ok=False)
SEEN = {'mocked_encoder_calls': 0, 'mocked_inspect_calls': 0, 'mocked_popen_calls': 0,
        'original_worker_bodies': 0, 'lease_enters': 0, 'lease_closes': 0}


def fixture_root():
    return Path(tempfile.mkdtemp(prefix='case-', dir=FIXTURES))


class MockedLifecycle:
    def __init__(self, root, exit_code=0):
        self.root, self.exit_code = root, exit_code
        self.jobs = root / 'run/jobs'
        self.jobs.mkdir(parents=True)
        self.entry = SimpleNamespace(STORY=root, OUTPUT=root / 'output',
            SCOPE='synthetic_filesystem_test_only', verify_state_inputs=lambda *_: None)
        self.args = SimpleNamespace(source_sha='a' * 64, authorization_sha='b' * 64)
        self.saved = {}
        self.context = SimpleNamespace(
            geometry=SimpleNamespace(write_new=write_new, identity=identity,
                                     now=lambda: 'synthetic-test-timestamp'),
            jobs=[{'plan': {'state': {'cell': cell}}, 'checkpoint': root / 'unread-models' / cell}
                  for cell in ACTUAL_CELLS], dataset=(), identities=(),
            digest=lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest())
        owner = self

        class Lease:
            def __enter__(self):
                SEEN['lease_enters'] += 1
                return (31, 32)  # Mock values: never opened, locked or inherited.

            def __exit__(self, *_):
                SEEN['lease_closes'] += 1

        self.context.parent = SimpleNamespace(handoff=lambda: None, leases=lambda _: Lease(),
            environment=lambda tokens: {'CUDA_VISIBLE_DEVICES': ','.join(tokens or [])})

        def encode(*_):
            SEEN['mocked_encoder_calls'] += 1
            return object(), {'synthetic_model_stub': True}

        def save(path, plan, *_):
            path.mkdir(parents=True, exist_ok=False)
            write_new(path / 'manifest.json', {'cell': plan['state']['cell'], 'synthetic_model_stub': True})
            owner.saved[str(path)] = {'manifest': identity(path / 'manifest.json'), 'synthetic_model_stub': True}
            return owner.saved[str(path)]

        def inspect(path, *_, expected_manifest_sha256):
            SEEN['mocked_inspect_calls'] += 1
            saved = owner.saved[str(path)]
            if identity(path / 'manifest.json')['sha256'] != expected_manifest_sha256:
                raise ValueError('Synthetic fixture anchor differs')
            return object(), saved

        self.context.vectors = SimpleNamespace(encode_state=encode, save_vectors=save, inspect_vectors=inspect)

        class Child:
            pid = -1  # Deliberately not an actual process handle.

            def __init__(self, command, **kwargs):
                SEEN['mocked_popen_calls'] += 1
                if kwargs['pass_fds'] != (31, 32):
                    raise AssertionError('Original two-descriptor routing changed')
                self.command = command

            def wait(self):
                if owner.exit_code == 0:
                    worker_args = dispatch.parse_args(self.command[3:])
                    SEEN['original_worker_bodies'] += 1
                    dispatch.worker(worker_args)
                return owner.exit_code

        self.Child = Child

    def __enter__(self):
        self.stack = ExitStack()
        replacements = {
            'RUN': self.root / 'run',
            'priority': lambda *_: {'synthetic_validation_priority_stub': True},
            'authenticate': lambda *_args, **_kwargs: (self.entry, self.context, {}),
            'process_identity': lambda pid, command: {
                'pid': pid, 'ppid': -1, 'start_ticks': -1, 'command': command,
                'synthetic_process_fixture': True},
        }
        for name, value in replacements.items():
            self.stack.enter_context(patch.object(dispatch, name, value))
        self.popen = self.stack.enter_context(patch.object(dispatch.subprocess, 'Popen', side_effect=self.Child))
        return self

    def __exit__(self, *_):
        self.stack.close()


class RecordLayoutTests(unittest.TestCase):
    def new_jobs(self):
        root = fixture_root() / 'jobs'
        root.mkdir()
        return root

    def prepared(self):
        root = self.new_jobs()
        layout.prepare_record_parents(root, ACTUAL_CELLS)
        return root

    def test_01_actual_input_protocol_and_authority_population_agree(self):
        self.assertEqual(layout.CELLS, ACTUAL_CELLS)
        self.assertEqual(ACTUAL_CELLS, tuple(AUTHORITY['state_order']))
        self.assertEqual(ACTUAL_CELLS, tuple(state['cell'] for state in PROTOCOL['states']))
        self.assertEqual((len(ACTUAL_CELLS), len(layout.RUNS)), (61, 12))

    def test_02_creates_only_twelve_required_parents(self):
        root = self.new_jobs()
        created = layout.prepare_record_parents(root, ACTUAL_CELLS)
        self.assertEqual({path.name for path in created}, set(layout.RUNS))
        self.assertEqual(len(tuple(root.iterdir())), 12)
        self.assertEqual(layout.inventory(root, ACTUAL_CELLS), {})

    def test_03_original_failure_precedes_any_child(self):
        with MockedLifecycle(fixture_root()) as fixture:
            with self.assertRaises(FileNotFoundError):
                dispatch.encode_one(fixture.args, fixture.entry, fixture.context, fixture.context.jobs[1])
            fixture.popen.assert_not_called()
            self.assertEqual(tuple(fixture.jobs.iterdir()), ())

    def test_04_original_single_state_and_worker_bodies_cover_all_61_cells(self):
        with MockedLifecycle(fixture_root()) as fixture:
            layout.prepare_record_parents(fixture.jobs, ACTUAL_CELLS)
            observed = {}
            for job in fixture.context.jobs:
                cell = job['plan']['state']['cell']
                observed[cell] = dispatch.encode_one(fixture.args, fixture.entry, fixture.context, job)
                write_new(fixture.jobs / f'{cell}.features-verified.json',
                          {'cell': cell, 'synthetic_feature_fixture': True})
            self.assertEqual(tuple(observed), ACTUAL_CELLS)
            self.assertEqual(fixture.popen.call_count, 61)
            self.assertEqual(len(layout.inventory(fixture.jobs, ACTUAL_CELLS)), 61 * 7)
            for role in layout.JSON_ROLES:
                paths = layout.existing_records(fixture.jobs, ACTUAL_CELLS, role)
                self.assertEqual(len(paths), 61)
                self.assertEqual({json.loads(path.read_bytes())['cell'] for path in paths}, set(ACTUAL_CELLS))
            self.assertEqual(len(tuple(fixture.jobs.glob('*.started.json'))), 1)
            self.assertEqual(len(tuple(fixture.jobs.glob('*.features-verified.json'))), 1)

    def test_05_original_worker_nonzero_stops_without_verification_or_retry(self):
        with MockedLifecycle(fixture_root(), exit_code=7) as fixture:
            layout.prepare_record_parents(fixture.jobs, ACTUAL_CELLS)
            job = fixture.context.jobs[1]
            with self.assertRaisesRegex(ValueError, 'no retry'):
                dispatch.encode_one(fixture.args, fixture.entry, fixture.context, job)
            self.assertEqual(fixture.popen.call_count, 1)
            exits = layout.existing_records(fixture.jobs, ACTUAL_CELLS, 'exited')
            self.assertEqual(len(exits), 1)
            self.assertEqual(json.loads(exits[0].read_bytes())['exit_code'], 7)
            self.assertEqual(layout.existing_records(fixture.jobs, ACTUAL_CELLS, 'verified'), ())

    def test_06_partial_population_is_not_reported_as_complete(self):
        root = self.prepared()
        cell = ACTUAL_CELLS[-1]
        write_new(root / f'{cell}.started.json', {'cell': cell, 'synthetic_fixture': True})
        self.assertEqual(len(layout.existing_records(root, ACTUAL_CELLS, 'started')), 1)
        self.assertEqual(layout.existing_records(root, ACTUAL_CELLS, 'exited'), ())

    def test_07_second_initialization_preserves_existing_tree(self):
        root = self.prepared()
        before = tuple(sorted(str(path) for path in root.iterdir()))
        with self.assertRaisesRegex(ValueError, 'fresh empty'):
            layout.prepare_record_parents(root, ACTUAL_CELLS)
        self.assertEqual(before, tuple(sorted(str(path) for path in root.iterdir())))

    def test_08_original_writer_still_refuses_overwrite(self):
        root = self.prepared()
        path = root / f'{ACTUAL_CELLS[1]}.verified.json'
        write_new(path, {'cell': ACTUAL_CELLS[1], 'synthetic_fixture': True})
        before = identity(path)
        with self.assertRaises(FileExistsError):
            write_new(path, {'cell': 'changed'})
        self.assertEqual(identity(path), before)

    def test_09_wrong_missing_extra_duplicate_reordered_and_unsafe_cells_refused_before_writes(self):
        invalid = [ACTUAL_CELLS[:-1], ACTUAL_CELLS + ('extra',), ACTUAL_CELLS[1:],
                   ACTUAL_CELLS[:1] + ACTUAL_CELLS[2:] + ACTUAL_CELLS[1:2],
                   ACTUAL_CELLS[:-1] + ACTUAL_CELLS[:1], 'pretrained']
        for name in ('../outside', '/absolute', 'x/../../outside', 'x//checkpoint-782',
                     'x/./checkpoint-782', 'x\\checkpoint-782', 'historical/checkpoint-782'):
            invalid.append(ACTUAL_CELLS[:-1] + (name,))
        for cells in invalid:
            with self.subTest(cells=repr(cells)[-100:]):
                root = self.new_jobs()
                with self.assertRaises(ValueError):
                    layout.prepare_record_parents(root, cells)
                self.assertEqual(tuple(root.iterdir()), ())

    def test_10_nonempty_root_is_preserved(self):
        root = self.new_jobs()
        write_new(root / 'failed.json', {'synthetic_original_failure': True})
        before = identity(root / 'failed.json')
        with self.assertRaisesRegex(ValueError, 'fresh empty'):
            layout.prepare_record_parents(root, ACTUAL_CELLS)
        self.assertEqual(identity(root / 'failed.json'), before)

    def test_11_symlink_root_or_ancestor_is_refused(self):
        base = fixture_root()
        target = base / 'target'
        target.mkdir()
        (base / 'link').symlink_to(target, target_is_directory=True)
        for root in (base / 'link', base / 'link/jobs'):
            with self.subTest(root=str(root)):
                with self.assertRaises(ValueError):
                    layout.prepare_record_parents(root, ACTUAL_CELLS)
        self.assertEqual(tuple(target.iterdir()), ())

    def test_12_missing_declared_directory_is_not_silent_zero(self):
        root = self.new_jobs()
        with self.assertRaisesRegex(ValueError, 'Missing declared'):
            layout.existing_records(root, ACTUAL_CELLS, 'started')

    def test_13_unknown_nested_file_or_directory_is_refused(self):
        for directory in (False, True):
            root = self.prepared()
            path = root / layout.RUNS[0] / 'unbound'
            path.mkdir() if directory else write_new(path, {'synthetic_fixture': True})
            with self.assertRaisesRegex(ValueError, 'nested'):
                layout.existing_records(root, ACTUAL_CELLS, 'started')

    def test_14_unknown_top_level_file_or_directory_is_refused(self):
        for directory in (False, True):
            root = self.prepared()
            path = root / 'unbound'
            path.mkdir() if directory else write_new(path, {'synthetic_fixture': True})
            with self.assertRaisesRegex(ValueError, 'top-level'):
                layout.existing_records(root, ACTUAL_CELLS, 'started')

    def test_15_symlink_record_is_refused_without_reading_target(self):
        root = self.prepared()
        path = root / f'{ACTUAL_CELLS[1]}.started.json'
        path.symlink_to(root / 'absent-target')
        with self.assertRaisesRegex(ValueError, 'nested'):
            layout.existing_records(root, ACTUAL_CELLS, 'started')

    def test_16_renamed_cell_payload_is_refused(self):
        root = self.prepared()
        write_new(root / f'{ACTUAL_CELLS[1]}.started.json', {'cell': ACTUAL_CELLS[2]})
        with self.assertRaisesRegex(ValueError, 'exact native filename'):
            layout.existing_records(root, ACTUAL_CELLS, 'started')

    def test_17_partial_json_is_not_counted(self):
        root = self.prepared()
        path = root / f'{ACTUAL_CELLS[1]}.started.json'
        with path.open('xb') as stream:
            stream.write(b'{')
        with self.assertRaises(json.JSONDecodeError):
            layout.existing_records(root, ACTUAL_CELLS, 'started')

    def test_18_unknown_record_family_is_refused(self):
        for role in ('log', '../started', 'started.json', 'completed', ''):
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, 'Unknown'):
                layout.existing_records(self.prepared(), ACTUAL_CELLS, role)

    def test_19_no_model_or_numerical_packages_imported(self):
        for name in ('torch', 'transformers', 'numpy', 'embed_optim'):
            self.assertNotIn(name, sys.modules)


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.completed_cases = []

    def stopTest(self, test):
        self.completed_cases.append(test.id())
        super().stopTest(test)


def main():
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(RecordLayoutTests))
    after = protected_identities()
    unchanged = after == PROTECTED_BEFORE
    suite = ET.Element('testsuite', name='offline-functional-record-layout',
        tests=str(result.testsRun), failures=str(len(result.failures)),
        errors=str(len(result.errors)), skipped=str(len(result.skipped)))
    failures = {test.id(): detail for test, detail in result.failures + result.errors}
    for name in result.completed_cases:
        case = ET.SubElement(suite, 'testcase', name=name)
        if name in failures:
            ET.SubElement(case, 'failure').text = failures[name]
    with (WORK / 'tests.xml').open('xb') as stream:
        stream.write(ET.tostring(suite, encoding='utf-8', xml_declaration=True))
    record = {
        'scope': 'offline_record_layout_component_and_mocked_original_lifecycle',
        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'tests_run': result.testsRun, 'failures': len(result.failures),
        'errors': len(result.errors), 'skipped': len(result.skipped),
        'candidate_source': identity(WORK / 'record_layout.py'),
        'test_source': identity(__file__), 'junit': identity(WORK / 'tests.xml'),
        'protected_sources_and_original_failure_unchanged': unchanged,
        'protected_files': after, 'declared_cells': list(ACTUAL_CELLS),
        'declared_record_families': list(layout.JSON_ROLES),
        'mocked_execution_counters': SEEN,
        'actual_gpu_jobs_launched': 0, 'model_encoding_executed': False,
        'native_numerical_acceptance_tested': False, 'real_lease_or_process_inspection': False,
        'baseline_reused_or_reencoded': False, 'recovery_coordinator_implemented': False,
        'recovery_execution_authorized': False, 'frozen_source_modified': False,
        'scientific_completion': False,
        'boundary': 'Only record directories/files and mocked original encode_one/worker bodies. '
                    'No model, primary data, original lock, process, finalizer, feature calculation or recovery launch.'}
    write_new(WORK / 'tests.json', record)
    print(json.dumps({key: record[key] for key in ('scope', 'observed_at_utc', 'tests_run',
          'failures', 'errors', 'skipped', 'protected_sources_and_original_failure_unchanged',
          'mocked_execution_counters', 'actual_gpu_jobs_launched', 'recovery_execution_authorized')},
          sort_keys=True), flush=True)
    raise SystemExit(0 if result.wasSuccessful() and unchanged else 1)


if __name__ == '__main__':
    main()
