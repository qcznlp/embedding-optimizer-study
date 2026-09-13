"""Small explicitly synthetic I/O controls; actual numerical evidence is separate."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_new_closed_replay_controls', HERE / 'replay.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


class InputIntegrity(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='synthetic-integrity-', dir=HERE / 'controls'))
        (self.root / 'data.json').write_text('{"explicitly_synthetic": true}\n')
        self.manifest = {'scope': 'closed-current-primary-numerical-and-document-inputs-v1',
            'native_primary_assembly_sha256':
                '1941d86b42972a53dbca8ff86c8cc346282cc10f17446d62f66d67fcedb826cc',
            'files': {'data.json': replay.identity(self.root / 'data.json')}}
        self.anchor()

    def anchor(self):
        write_json(self.root / 'manifest.json', self.manifest)
        self.sha = replay.identity(self.root / 'manifest.json')['sha256']

    def test_integrity_only_positive_is_not_scientific_admission(self):
        self.assertEqual(replay.authenticate(self.root, self.sha), self.manifest)

    def test_wrong_external_anchor(self):
        with self.assertRaisesRegex(ValueError, 'anchor'):
            replay.authenticate(self.root, '0' * 64)

    def test_changed_payload(self):
        (self.root / 'data.json').write_text('{"explicitly_synthetic": false}\n')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            replay.authenticate(self.root, self.sha)

    def test_additional_file(self):
        (self.root / 'extra.json').write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'additional'):
            replay.authenticate(self.root, self.sha)

    def test_missing_payload(self):
        (self.root / 'data.json').rename(self.root / 'renamed.json')
        with self.assertRaisesRegex(ValueError, 'Ordinary'):
            replay.authenticate(self.root, self.sha)

    def test_symlink(self):
        (self.root / 'data.json').rename(self.root / 'original.json')
        (self.root / 'data.json').symlink_to(self.root / 'original.json')
        with self.assertRaisesRegex(ValueError, 'Ordinary'):
            replay.authenticate(self.root, self.sha)

    def test_nonlocal_role_even_with_new_manifest_digest(self):
        self.manifest['files'] = {'../outside.json': self.manifest['files']['data.json']}
        self.anchor()
        with self.assertRaisesRegex(ValueError, 'Nonlocal'):
            replay.authenticate(self.root, self.sha)


class RuntimeBoundary(unittest.TestCase):
    def attempt(self, content, *, corrupt_anchor=False, existing_output=False):
        root = Path(tempfile.mkdtemp(prefix='synthetic-runtime-', dir=HERE / 'controls'))
        shutil.copyfile(HERE / 'run_isolated.py', root / 'run_isolated.py')
        (root / 'replay.py').write_text('# Explicitly synthetic refusal control, not scientific results.\n' + content)
        write_json(root / 'manifest.json', {'explicitly_synthetic_io_control': True})
        anchor = hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
        output = root / 'output'
        if existing_output:
            output.mkdir()
            (output / 'preserve.txt').write_text('preserve synthetic prior attempt\n')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': ''}
        result = subprocess.run([sys.executable, '-B', str(root / 'run_isolated.py'),
            '--manifest-sha256', '0' * 64 if corrupt_anchor else anchor,
            '--output', str(output)], cwd=root, env=env, capture_output=True, text=True, timeout=20)
        write_json(root / 'actual-execution.json', {'returncode': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr, 'synthetic_control': True})
        return output, result

    def test_old_producer_read_refused_before_file_lookup(self):
        output, result = self.attempt("from pathlib import Path\nPath('/root/embedding-optimizer-PORTABILITY-CANARY-NOT-A-REAL-PRODUCER').read_bytes()\n")
        self.assertNotEqual(result.returncode, 0)
        record = json.loads((output / 'io-boundary.json').read_bytes())
        self.assertEqual(record['failure']['exception_type'], 'PermissionError')
        self.assertEqual(len(record['producer_reads_refused']), 1)

    def test_network_refused_before_connection(self):
        output, result = self.attempt("import socket\nsocket.socket().connect(('127.0.0.1', 1))\n")
        self.assertNotEqual(result.returncode, 0)
        record = json.loads((output / 'io-boundary.json').read_bytes())
        self.assertEqual(record['failure']['exception_type'], 'PermissionError')
        self.assertEqual(record['network_refused'], 1)

    def test_bad_external_anchor_preserves_absent_output(self):
        output, result = self.attempt("raise AssertionError('Must not execute')\n", corrupt_anchor=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())

    def test_existing_attempt_is_preserved(self):
        output, result = self.attempt("raise AssertionError('Must not execute')\n", existing_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((output / 'preserve.txt').read_text(), 'preserve synthetic prior attempt\n')
        self.assertFalse((output / 'io-boundary.json').exists())


if __name__ == '__main__':
    (HERE / 'controls').mkdir(exist_ok=False)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    write_json(HERE / 'controls-result.json', {'scope': 'explicitly-synthetic-portability-boundary-controls',
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'skips': len(result.skipped), 'real_scientific_outputs': False,
        'source_bindings': {name: replay.identity(HERE / name) for name in
                            ('test_portability.py', 'replay.py', 'run_isolated.py')}})
    sys.exit(not result.wasSuccessful())
