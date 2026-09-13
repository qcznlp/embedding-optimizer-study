"""Bounded new-entry controls; synthetic identities are never scientific data."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('new_functional_entry', Path(__file__).with_name('inference.py'))
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class Controls(unittest.TestCase):
    def setUp(self):
        self.jobs, self.joined = [], []
        for i in range(60):
            run, step, stage = f'synthetic-{i // 5}', i % 5 + 1, i % 5 + 1
            model = {'bytes': 2, 'sha256': str(i).zfill(64)}
            seal = {'bytes': 3, 'sha256': str(i + 100).zfill(64)}
            sha = str(i // 5 + 1000).zfill(64)
            self.joined.append({'run_id': run, 'step': step, 'stage': stage,
                'run_identity_sha256': sha, 'checkpoint_seal': seal, 'model_safetensors': model})
            self.jobs.append({'plan': {'state': {'meta': {'run_id': run, 'step': step}, 'stage': stage},
                'model': {'run_identity_sha256': sha, 'checkpoint': {'step': step,
                    'run_identity_sha256': sha, 'checkpoint_seal': copy.deepcopy(seal),
                    'files': [{'path': 'model.safetensors', **model}]}}}})

    def test_complete_distinct_synthetic_join(self):
        entry.join_checkpoints(self.jobs, self.joined)

    def test_missing_original_row(self):
        with self.assertRaises(ValueError): entry.join_checkpoints(self.jobs, self.joined[:-1])

    def test_missing_vector_state(self):
        with self.assertRaises(ValueError): entry.join_checkpoints(self.jobs[:-1], self.joined)

    def test_duplicate_vector_state(self):
        self.jobs[1] = copy.deepcopy(self.jobs[0])
        with self.assertRaises(ValueError): entry.join_checkpoints(self.jobs, self.joined)

    def test_wrong_weight(self):
        self.jobs[0]['plan']['model']['checkpoint']['files'][0]['sha256'] = 'a' * 64
        with self.assertRaises(ValueError): entry.join_checkpoints(self.jobs, self.joined)

    def test_wrong_seal(self):
        self.jobs[0]['plan']['model']['checkpoint']['checkpoint_seal']['sha256'] = 'a' * 64
        with self.assertRaises(ValueError): entry.join_checkpoints(self.jobs, self.joined)

    def test_wrong_run(self):
        self.jobs[0]['plan']['model']['run_identity_sha256'] = 'a' * 64
        with self.assertRaises(ValueError): entry.join_checkpoints(self.jobs, self.joined)

    def test_wrong_stage(self):
        self.jobs[0]['plan']['state']['stage'] = 2
        with self.assertRaises(ValueError): entry.join_checkpoints(self.jobs, self.joined)

    def test_cuda_must_be_hidden(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
            with self.assertRaisesRegex(ValueError, 'CPU-only'):
                entry.source_admission(SimpleNamespace(source_sha=entry.identity(entry.__file__)['sha256']))

    def test_source_must_be_externally_bound(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            with self.assertRaisesRegex(ValueError, 'Changed new readout'):
                entry.source_admission(SimpleNamespace(source_sha='0' * 64))

    def test_json_duplicate_and_nonfinite_refused(self):
        with tempfile.TemporaryDirectory(prefix='functional-inference-control-') as directory:
            for i, data in enumerate((b'{"x":1,"x":2}', b'{"x":NaN}')):
                path = Path(directory) / str(i)
                with path.open('xb') as stream: stream.write(data)
                with self.assertRaises(ValueError): entry.read(path)

    def test_changed_external_receipt_refused(self):
        with self.assertRaisesRegex(ValueError, 'External binding'):
            entry.read(Path(entry.__file__).parent / 'owner-approval.json', '0' * 64)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Controls)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = {'source': entry.identity(entry.__file__), 'test_source': entry.identity(__file__),
        'tests_run': result.testsRun, 'errors': len(result.errors), 'failures': len(result.failures),
        'skipped': len(result.skipped), 'scope': 'new_entry_synthetic_identity_controls_only',
        'scientific_completion': False}
    entry.write(Path(__file__).with_name('tests.json'), record)
    print(json.dumps(record, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
