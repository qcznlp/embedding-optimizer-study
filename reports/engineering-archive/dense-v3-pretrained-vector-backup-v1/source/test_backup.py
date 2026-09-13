"""Bounded data-transport tests; no model, GPU, network or live scheduler."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import backup
import verify_recovered as reader

WORK = Path(__file__).parent


class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = WORK / 'staging'
        cls.anchor = json.loads((WORK / 'preflight.json').read_text())['manifest']['sha256']

    def test_accepted_complete_snapshot(self):
        value = reader.verify(self.root, self.anchor)
        self.assertEqual(value['snapshot_files'], 10)
        self.assertEqual(value['accepted_vector_states'], 1)
        self.assertEqual(value['feature_states'], 0)
        self.assertFalse(value['scientific_completion'])

    def test_original_input_copies_match(self):
        selected = backup.sources(backup.transport())
        self.assertEqual(len(selected), 9)

    def test_prepared_source_and_inventory(self):
        _, expected = backup.prepared(backup.transport(), WORK)
        self.assertEqual(len(expected), 10)

    def test_wrong_external_anchor(self):
        with self.assertRaises(ValueError):
            reader.verify(self.root, '0' * 64)

    def test_missing_external_anchor(self):
        with self.assertRaises(ValueError):
            reader.verify(self.root, '')

    def test_each_payload_corruption_refused(self):
        for name in (*reader.ORIGINALS, 'README.md', 'artifact_manifest.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix='pretrained-backup-test-') as tmp:
                root = Path(tmp) / 'snapshot'
                shutil.copytree(self.root, root)
                with (root / name).open('ab') as stream:
                    stream.write(b'CORRUPTED')
                with self.assertRaises(ValueError):
                    reader.verify(root, self.anchor)

    def test_extra_file_refused(self):
        with tempfile.TemporaryDirectory(prefix='pretrained-backup-test-') as tmp:
            root = Path(tmp) / 'snapshot'
            shutil.copytree(self.root, root)
            (root / 'unexpected.txt').touch()
            with self.assertRaises(ValueError):
                reader.verify(root, self.anchor)

    def test_symlink_root_refused(self):
        with tempfile.TemporaryDirectory(prefix='pretrained-backup-test-') as tmp:
            root = Path(tmp) / 'snapshot'
            root.symlink_to(self.root, target_is_directory=True)
            with self.assertRaises(ValueError):
                reader.verify(root, self.anchor)


class UploadBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.names = set(reader.ORIGINALS) | {'README.md', 'artifact_manifest.json'}
        self.prefix = backup.ADDITION + '/fixture'
        self.modes = {self.prefix + '/' + n: {
            'mode': 'lfs' if n == 'vectors/vectors.npz' else 'regular',
            'ignored': False, 'remote_oid': None} for n in self.names}

    def test_exact_one_npz_lfs_allowed(self):
        backup.check_modes(self.modes, self.prefix, self.names)

    def test_npz_regular_refused(self):
        self.modes[self.prefix + '/vectors/vectors.npz']['mode'] = 'regular'
        with self.assertRaises(ValueError):
            backup.check_modes(self.modes, self.prefix, self.names)

    def test_other_lfs_refused(self):
        self.modes[self.prefix + '/vectors/manifest.json']['mode'] = 'lfs'
        with self.assertRaises(ValueError):
            backup.check_modes(self.modes, self.prefix, self.names)

    def test_existing_remote_object_refused(self):
        self.modes[self.prefix + '/README.md']['remote_oid'] = 'existing'
        with self.assertRaises(ValueError):
            backup.check_modes(self.modes, self.prefix, self.names)

    def test_ignored_file_refused(self):
        self.modes[self.prefix + '/README.md']['ignored'] = True
        with self.assertRaises(ValueError):
            backup.check_modes(self.modes, self.prefix, self.names)

    def test_additional_source_file_refused(self):
        with self.assertRaises(ValueError):
            backup.check_modes(self.modes, self.prefix, self.names | {'backup.py'})

    def test_missing_remote_entry_refused(self):
        del self.modes[self.prefix + '/README.md']
        with self.assertRaises(ValueError):
            backup.check_modes(self.modes, self.prefix, self.names)

    def test_preserves_nine_subtrees_and_root(self):
        before = {'README.md': {'blob_id': 'old'}, '.gitattributes': {'blob_id': 'attrs'},
                  backup.NAMESPACE: {'kind': 'RepoFolder', 'tree_id': 'before'}}
        old = {backup.NAMESPACE + '/' + str(i): {'tree_id': str(i)} for i in range(9)}
        after = copy.deepcopy(before)
        after[backup.NAMESPACE]['tree_id'] = 'after'
        new = {**old, backup.ADDITION: {'tree_id': 'new'}}
        backup.check_preservation(before, after, old, new)
        for key in ('README.md', '.gitattributes'):
            changed = copy.deepcopy(after)
            changed[key]['blob_id'] = 'changed'
            with self.subTest(key=key), self.assertRaises(ValueError):
                backup.check_preservation(before, changed, old, new)
        changed = copy.deepcopy(new)
        changed[next(iter(old))]['tree_id'] = 'changed'
        with self.assertRaises(ValueError):
            backup.check_preservation(before, after, old, changed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
