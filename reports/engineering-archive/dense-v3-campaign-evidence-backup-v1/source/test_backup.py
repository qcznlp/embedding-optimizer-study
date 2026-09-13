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
        self.assertEqual(value['files'], 41)
        self.assertEqual(value['training_runs'], 12)
        self.assertEqual(value['original_exit_unobserved_runs'], 2)
        self.assertFalse(value['scientific_completion'])

    def test_original_input_copies_match(self):
        selected = backup.sources(backup.transport())
        self.assertEqual(len(selected), 40)

    def test_prepared_source_and_inventory(self):
        _, expected = backup.prepared(backup.transport(), WORK)
        self.assertEqual(len(expected), 41)

    def test_wrong_external_anchor(self):
        with self.assertRaises(ValueError):
            reader.verify(self.root, '0' * 64)

    def test_missing_external_anchor(self):
        with self.assertRaises(ValueError):
            reader.verify(self.root, '')

    def test_each_payload_corruption_refused(self):
        for name in (*reader.ORIGINALS, 'README.md', 'artifact_manifest.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix='campaign-backup-test-') as tmp:
                root = Path(tmp) / 'snapshot'
                shutil.copytree(self.root, root)
                with (root / name).open('ab') as stream:
                    stream.write(b'CORRUPTED')
                with self.assertRaises(ValueError):
                    reader.verify(root, self.anchor)

    def test_extra_file_refused(self):
        with tempfile.TemporaryDirectory(prefix='campaign-backup-test-') as tmp:
            root = Path(tmp) / 'snapshot'
            shutil.copytree(self.root, root)
            (root / 'unexpected.txt').touch()
            with self.assertRaises(ValueError):
                reader.verify(root, self.anchor)

    def test_symlink_root_refused(self):
        with tempfile.TemporaryDirectory(prefix='campaign-backup-test-') as tmp:
            root = Path(tmp) / 'snapshot'
            root.symlink_to(self.root, target_is_directory=True)
            with self.assertRaises(ValueError):
                reader.verify(root, self.anchor)


    def test_empty_extra_directory_refused(self):
        with tempfile.TemporaryDirectory(prefix='campaign-backup-test-') as tmp:
            root = Path(tmp) / 'snapshot'
            shutil.copytree(self.root, root)
            (root / 'empty-extra').mkdir()
            with self.assertRaises(ValueError):
                reader.verify(root, self.anchor)

    def test_git_identity_is_verified_not_ignored(self):
        import hashlib
        for mode in ('missing', 'wrong'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix='campaign-backup-test-') as tmp:
                root = Path(tmp) / 'snapshot'
                shutil.copytree(self.root, root)
                path = root / 'artifact_manifest.json'
                value = json.loads(path.read_text())
                if mode == 'missing':
                    del value['files']['README.md']['git_blob_sha1']
                else:
                    value['files']['README.md']['git_blob_sha1'] = '0' * 40
                raw = (json.dumps(value, sort_keys=True) + '\n').encode()
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    reader.verify(root, hashlib.sha256(raw).hexdigest())


class UploadBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.names = set(reader.ORIGINALS) | {'README.md', 'artifact_manifest.json'}
        self.prefix = backup.ADDITION + '/fixture'
        self.modes = {self.prefix + '/' + n: {
            'mode': 'regular',
            'ignored': False, 'remote_oid': None} for n in self.names}

    def test_exact_regular_modes_allowed(self):
        backup.check_modes(self.modes, self.prefix, self.names)

    def test_readme_lfs_refused(self):
        self.modes[self.prefix + '/README.md']['mode'] = 'lfs'
        with self.assertRaises(ValueError):
            backup.check_modes(self.modes, self.prefix, self.names)

    def test_other_lfs_refused(self):
        self.modes[self.prefix + '/anchors/input-view-audit.json']['mode'] = 'lfs'
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

    def test_preserves_ten_subtrees_and_root(self):
        before = {'README.md': {'blob_id': 'old'}, '.gitattributes': {'blob_id': 'attrs'},
                  backup.NAMESPACE: {'kind': 'RepoFolder', 'tree_id': 'before'}}
        old = {backup.NAMESPACE + '/' + str(i): {'tree_id': str(i)} for i in range(10)}
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
