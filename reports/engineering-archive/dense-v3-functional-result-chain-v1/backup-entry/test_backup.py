"""New backup entry controls only; no remote writes or model execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('new_backup_entry',Path(__file__).with_name('backup.py'))
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


class Controls(unittest.TestCase):
    def fixture(self):
        temp=tempfile.TemporaryDirectory(prefix='factorial-backup-control-')
        self.addCleanup(temp.cleanup)
        return Path(temp.name)

    def test_new_json_payload(self):
        root=self.fixture();entry.write(root/'a.json',{'synthetic':True})
        rows=entry.data_files(root)
        self.assertEqual(rows,[{'path':'a.json',**entry.identity(root/'a.json')}])

    def test_existing_receipt_preserved(self):
        root=self.fixture();entry.write(root/'a.json',{'synthetic':True})
        before=entry.identity(root/'a.json')
        with self.assertRaises(FileExistsError):entry.write(root/'a.json',{'changed':True})
        self.assertEqual(entry.identity(root/'a.json'),before)

    def test_source_extension_refused(self):
        root=self.fixture();entry.write(root/'code.py',{'synthetic':True})
        with self.assertRaises(ValueError):entry.data_files(root)

    def test_credential_shape_refused(self):
        root=self.fixture();entry.write(root/'config.json',{'synthetic':'hf_'+'A'*24})
        with self.assertRaises(ValueError):entry.data_files(root)

    def test_symlink_refused(self):
        root=self.fixture();entry.write(root/'a.json',{'synthetic':True})
        (root/'b.json').symlink_to(root/'a.json')
        with self.assertRaises(ValueError):entry.data_files(root)

    def test_missing_external_binding_refused(self):
        root=self.fixture();entry.write(root/'a.json',{'synthetic':True})
        with self.assertRaises(ValueError):entry.read(root/'a.json','0'*64)

    def test_empty_tree_refused(self):
        with self.assertRaises(ValueError):entry.data_files(self.fixture())


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    receipt={'source':entry.identity(entry.__file__),'test_source':entry.identity(__file__),
        'tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),
        'skips':len(result.skipped),'actual_remote_calls':0,'scientific_completion':False}
    entry.write(Path(__file__).with_name('tests.json'),receipt)
    print(json.dumps(receipt,sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
