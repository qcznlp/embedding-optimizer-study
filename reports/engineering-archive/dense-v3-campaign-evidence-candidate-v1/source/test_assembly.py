"""Bounded composition refusal cases; all mutations are in private temp copies."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from assemble_snapshot import assemble

RECOVERED = Path(sys.argv.pop(1))
ADDITIONAL = Path(sys.argv.pop(1))


class AssemblyTests(unittest.TestCase):
    def test_existing_destination_preserved(self):
        with tempfile.TemporaryDirectory(prefix='campaign-existing-') as tmp:
            root = Path(tmp); marker = root / 'keep.txt'; marker.write_text('preserve')
            with self.assertRaises(ValueError):
                assemble(RECOVERED, ADDITIONAL, root)
            self.assertEqual(marker.read_text(), 'preserve')

    def test_missing_original_receipt_refused_before_output(self):
        with tempfile.TemporaryDirectory(prefix='campaign-missing-') as tmp:
            root = Path(tmp); addition = root / 'additional'; shutil.copytree(ADDITIONAL, addition)
            (addition / 'receipts/verified-v3-adamw-1e-6.exited.json').unlink()
            with self.assertRaises((ValueError, OSError)):
                assemble(RECOVERED, addition, root / 'output')
            self.assertFalse((root / 'output').exists())

    def test_extra_record_refused_before_output(self):
        with tempfile.TemporaryDirectory(prefix='campaign-extra-') as tmp:
            root = Path(tmp); addition = root / 'additional'; shutil.copytree(ADDITIONAL, addition)
            (addition / 'receipts/verified-v3-normuon-1e-4.exited.json').write_text('{"exit_code":0}')
            with self.assertRaises(ValueError):
                assemble(RECOVERED, addition, root / 'output')
            self.assertFalse((root / 'output').exists())

    def test_wrong_recovered_manifest_refused(self):
        with tempfile.TemporaryDirectory(prefix='campaign-wrong-manifest-') as tmp:
            root = Path(tmp); recovered = root / 'wrong'; recovered.mkdir()
            (recovered / 'artifact_manifest.json').write_text('{}')
            with self.assertRaises(ValueError):
                assemble(recovered, ADDITIONAL, root / 'output')
            self.assertFalse((root / 'output').exists())

    def test_symlinked_input_root_refused(self):
        with tempfile.TemporaryDirectory(prefix='campaign-input-link-') as tmp:
            root = Path(tmp); (root / 'linked').symlink_to(ADDITIONAL, target_is_directory=True)
            with self.assertRaises(ValueError):
                assemble(RECOVERED, root / 'linked', root / 'output')
            self.assertFalse((root / 'output').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
