"""Bounded rendering tests; fixtures are never scientific results."""
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('actual_result_render', HERE / 'render.py')
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)
BUNDLE = Path(os.environ['DENSE_RESULT_BUNDLE'])


class RenderChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functional = R.read(BUNDLE / 'inputs/functional_controls/result.json')
        cls.outputs = R.generate(BUNDLE)

    def controls(self, value):
        return R.checked_controls(value, R.FUNCTIONAL_LABELS, 24)

    def test_actual_complete_controls(self):
        value = self.controls(self.functional)
        self.assertEqual(len(value['rows']), 24)
        self.assertEqual(value['survivors'], [])

    def bad_control(self, mutate):
        value = copy.deepcopy(self.functional)
        mutate(value)
        with self.assertRaises((ValueError, TypeError, KeyError)):
            self.controls(value)

    def test_missing_summary(self):
        self.bad_control(lambda v: v['summaries'].pop())

    def test_duplicate_summary(self):
        self.bad_control(lambda v: v['summaries'].__setitem__(1, v['summaries'][0]))

    def test_false_positive_flag(self):
        self.bad_control(lambda v: v['summaries'][0].__setitem__('diagnostic_flag', True))

    def test_edited_rmse(self):
        self.bad_control(lambda v: v['summaries'][0].__setitem__('pooled_feature_rmse', 0.0))

    def test_missing_fold(self):
        self.bad_control(lambda v: v['folds'].pop())

    def test_duplicate_fold(self):
        self.bad_control(lambda v: v['folds'].__setitem__(1, v['folds'][0]))

    def test_edited_improved_count(self):
        self.bad_control(lambda v: v['summaries'][0].__setitem__('improved_folds', 4))

    def test_edited_population(self):
        self.bad_control(lambda v: v['folds'][0].__setitem__('test_rows', 14))

    def test_unresolved_not_hidden(self):
        self.bad_control(lambda v: v['summaries'][0].__setitem__('defined_folds', 3))

    def test_edited_pooled_exact_mse(self):
        self.bad_control(lambda v: v['summaries'][0].__setitem__('pooled_feature_mse_exact', '1'))

    def test_functional_original_bytes_retained(self):
        raw = (BUNDLE / 'inputs/functional/results.tex').read_bytes()
        self.assertEqual(self.outputs['functional-inference.tex'], raw)
        self.assertEqual(self.outputs['dimension-utilization.tex'].replace(
            b'\\label{tab:dimension-utilization}\n', b'', 1), raw)

    def test_primary_prefix_retained(self):
        self.assertTrue(self.outputs['optimizer-primary.tex'].startswith(
            self.outputs['original-optimizer-primary.tex']))

    def test_all_primary_contrasts_preserved(self):
        summary = json.loads(self.outputs['primary-summary.json'])
        self.assertEqual(len(summary['primary']), 3)
        self.assertEqual(len(summary['secondary']), 3)
        self.assertTrue(all(r['support'] == 'inconclusive' for r in summary['primary']))
        self.assertEqual(sum(r['support'] == 'positive' for r in summary['secondary']), 1)

    def test_all_exact_features_preserved(self):
        self.assertEqual(len(json.loads(self.outputs['exact-summary.json'])['rows']), 5)

    def test_all_recipe_cells_present(self):
        text = self.outputs['recipe-sensitivity.tex'].decode()
        self.assertEqual(text.count('/4)'), 108)
        self.assertIn('0 of 14', text)
        self.assertIn('0 of four', text)
        self.assertIn('post-result exploratory', text)

    def test_actual_source_and_input_authentication(self):
        manifest = R.authenticate(BUNDLE)
        self.assertEqual(len(manifest['records']), 72)

    def test_input_corruption_and_missing_refused(self):
        with tempfile.TemporaryDirectory(prefix='dense-results-auth-') as tmp:
            root = Path(tmp) / 'bundle'
            shutil.copytree(BUNDLE, root)
            target = root / 'inputs/trajectory/primary_summary.csv'
            target.write_bytes(target.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'Bound rendering input differs'):
                R.authenticate(root)
            target.unlink()
            with self.assertRaisesRegex(ValueError, 'Nonordinary input'):
                R.authenticate(root)

    def test_symlinked_input_refused(self):
        with tempfile.TemporaryDirectory(prefix='dense-results-link-') as tmp:
            link = Path(tmp) / 'linked'
            link.symlink_to(BUNDLE / 'inputs.json')
            with self.assertRaisesRegex(ValueError, 'Nonordinary input'):
                R.identity(link)

    def test_clean_reconstruction_and_edited_output_refusal(self):
        with tempfile.TemporaryDirectory(prefix='dense-results-cold-') as tmp:
            output = Path(tmp) / 'actual'
            env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
            env['CUDA_VISIBLE_DEVICES'] = ''
            command = [sys.executable, '-B', str(HERE / 'render.py'), '--bundle', str(BUNDLE),
                       '--output', str(output)]
            generated = subprocess.run(command, cwd=tmp, env=env, capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            verified = subprocess.run(command + ['--verify'], cwd=tmp, env=env, capture_output=True, text=True)
            self.assertEqual(verified.returncode, 0, verified.stderr)
            target = output / 'optimizer-primary.tex'
            target.write_bytes(target.read_bytes() + b'% edited\n')
            refused = subprocess.run(command + ['--verify'], cwd=tmp, env=env, capture_output=True, text=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn('Freshly reconstructed bytes differ', refused.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
