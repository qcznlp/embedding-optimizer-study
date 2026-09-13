"""CPU filesystem/native serializer controls; no model encoding or feature finding."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('new_feature_entry', Path(__file__).with_name('features.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Tests(unittest.TestCase):
    def test_actual_completed_vector_and_preserved_failed_attempt_bindings(self):
        self.assertEqual(m.require_inputs()['all_states_encoded_and_native_readback_verified'], 61)

    def test_every_actual_nested_state_parent(self):
        manifest = m.read(m.VECTOR_ROOT / 'manifest.json')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for cell in manifest['states']:
                path = root / cell
                m.prepare_parent(path, root)
                self.assertTrue(path.parent.is_dir())
                self.assertFalse(path.exists())
            self.assertEqual(len(manifest['states']), 61)

    def test_every_actual_nested_record_parent(self):
        manifest = m.read(m.VECTOR_ROOT / 'manifest.json')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for cell in manifest['states']:
                path = root / f'{cell}.features-verified.json'
                m.prepare_parent(path, root)
                self.assertTrue(path.parent.is_dir())

    def test_no_existing_output_adoption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'state'
            path.mkdir()
            with self.assertRaises(ValueError): m.prepare_parent(path, root)

    def test_external_or_symlinked_parent_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError): m.prepare_parent(root.parent / 'outside', root)
            target = root / 'target'; target.mkdir()
            (root / 'alias').symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError): m.prepare_parent(root / 'alias/state', root)

    def test_same_actual_numerical_and_readback_loop(self):
        old = ast.parse((m.PRIOR / 'recovery.py').read_text())
        new = ast.parse(Path(m.__file__).read_text())
        old_fn = next(x for x in old.body if isinstance(x, ast.FunctionDef) and x.name == 'feature_worker')
        new_fn = next(x for x in new.body if isinstance(x, ast.FunctionDef) and x.name == 'compute')
        def numerical_loop(fn):
            loop = next(x for x in fn.body if isinstance(x, ast.For))
            # Operational difference only: new parent preparation and named receipt path.
            loop.body = [x for x in loop.body if not (
                isinstance(x, ast.Expr) and isinstance(x.value, ast.Call)
                and isinstance(x.value.func, ast.Name) and x.value.func.id == 'prepare_parent')
                and not (isinstance(x, ast.Assign) and isinstance(x.targets[0], ast.Name)
                         and x.targets[0].id == 'receipt_path')]
            for x in ast.walk(loop):
                if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == 'write_new':
                    x.args[0] = ast.Constant(value='new-operational-record-path')
            return ast.dump(loop, include_attributes=False)
        self.assertEqual(numerical_loop(old_fn), numerical_loop(new_fn))

    def test_real_native_serializer_under_nested_parent(self):
        from embed_optim import dimension_intervention_io as io
        result = {'tables': {name: [{'mock': 1}] for name in
            ('checkpoint_summary', 'task_summary', 'random_removal', 'rotation_summary')},
            'numerical_policy': {'fixture': True}, 'rotation_checks': [],
            'attributions': {'task_groups': np.array(['mock-task']),
                'ndcg_removal_gain': np.array([[0.1, 0.2]]),
                'margin_removal_gain': np.array([[0.3, -0.1]])}, 'rotated_attributions': []}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root / 'mock-run/checkpoint-79'
            with self.assertRaises(ValueError): io.save_state(output, {'fixture': True}, result)
            m.prepare_parent(output, root)
            saved = io.save_state(output, {'fixture': True}, result)
            self.assertTrue(saved['fresh_numerics_verified'])
            self.assertTrue(io.inspect_state(output, {'fixture': True}, result)['fresh_numerics_verified'])
            with self.assertRaises(ValueError): io.save_state(output, {'fixture': True}, result)
            result['attributions']['ndcg_removal_gain'][0, 0] = 0.7
            with self.assertRaises(ValueError): io.inspect_state(output, {'fixture': True}, result)

    def test_no_gpu_launch_cli(self):
        args = m.parse(['--source-sha', 's', '--approval-sha', 'a', '--tests-sha', 't', '--compute'])
        self.assertFalse(hasattr(args, 'gpu_token'))
        self.assertFalse(hasattr(args, 'lease_fd'))


if __name__ == '__main__':
    import json
    from datetime import datetime, timezone
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    summary = {'source_sha256': m.identity(Path(m.__file__))['sha256'],
        'test_source': m.identity(Path(__file__)), 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
        'actual_native_serializer_fixture': 'tiny explicit synthetic data, not feature findings',
        'finished_at_utc': datetime.now(timezone.utc).isoformat()}
    with Path(__file__).with_name('tests.json').open('x') as stream:
        stream.write(json.dumps(summary, sort_keys=True, indent=2) + '\n')
    raise SystemExit(not result.wasSuccessful())
