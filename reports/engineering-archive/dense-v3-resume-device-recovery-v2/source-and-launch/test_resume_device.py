"""Restoration callback controls; no genuine GPU endpoint is claimed."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import resume as r


def trainer_fixture():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = torch.optim.Optimizer([
        {'params': [parameter], 'param_names': ['matrix'], 'algorithm': 'adamw'}
    ], defaults={})
    optimizer.state[parameter] = dict(step=torch.tensor(313.), exp_avg=torch.ones_like(parameter),
                                    exp_avg_sq=torch.ones_like(parameter))
    optimizer.completed_steps = 313
    return SimpleNamespace(_raw_optimizer=lambda: optimizer, component_identity=lambda: {'anchor': 'unchanged'}), parameter


class DeviceRecoveryControls(unittest.TestCase):
    def test_callback_runs_once_without_control_change(self):
        trainer, parameter = trainer_fixture()
        with TemporaryDirectory(prefix='device-callback-') as tmp:
            output = Path(tmp) / 'placed.json'
            callback = r.device_restore_callback(trainer, output, 1, lambda f: f())
            control = object()
            moments = trainer._raw_optimizer().state[parameter]
            old_avg = moments['exp_avg']
            self.assertIs(callback.on_train_begin(None, None, control), control)
            self.assertIs(moments['exp_avg'], old_avg)
            record = json.loads(output.read_text())
            self.assertEqual(record['placement']['named_adam_counters'], 1)
            self.assertEqual(record['placement']['counters_moved'], 0)
            self.assertTrue(record['moments_not_reset'])
            with self.assertRaises(ValueError): callback.on_train_begin(None, None, control)

    def test_unrestored_optimizer_rejected(self):
        trainer, _ = trainer_fixture()
        trainer._raw_optimizer().completed_steps = 0
        with TemporaryDirectory(prefix='device-callback-') as tmp:
            output = Path(tmp) / 'placed.json'
            callback = r.device_restore_callback(trainer, output, 1, lambda f: f())
            with self.assertRaises(ValueError): callback.on_train_begin(None, None, None)
            self.assertFalse(output.exists())

    def test_wrong_counter_population_rejected(self):
        trainer, _ = trainer_fixture()
        with TemporaryDirectory(prefix='device-callback-') as tmp:
            output = Path(tmp) / 'placed.json'
            callback = r.device_restore_callback(trainer, output, 134, lambda f: f())
            with self.assertRaises(ValueError): callback.on_train_begin(None, None, None)
            self.assertFalse(output.exists())

    def test_wrong_counter_value_rejected(self):
        trainer, parameter = trainer_fixture()
        trainer._raw_optimizer().state[parameter]['step'].add_(1)
        with TemporaryDirectory(prefix='device-callback-') as tmp:
            output = Path(tmp) / 'placed.json'
            callback = r.device_restore_callback(trainer, output, 1, lambda f: f())
            with self.assertRaises(ValueError): callback.on_train_begin(None, None, None)
            self.assertFalse(output.exists())

    def test_component_change_rejected(self):
        trainer, _ = trainer_fixture()
        calls = iter([{'anchor': 'before'}, {'anchor': 'after'}])
        trainer.component_identity = lambda: next(calls)
        with TemporaryDirectory(prefix='device-callback-') as tmp:
            output = Path(tmp) / 'placed.json'
            callback = r.device_restore_callback(trainer, output, 1, lambda f: f())
            with self.assertRaises(ValueError): callback.on_train_begin(None, None, None)
            self.assertFalse(output.exists())

    def test_native_loader_and_comparator_not_overridden(self):
        tree = ast.parse(Path(r.__file__).read_text())
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        self.assertEqual([n.name for n in classes], ['PlaceRestoredAdamCounters'])
        functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        self.assertNotIn('_load_optimizer_and_scheduler', functions)
        self.assertNotIn('load_state_dict', functions)
        self.assertNotIn('_adamw_step', functions)
        old = ast.parse((r.ORIGINAL_RESUME / 'resume.py').read_text())
        original_comparison = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'recursive_equal')
        current_comparison = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'recursive_equal')
        self.assertEqual(ast.dump(original_comparison), ast.dump(current_comparison))

    def test_original_download_not_relabelled(self):
        real = r.read
        with patch.object(r, 'read') as reader:
            def changed(path, *args):
                value = real(path, *args)
                if Path(path).name == 'downloaded.json':
                    value['authorization_sha256'] = '0' * 64
                return value
            reader.side_effect = changed
            with self.assertRaises(ValueError): r.original_download()

    def test_no_download_cli_surface(self):
        tree = ast.parse(Path(r.__file__).read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        modes = [ast.literal_eval(k.value) for n in ast.walk(main) if isinstance(n, ast.Call)
                 for k in n.keywords if k.arg == 'choices' and isinstance(k.value, ast.List)]
        self.assertIn(['prepare', 'coordinate', 'worker', 'compare'], modes)
        self.assertFalse(any('download' in values for values in modes))


if __name__ == '__main__':
    unittest.main()
