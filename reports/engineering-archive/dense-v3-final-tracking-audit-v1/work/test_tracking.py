"""Small synthetic refusal controls, separate from the 24 actual observations."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
import tracking


def fixture():
    local = {'global_step': 20, 'max_steps': 20, 'epoch': 1,
             'log_history': [{'step': s, 'loss': .25, 'grad_norm': 1.0,
                              'learning_rate': lr} for s, lr in ((1, 0.0), (10, .01 * (11 / 18)), (20, .01 / 18))]}
    remote = [{'train/' + k if k != 'step' else 'train/global_step': v for k, v in r.items()}
              for r in local['log_history']]
    return local, remote


class TrackingControls(unittest.TestCase):
    def test_exact(self):
        a, b = fixture()
        r = tracking.metric_comparison(a, b, 20, .01)
        self.assertTrue(r['exact_native_online_match'])
        self.assertTrue(r['native_lr_schedule_exact'])

    def test_missing_row(self):
        a, b = fixture()
        with self.assertRaises(ValueError):
            tracking.metric_comparison(a, b[:-1], 20, .01)

    def test_duplicate_row(self):
        a, b = fixture()
        b[-1] = b[-2]
        with self.assertRaises(ValueError):
            tracking.metric_comparison(a, b, 20, .01)

    def test_wrong_order(self):
        a, b = fixture()
        with self.assertRaises(ValueError):
            tracking.metric_comparison(a, b[::-1], 20, .01)

    def test_boolean_step(self):
        a, b = fixture()
        b[0]['train/global_step'] = True
        with self.assertRaises(ValueError):
            tracking.metric_comparison(a, b, 20, .01)

    def test_nonfinite(self):
        a, b = fixture()
        b[1]['train/loss'] = math.nan
        with self.assertRaises(ValueError):
            tracking.metric_comparison(a, b, 20, .01)

    def test_boolean_metric(self):
        a, b = fixture()
        b[1]['train/grad_norm'] = True
        with self.assertRaises(ValueError):
            tracking.metric_comparison(a, b, 20, .01)

    def test_native_endpoint(self):
        a, b = fixture()
        a['global_step'] = 19
        with self.assertRaises(ValueError):
            tracking.metric_comparison(a, b, 20, .01)

    def test_one_ulp_is_not_exact(self):
        a, b = fixture()
        b[1]['train/loss'] = math.nextafter(.25, 1.0)
        r = tracking.metric_comparison(a, b, 20, .01)
        self.assertFalse(r['exact_native_online_match'])
        self.assertEqual(r['max_online_ulp_distance'], 1)

    def test_schedule_is_independent(self):
        a, b = fixture()
        a['log_history'][1]['learning_rate'] *= 2
        b[1]['train/learning_rate'] *= 2
        r = tracking.metric_comparison(a, b, 20, .01)
        self.assertTrue(r['exact_native_online_match'])
        self.assertFalse(r['native_lr_schedule_exact'])

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory(prefix='dense-tracking-control-') as temp:
            p = Path(temp) / 'output.json'
            tracking.write(p, {'synthetic': True})
            with self.assertRaises(FileExistsError):
                tracking.write(p, {'synthetic': False})

    def test_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory(prefix='dense-tracking-control-') as temp:
            p = Path(temp) / 'input.json'
            p.write_text('{"x":1,"x":2}')
            with self.assertRaises(ValueError):
                tracking.read(p)


if __name__ == '__main__':
    unittest.main(verbosity=2)
