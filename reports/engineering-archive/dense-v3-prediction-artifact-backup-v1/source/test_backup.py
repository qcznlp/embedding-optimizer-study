"""Focused transport and offline reconstruction controls; no network or model calls."""
import copy
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

WORK = Path(__file__).parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


b = module('backup')
r = module('verify_recovered')
t = b.load_transport()
STAGING = WORK / 'staging'
SHA = '52930a02e24ec8e0d622aa6dddcd6b379dd41f8a3ae7bd33b3263fa58fcd7b3d'


class BackupTests(unittest.TestCase):
    def test_full_actual_reconstruction(self):
        result = r.verify(STAGING, SHA)
        self.assertEqual((result['files'], result['locked_predictions'], result['sensitivity_predictions'],
                          result['exact_fold_errors_reconstructed'], result['plot_points_matched']),
                         (47, 840, 5040, 392, 98))

    def test_same_complete_population_independently_defined(self):
        self.assertEqual(b.payload_names(), r.names())
        self.assertEqual(len(b.select_sources(t)), 46)
        b.prepared(t, WORK)

    def test_no_repeat_prepare(self):
        with self.assertRaisesRegex(ValueError, 'Preserve existing staging'):
            b.prepare(t, WORK)

    def test_unselected_source_refused(self):
        with self.assertRaisesRegex(ValueError, 'Unselected payload'):
            b.inspect_payload(t, 'source.py', STAGING/'README.md', {})

    def test_wrong_manifest_refused(self):
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            r.inventory(STAGING, '0'*64)

    def test_relative_snapshot_refused(self):
        with self.assertRaisesRegex(ValueError, 'absolute directory'):
            r.inventory(Path('staging'), SHA)

    def test_extra_snapshot_file_refused(self):
        with tempfile.TemporaryDirectory(prefix='prediction-backup-test-') as directory:
            root = Path(directory)/'snapshot'; shutil.copytree(STAGING, root)
            (root/'unselected.py').write_text('not published\n')
            with self.assertRaisesRegex(ValueError, 'extra payload'):
                r.inventory(root, SHA)

    def test_missing_snapshot_file_refused(self):
        with tempfile.TemporaryDirectory(prefix='prediction-backup-test-') as directory:
            root = Path(directory)/'snapshot'
            shutil.copytree(STAGING, root, ignore=shutil.ignore_patterns('predictions.csv'))
            with self.assertRaisesRegex(ValueError, 'extra payload'):
                r.inventory(root, SHA)

    def test_symlink_refused(self):
        with tempfile.TemporaryDirectory(prefix='prediction-backup-test-') as directory:
            path = Path(directory)/'link'; path.symlink_to(STAGING/'README.md')
            with self.assertRaisesRegex(ValueError, 'symlinked'):
                r.identity(path)

    def test_changed_original_data_refused(self):
        with tempfile.TemporaryDirectory(prefix='prediction-backup-test-') as directory:
            path = Path(directory)/'changed.csv'; path.write_text('wrong\n')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                r.bound(path, r.identity(STAGING/'locked/actual/original/bridge_rows.csv'))

    def test_embedded_source_refused(self):
        with tempfile.TemporaryDirectory(prefix='prediction-backup-test-') as directory:
            path = Path(directory)/'source.json'
            path.write_text(json.dumps({'payload': 'def executable():\n    return 1\n'}))
            with self.assertRaisesRegex(ValueError, 'Embedded executable'):
                t.scan_text(path)

    def test_credential_field_refused(self):
        with tempfile.TemporaryDirectory(prefix='prediction-backup-test-') as directory:
            path = Path(directory)/'credential.json'; path.write_text('{"access_token":"synthetic-not-a-real-token"}')
            with self.assertRaisesRegex(ValueError, 'Credential-bearing'):
                t.scan_text(path)

    def test_svg_actual_inline_pngs_accepted(self):
        for path in (STAGING/'locked/figures/held_dose_prediction.svg',
                     STAGING/'sensitivity/figures/comparator_sensitivity.svg'):
            b.inspect_svg(path.read_bytes())

    def test_svg_active_and_external_refused(self):
        for body in ('<script/>', '<foreignObject/>', '<image href="https://invalid.example/a.png"/>',
                     '<image href="data:text/html;base64,Zm9v"/>', '<rect onclick="bad()"/>',
                     '<style>@import url(https://invalid.example/style)</style>',
                     '<image href="data:image/png;base64,YmFk"/>'):
            with self.subTest(body=body), self.assertRaises(ValueError):
                b.inspect_svg(('<svg xmlns="http://www.w3.org/2000/svg">'+body+'</svg>').encode())

    def test_modes_require_exact_population(self):
        with self.assertRaisesRegex(ValueError, 'population'):
            b.check_modes({}, 'prefix')

    def test_modes_only_two_original_png_lfs(self):
        modes = {'prefix/'+n: {'mode': 'lfs' if n.endswith('.png') else 'regular',
                  'ignored': False, 'remote_oid': None} for n in b.payload_names()}
        b.check_modes(modes, 'prefix')
        for field, value in (('ignored', True), ('remote_oid', 'existing-object'), ('mode', 'lfs')):
            with self.subTest(field=field):
                bad = copy.deepcopy(modes); bad['prefix/README.md'][field] = value
                with self.assertRaisesRegex(ValueError, 'Unexpected upload mode'):
                    b.check_modes(bad, 'prefix')

    def test_preservation_rejects_old_content_or_extra_subtree(self):
        before = {b.NAMESPACE: {'kind': 'RepoFolder'}, 'README.md': {'sha': 'old'}}
        sub = {b.NAMESPACE+'/old': {'sha': 'old'}}
        after_sub = {**sub, b.ADDITION: {'sha': 'new'}}
        b.check_preservation(before, copy.deepcopy(before), sub, after_sub)
        bad = copy.deepcopy(before); bad['README.md']['sha'] = 'new'
        with self.assertRaisesRegex(ValueError, 'Old root'):
            b.check_preservation(before, bad, sub, after_sub)
        with self.assertRaisesRegex(ValueError, 'Unexpected corrected'):
            b.check_preservation(before, before, sub, {**after_sub, 'extra': {}})

    def test_bad_predictions_folds_and_decisions_refused(self):
        tables = r.read(STAGING/'locked/actual/tables.json')
        panel = tables['original']; states = {(x['run_id'], x['stage']): x for x in panel['bridge_rows']}
        for kind in ('missing', 'duplicate', 'outcome', 'prediction', 'fold', 'decision'):
            with self.subTest(kind=kind):
                p, f, s = (copy.deepcopy(panel[k]) for k in
                           ('held_out_predictions', 'leave_dose_fold_metrics', 'feature_prediction_summary'))
                if kind == 'missing': p.pop()
                elif kind == 'duplicate': p[-1] = p[0]
                elif kind == 'outcome': p[0]['observed'] += .01
                elif kind == 'prediction': p[0]['feature_prediction_exact'] = '0'
                elif kind == 'fold': f[0]['feature_mse_exact'] = '0'
                elif kind == 'decision': s[0]['predictively_useful'] = not s[0]['predictively_useful']
                with self.assertRaises(ValueError):
                    r.check_predictions(p, f, s, states, False)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BackupTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {'scope': 'bounded-actual-prediction-backup-controls', 'tests': result.testsRun,
               'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
               'actual_staged_snapshot_reconstructed': result.wasSuccessful(),
               'remote_mutations': False, 'model_execution': False}
    with (WORK/'tests.json').open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
