"""Synthetic adapter refusal controls; no fixture is scientific admission."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('tested_native_primary', Path(__file__).with_name('native_primary.py'))
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


class Guards(unittest.TestCase):
    def setUp(self):
        self.native = {'whole_run_artifacts_verified': True, 'scientific_completion': False,
            'steps': [782, 1563, 2345, 3126, 3907], 'dataset_fingerprint': '0c29f5d460d1b4d7',
            'fixture_only': {'all_native_fields_compared': [1, 2, 3]}}
        self.accepted = {**copy.deepcopy(self.native), 'explicit_two_selection_view_audit_passed': True,
            'original_single_selection_guard_passed': False, 'input_view_audit_sha256': reader.VIEW_SHA}

    def test_synthetic_matching_fields(self):
        reader.compare_native_proof(self.native, self.accepted)

    def test_each_amendment_field_required(self):
        for key in ('explicit_two_selection_view_audit_passed', 'original_single_selection_guard_passed',
                    'input_view_audit_sha256'):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.accepted)
                del changed[key]
                with self.assertRaises(ValueError):
                    reader.compare_native_proof(self.native, changed)

    def test_content_mismatch_despite_matching_tag(self):
        changed = copy.deepcopy(self.accepted)
        changed['fixture_only']['all_native_fields_compared'][2] = 4
        with self.assertRaises(ValueError):
            reader.compare_native_proof(self.native, changed)

    def test_fabricated_old_guard_pass(self):
        self.accepted['original_single_selection_guard_passed'] = True
        with self.assertRaises(ValueError):
            reader.compare_native_proof(self.native, self.accepted)

    def test_incomplete_horizon(self):
        self.native['steps'].pop()
        with self.assertRaises(ValueError):
            reader.compare_native_proof(self.native, self.accepted)

    def test_no_fingerprint_substitution(self):
        for tag in ('0c6bd82f699a563c', '5a2cdf9a1ae149bc', 'other'):
            with self.subTest(tag=tag):
                self.native['dataset_fingerprint'] = self.accepted['dataset_fingerprint'] = tag
                with self.assertRaises(ValueError):
                    reader.compare_native_proof(self.native, self.accepted)

    def test_no_deep_reader_absence(self):
        self.native['whole_run_artifacts_verified'] = False
        with self.assertRaises(ValueError):
            reader.compare_native_proof(self.native, self.accepted)

    def test_no_scientific_scope_inflation(self):
        self.native['scientific_completion'] = True
        with self.assertRaises(ValueError):
            reader.compare_native_proof(self.native, self.accepted)

    def test_population_coverage(self):
        primary = SimpleNamespace(inputs={'runs': [{'run_id': str(i)} for i in range(12)]},
            payload={'evaluation': {'tasks': [str(i) for i in range(14)]}})
        runs = primary.inputs['runs']
        evaluations = [{'run_id': r['run_id'], 'stage': s} for r in runs for s in range(1, 6)]
        scores = [{**e, 'task': t} for e in evaluations for t in primary.payload['evaluation']['tasks']]
        reader.require_coverage(primary, runs, evaluations, scores)
        for family in ('runs', 'evaluations', 'scores'):
            for mutation in ('omit', 'duplicate', 'replace'):
                with self.subTest(family=family, mutation=mutation):
                    groups = copy.deepcopy({'runs': runs, 'evaluations': evaluations, 'scores': scores})
                    rows = groups[family]
                    if mutation == 'omit':
                        rows.pop()
                    elif mutation == 'duplicate':
                        rows[-1] = rows[0]
                    else:
                        rows[-1]['run_id'] = 'undeclared'
                    with self.assertRaises(ValueError):
                        reader.require_coverage(primary, **groups)


if __name__ == '__main__':
    unittest.main()
