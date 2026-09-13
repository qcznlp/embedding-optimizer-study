"""Bounded real-evidence and deliberate-corruption checks; never model execution."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import campaign_evidence as c

SOURCE = Path(__file__).parent / 'snapshot'


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.anchors = {name: c.read_local(SOURCE, f'anchors/{name}', sha, size)
                       for name, (size, sha) in c.ANCHORS.items()}
        cls.roles = c._roles(cls.anchors)
        cls.payloads = {name: c.read_local(SOURCE, name, b['sha256'], b.get('bytes'))
                        for name, b in cls.roles.items()}

    def semantic(self, run):
        anchors = copy.deepcopy(self.anchors)
        row = next(r for r in anchors['admission.json']['actual_completion']['rows'] if r['run_id'] == run)
        records = {kind: copy.deepcopy(self.payloads[f'receipts/{run}.{kind}.json']) for kind in
                   ('proof', 'started', 'terminated' if run in c.TERMINAL else 'exited')}
        native = {name.removeprefix(f'native/{run}/'): copy.deepcopy(value)
                  for name, value in self.payloads.items() if name.startswith(f'native/{run}/')}
        return row, anchors, records, native

    def alter_proof(self, case, mutate):
        row, anchors, records, native = case
        mutate(row['proof'])
        anchors['admission.json']['admitted']['complete_runs'][row['run_id']] = copy.deepcopy(row['proof'])
        records['proof'] = copy.deepcopy(row['proof'])
        with self.assertRaises((ValueError, KeyError)):
            c._validate_run(row, anchors, records, native)

    def test_real_full_population(self):
        result = c.read_complete_training_population(SOURCE)
        self.assertEqual((result['runs'], result['checkpoints'], result['files_authenticated']), (12, 60, 172))
        self.assertEqual(len(result['original_exit_zero_runs']), 10)
        self.assertEqual(len(result['original_exit_unobserved_runs']), 2)
        for field in ('scientific_completion', 'committed_source_release', 'fresh_model_or_data_validation',
                      'formal_primary_contract_admission', 'original_single_selection_guard_passed',
                      'original_absolute_paths_opened'):
            self.assertIs(result[field], False)
        c.same(result['original_completion_rows'], self.anchors['admission.json']['actual_completion']['rows'],
               'Returned provenance was rewritten')

    def test_both_native_proof_types(self):
        for run in c.RUNS:
            with self.subTest(run=run):
                c._validate_run(*self.semantic(run))

    def test_no_original_path_fallback(self):
        case = self.semantic(c.RUNS[0])
        case[0]['binding']['path'] = '/not-present-original-host/never-open-this.json'
        c._validate_run(*case)

    def test_terminal_exit_cannot_be_inferred(self):
        for value in (0, 1, False, 'unknown'):
            case = self.semantic('verified-v3-normuon-1e-4')
            case[0]['binding']['exit_code'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                c._validate_run(*case)

    def test_terminal_pid_reuse(self):
        case = self.semantic('verified-v3-normuon-3e-3')
        case[2]['terminated']['start_ticks'] += 1
        with self.assertRaises(ValueError):
            c._validate_run(*case)

    def test_terminal_forged_original_exit(self):
        self.alter_proof(self.semantic('verified-v3-normuon-1e-4'),
                         lambda p: p.update(original_supervisor_exit_receipt_present=True))

    def test_observed_nonzero_and_boolean_exit(self):
        for value in (1, None, False):
            case = self.semantic(c.RUNS[0]); case[2]['exited']['exit_code'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                c._validate_run(*case)

    def test_wrong_proof_type(self):
        self.alter_proof(self.semantic(c.RUNS[0]), lambda p: p.update(scope='dense_primary_correctness_v3'))

    def test_historical_flags_preserved(self):
        for field in ('scientific_completion', 'committed_source_release', 'original_single_selection_guard_passed'):
            with self.subTest(field=field):
                self.alter_proof(self.semantic(c.RUNS[0]), lambda p: p.update({field: True}))

    def test_wrong_actual_fingerprint(self):
        self.alter_proof(self.semantic(c.RUNS[0]), lambda p: p.update(dataset_fingerprint='5a2cdf9a1ae149bc'))

    def test_missing_equivalence_flag(self):
        self.alter_proof(self.semantic(c.RUNS[0]), lambda p: p.update(explicit_two_selection_view_audit_passed=False))

    def test_missing_checkpoint(self):
        self.alter_proof(self.semantic(c.RUNS[0]), lambda p: p['checkpoints'].pop())

    def test_missing_tensor_state_proof(self):
        self.alter_proof(self.semantic(c.RUNS[0]), lambda p: p['deep_checkpoint_checks'][0].update(parameter_states=133))

    def test_missing_optimizer_inventory(self):
        self.alter_proof(self.semantic(c.RUNS[0]), lambda p: p['checkpoints'][0].update(
            files=[f for f in p['checkpoints'][0]['files'] if f['path'] != 'optimizer.pt']))

    def test_changed_run_identity(self):
        case = self.semantic(c.RUNS[0]); case[0]['expected']['recipe']['temperature'] = .1
        with self.assertRaises(ValueError):
            c._validate_run(*case)

    def test_changed_saved_step(self):
        case = self.semantic(c.RUNS[0]); case[3]['checkpoint-782/trainer_state.json']['global_step'] = 783
        with self.assertRaises(ValueError):
            c._validate_run(*case)

    def test_changed_recorded_source(self):
        case = self.semantic(c.RUNS[0]); case[1]['source-assembly.json']['files']['src/embed_optim/train.py']['identity']['bytes'] += 1
        with self.assertRaises(ValueError):
            c._validate_run(*case)

    def test_anchor_population_order_and_missing_run(self):
        for mutation in (lambda rows: rows.reverse(), lambda rows: rows.pop(), lambda rows: rows.append(copy.deepcopy(rows[0]))):
            anchors = copy.deepcopy(self.anchors)
            mutation(anchors['admission.json']['actual_completion']['rows'])
            with self.assertRaises(ValueError):
                c._validate_anchors(anchors)

    def test_view_audit_not_sampled_or_rewritten(self):
        for field, value in [('rows', 499999), ('entire_arrow_tables_equal_including_metadata', False),
                             ('fingerprints', ['0c29f5d460d1b4d7'] * 3)]:
            anchors = copy.deepcopy(self.anchors)
            anchors['input-view-audit.json']['audit']['semantic_view'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                c._validate_anchors(anchors)

    def test_local_path_refusals(self):
        for name in ('../x', '/tmp/x', 'x/../y', './x', 'x//y', 'x\\y', 'x\0y', ''):
            with self.subTest(name=repr(name)), self.assertRaises(ValueError):
                c.relative(name)

    def test_duplicate_json_and_nonfinite(self):
        with tempfile.TemporaryDirectory(prefix='campaign-json-') as tmp:
            root = Path(tmp)
            for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
                (root / 'input.json').write_bytes(raw)
                with self.assertRaises(ValueError):
                    c.read_local(root, 'input.json', hashlib.sha256(raw).hexdigest())

    def test_file_and_parent_symlinks(self):
        with tempfile.TemporaryDirectory(prefix='campaign-links-') as tmp:
            root = Path(tmp); (root / 'file.json').symlink_to(SOURCE / 'anchors/input-view-audit.json')
            (root / 'parent').symlink_to(SOURCE / 'anchors', target_is_directory=True)
            for name in ('file.json', 'parent/input-view-audit.json'):
                with self.subTest(name=name), self.assertRaises((ValueError, OSError)):
                    c.read_local(root, name, c.ANCHORS['input-view-audit.json'][1])
            with self.assertRaises(ValueError):
                c.inventory(root)

    def test_full_reader_tree_refusals_and_relocation(self):
        with tempfile.TemporaryDirectory(prefix='campaign-relocated-') as tmp:
            root = Path(tmp) / 'evidence'; shutil.copytree(SOURCE, root)
            c.same(c.read_complete_training_population(root), c.read_complete_training_population(SOURCE),
                   'Local-role relocation changed evidence')
            mutations = {
                'extra_file': lambda: (root / 'unexpected.json').write_text('{}'),
                'extra_empty_directory': lambda: (root / 'unexpected-dir').mkdir(),
                'changed_metadata': lambda: (root / f'native/{c.RUNS[0]}/completed.json').write_text('{}'),
                'changed_receipt': lambda: (root / f'receipts/{c.RUNS[0]}.exited.json').write_text('{}'),
                'changed_anchor': lambda: (root / 'anchors/input-view-audit.json').write_text('{}'),
                'missing_receipt': lambda: (root / f'receipts/{c.RUNS[0]}.exited.json').unlink(),
                'fake_terminal_exit': lambda: (root / 'receipts/verified-v3-normuon-1e-4.exited.json').write_text('{"exit_code":0}'),
            }
            for label, mutation in mutations.items():
                # Each deliberately corrupt copy is separate and never changes source evidence.
                saved = Path(tmp) / label; shutil.copytree(SOURCE, saved)
                root = saved
                mutation()
                with self.subTest(case=label), self.assertRaises((ValueError, OSError)):
                    c.read_complete_training_population(root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
