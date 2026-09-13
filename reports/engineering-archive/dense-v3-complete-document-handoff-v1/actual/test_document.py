"""Document/consumer controls. Synthetic sign branches are never study outcomes."""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import author as a
import document_component as d


class DocumentControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from embed_optim.primary_v3_contract import PrimaryV3Contract
        primary = PrimaryV3Contract.load(a.STORY / 'configs/dense_primary_v3_protocol.json', a.STORY, a.PRIMARY)
        cls.constants = d.original.render_constants(primary)
        cls.main = (a.TEMPLATE / 'main.tex').read_text()
        from embed_optim import state_operator_factorial_publication as renderer
        cls.renderer = renderer
        cls.pending = (a.TEMPLATE / 'generated/state-operator-factorial.tex').read_bytes()
        cls.original_includes = {
            'generated/' + name: (a.ASSEMBLY / 'exact-publication' / name).read_bytes()
            for name in ('optimizer-primary.tex', 'dimension-utilization.tex')}
        cls.original_includes['generated/recipe-sensitivity.tex'] = (a.ASSEMBLY / 'recipe-sensitivity.tex').read_bytes()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='dense-document-control-')
        self.root = Path(self.directory.name)
        self.paper = self.root / 'paper'
        self.paper.mkdir()
        for name in d.BUILD_INPUTS:
            source = a.TEMPLATE / name
            target = self.paper / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        (self.paper / 'results.tex').write_bytes(self.constants)
        self.includes = {**self.original_includes, 'generated/state-operator-factorial.tex': self.pending}
        for name, value in self.includes.items():
            (self.paper / name).write_bytes(value)

    def tearDown(self):
        self.directory.cleanup()

    def fixture(self, signs=(0, 0, 0)):
        # Pure formatter coverage, in a disposable test directory only.
        rows = {}
        for name, sign in zip(self.renderer.ESTIMANDS, signs, strict=True):
            point, low, high = (0.0, -.001, .001) if not sign else (sign * .002, sign * .002 - .001, sign * .002 + .001)
            rows[name] = {'estimand': name, 'point_estimate': point,
                'bootstrap_ci_95_lower': low, 'bootstrap_ci_95_upper': high,
                'decision': {0: 'inconclusive', 1: 'supported_positive', -1: 'supported_negative'}[sign]}
        self.includes['generated/state-operator-factorial.tex'] = self.renderer._render_latex(rows).encode()
        (self.paper / 'generated/state-operator-factorial.tex').write_bytes(self.includes['generated/state-operator-factorial.tex'])

    def inspect(self):
        return d.inspect_sources(self.root, self.includes, self.constants)

    def main_edit(self, old, new):
        text = (self.paper / 'main.tex').read_text()
        self.assertIn(old, text)
        (self.paper / 'main.tex').write_text(text.replace(old, new))

    def test_actual_pending_draft_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unresolved active'):
            self.inspect()

    def test_all_twenty_seven_formatter_branches(self):
        for signs in itertools.product((-1, 0, 1), repeat=3):
            with self.subTest(signs=signs):
                self.fixture(signs)
                result = self.inspect()
                self.assertLessEqual(result['actual_abstract']['words_conservative'], 200)
                self.assertFalse(result['final_release_verified'])

    def test_clean_actual_constants_only(self):
        self.assertEqual(set(d.original.definitions(self.constants.decode())), set(d.original.CONSTANTS))

    def test_missing_control_include_rejected(self):
        self.fixture()
        self.main_edit(r'\input{generated/recipe-sensitivity}', '')
        with self.assertRaisesRegex(ValueError, 'input topology'):
            self.inspect()

    def test_unbound_external_figure_rejected(self):
        self.fixture()
        self.main_edit('figures/weight-to-retrieval-map.pdf', 'figures/unbound.pdf')
        with self.assertRaisesRegex(ValueError, 'external figure'):
            self.inspect()

    def test_hidden_control_table_rejected(self):
        self.fixture()
        self.main_edit(r'\WeightRecipeSensitivityTable', '')
        with self.assertRaises(ValueError):
            self.inspect()

    def test_duplicate_control_call_rejected(self):
        self.fixture()
        self.main_edit(r'\RecipeSensitivityFinding', r'\RecipeSensitivityFinding\RecipeSensitivityFinding')
        with self.assertRaisesRegex(ValueError, 'duplicated control'):
            self.inspect()

    def test_implementation_narrative_rejected(self):
        self.fixture()
        self.main_edit(r'\section{Introduction}', r'\section{Introduction} A padding problem occurred.')
        with self.assertRaisesRegex(ValueError, 'Implementation narrative'):
            self.inspect()

    def test_unbound_macro_override_rejected(self):
        self.fixture()
        self.main_edit(r'\begin{document}', r'\renewcommand{\CorrectedAbstractFinding}{Changed.}\begin{document}')
        with self.assertRaisesRegex(ValueError, 'redefine'):
            self.inspect()

    def test_literal_pending_in_main_rejected(self):
        self.fixture()
        self.main_edit(r'\section{Introduction}', r'\section{Introduction} PENDING')
        with self.assertRaisesRegex(ValueError, 'Unresolved'):
            self.inspect()

    def test_indirect_input_rejected(self):
        self.fixture()
        self.main_edit(r'\begin{document}', r'\input hidden.tex \begin{document}')
        with self.assertRaisesRegex(ValueError, 'Indirect'):
            self.inspect()

    def test_changed_generated_bytes_rejected(self):
        self.fixture()
        target = self.paper / 'generated/optimizer-primary.tex'
        target.write_bytes(target.read_bytes() + b'\nchanged\n')
        with self.assertRaisesRegex(ValueError, 'Generated evidence differs'):
            self.inspect()

    def test_historical_constant_rejected(self):
        self.fixture()
        self.constants = self.constants + b'\\newcommand{\\OldFinding}{Historical}\n'
        (self.paper / 'results.tex').write_bytes(self.constants)
        with self.assertRaises(ValueError):
            self.inspect()

    def test_overlong_abstract_rejected(self):
        self.fixture()
        self.main_edit(r'\begin{abstract}', r'\begin{abstract} ' + 'extra ' * 201)
        with self.assertRaisesRegex(ValueError, '200 words'):
            self.inspect()

    def test_actual_complete_caption_inventory(self):
        aux = (a.TEMPLATE / 'build/main.aux').read_text()
        observed = d.caption_inventory(aux, 8, 12)
        self.assertEqual((len(observed['figure']), len(observed['table'])), (4, 9))

    def test_ninth_table_cannot_be_dropped(self):
        aux = (a.TEMPLATE / 'build/main.aux').read_text()
        lines = [line for line in aux.splitlines() if not (r'\@writefile{lot}' in line and r'\numberline {9}' in line)]
        self.assertLess(len(lines), len(aux.splitlines()))
        with self.assertRaises(ValueError):
            d.caption_inventory('\n'.join(lines), 8, 12)

    def test_appendix_cannot_move_into_main(self):
        aux = (a.TEMPLATE / 'build/main.aux').read_text()
        with self.assertRaises(ValueError):
            d.caption_inventory(aux, 11, 12)

    def test_actual_float_layout_all_twelve_labels(self):
        result = d.layout(a.TEMPLATE, 12)
        self.assertEqual(result['main_end_page'], 8)
        self.assertEqual(len(result['main_float_pages']) + len(result['appendix_float_pages']), 12)

    def test_duplicate_json_rejected(self):
        target = self.root / 'duplicate.json'
        target.write_text('{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            a.read(target)

    def test_nonfinite_json_rejected(self):
        target = self.root / 'nan.json'
        target.write_text('{"x":NaN}')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            a.read(target)

    def test_changed_binding_rejected(self):
        target = self.root / 'data.json'
        target.write_text('{"x":1}')
        previous = a.identity(target)
        target.write_text('{"x":2}')
        with self.assertRaisesRegex(ValueError, 'Bound content'):
            a.read(target, previous)

    def test_symlink_input_rejected(self):
        link = self.root / 'linked.json'
        link.symlink_to(a.MAPS / 'completed.json')
        with self.assertRaisesRegex(ValueError, 'ordinary bound'):
            a.read(link)

    def test_copy_does_not_overwrite(self):
        target = self.root / 'copy.json'
        source = a.MAPS / 'completed.json'
        a.copy_file(source, target, a.identity(source))
        with self.assertRaises(FileExistsError):
            a.copy_file(source, target, a.identity(source))

    def test_compile_will_not_accept_pending(self):
        with self.assertRaisesRegex(ValueError, 'Unresolved'):
            d.compile_fresh(self.root, self.includes, self.constants)
        self.assertFalse((self.paper / 'build').exists())


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DocumentControls)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    a.write(a.HERE / 'tests.json', {'scope': 'document controls, not scientific evidence',
        'source': a.identity(__file__), 'component': a.identity(a.HERE / 'document_component.py'),
        'author': a.identity(a.HERE / 'author.py'), 'tests_run': result.testsRun,
        'synthetic_formatter_branches': 27, 'failures': len(result.failures),
        'errors': len(result.errors), 'skipped': len(result.skipped),
        'gpu_access': False, 'full_goal_complete': False})
    raise SystemExit(not result.wasSuccessful())
