"""Compiler-only controls: actual primary text, explicitly synthetic factorial cells.

No authoring/summary contract is manufactured. These disposable document-component
outputs cannot enter author.py, the authoritative paper, or scientific results.
"""
import itertools
from pathlib import Path

import author as a
import document_component as d
from embed_optim import state_operator_factorial_publication as renderer

auth = a.read(a.HERE / 'authorization.json', 'ca7a0503b9324d51ce62a9c19cd1ec380e06311a0ff498b84ea30e683670e0e4')
base = a.HERE / 'layout-controls-NOT-SCIENTIFIC-RESULTS'
base.mkdir(exist_ok=False)
observations = []
for label, signs in (('all-positive', (1, 1, 1)), ('all-negative', (-1, -1, -1))):
    root = base / label
    root.mkdir()
    a.write(root / 'TEST-ONLY.json', {'synthetic_factorial_cells': True,
        'purpose': 'document-component layout control, never scientific admission',
        'full_goal_complete': False})
    for name, value in auth['paper_inputs'].items():
        a.copy_file(value['path'], root / 'paper' / name, value)
    rows = {name: {'estimand': name, 'point_estimate': sign * .002,
        'bootstrap_ci_95_lower': sign * .002 - .001, 'bootstrap_ci_95_upper': sign * .002 + .001,
        'decision': 'supported_positive' if sign > 0 else 'supported_negative'}
        for name, sign in zip(renderer.ESTIMANDS, signs, strict=True)}
    with (root / 'paper/generated/state-operator-factorial.tex').open('xb') as stream:
        stream.write(renderer._render_latex(rows).encode())
    expected = {name: (root / 'paper' / name).read_bytes() for name in d.RESULT_FILES}
    constants = (root / 'paper/results.tex').read_bytes()
    try:
        result = d.compile_fresh(root, expected, constants)
        observations.append({'case': label, 'main_end_page': result['layout']['main_end_page'],
            'abstract_words': result['source_inspection']['actual_abstract']['words_conservative'],
            'pdf_pages': result['pdf_pages'], 'font_count': result['font_count'],
            'document_component_passed': True, 'upstream_scientific_admission_performed': False})
    except BaseException as error:
        a.write(root / 'failed.json', {'exception_type': type(error).__name__, 'message': str(error),
            'synthetic_factorial_cells': True, 'scientific_result': False})
        raise
a.write(base / 'observations.json', {'source': a.identity(__file__),
    'cases': observations, 'synthetic_factorial_cells': True, 'scientific_result': False,
    'real_factorial_complete': False, 'authoritative_manuscript_installed': False})
print(observations, flush=True)
