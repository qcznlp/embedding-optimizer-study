"""Reconstruct actual scientific result fragments, never install or admit a paper.

Only the explicitly copied inputs are read. Historical admission/authoring entry
points are not called. The original bridge numerical functions are replayed;
other already verified statistics remain authenticated inputs, not new analyses.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

MANIFEST_SHA = 'fe7256289ffd3eb2ff3d209bff4420fef2e5751ac4cb2bc0449da7b10860de4c'
STEPS = (782, 1563, 2345, 3126, 3907)
COMPARATORS = (('B0_original', False), ('B1_raw_rate', False),
               ('B2_optimizer_stage_rate', False), ('B3_nominal_schedule', False),
               ('B0_original', True), ('B3_nominal_schedule', True))
FUNCTIONAL_LABELS = {
    'margin_helpful_mass_share': 'Helpful mass share',
    'margin_helpful_attribution_participation_ratio': 'Helpful participation',
    'margin_degrading_attribution_mass': 'Degrading mass',
    'random_50pct_relative_shortlist_ndcg': '50% removal retention',
}


def need(ok, why):
    if not ok:
        raise ValueError(why)


def identity(path):
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Nonordinary input')
    before = path.stat()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need(all(getattr(before, k) == getattr(after, k) for k in
             ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')), 'Input changed')
    return dict(bytes=after.st_size, sha256=sha)


def read(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            need(key not in out, 'Duplicate JSON key')
            out[key] = value
        return out
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(path.read_bytes(), object_pairs_hook=unique, parse_constant=invalid)


def authenticate(bundle):
    need(identity(bundle / 'inputs.json')['sha256'] == MANIFEST_SHA, 'Rendering input manifest differs')
    manifest = read(bundle / 'inputs.json')
    for name, row in manifest['records'].items():
        path = Path(name)
        need(not path.is_absolute() and '..' not in path.parts and path.as_posix() == name,
             'Noncanonical input path')
        need(identity(bundle / path) == row['binding'], 'Bound rendering input differs: ' + name)
    return manifest


def source_functions(bundle, filename, names, dependencies):
    path = bundle / 'source-original/src/embed_optim' / (filename + '.py')
    tree = ast.parse(path.read_bytes())
    picked, seen = [], set()
    for node in tree.body:
        key = [node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else (
            [t.id for t in node.targets if isinstance(t, ast.Name)] if isinstance(node, ast.Assign) else [])
        if any(name in names for name in key):
            picked.append(node)
            seen.update(key)
    need(seen == set(names), 'Original function inventory differs: ' + filename)
    namespace = dict(math=math, statistics=statistics, defaultdict=defaultdict, Counter=Counter,
                     Fraction=Fraction, **dependencies)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *picked], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


def table(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def checked_controls(result, labels, expected):
    """All cells and exact strict flags, never selecting a favorable baseline."""
    summaries, folds = result['summaries'], result['folds']
    wanted = {(b, c, f) for b, c in COMPARATORS for f in labels}
    actual = Counter((row['baseline'], row['conditioned_on_displacement'], row['feature'])
                     for row in summaries)
    need(actual == Counter(wanted) and len(summaries) == expected,
         'Incomplete/duplicated recipe-control population')
    need(len(folds) == 4 * expected, 'Incomplete recipe-control folds')
    for row in summaries:
        need(row['total_folds'] == row['defined_folds'] == 4, 'Unresolved control must not be hidden')
        group = [f for f in folds if (f['baseline'], f['conditioned_on_displacement'], f['feature'])
                 == (row['baseline'], row['conditioned_on_displacement'], row['feature'])]
        need(len(group) == 4 and len({f['fold'] for f in group}) == 4,
             'Incomplete/duplicated held-dose folds')
        improvements = []
        for fold in group:
            better = Fraction(fold['feature_mse_exact']) < Fraction(fold['baseline_mse_exact'])
            need(fold['improves'] is better and fold['train_rows'] == 45 and fold['test_rows'] == 15,
                 'Fold comparison or held-out population differs')
            improvements.append(better)
        need(row['improved_folds'] == sum(improvements), 'Improved-fold count differs')
        for field in ('baseline', 'feature'):
            pooled = sum((Fraction(f[field + '_mse_exact']) for f in group), Fraction()) / 4
            need(pooled == Fraction(row['pooled_' + field + '_mse_exact']), 'Pooled exact MSE differs')
        lower = Fraction(row['pooled_feature_mse_exact']) < Fraction(row['pooled_baseline_mse_exact'])
        need(row['diagnostic_flag'] is (lower and row['improved_folds'] >= 3), 'Exact predictive flag differs')
        for mse, rmse in (('pooled_feature_mse_exact', 'pooled_feature_rmse'),
                          ('pooled_baseline_mse_exact', 'pooled_baseline_rmse')):
            need(math.isclose(math.sqrt(float(Fraction(row[mse]))), row[rmse], rel_tol=1e-14),
                 'Control RMSE does not match exact MSE')
    indexed = {(r['baseline'], r['conditioned_on_displacement'], r['feature']): r for r in summaries}
    survivors = [f for f in labels if all(indexed[b, c, f]['diagnostic_flag'] for b, c in COMPARATORS[:4])]
    return dict(rows=summaries, survivors=survivors, indexed=indexed)


def controls_tex(weight, functional, weight_labels, escape):
    def render_one(info, labels, macro, label):
        rows = []
        for feature, name in labels.items():
            cells = []
            for b, c in COMPARATORS:
                row = info['indexed'][b, c, feature]
                value = row['pooled_feature_rmse'] * 100
                mark = '*' if row['diagnostic_flag'] else ''
                cells.append(f"{value:.3f}{mark} ({row['improved_folds']}/4)")
            rows.append(escape(name) + ' & ' + ' & '.join(cells) + r' \\')
        baseline = []
        for b, c in COMPARATORS:
            values = {info['indexed'][b, c, f]['pooled_baseline_rmse'] for f in labels}
            need(len(values) == 1, 'Control family uses inconsistent baseline outcomes')
            baseline.append(f"{values.pop() * 100:.3f}")
        return ('\\newcommand{\\' + macro + '}{%\n'
                '\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{2pt}\n'
                '\\begin{tabular}{p{4.8cm}rrrrrr}\\toprule\n'
                'Measurement & B0 & B1 & B2 & B3 & B0+D & B3+D \\\\\n\\midrule\n'
                'Comparator alone & ' + ' & '.join(baseline) + ' \\\\\n\\midrule\n'
                + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n'
                '\\caption{Complete post-result exploratory recipe-control sensitivity. '
                'Cells show pooled held-dose RMSE in nDCG@10 points (lower is better), '
                'and the number of improved folds. An asterisk marks strictly lower exact '
                'pooled MSE and improvement in at least three of four folds; it is not a '
                'significance test. D is cumulative weight displacement. All six comparator '
                'variants are retained.}\\label{' + label + '}\n\\end{table*}%\n}\n')
    counts = (len(weight['survivors']), len(functional['survivors']))
    finding = (f"In post-result exploratory controls, {counts[0]} of 14 weight measurements and "
               f"{counts[1]} of four functional measurements meet the predictive criterion under "
               "all four recipe-only comparators. Predictive findings depend on the comparator; "
               "we retain every baseline rather than select the one with lowest observed error. "
               "These controls neither identify a mediator nor establish equivalence.")
    return ('% Generated from all actual exploratory comparisons; no model admission.\n'
            '\\newcommand{\\RecipeSensitivityFinding}{' + escape(finding) + '}\n'
            + render_one(weight, weight_labels, 'WeightRecipeSensitivityTable', 'tab:weight-recipe-sensitivity')
            + render_one(functional, FUNCTIONAL_LABELS, 'FunctionalRecipeSensitivityTable', 'tab:functional-recipe-sensitivity'))


def generate(bundle):
    manifest = authenticate(bundle)
    path = bundle / 'source/weight_readout.py'
    spec = importlib.util.spec_from_file_location('_manuscript_weight_numeric', path)
    numeric = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = numeric
    spec.loader.exec_module(numeric)
    kernels = numeric.pure_kernels(bundle / 'source-original')
    core = source_functions(bundle, 'corrected_publication',
        ['OPTIMIZERS', 'CONTRASTS', 'LABELS', 'FEATURE_LABELS', '_finite', '_validate_contrasts',
         '_validate_optimizer_stage', '_final_geometry', '_ci', '_tex_escape',
         'build_corrected_finding', 'build_weight_space_finding'], {})
    bridge = source_functions(bundle, 'primary_v3_publication_bridge',
        ['checked_bridge', 'delta_text', 'finding', 'render_bridge'],
        {**core, 'evaluate': kernels['numerical']['evaluate'], 'require_same': lambda a, b: need(a == b, 'Original bridge replay differs')})
    render = source_functions(bundle, 'primary_v3_publication_render',
        ['FUNCTIONAL_TABLE_LABEL', 'FUNCTIONAL_TABLE_CAPTION', 'bind_functional_table',
         'conclusion', 'render', 'findings'], {**core, 'bridge_finding': bridge['finding']})
    weight = read(bundle / 'inputs/weight/tables.json')
    bridge_display = bridge['render_bridge'](weight['original'])
    trajectory = bundle / 'inputs/trajectory'
    geometry = kernels['parser']['typed_csv'](bundle / 'inputs/weight/checkpoint_geometry.csv')
    # Complete population joins, not a best-cell-only excerpt.
    panel = weight['original']['bridge_rows']
    expected = {(r['run_id'], int(r['stage'])) for r in panel}
    need(len(expected) == len(panel) == 60, 'Incomplete bridge panel')
    runs = {r['run_id'] for r in panel}
    need(len(runs) == 12 and expected == {(run, stage) for run in runs for stage in range(1, 6)},
         'Incomplete run-stage grid')
    need({(r['run_id'], int(r['stage'])) for r in geometry} == expected, 'Geometry/retrieval population differs')
    scores = table(trajectory / 'all_task_scores.csv')
    need(len(scores) == 840 and len({(r['run_id'], r['stage'], r['task']) for r in scores}) == 840,
         'Incomplete full-task population')
    need({(r['run_id'], int(r['stage'])) for r in scores} == expected, 'Score population differs')
    functional = read(bundle / 'inputs/functional/tables.json')
    need(len(functional['primary_contrasts']) == 9 and len(functional['rotation_contrasts']) == 27
         and len(functional['held_out_predictions']) == 240, 'Incomplete functional statistics')
    need({(r['run_id'], int(r['stage'])) for r in functional['bridge_rows']} == expected,
         'Functional/weight population differs')
    summary = dict(primary=core['_validate_contrasts'](table(trajectory / 'primary_summary.csv'), secondary=False),
        secondary=core['_validate_contrasts'](table(trajectory / 'secondary_summary.csv'), secondary=True),
        optimizer_stage=core['_validate_optimizer_stage'](table(trajectory / 'optimizer_stage_scores.csv')),
        final_geometry=core['_final_geometry'](geometry), bridge=bridge_display['rows'], bridge_latex=bridge_display['latex'])
    raw_primary = render['render'](summary)
    # Exact counterpart formatting uses all five already independently checked comparisons.
    exact_render = source_functions(bundle, 'primary_v3_exact_publication_render',
        ['LABELS', 'INTERPRETATION', 'WRAPPER', 'finding', 'render', 'combine'],
        {'_tex_escape': core['_tex_escape'], 'delta_text': bridge['delta_text'],
         'require_same': lambda a, b: need(a == b, 'Exact rendering population differs')})
    exact = weight['exact']
    old = {r['feature']: r for r in bridge_display['rows']}
    comparisons = {r['exact_feature']: r for r in exact['predictive_sensitivity_summary']}
    associations = {r['feature']: r for r in exact['residual_associations']}
    summaries = {r['feature']: r for r in exact['feature_prediction_summary']}
    need(set(comparisons) == set(associations) == set(summaries) == set(exact_render['LABELS']),
         'Incomplete exact counterparts')
    rows = []
    for feature, label in exact_render['LABELS'].items():
        row, comparison = summaries[feature], comparisons[feature]
        status = Counter(r['status'] for r in exact['leave_dose_fold_metrics'] if r['feature'] == feature)
        criterion = 'undefined' if row['predictively_useful'] is None else 'supported' if row['predictively_useful'] else 'not supported'
        if status == {'baseline_equivalent': 4}:
            criterion = 'baseline equivalent'
        rows.append(dict(feature=feature, label=label, original_feature=comparison['original_feature'],
            original=old[comparison['original_feature']], comparison=comparison,
            exact={**row, 'criterion': criterion, 'fold_status_counts': dict(sorted(status.items())),
                   'association': associations[feature]}))
    exact_summary = dict(rows=rows, interpretation=exact_render['INTERPRETATION'])
    exact_tex = exact_render['render'](exact_summary)
    functional_text = (bundle / 'inputs/functional/results.tex').read_text()
    exact_control_labels = {feature: 'Exact: ' + label for feature, label in exact_render['LABELS'].items()}
    exact_control_labels['full_spectrum_nonzero_saved_segment_entropy_rank_fraction'] = 'Full-spectrum segment entropy'
    weight_labels = {**core['FEATURE_LABELS'], **exact_control_labels}
    weight_controls = checked_controls(read(bundle / 'inputs/weight_controls/result.json'), weight_labels, 84)
    functional_controls = checked_controls(read(bundle / 'inputs/functional_controls/result.json'), FUNCTIONAL_LABELS, 24)
    tex = controls_tex(weight_controls, functional_controls, weight_labels, core['_tex_escape'])
    findings = render['findings'](summary)
    findings['exact_geometry_prediction'] = exact_render['finding'](rows)
    findings['recipe_controls'] = dict(weight_all_four=weight_controls['survivors'],
                                     functional_all_four=functional_controls['survivors'], exploratory=True)
    values = {'primary-summary.json': summary, 'exact-summary.json': exact_summary, 'findings.json': findings}
    outputs = {name: (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
               for name, value in values.items()}
    outputs.update({'original-optimizer-primary.tex': raw_primary.encode(),
        'optimizer-primary.tex': exact_render['combine'](raw_primary.encode(), exact_tex.encode()),
        'exact-sensitivity.tex': exact_tex.encode(), 'functional-inference.tex': functional_text.encode(),
        'dimension-utilization.tex': render['bind_functional_table'](functional_text).encode(),
        'recipe-sensitivity.tex': tex.encode()})
    for name, content in outputs.items():
        if name.endswith('.tex'):
            need(not any(s in content.lower() for s in (b'packing', b'padding', b'execution incident', b'resultpending')),
                 'Nonscientific or pending text in completed fragment')
    need(not any(name == 'embed_optim' or name.startswith('embed_optim.') for name in sys.modules),
         'Unexpected project-package import')
    need(authenticate(bundle) == manifest, 'Inputs changed during formatting')
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only renderer must hide CUDA')
    need(args.bundle.is_absolute() and args.output.is_absolute(), 'Use absolute paths')
    need(not any(p.is_symlink() for p in (args.output, *args.output.parents)), 'Symlinked destination')
    need(args.output != args.bundle and args.bundle not in args.output.parents, 'Output overlaps source bundle')
    outputs = generate(args.bundle)
    if args.verify:
        need(args.output.is_dir() and {p.name for p in args.output.iterdir()} == {*outputs, 'verification.json'},
             'Generated inventory differs')
        for name, content in outputs.items():
            identity(args.output / name)
            need((args.output / name).read_bytes() == content, 'Freshly reconstructed bytes differ: ' + name)
    else:
        need(not args.output.exists(), 'Output already exists; never overwrite results')
        args.output.mkdir(parents=True, exist_ok=False)
        for name, content in outputs.items():
            with (args.output / name).open('xb') as stream:
                stream.write(content)
    receipt = dict(scope='actual-scientific-result-fragment-reconstruction', input_manifest_sha256=MANIFEST_SHA,
        renderer=identity(Path(__file__)), outputs={n: identity(args.output / n) for n in outputs},
        full_primary_60_states_840_tasks_retained=True, all_9_original_bridge_features_replayed=True,
        other_statistics_recomputed=False, original_functional_latex_reused_with_origin=True,
        all_5_exact_counterparts_retained=True, all_84_and_24_exploratory_controls_retained=True,
        numerical_input_admission_not_relabelled=True, model_admission=False,
        manuscript_installed=False, source_publication=False, scientific_completion=False)
    if args.verify:
        need(read(args.output / 'verification.json') == receipt, 'Reconstructed receipt differs')
    else:
        with (args.output / 'verification.json').open('x') as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(generated_files=len(outputs), output=str(args.output), verified=args.verify,
                          receipt=identity(args.output / 'verification.json'))))


if __name__ == '__main__':
    main()
