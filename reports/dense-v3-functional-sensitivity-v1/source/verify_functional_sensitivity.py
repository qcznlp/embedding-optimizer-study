"""Independent full-matrix rational verification of the 24 functional comparisons."""
import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp

PARENT = '608965f5297fc196adcce6ca7417bd40e7f35d05e26133f20c0811dd1fd7027e'
ORACLE = '176ce4c6134ac728351f9099c5529ef54f84f59abe12efc111c0e7808c0f7100'
STEPS = (782, 1563, 2345, 3126, 3907)
OPTIMIZERS = ('adamw', 'muon', 'normuon')
BASELINES = ('B0_original', 'B1_raw_rate', 'B2_optimizer_stage_rate', 'B3_nominal_schedule')
DISPLACEMENT = 'cumulative_displacement_to_weight_ratio'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def bound(path, expected):
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Missing/symlinked evidence')
    raw = path.read_bytes()
    got = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    need(got['sha256'] == expected if isinstance(expected, str) else got == expected, 'Identity differs')
    return json.loads(raw) if path.suffix == '.json' else path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'analysis', 'oracle', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--receipt-sha256', required=True)
    a = parser.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not a.output.exists(), 'CPU-only new output')
    parent = bound(a.parent / 'readout.json', PARENT)
    prior = bound(a.parent / 'tables.json', parent['payloads']['tables.json'])
    receipt = bound(a.analysis / 'readout.json', a.receipt_sha256)
    need(receipt['parent_readout_sha256'] == PARENT and receipt['exploratory_post_result'] is True,
         'Different declared scope')
    for name, binding in receipt['outputs'].items():
        bound(a.analysis / name, binding)
    result = json.loads((a.analysis / 'result.json').read_text())
    bound(a.oracle, ORACLE)
    spec = importlib.util.spec_from_file_location('_original_independent_full_OLS', a.oracle)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    rows = prior['bridge_rows']
    features = [r['feature'] for r in prior['feature_prediction_summary']]
    need(len(rows) == 60 and len(set(features)) == len(features) == 4, 'Different actual panel')
    y = [oracle.q(r['mean_ndcg_at_10']) for r in rows]
    matrices = {name: [] for name in BASELINES}
    covariates = []
    for row in rows:
        o, s, x = row['optimizer'], row['stage'], row['centered_log10_learning_rate']
        unit = {'adamw': 1e-5, 'muon': 1e-3, 'normuon': 1e-3}[o]
        rate = row['learning_rate'] / unit
        step = STEPS[s - 1]
        mass = sum((Fraction(i, 391) if i < 391 else Fraction(3907-i, 3516)
                    for i in range(step)), Fraction()) / Fraction(3907, 2)
        budget = rate * float(mass)
        b0 = [1., float(o == 'muon'), float(o == 'normuon'), *[float(s == j) for j in range(2, 6)], x]
        raw = [rate * int(o == t) for t in OPTIMIZERS]
        b2 = [float(o == t and s == j) for t in OPTIMIZERS for j in range(1, 6)]
        b2 += [x * int(o == t) for t in OPTIMIZERS] + raw
        assembled = {'B0_original': b0, 'B1_raw_rate': b0 + raw, 'B2_optimizer_stage_rate': b2,
                     'B3_nominal_schedule': b2 + [budget * int(o == t) for t in OPTIMIZERS]}
        for name, values in assembled.items():
            matrices[name].append(values)
        covariates.append(dict(run_id=row['run_id'], stage=s, step=step, raw_rate_unit=unit,
            scaled_raw_rate=rate, nominal_schedule_mass=float(mass), scaled_nominal_budget=budget))
    need(matrices == result['design_values'] and covariates == result['covariates'], 'Independent recipe design differs')
    splits = {f: ([i for i, r in enumerate(rows) if r['dose_index'] != f],
                  [i for i, r in enumerate(rows) if r['dose_index'] == f]) for f in range(1, 5)}
    fold_index = {(r['baseline'], r['conditioned_on_displacement'], r['feature'], r['fold']): r
                  for r in result['folds']}
    pred_index = {(r['baseline'], r['conditioned_on_displacement'], r['feature'], r['run_id'], r['stage']): r
                  for r in result['predictions']}
    expected_cells = {(b, c, f) for b in BASELINES
                      for c in ([False, True] if b in ('B0_original', 'B3_nominal_schedule') else [False])
                      for f in features}
    need(len(result['summaries']) == len(expected_cells) == 24 and
         {(r['baseline'], r['conditioned_on_displacement'], r['feature']) for r in result['summaries']} == expected_cells,
         'Missing or duplicate comparison')
    need(len(fold_index) == len(result['folds']) == 96 and len(pred_index) == len(result['predictions']) == 1440,
         'Missing or duplicate predictions/folds')
    contexts = {}
    counts = dict(predictions=0, folds=0, comparisons=0)
    for summary in result['summaries']:
        key = summary['baseline'], summary['conditioned_on_displacement']
        if key not in contexts:
            design = [[oracle.q(v) for v in b] + ([oracle.q(r[DISPLACEMENT])] if key[1] else [])
                      for b, r in zip(matrices[key[0]], rows, strict=True)]
            contexts[key] = design, {f: oracle.solve(design, y, tr) for f, (tr, te) in splits.items()}
        design, baseline = contexts[key]
        feature = summary['feature']
        values = [oracle.q(r[feature]) for r in rows]
        augmented = [b + [v] for b, v in zip(design, values, strict=True)]
        pooled_b, pooled_a = [None] * 60, [None] * 60
        improved = 0
        for f, (tr, te) in splits.items():
            saved = fold_index[*key, feature, f]
            need(saved['status'] == 'resolved', 'Unresolved actual fit needs an explicit undefined-case audit')
            ap, bp = oracle.solve(augmented, y, tr), baseline[f]
            train_values = [values[i] for i in tr]
            mean = sum(train_values) / len(train_values)
            centered = [v - mean for v in train_values]
            amplitude = max(abs(v) for v in centered)
            z = np.asarray([float(v / amplitude) for v in centered])
            z /= np.sqrt(np.mean(z * z))
            numeric = np.column_stack([np.asarray([design[i] for i in tr], dtype=float), z])
            singular = np.linalg.svd(numeric, compute_uv=False)
            need(np.count_nonzero(singular > np.finfo(float).eps * max(numeric.shape) * singular[0]) == numeric.shape[1],
                 'Independent numerical resolution differs')
            bm = oracle.error([y[i] for i in te], [bp[i] for i in te])
            am = oracle.error([y[i] for i in te], [ap[i] for i in te])
            need(bm == sp.Rational(saved['baseline_mse_exact']) and am == sp.Rational(saved['feature_mse_exact']),
                 'Fold exact MSE differs')
            improves = bool(bm > am)
            need(saved['improves'] is improves, 'Fold decision differs')
            improved += improves
            counts['folds'] += 1
            for i in te:
                recorded = pred_index[*key, feature, rows[i]['run_id'], rows[i]['stage']]
                need(recorded['fold'] == f and recorded['observed'] == rows[i]['mean_ndcg_at_10'], 'Prediction identity differs')
                need(sp.Rational(recorded['baseline_prediction_exact']) == bp[i] and
                     sp.Rational(recorded['feature_prediction_exact']) == ap[i], 'Independent exact prediction differs')
                pooled_b[i], pooled_a[i] = bp[i], ap[i]
                counts['predictions'] += 1
        bm, am = oracle.error(y, pooled_b), oracle.error(y, pooled_a)
        need(bm == sp.Rational(summary['pooled_baseline_mse_exact']) and
             am == sp.Rational(summary['pooled_feature_mse_exact']), 'Pooled exact MSE differs')
        need(summary['defined_folds'] == 4 and summary['improved_folds'] == improved and
             summary['diagnostic_flag'] is bool(bm > am and improved >= 3), 'Summary decision differs')
        for field, value in (('pooled_baseline_rmse', sp.sqrt(bm)), ('pooled_feature_rmse', sp.sqrt(am)),
                             ('pooled_rmse_reduction', sp.sqrt(bm) - sp.sqrt(am))):
            need(summary[field] == float(sp.N(value, 80)), 'Display arithmetic differs')
        counts['comparisons'] += 1
        if counts['comparisons'] % 4 == 0:
            print(json.dumps(dict(**counts, baseline=key)), flush=True)
    need(counts == dict(predictions=1440, folds=96, comparisons=24), 'Incomplete independent verification')
    output = dict(scope='independent-full-rational-OLS-and-recipe-designs-for-functional-sensitivity',
        completed_at_utc=datetime.now(timezone.utc).isoformat(), parent_readout_sha256=PARENT,
        producer_readout_sha256=a.receipt_sha256, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        unchanged_oracle_sha256=ORACLE, independent_design_rows=240, independent_schedule_covariates=60,
        **counts, same_measurements_not_independent_experiment=True, exploratory_post_result=True,
        formal_primary_admission=False, scientific_completion=False)
    with a.output.open('x') as stream:
        stream.write(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
