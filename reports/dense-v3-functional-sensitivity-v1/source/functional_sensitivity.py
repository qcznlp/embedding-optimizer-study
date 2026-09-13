"""Post-result sensitivity of all four functional predictors; no new training.

The recipe designs and exact solver are reused unchanged from the completed
weight-predictor sensitivity. This is a new, explicitly four-feature analysis,
not the original fourteen-feature consumer with an altered population guard.
"""
import argparse
import importlib.util
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

PARENT = '608965f5297fc196adcce6ca7417bd40e7f35d05e26133f20c0811dd1fd7027e'
DESIGNS = '6a87b441917c95d1fb0355c340beb8715acee89e66c102b51ecdfce6265d9141'
ARITHMETIC = '0d66ce6e99aa42b8d93c8295cb0f5f3015ea7b7907a9f10fb095d95be2d646b4'
FEATURES = ('margin_helpful_mass_share', 'margin_helpful_attribution_participation_ratio',
            'margin_degrading_attribution_mass', 'random_50pct_relative_shortlist_ndcg')


def load(path, digest, name):
    import hashlib
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Missing or symlinked source')
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('Source identity differs')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calculate(tables, source, k):
    rows = tables['bridge_rows']
    source.need(len(rows) == 60 and len({(r['run_id'], r['stage']) for r in rows}) == 60,
                'Incomplete or duplicate functional panel')
    source.need(set(FEATURES) == {r['feature'] for r in tables['feature_prediction_summary']},
                'Functional feature population differs')
    source.need(all(type(r[f]) in (float, int) and math.isfinite(r[f])
                    for r in rows for f in (*FEATURES, source.DISPLACEMENT)),
                'Undefined input: no available-case analysis')
    numeric, names, covariates = source.designs(rows)
    designs = {b: [[k.rational(v) for v in r] for r in d] for b, d in numeric.items()}
    y = [k.rational(r['mean_ndcg_at_10']) for r in rows]
    values = {f: [k.rational(r[f]) for r in rows] for f in (*FEATURES, source.DISPLACEMENT)}
    splits = {f: ([i for i, r in enumerate(rows) if r['dose_index'] != f],
                  [i for i, r in enumerate(rows) if r['dose_index'] == f]) for f in range(1, 5)}
    source.need(all(len(tr) == 45 and len(te) == 15 for tr, te in splits.values()), 'Different folds')
    summaries, folds, predictions, scores, diagnostics = [], [], [], [], []
    for baseline in source.BASELINES:
        for conditioned in ([False, True] if baseline in ('B0_original', 'B3_nominal_schedule') else [False]):
            design = [b + ([v] if conditioned else [])
                      for b, v in zip(designs[baseline], values[source.DISPLACEMENT], strict=True)]
            contexts = {f: source.ExactBaseline(design, tr, k) for f, (tr, te) in splits.items()}
            pooled_base = [None] * 60
            for f, (tr, te) in splits.items():
                fitted, _ = contexts[f].project(y)
                for i in te:
                    pooled_base[i] = fitted[i]
            bm = k.mse(y, pooled_base)
            common = dict(baseline=baseline, conditioned_on_displacement=conditioned)
            scores.append(dict(**common, columns=len(design[0]), rows=60,
                               pooled_mse_exact=str(bm), pooled_rmse=k.root(bm)))
            for feature in FEATURES:
                pooled_added = [None] * 60
                improved = defined = 0
                for f, (tr, te) in splits.items():
                    result = contexts[f].add(values[feature], y)
                    bp = [result['baseline'][i] for i in te]
                    ap = None if result['added'] is None else [result['added'][i] for i in te]
                    fbm = k.mse([y[i] for i in te], bp)
                    fam = None if ap is None else k.mse([y[i] for i in te], ap)
                    improves = None if fam is None else fbm > fam
                    defined += int(fam is not None)
                    improved += int(improves is True)
                    key = dict(**common, feature=feature, fold=f)
                    folds.append(dict(**key, status=result['status'], baseline_mse_exact=str(fbm),
                        feature_mse_exact=None if fam is None else str(fam), baseline_rmse=k.root(fbm),
                        feature_rmse=None if fam is None else k.root(fam), improves=improves,
                        training_residual_energy_exact=result['energy'], train_rows=45, test_rows=15))
                    diagnostics.append(dict(**key, baseline_health=contexts[f].health, feature_health=result['health']))
                    for i in te:
                        ap_i = None if result['added'] is None else result['added'][i]
                        pooled_added[i] = ap_i
                        predictions.append(dict(**key, run_id=rows[i]['run_id'], stage=rows[i]['stage'],
                            status=result['status'], observed=rows[i]['mean_ndcg_at_10'],
                            baseline_prediction_exact=str(result['baseline'][i]),
                            feature_prediction_exact=None if ap_i is None else str(ap_i)))
                am = k.mse(y, pooled_added) if defined == 4 else None
                summaries.append(dict(**common, feature=feature, columns=len(design[0]),
                    pooled_baseline_mse_exact=str(bm), pooled_feature_mse_exact=None if am is None else str(am),
                    pooled_baseline_rmse=k.root(bm), pooled_feature_rmse=None if am is None else k.root(am),
                    pooled_rmse_reduction=None if am is None else k.rmse_reduction(bm, am),
                    improved_folds=improved, defined_folds=defined, total_folds=4,
                    diagnostic_flag=None if am is None else bm > am and improved >= 3))
            print(json.dumps(dict(**common, completed_comparisons=len(summaries))), flush=True)
    source.need((len(summaries), len(folds), len(predictions)) == (24, 96, 1440), 'Incomplete sensitivity')
    original_s = {r['feature']: r for r in tables['feature_prediction_summary']}
    original_p = {(r['feature'], r['run_id'], r['stage']): r for r in tables['held_out_predictions']}
    for r in summaries:
        if r['baseline'] == 'B0_original' and not r['conditioned_on_displacement']:
            old = original_s[r['feature']]
            for field in ('pooled_baseline_mse_exact', 'pooled_feature_mse_exact', 'pooled_baseline_rmse',
                          'pooled_feature_rmse', 'pooled_rmse_reduction'):
                source.need(r[field] == old[field], 'Original functional B0 differs: ' + field)
            source.need(r['diagnostic_flag'] is old['predictively_useful'] and
                        r['improved_folds'] == old['folds_improved'], 'Original decision differs')
    checked = 0
    for r in predictions:
        if r['baseline'] == 'B0_original' and not r['conditioned_on_displacement']:
            old = original_p[r['feature'], r['run_id'], r['stage']]
            source.need(all(r[f] == old[f] for f in ('baseline_prediction_exact', 'feature_prediction_exact')),
                        'Original exact prediction differs')
            checked += 1
    source.need(checked == 240, 'Incomplete B0 reproduction')
    return dict(summaries=summaries, folds=folds, predictions=predictions, baseline_scores=scores,
                diagnostics=diagnostics, covariates=covariates, design_columns=names, design_values=numeric)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'designs', 'arithmetic', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    s = load(a.designs, DESIGNS, '_unchanged_weight_sensitivity_designs')
    k = load(a.arithmetic, ARITHMETIC, '_unchanged_exact_arithmetic')
    s.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only analysis')
    s.need(a.output.is_absolute() and not a.output.exists(), 'New absolute output required')
    bindings = {str(a.designs): s.identity(a.designs), str(a.arithmetic): s.identity(a.arithmetic)}
    receipt_path = a.parent / 'readout.json'
    bindings[str(receipt_path)] = s.bound(receipt_path, PARENT)
    receipt = json.loads(receipt_path.read_text())
    panel_path = a.parent / 'tables.json'
    bindings[str(panel_path)] = s.bound(panel_path, receipt['payloads']['tables.json'])
    tables = json.loads(panel_path.read_text())
    result = calculate(tables, s, k)
    for path, binding in bindings.items():
        s.bound(Path(path), binding)
    a.output.mkdir(parents=True, exist_ok=False)
    s.dump(a.output / 'result.json', result)
    for name in ('summaries', 'folds', 'predictions', 'baseline_scores', 'covariates'):
        s.table(a.output / (name + '.csv'), result[name])
    outputs = {p.name: s.identity(p) for p in a.output.iterdir()}
    record = dict(scope='post-result-all-four-functional-predictors-recipe-control-sensitivity',
        completed_at_utc=datetime.now(timezone.utc).isoformat(), parent_readout_sha256=PARENT,
        source=s.identity(Path(__file__)), input_bindings=bindings, outputs=outputs,
        comparisons=24, folds=96, heldout_prediction_rows=1440, original_predictions_exactly_reproduced=240,
        exploratory_post_result=True, recipe_designs_and_solver_unchanged=True,
        original_results_modified=False, new_model_or_retrieval_execution=False,
        causal_mechanism_claim=False, formal_primary_admission=False, scientific_completion=False)
    s.dump(a.output / 'readout.json', record)
    print(json.dumps({k: v for k, v in record.items() if k not in ('input_bindings', 'outputs')}))


if __name__ == '__main__':
    main()
