"""Offline, stdlib-only verification of recovered prediction data; no fitting or project imports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction as Q
from pathlib import Path, PurePosixPath

PROTOCOL = '4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b'
ORIGIN = {'locked': '243b001980feca61e6cca4852f544e9922fa7ca90b2a157981836d47b16d8621',
          'sensitivity': '1e61a623504a58d3bed2a921857d5f637bf4dc4ee7e3ace5c5af452db3d79637'}
FEATURES = (
    'log_saved_segment_to_weight_ratio', 'saved_segment_stable_rank_fraction',
    'saved_segment_sketch_effective_rank_fraction', 'saved_segment_row_norm_cv',
    'saved_segment_top_1pct_row_energy', 'cumulative_displacement_to_weight_ratio',
    'cumulative_stable_rank_fraction', 'mean_saved_segment_subspace_overlap_to_adamw',
    'mean_cumulative_subspace_overlap_to_adamw',
    'exact_nonzero_saved_segment_stable_rank_fraction',
    'full_spectrum_nonzero_saved_segment_entropy_rank_fraction',
    'exact_nonzero_cumulative_stable_rank_fraction',
    'exact_mean_saved_segment_overlap_to_adamw', 'exact_mean_cumulative_overlap_to_adamw')
BASELINES = ('B0_original', 'B1_raw_rate', 'B2_optimizer_stage_rate', 'B3_nominal_schedule')
GROUPS = tuple((b, False) for b in BASELINES) + (('B0_original', True), ('B3_nominal_schedule', True))
STEPS = (782, 1563, 2345, 3126, 3907)
SECRET = re.compile(rb'wandb_v1_[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
         'Missing or symlinked file')
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    need(all(getattr(before, k) == getattr(after, k) for k in
             ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')), 'File changed during read')
    need(SECRET.search(raw) is None, 'Credential-shaped payload')
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'git_blob_sha1': hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()}


def bound(path, expected):
    value = identity(path)
    need(all(value[k] == v for k, v in expected.items()), 'Payload identity mismatch: ' + path.name)
    return value


def read(path):
    identity(path)
    return json.loads(path.read_bytes())


def names():
    common = ('bridge_rows', 'feature_prediction_summary', 'held_out_predictions',
              'leave_dose_fold_metrics', 'residual_associations')
    out = {f'locked/actual/{b}/{n}.csv' for b in ('original', 'exact') for n in common}
    out |= {f'locked/actual/exact/{n}.csv' for n in ('measurement_comparison', 'predictive_sensitivity_summary')}
    out |= {'locked/actual/' + n for n in ('diagnostics.json', 'joined-checkpoints.json',
             'recipes.json', 'tables.json', 'summary.md', 'readout.json')}
    out |= {'locked/figures/' + n for n in ('figure_data.csv', 'figure_manifest.json',
             'held_dose_prediction.pdf', 'held_dose_prediction.png', 'held_dose_prediction.svg')}
    out |= {'sensitivity/actual/' + n for n in ('baseline_scores.csv', 'covariates.csv',
             'folds.csv', 'predictions.csv', 'readout.json', 'result.json', 'summaries.csv', 'summary.md')}
    out |= {'sensitivity/figures/' + n for n in ('comparator_sensitivity.pdf',
             'comparator_sensitivity.png', 'comparator_sensitivity.svg', 'descriptive.json',
             'figure_data.csv', 'manifest.json')}
    for b in ORIGIN:
        out.add(b + '/plan.md')
        out |= {b + '/provenance/' + n for n in
                ('verification.json', 'independent.json', 'source-relocated-readout.json')}
    return out | {'README.md', 'artifact_manifest.json'}


def inventory(root, manifest_sha):
    need(root.is_absolute() and root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)),
         'Snapshot must be an ordinary absolute directory')
    need(re.fullmatch('[0-9a-f]{64}', manifest_sha) is not None, 'Require external manifest SHA-256')
    bound(root / 'artifact_manifest.json', {'sha256': manifest_sha})
    manifest = read(root / 'artifact_manifest.json')
    need(manifest['scope'] == 'complete-v3-weight-retrieval-predictions-backup'
         and manifest['primary_protocol_sha256'] == PROTOCOL
         and manifest['source_code_included'] is False and manifest['new_inference'] is False
         and manifest['scientific_completion'] is False and manifest['sensitivity_is_post_result'] is True
         and manifest['source_verification_sha256'] == ORIGIN['locked']
         and manifest['sensitivity_verification_sha256'] == ORIGIN['sensitivity']
         and (manifest['checkpoint_count'], manifest['locked_features'], manifest['locked_predictions'],
              manifest['exploratory_comparisons'], manifest['exploratory_predictions']) == (60, 14, 840, 84, 5040),
         'Wrong manifest scientific boundary or population')
    expected = names()
    entries = list(root.rglob('*'))
    need(all(not p.is_symlink() and (p.is_file() or p.is_dir()) for p in entries), 'Nonordinary snapshot entry')
    need({p.relative_to(root).as_posix() for p in entries if p.is_file()} == expected
         and set(manifest['files']) == expected - {'artifact_manifest.json'}, 'Incomplete or extra payload population')
    total = 0
    for name, expected_id in manifest['files'].items():
        need(PurePosixPath(name).as_posix() == name and not PurePosixPath(name).is_absolute()
             and '..' not in PurePosixPath(name).parts and '\\' not in name, 'Unsafe file path')
        total += bound(root / name, expected_id)['bytes']
    need(len(manifest['files']) == manifest['payload_files'] == 46 and total == manifest['payload_bytes'],
         'Manifest totals disagree')
    for b, sha in ORIGIN.items():
        bound(root / b / 'provenance/verification.json', {'sha256': sha})
        original = read(root / b / 'provenance/verification.json')
        need(original['scientific_completion'] is False, 'Original scope changed')
        for name in expected:
            if not name.startswith(b + '/') or name.endswith('/provenance/verification.json'):
                continue
            rel = name.removeprefix(b + '/')
            if rel == 'provenance/independent.json':
                rel = 'independent-predictions.json' if b == 'locked' else 'independent.json'
            elif rel == 'provenance/source-relocated-readout.json':
                rel = 'source-relocated/readout.json'
            elif b == 'sensitivity' and rel.startswith('figures/'):
                rel = rel.replace('figures/', 'figures-v2/', 1)
            bound(root / name, original['payloads'][rel])
        actual = read(root / b / 'actual/readout.json')
        copied = read(root / b / 'provenance/source-relocated-readout.json')
        need(actual['outputs'] == copied['outputs'] and actual['scientific_completion'] is False
             and actual['formal_primary_admission'] is False, 'Replay or scientific boundary changed')
        for name, item in actual['outputs'].items():
            bound(root / b / 'actual' / name, item)
        figure = read(root / b / 'figures' / ('figure_manifest.json' if b == 'locked' else 'manifest.json'))
        need(figure['input_readout_sha256'] == identity(root / b / 'actual/readout.json')['sha256']
             and figure['statistical_recalculation'] is False, 'Figure input differs')
        for name, item in figure['files'].items():
            bound(root / b / 'figures' / name, item)
    need(read(root / 'sensitivity/actual/readout.json')['exploratory_post_result'] is True
         and read(root / 'sensitivity/provenance/independent.json')['exploratory_post_result'] is True,
         'Exploratory analysis must remain post-result')
    return manifest


def csv_equal(path, rows):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        cells = list(reader)
        keys = list(dict.fromkeys(k for row in rows for k in row))
        need(len(reader.fieldnames) == len(set(reader.fieldnames)) and set(reader.fieldnames) == set(keys),
             'CSV columns differ')
    need(cells == [{k: '' if r.get(k) is None else str(r[k]) for k in keys} for r in rows],
         'CSV and typed table differ: ' + path.name)
    return len(cells)


def rms(mse):
    with localcontext() as context:
        context.prec = 80
        return float((Decimal(mse.numerator) / Decimal(mse.denominator)).sqrt())


def rms_reduction(base, feature):
    with localcontext() as context:
        context.prec = 80
        b = Decimal(base.numerator) / Decimal(base.denominator)
        f = Decimal(feature.numerator) / Decimal(feature.denominator)
        return float(b.sqrt() - f.sqrt())


def check_predictions(predictions, folds, summaries, states, exploratory):
    fold_key = 'fold' if exploratory else 'held_out_dose_index'
    groups = GROUPS if exploratory else (('locked', False),)
    features = tuple(r['feature'] for r in summaries) if not exploratory else FEATURES
    need(len(features) == len(set(features)), 'Duplicate feature summary')
    wanted = {(b, c, f, d) for b, c in groups for f in features for d in range(1, 5)}
    buckets = defaultdict(list)
    unique, original_rows = set(), {}
    for p in predictions:
        key = ((p['baseline'], p['conditioned_on_displacement']) if exploratory else ('locked', False))
        b, c = key
        need(type(c) is bool and (b, c) in groups and p['feature'] in features, 'Unknown prediction population')
        state = states[(p['run_id'], p['stage'])]
        d = p[fold_key]
        need(d == state['dose_index'] and p['observed'] == state['mean_ndcg_at_10'], 'Prediction outcome or held fold differs')
        row_key = (*key, p['feature'], p['run_id'], p['stage'])
        need(row_key not in unique, 'Duplicate prediction')
        unique.add(row_key)
        need(p['status'] in ('resolved', 'baseline_equivalent'), 'Undefined actual prediction')
        base, feature = Q(p['baseline_prediction_exact']), Q(p['feature_prediction_exact'])
        if p['status'] == 'baseline_equivalent':
            need(exploratory and c and p['feature'] == FEATURES[5] and base == feature, 'Unexpected redundant prediction')
        if not exploratory:
            need(p['baseline_prediction'] == float(base) and p['feature_prediction'] == float(feature),
                 'Displayed prediction differs')
        observed = Q(p['observed'])
        buckets[(*key, p['feature'], d)].append(((base - observed)**2, (feature - observed)**2))
        original_rows[(p['feature'], p['run_id'], p['stage'])] = (base, feature)
    need(set(buckets) == wanted and len(unique) == len(wanted)*15, 'Incomplete held-out prediction grid')
    metrics = {}
    for key, errors in buckets.items():
        need(len(errors) == 15, 'Wrong held-out fold size')
        metrics[key] = tuple(sum((e[j] for e in errors), Q(0))/15 for j in (0, 1))
    seen = set()
    redundant = 0
    for r in folds:
        key = (*((r['baseline'], r['conditioned_on_displacement']) if exploratory else ('locked', False)),
               r['feature'], r[fold_key])
        need(key not in seen and key in metrics, 'Duplicate or unexpected fold')
        seen.add(key)
        b, f = metrics[key]
        need(Q(r['baseline_mse_exact']) == b and Q(r['feature_mse_exact']) == f
             and r['baseline_rmse'] == rms(b) and r['feature_rmse'] == rms(f)
             and r['train_rows'] == 45 and r['test_rows'] == 15
             and r['improves' if exploratory else 'feature_improves'] is (f < b), 'Fold arithmetic differs')
        redundant += r['status'] == 'baseline_equivalent'
        if not exploratory:
            need(Q(r['mse_reduction_exact']) == b-f and r['rmse_reduction'] == rms_reduction(b, f), 'Fold reduction differs')
    need(seen == wanted, 'Missing fold metrics')
    result, seen = {}, set()
    for r in summaries:
        key = (*((r['baseline'], r['conditioned_on_displacement']) if exploratory else ('locked', False)), r['feature'])
        need(key not in seen, 'Duplicate pooled comparison')
        seen.add(key)
        values = [metrics[(*key, d)] for d in range(1, 5)]
        b, f = (sum((v[j] for v in values), Q(0))/4 for j in (0, 1))
        improved = sum(v[1] < v[0] for v in values)
        flag = f < b and improved >= 3
        need(Q(r['pooled_baseline_mse_exact']) == b and Q(r['pooled_feature_mse_exact']) == f
             and r['pooled_baseline_rmse'] == rms(b) and r['pooled_feature_rmse'] == rms(f)
             and r['improved_folds' if exploratory else 'folds_improved'] == improved
             and r['defined_folds' if exploratory else 'folds_defined'] == 4
             and r['total_folds' if exploratory else 'folds_total'] == 4
             and r['diagnostic_flag' if exploratory else 'predictively_useful'] is flag
             and r['pooled_rmse_reduction'] == rms_reduction(b, f), 'Pooled arithmetic or decision differs')
        if not exploratory:
            need(Q(r['pooled_mse_reduction_exact']) == b-f and r['pooled_rows'] == 60, 'Pooled reduction differs')
        result[key] = (rms(b), rms(f), improved, flag)
    need(seen == {k[:3] for k in wanted}, 'Missing pooled comparison')
    return result, original_rows, redundant


def verify(root, manifest_sha):
    manifest = inventory(root, manifest_sha)
    tables = read(root / 'locked/actual/tables.json')
    result = read(root / 'sensitivity/actual/result.json')
    csv_rows = 0
    for b in ('original', 'exact'):
        for name, rows in tables[b].items():
            csv_rows += csv_equal(root / f'locked/actual/{b}/{name}.csv', rows)
    for name in ('baseline_scores', 'covariates', 'folds', 'predictions', 'summaries'):
        csv_rows += csv_equal(root / f'sensitivity/actual/{name}.csv', result[name])
    recipes = read(root / 'locked/actual/recipes.json')
    joined = read(root / 'locked/actual/joined-checkpoints.json')
    rows = tables['exact']['bridge_rows']
    states = {(r['run_id'], r['stage']): r for r in rows}
    grid = {(run, s) for run in recipes for s in range(1, 6)}
    need(len(recipes) == 12 and len(rows) == len(states) == len(joined) == 60 and set(states) == grid,
         'Wrong complete checkpoint grid')
    need(len(tables['original']['bridge_rows']) == 60, 'Original row count differs')
    for r in tables['original']['bridge_rows']:
        need(all(states[(r['run_id'], r['stage'])][k] == v for k, v in r.items()), 'Original feature overwritten')
    for r in joined:
        need(r['step'] == STEPS[r['stage']-1]
             and r['mean_ndcg_at_10'] == states[(r['run_id'], r['stage'])]['mean_ndcg_at_10'], 'Joined outcome differs')
    need({(r['run_id'], r['stage']) for r in joined} == grid, 'Joined checkpoint population differs')
    for r in rows:
        config = recipes[r['run_id']]['optimizer']
        rates = sorted({x['optimizer']['lr'] for x in recipes.values() if x['optimizer']['name'] == config['name']})
        need(len(rates) == 4 and r['dose_index'] == rates.index(config['lr'])+1
             and r['learning_rate'] == config['lr'] and r['optimizer'] == config['name']
             and all(math.isfinite(r[f]) for f in FEATURES), 'Recipe or feature differs')
    locked, originals = {}, {}
    for b, fs in (('original', FEATURES[:9]), ('exact', FEATURES[9:])):
        t = tables[b]
        need(tuple(r['feature'] for r in t['feature_prediction_summary']) == fs, 'Feature order differs')
        summaries, predictions, redundant = check_predictions(t['held_out_predictions'],
            t['leave_dose_fold_metrics'], t['feature_prediction_summary'], states, False)
        need(redundant == 0, 'Unexpected undefined original fold')
        locked.update(summaries); originals.update(predictions)
    sensitivity, _, redundant = check_predictions(result['predictions'], result['folds'], result['summaries'], states, True)
    need(redundant == 8 and len(result['predictions']) == 5040 and len(result['summaries']) == 84, 'Sensitivity count differs')
    b0_count = 0
    for p in result['predictions']:
        if p['baseline'] == 'B0_original' and not p['conditioned_on_displacement']:
            need((Q(p['baseline_prediction_exact']), Q(p['feature_prediction_exact']))
                 == originals[(p['feature'], p['run_id'], p['stage'])], 'Original B0 prediction changed')
            b0_count += 1
    need(b0_count == 840, 'B0 prediction population differs')
    for r in result['baseline_scores']:
        matches = [s for (b, c, _), s in sensitivity.items() if (b, c) == (r['baseline'], r['conditioned_on_displacement'])]
        need(len(matches) == 14 and all(s[0] == r['pooled_rmse'] for s in matches)
             and rms(Q(r['pooled_mse_exact'])) == r['pooled_rmse'] and r['rows'] == 60, 'Baseline table differs')
    need(len(result['baseline_scores']) == 6, 'Baseline table incomplete')
    plot_rows = 0
    for b, values in (('locked', locked), ('sensitivity', sensitivity)):
        with (root / b / 'figures/figure_data.csv').open(newline='') as stream:
            plotted = list(csv.DictReader(stream))
        seen = set()
        for p in plotted:
            key = ('locked', False, p['feature']) if b == 'locked' else (
                p['baseline'], {'True': True, 'False': False}[p['conditioned_on_displacement']], p['feature'])
            need(key in values and key not in seen, 'Extra or duplicate plot point')
            seen.add(key); base, feature, improved, flag = values[key]
            need(float(p['baseline_rmse_points']) == 100*base and float(p['feature_rmse_points']) == 100*feature
                 and int(p['improved_folds']) == improved
                 and p['predictive_support' if b == 'locked' else 'diagnostic_flag'] == str(flag), 'Plot differs from numeric result')
            if b == 'sensitivity':
                need(float(p['feature_minus_baseline_rmse_points']) == 100*(feature-base), 'Plot delta differs')
        need(seen == set(values), 'Missing plotted result')
        plot_rows += len(plotted)
    descriptive = read(root / 'sensitivity/figures/descriptive.json')
    all_four = [f for f in FEATURES if all(sensitivity[(b, False, f)][3] for b in BASELINES)]
    need(all_four == descriptive['features_passing_all_four_unconditioned_baselines'] == [], 'All-baseline interpretation differs')
    need(len(descriptive['groups']) == 6, 'Missing descriptive group')
    for row in descriptive['groups']:
        values = [sensitivity[(row['baseline'], row['conditioned_on_displacement'], f)] for f in FEATURES]
        need(row['flags_true'] == sum(x[3] for x in values)
             and all(100*x[0] == row['baseline_rmse_points'] for x in values), 'Descriptive group differs')
    return {'scope': 'independent-recovered-prediction-data-and-error-reconstruction',
            'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'snapshot_root': str(root),
            'manifest_sha256': manifest_sha, 'files': 47,
            'bytes': manifest['payload_bytes'] + identity(root / 'artifact_manifest.json')['bytes'],
            'csv_rows_matched': csv_rows, 'checkpoint_outcomes_joined': 60,
            'locked_predictions': 840, 'sensitivity_predictions': 5040,
            'exact_fold_errors_reconstructed': 392, 'pooled_comparisons_reconstructed': 98,
            'original_B0_predictions_unchanged': b0_count, 'redundant_self_addition_folds': redundant,
            'plot_points_matched': plot_rows, 'all_four_baseline_supported_features': all_four,
            'only_supplied_snapshot_read': True, 'numerical_fitting_or_new_bootstrap': False,
            'model_or_network_execution': False, 'scientific_completion': False,
            'source_release': False, 'same_physical_host': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(args.output.is_absolute() and not args.output.exists()
         and not args.output.is_relative_to(args.snapshot)
         and not any(p.is_symlink() for p in (args.output, *args.output.parents)), 'Unsafe or existing receipt')
    value = verify(args.snapshot, args.manifest_sha256)
    with args.output.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(value, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
