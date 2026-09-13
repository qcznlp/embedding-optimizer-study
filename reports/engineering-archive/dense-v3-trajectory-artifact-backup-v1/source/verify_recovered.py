"""Independent stdlib reconstruction of a supplied trajectory snapshot; no models/network."""
import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path, PurePosixPath

PROTOCOL = '4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b'
READOUT_SHA = 'dd48e622c9cd1f28f8e93cab670349fc937e8e64aa50b98947104151dbda42c2'
INDEX_SHA = '0418c68b9a78cc80173c9cf6af5d5bb57d223d7e763bc6da03d2fe4f682bbf96'
ENDPOINT_SHA = '592c902f94b8f190f3165f4075dfddc1e8c3f52add760805da639b541d27716f'
RATES = {'adamw': ('1e-6', '3e-6', '1e-5', '3e-5'),
         'muon': ('1e-4', '3e-4', '1e-3', '3e-3'),
         'normuon': ('1e-4', '3e-4', '1e-3', '3e-3')}
RUNS = tuple('verified-v3-' + o + '-' + r for o, rs in RATES.items() for r in rs)
STEPS = (782, 1563, 2345, 3126, 3907)
TASKS = ('ArguAna', 'ClimateFEVER', 'DBPedia', 'FEVER', 'FiQA2018', 'HotpotQA',
         'MSMARCO', 'NFCorpus', 'NQ', 'QuoraRetrieval', 'SCIDOCS', 'SciFact', 'TRECCOVID', 'Touche2020')
SELECTED = {'adamw': 'verified-v3-adamw-3e-5', 'muon': 'verified-v3-muon-3e-4',
            'normuon': 'verified-v3-normuon-3e-4'}
COUNTS = {'all_task_scores.csv': 840, 'optimizer_stage_scores.csv': 15,
          'primary_summary.csv': 3, 'primary_task_effects.csv': 14,
          'run_observed_auc.csv': 12, 'run_stage_scores.csv': 60,
          'secondary_summary.csv': 3, 'secondary_task_effects.csv': 14}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe(name):
    p = PurePosixPath(name)
    require(bool(p.parts) and not p.is_absolute() and '..' not in p.parts
            and p.as_posix() == name and '\\' not in name, 'Unsafe relative path')
    return name


def identity(path):
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Ordinary files required')
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'git_blob_sha1': hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()}


def read_json(path, digest=None):
    got = identity(path)
    require(digest is None or got['sha256'] == digest, 'External input anchor differs')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Non-finite JSON value: ' + value)
    return json.loads(path.read_bytes(), object_pairs_hook=unique, parse_constant=invalid)


def table(path, count):
    rows = csv.DictReader(io.StringIO(path.read_text()))
    require(rows.fieldnames and len(rows.fieldnames) == len(set(rows.fieldnames)), 'Bad CSV header')
    values = list(rows)
    require(len(values) == count and all(None not in r and None not in r.values() for r in values), 'Bad CSV population')
    return values


def close(actual, expected, tolerance=1e-15):
    actual = float(actual)
    require(math.isfinite(actual) and abs(actual - float(expected)) <= tolerance, 'Reconstructed scalar differs')


def load_scores(rows):
    scores = {}
    for row in rows:
        run, stage, task = row['run_id'], int(row['stage']), row['task']
        require(run in RUNS and stage in range(1, 6) and task in TASKS, 'Wrong score identity')
        optimizer, rate = run.removeprefix('verified-v3-').split('-', 1)
        require(row['model_family'] == 'dense' and row['optimizer'] == optimizer
                and float(row['learning_rate']) == float(rate) and int(row['step']) == STEPS[stage-1]
                and float(row['fraction']) == stage / 5, 'Wrong score recipe')
        value = float(row['ndcg_at_10'])
        require(math.isfinite(value) and 0 <= value <= 1 and (run, stage, task) not in scores, 'Invalid/duplicate score')
        scores[(run, stage, task)] = Fraction.from_float(value)
    require(set(scores) == {(r, s, t) for r in RUNS for s in range(1, 6) for t in TASKS}, 'Missing score cell')
    return scores


def reconstruct_dynamics(scores, tables):
    means = {(run, stage): sum((scores[(run, stage, task)] for task in TASKS), Fraction()) / 14
             for run in RUNS for stage in range(1, 6)}
    seen = set()
    for row in tables['run_stage_scores.csv']:
        run, stage = row['run_id'], int(row['stage'])
        key = run, stage
        require(key in means and key not in seen, 'Run-stage population differs')
        seen.add(key)
        opt, rate = run.removeprefix('verified-v3-').split('-', 1)
        require(row['optimizer'] == opt and float(row['learning_rate']) == float(rate)
                and int(row['tasks']) == 14 and float(row['progress_fraction']) == stage/5, 'Run-stage recipe differs')
        close(row['mean_ndcg_at_10'], means[key])
        close(row['median_ndcg_at_10'], statistics.median(scores[(run, stage, t)] for t in TASKS))
    require(seen == set(means), 'Missing run-stage mean')
    seen = set()
    for row in tables['optimizer_stage_scores.csv']:
        opt, stage = row['optimizer'], int(row['stage'])
        require(opt in RATES and stage in range(1, 6) and (opt, stage) not in seen, 'Optimizer-stage population differs')
        seen.add((opt, stage))
        values = [means[('verified-v3-' + opt + '-' + rate, stage)] for rate in RATES[opt]]
        require(int(row['learning_rates']) == 4 and float(row['progress_fraction']) == stage/5, 'Rate grid differs')
        close(row['mean_ndcg_at_10_across_rates'], sum(values, Fraction()) / 4)
        close(row['median_ndcg_at_10_across_rates'], statistics.median(values))
    require(len(seen) == 15, 'Missing optimizer-stage summary')
    seen = set()
    for row in tables['run_observed_auc.csv']:
        run = row['run_id']
        require(run in RUNS and run not in seen, 'Area run differs')
        seen.add(run)
        opt, rate = run.removeprefix('verified-v3-').split('-', 1)
        require(row['optimizer'] == opt and float(row['learning_rate']) == float(rate), 'Area recipe differs')
        area = sum(((means[(run, s)] + means[(run, s+1)]) / 10 for s in range(1, 5)), Fraction())
        close(row['observed_auc_20_to_100'], area)
        close(row['observed_mean_20_to_100'], area * Fraction(5, 4))
    require(seen == set(RUNS), 'Missing observed area')
    return means


def verify(root, manifest_sha):
    require(root.is_absolute() and root.is_dir(), 'Absolute snapshot root required')
    require(not any(p.is_symlink() for p in (root, *root.parents, *root.rglob('*'))), 'Snapshot symlink refused')
    require(re.fullmatch('[0-9a-f]{64}', manifest_sha) is not None, 'External manifest SHA required')
    manifest = read_json(root / 'artifact_manifest.json', manifest_sha)
    require(manifest['scope'] == 'complete-v3-retrieval-trajectories-backup'
            and manifest['primary_protocol_sha256'] == PROTOCOL
            and manifest['checkpoint_count'] == 60 and manifest['task_score_count'] == 840
            and manifest['raw_score_snapshots'] == 5 and manifest['raw_score_index_sha256'] == INDEX_SHA
            and manifest['source_code_included'] is False and manifest['new_inference'] is False
            and manifest['scientific_completion'] is False, 'Wrong snapshot scope')
    files = manifest['files']
    require(len(files) == manifest['payload_files'] == 25, 'Wrong file count')
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
            == set(files) | {'artifact_manifest.json'}, 'Incomplete or extra snapshot files')
    for name, expected in files.items():
        require(identity(root / safe(name)) == expected, 'Payload bytes differ')
    require(sum(f['bytes'] for f in files.values()) == manifest['payload_bytes'], 'Payload byte total differs')
    native = read_json(root / 'tables/readout.json', READOUT_SHA)
    endpoint = read_json(root / 'provenance/accepted-endpoint-readout.json', ENDPOINT_SHA)
    index = read_json(root / 'provenance/raw-score-index.json', INDEX_SHA)
    require(native['checkpoint_count'] == 60 and native['task_score_count'] == 840
            and native['selected'] == endpoint['selected'] == SELECTED
            and native['all_six_endpoint_contrasts_unchanged'] is True
            and native['original_whole_grid_consumer_called'] is False
            and native['original_whole_grid_admission_passed'] is False
            and native['scientific_completion'] is False, 'Wrong native readout scope')
    require(set(native['outputs']) == set(COUNTS), 'Missing native output')
    tables = {}
    for name, count in COUNTS.items():
        got = identity(root / 'tables' / name)
        require(native['outputs'][name] == {**{k: got[k] for k in ('bytes', 'sha256')}, 'rows': count}, 'Native table binding differs')
        tables[name] = table(root / 'tables' / name, count)
    scores = load_scores(tables['all_task_scores.csv'])
    means = reconstruct_dynamics(scores, tables)
    indexed = {}
    require(index['steps'] == list(STEPS) and index['runs'] == list(RUNS)
            and index['tasks'] == list(TASKS) and len(index['snapshots']) == 5, 'Wrong recovery index')
    for stage, snapshot in enumerate(index['snapshots'], 1):
        require(snapshot['step'] == STEPS[stage-1] and len(snapshot['checkpoints']) == 12, 'Index stage differs')
        require(snapshot['repo_id'] == 'qcz/embedding-optimizer-study-analysis-artifacts'
                and snapshot['repo_type'] == 'dataset' and re.fullmatch('[0-9a-f]{40}', snapshot['revision']), 'Index remote target differs')
        safe(snapshot['prefix'])
        for checkpoint in snapshot['checkpoints']:
            run = checkpoint['run_id']
            require(run in RUNS and checkpoint['step'] == snapshot['step']
                    and [r['task'] for r in checkpoint['task_files']] == list(TASKS), 'Index checkpoint differs')
            safe(checkpoint['native_complete']['path'])
            for entry in checkpoint['task_files']:
                safe(entry['path'])
                key = run, stage, entry['task']
                require(key not in indexed, 'Duplicate index score')
                indexed[key] = Fraction.from_float(entry['ndcg_at_10'])
            close(checkpoint['macro_ndcg_at_10'], means[(run, stage)])
            close(checkpoint['macro_score_0_to_100'], means[(run, stage)] * 100, 1e-12)
    require(indexed == scores, 'Index score and full table differ')

    contrast_count = 0
    for family in ('primary', 'secondary'):
        effect_rows = tables[family + '_task_effects.csv']
        require([r['task'] for r in effect_rows] == list(TASKS), 'Effect task population differs')
        for row in effect_rows:
            task = row['task']
            values = {opt: (sum((scores[('verified-v3-' + opt + '-' + rate, 5, task)] for rate in RATES[opt]), Fraction()) / 4
                            if family == 'primary' else scores[(SELECTED[opt], 5, task)]) for opt in RATES}
            for opt, value in values.items():
                close(row[opt + '_ndcg_at_10'], value)
            for treatment, baseline in (('muon', 'adamw'), ('normuon', 'adamw'), ('normuon', 'muon')):
                close(row[treatment + '_minus_' + baseline], values[treatment] - values[baseline])
        for filename in (family + '_summary.csv', family + '_task_effects.csv'):
            original = endpoint['tables'][filename]
            require(len(tables[filename]) == len(original), 'Original endpoint rows differ')
            for row, reference in zip(tables[filename], original, strict=True):
                for name, value in reference.items():
                    require(float(row[name]) == value if type(value) in (int, float)
                            else row[name] == value, 'Original endpoint value/decision differs')
        for row in tables[family + '_summary.csv']:
            key = row['treatment'] + '_minus_' + row['baseline']
            expected = sum(Fraction.from_float(float(r[key])) for r in effect_rows) / 14
            close(row['mean_delta_ndcg_at_10'], expected)
            lo, hi = float(row['simultaneous_ci_95_lower']), float(row['simultaneous_ci_95_upper'])
            require(row['support'] == ('positive' if lo > 0 else 'negative' if hi < 0 else 'inconclusive'), 'Support differs')
            contrast_count += 1
    plotted = table(root / 'figures/figure_data.csv', 60)
    require({(r['run_id'], int(r['stage'])) for r in plotted} == set(means), 'Figure population differs')
    for row in plotted:
        run, stage = row['run_id'], int(row['stage'])
        opt, rate = run.removeprefix('verified-v3-').split('-', 1)
        require(row['optimizer'] == opt and float(row['learning_rate']) == float(rate)
                and int(row['step']) == STEPS[stage-1] and int(row['training_progress_percent']) == stage*20
                and row['validation_selected'] == str(run == SELECTED[opt]), 'Figure recipe/selection differs')
        close(row['mean_ndcg_at_10'], means[(run, stage)])
        close(row['plotted_score_0_to_100'], means[(run, stage)] * 100, 1e-12)
    selected_rows = table(root / 'descriptive/selected_stage_contrasts.csv', 5)
    require([int(r['stage']) for r in selected_rows] == list(range(1, 6)), 'Selected stage population differs')
    for row in selected_rows:
        stage = int(row['stage'])
        require(int(row['training_progress_percent']) == stage*20, 'Selected progress differs')
        values = {opt: means[(SELECTED[opt], stage)] * 100 for opt in RATES}
        for opt, value in values.items():
            close(row[opt + '_score_0_to_100'], value, 1e-12)
        for opt in ('muon', 'normuon'):
            close(row[opt + '_minus_adamw'], values[opt] - values['adamw'], 1e-12)
            require(row[opt + '_above_selected_adamw_final'] == str(values[opt] > means[(SELECTED['adamw'], 5)] * 100), 'Crossing indicator differs')
    observer = read_json(root / 'observations/observer-completed.json')
    observed = {(r['run_id'], STEPS.index(r['step'])+1, r['task']): Fraction.from_float(r['ndcg_at_10'])
                for r in observer['task_rows'] if r['baseline'] is False}
    require(observed == scores and observer['primary_tasks_accepted'] == 840
            and observer['baseline_tasks_accepted'] == 14, 'Observer table differs')
    return {'scope': 'independent_recovered_complete_trajectory_numeric_reconstruction',
            'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest': identity(root / 'artifact_manifest.json'),
            'files_verified': 26, 'bytes_verified': manifest['payload_bytes'] + identity(root / 'artifact_manifest.json')['bytes'],
            'task_scores_crosschecked_with_index_and_observer': 840,
            'run_stage_means_and_medians_reconstructed': 120,
            'optimizer_stage_means_and_medians_reconstructed': 30,
            'areas_and_normalized_means_reconstructed': 24, 'endpoint_task_numeric_fields_reconstructed': 168,
            'endpoint_contrasts_matched': contrast_count, 'plotted_checkpoints_reconstructed': 60,
            'selected_stage_comparisons_reconstructed': 5, 'indexed_raw_snapshots': 5,
            'reads_only_supplied_snapshot': True, 'new_bootstrap_or_model_execution': False,
            'raw_remote_files_downloaded_by_this_reader': False, 'scientific_completion': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute() and not args.output.exists(), 'New absolute output required')
    result = verify(args.root, args.manifest_sha256)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
