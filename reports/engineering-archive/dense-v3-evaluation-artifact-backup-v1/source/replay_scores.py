"""Offline standard-library reconstruction from complete recovered raw score files."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path, PurePosixPath

MANIFEST_SHA = 'bf443d5623c1312f8661a918ab93f9aea5f119058307733d477c8eac0c6da74f'
METRICS = ('contrastive_loss', 'positive_score', 'hardest_negative_score',
           'positive_margin', 'reciprocal_rank', 'top1_accuracy')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    require(root.is_absolute() and root.is_dir() and not args.output.exists(), 'Wrong roots or prior output')
    require(not any(p.is_symlink() for p in (root, *root.parents)), 'Symlinked root')
    raw = (root / 'artifact_manifest.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == MANIFEST_SHA, 'Transport manifest changed')
    manifest = json.loads(raw)
    expected = set(manifest['files']) | {'artifact_manifest.json'}
    actual = set()
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'Symlinked payload')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    require(actual == expected and len(actual) == 660, 'Incomplete or extra recovered population')
    for name, item in manifest['files'].items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and '..' not in relative.parts and relative.as_posix() == name,
                'Invalid manifest path')
        payload = (root / name).read_bytes()
        require(len(payload) == item['bytes'] and hashlib.sha256(payload).hexdigest() == item['sha256'] and
                hashlib.sha1(b'blob ' + str(len(payload)).encode() + bytes([0]) + payload).hexdigest() == item['git_blob_sha1'],
                'Recovered payload differs')
    require(manifest['scientific_completion'] is False and manifest['source_code_included'] is False,
            'Wrong snapshot scope')

    # Reconstruct validation before opening BEIR; do not replace original FP32 metrics.
    selection = read(root / 'native/validation-selection/all-twelve-validation-selection.json')
    validation = {}
    maxima = {name: 0.0 for name in METRICS}
    record_count = group_metric_count = 0
    paths = sorted((root / 'native/validation').glob('*/manifest.json'))
    require(len(paths) == 12, 'Missing validation models')
    for path in paths:
        native = read(path)
        plan = native['plan']
        run = plan['run_id']
        require(run not in validation and native['status'] == 'complete' and
                read(path.parent / 'admission.json') == plan, 'Wrong native validation identity')
        require(plan['settings']['temperature'] == 0.02 and
                plan['settings']['reference_replay']['atol'] == 2e-5 and
                plan['settings']['reference_replay']['rtol'] == 2e-6 and
                plan['settings']['beir_is_a_selection_input'] is False, 'Validation settings differ')
        records = [json.loads(line) for line in (path.parent / 'sample_scores.jsonl').read_text().splitlines()]
        require(len(records) == 4096, 'Partial validation model')
        groups = defaultdict(list)
        seen = set()
        for position, record in enumerate(records):
            require(record['row']['position'] == position and len(record['scores']) == 8 and
                    set(record['metrics']) == set(METRICS), 'Wrong scored row schema')
            key = (record['row']['source'], record['row']['query_id'])
            require(key not in seen, 'Duplicate validation query')
            seen.add(key)
            scores = record['scores']
            require(all(math.isfinite(v) for v in scores), 'Nonfinite score')
            logits = [v / 0.02 for v in scores]
            pivot = max(logits)
            hardest = max(scores[1:])
            rank = 1 + sum(v >= scores[0] for v in scores[1:])
            scalar = {'contrastive_loss': pivot - logits[0] + math.log(sum(math.exp(v - pivot) for v in logits)),
                      'positive_score': scores[0], 'hardest_negative_score': hardest,
                      'positive_margin': scores[0] - hardest, 'reciprocal_rank': 1 / rank,
                      'top1_accuracy': float(rank == 1)}
            for name, value in scalar.items():
                stored = record['metrics'][name]
                require(math.isclose(stored, value, abs_tol=2e-5, rel_tol=2e-6), 'Scalar validation replay differs')
                maxima[name] = max(maxima[name], abs(stored - value))
            groups['__all__'].append(record)
            groups[record['row']['source']].append(record)
        summary = read(path.parent / 'summary.json')
        require(summary['records'] == 4096 and len(summary['groups']) == len(groups), 'Wrong group summary')
        for group in summary['groups']:
            values = groups[group['group']]
            require(group['samples'] == len(values), 'Changed group denominator')
            for name in METRICS:
                mean = sum(r['metrics'][name] for r in values) / len(values)
                require(mean == group[name], 'Ordered raw-metric group mean differs')
                group_metric_count += 1
        declared = next(r for r in selection['run_metrics'] if r['run_id'] == run)
        overall = next(r for r in summary['groups'] if r['group'] == '__all__')
        require(all(declared[name] == overall[name] for name in METRICS), 'Original selector inputs differ')
        validation[run] = declared
        record_count += len(records)
    chosen = {}
    for optimizer in ('adamw', 'muon', 'normuon'):
        group = [r for r in validation.values() if r['optimizer'] == optimizer]
        require(len(group) == 4, 'Incomplete rate population')
        chosen[optimizer] = min(group, key=lambda r: (r['contrastive_loss'], r['learning_rate']))['run_id']
    require(chosen == selection['selected'] == read(root / 'native/validation-selection/completed.json')['selection'],
            'Original loss-only choice differs')

    beir = {}
    beir_tasks = {}
    original_root = Path('/root/embedding-optimizer-v3-experiment/evaluations/dense-primary-v3')
    for path in sorted((root / 'native/beir-final').glob('*/all-fourteen-tasks-verified.json')):
        native = read(path)
        run = native['plan']['checkpoint']['run_id']
        require(run not in beir and native['plan']['checkpoint']['step'] == 3907 and
                native['complete_task_files_verified'] is True and native['scientific_completion'] is False,
                'Wrong complete BEIR state')
        expected_tasks = native['plan']['tasks']
        require(len(expected_tasks) == len(set(expected_tasks)) == 14 and len(native['tasks']) == 14,
                'Incomplete task denominator')
        scores = {}
        for task in native['tasks']:
            files = [item for item in task['files'] if Path(item['path']).name.endswith('Decontaminated.json')]
            require(len(files) == 1, 'Wrong raw task score file')
            relative = Path(files[0]['path']).relative_to(original_root)
            value = read(root / 'native/beir-final' / relative)
            name = value['task_name'].removesuffix('Decontaminated')
            split = 'dev' if name == 'MSMARCO' else 'test'
            require(name == task['task'] and name not in scores and set(value['scores']) == {split} and
                    len(value['scores'][split]) == 1, 'Wrong raw task/split coverage')
            scored = value['scores'][split][0]
            score = scored['ndcg_at_10']
            require(scored['hf_subset'] == 'default' and math.isfinite(score) and 0 <= score <= 1 and
                    score == task['ndcg_at_10'] and abs(score - scored['main_score']) <= 1e-12,
                    'Raw/native BEIR metric differs')
            scores[name] = score
            beir_tasks[(run, name)] = score
        require(set(scores) == set(expected_tasks), 'Missing BEIR task')
        mean = sum((Fraction.from_float(v) for v in scores.values()), Fraction()) / 14
        beir[run] = {'macro_ndcg_at_10': float(mean), 'macro_score_0_to_100': float(100 * mean)}
    require(set(beir) == set(validation) and len(beir_tasks) == 168, 'Incomplete joined state grid')
    task_table = rows(root / 'tables/final-checkpoints/task-scores.csv')
    mean_table = rows(root / 'tables/final-checkpoints/summary.csv')
    require(len(task_table) == 168 and len(mean_table) == 12, 'Wrong BEIR table population')
    for row in task_table:
        require(float(row['ndcg_at_10']) == beir_tasks[(row['run_id'], row['task'])], 'Task CSV differs')
    for row in mean_table:
        require(all(float(row[name]) == value for name, value in beir[row['run_id']].items()), 'BEIR mean CSV differs')
    all_validation = rows(root / 'tables/validation/all-validation-metrics.csv')
    endpoints = rows(root / 'tables/validation/selected-endpoints.csv')
    require(len(all_validation) == 12 and len(endpoints) == 3, 'Wrong validation table population')
    for row in all_validation:
        require(all(float(row[name]) == validation[row['run_id']][name] for name in METRICS), 'Validation CSV differs')
        require((row['selected_by_validation'] == 'True') == (chosen[row['optimizer']] == row['run_id']), 'Selection label differs')
    adam = beir[chosen['adamw']]['macro_score_0_to_100']
    for row in endpoints:
        run = chosen[row['optimizer']]
        require(row['run_id'] == run and float(row['validation_mean_loss']) == validation[run]['contrastive_loss'] and
                float(row['beir_macro_ndcg_x100']) == beir[run]['macro_score_0_to_100'] and
                float(row['difference_from_selected_adamw_points']) == beir[run]['macro_score_0_to_100'] - adam,
                'Selected endpoint table differs')
    result = {'scope': 'offline_complete_recovered_endpoint_and_validation_score_reconstruction',
              'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'input_root': str(root),
              'manifest_sha256': MANIFEST_SHA, 'verified_files': 660, 'beir_task_scores': 168,
              'beir_endpoint_means': 12, 'validation_rows': record_count,
              'scalar_validation_checks': 6 * record_count, 'validation_group_metric_means': group_metric_count,
              'maximum_scalar_replay_absolute_errors': maxima, 'selected': chosen,
              'tables_verified': 4, 'selected_endpoint_rows': 3, 'numeric_replay_passed': True,
              'model_forward_or_query_level_beir_recomputed': False,
              'raw_query_document_text_reconstructed': False, 'source_publication': False,
              'scientific_completion': False, 'network_access': False}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
