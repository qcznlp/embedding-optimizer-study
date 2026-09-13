"""Independently reconstruct all twelve step-2345 checkpoint outcomes.

Host-bound raw-file verification, not model inference or an optimizer comparison.
The existing 42-state readback must remain exactly unchanged inside the new one.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

STORY = Path('/root/embedding-optimizer-story-refactor')
EXPERIMENT = Path('/root/embedding-optimizer-v3-experiment')
PRIMARY = Path('/root/embedding-optimizer-primary-v3')
PRIOR = STORY / 'reports/engineering-archive/dense-v3-third-stage-pool-b-evaluations-v1/actual/complete-checkpoint-readback.json'
PRIOR_SHA = '77a0d0bdb3251b86a3e1ab9d6c6fe56e96f9e8af388c16f3acd7bd4d035f3d24'
AUTH_SHA = '2351e225387f42fe607d7e002a0717de79a78cba9f3c93f66652510083f0be7c'
PROTOCOL_SHA = '4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b'
READER_SHA = '765f00465728ad6e18d299f2e064070d6cc525775a5bf1c7e71273c9497ca5b2'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f'Not a regular input: {path}')
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(all(getattr(before, f) == getattr(after, f) for f in fields), 'Moving input')
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}, raw


def pinned_json(path, expected_sha):
    bound, raw = snapshot(path)
    require(bound['sha256'] == expected_sha, 'Changed pinned parent')
    return bound, json.loads(raw)


def verify_embedded(item):
    bound, raw = snapshot(item['path'])
    require(raw == item['text'].encode(), 'Embedded raw bytes differ')
    require(all(bound[k] == item[k] for k in ('bytes', 'sha256')), 'Embedded raw identity differs')
    return raw


def order(key):
    optimizer, rate = key[0].removeprefix('verified-v3-').split('-', 1)
    return (['adamw', 'muon', 'normuon'].index(optimizer), float(rate), key[1])


def write_new(path, raw):
    with Path(path).open('xb') as stream:
        stream.write(raw)
    return snapshot(path)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_directory.exists(), 'Preserve previous analysis outputs')
    current_bound, raw = snapshot(args.bundle)
    current = json.loads(raw)
    prior_bound, prior = pinned_json(PRIOR, PRIOR_SHA)
    auth_bound, auth = pinned_json(EXPERIMENT / 'launch/evaluation-handoff/authorization.json', AUTH_SHA)
    protocol_bound, protocol = pinned_json(PRIMARY / 'configs/dense_primary_v3_protocol.json', PROTOCOL_SHA)
    require(current['source']['sha256'] == READER_SHA, 'Different native reader')
    require(current['protocol_sha256'] == PROTOCOL_SHA, 'Different primary protocol')
    require(current['scope'] == 'descriptive_complete_fourteen_task_checkpoint_readback', 'Different native scope')
    require(current['dispatcher_source'] == prior['dispatcher_source'], 'Changed original dispatcher snapshot')
    require(current['assembly'] == prior['assembly'], 'Changed original assembly snapshot')
    runs = {r for queue in auth['queues'].values() for r in queue}
    old_keys = {(r, step) for r in runs for step in (782, 1563, 3907)}
    old_keys |= {(r, 2345) for r in auth['queues']['b']}
    new_keys = {(r, 2345) for r in auth['queues']['a']}
    third_stage_keys = {(r, 2345) for r in runs}
    records = {(r['run_id'], r['step']): r for r in current['checkpoints']}
    old_records = {(r['run_id'], r['step']): r for r in prior['checkpoints']}
    require(len(runs) == 12 and len(new_keys) == 6, 'Different declared configuration coverage')
    require(len(old_records) == len(prior['checkpoints']) == 42 and set(old_records) == old_keys, 'Different prior cohort')
    require(len(records) == len(current['checkpoints']) == 48 and set(records) == old_keys | new_keys, 'Different new cohort')
    require(all(records[k] == v for k, v in old_records.items()), 'A previous complete checkpoint changed')
    expected_tasks = protocol['evaluation']['tasks']
    require(len(expected_tasks) == len(set(expected_tasks)) == 14, 'Different declared task list')
    scores = workers = snapshots = 0
    task_table = []
    for key in sorted(records, key=order):
        record = records[key]
        original = json.loads(verify_embedded(record['original_complete_receipt']))
        operational = json.loads(verify_embedded(record['original_operational_receipt']))
        require(original == record['native_complete_reread'], 'Original complete receipt differs')
        require(original['complete_task_files_verified'] is True and original['scientific_completion'] is False, 'Changed native acceptance scope')
        require(operational['source_sha256'] == current['dispatcher_source']['sha256'] and operational['authorization_sha256'] == AUTH_SHA, 'Wrong operational source binding')
        rows = record['native_complete_reread']['tasks']
        require(record['task_count'] == 14 and [r['task'] for r in rows] == expected_tasks, 'Incomplete or reordered task coverage')
        rawfiles = {}
        for item in record['raw_score_and_metadata_snapshots']:
            require(item['path'] not in rawfiles, 'Duplicate raw score/metadata snapshot')
            rawfiles[item['path']] = (item, verify_embedded(item))
            snapshots += 1
        values = []
        for row in rows:
            candidates = [f for f in row['files'] if f['path'].endswith('Decontaminated.json')]
            require(len(candidates) == 1, 'Ambiguous native task score file')
            score_file = candidates[0]
            item, score_raw = rawfiles[score_file['path']]
            source = json.loads(score_raw)
            require(all(item[k] == score_file[k] for k in ('bytes', 'sha256')), 'Different task score identity')
            require(source['task_name'] == row['task'] + 'Decontaminated', 'Different canonical task name')
            require(list(source['scores']) == (['dev'] if row['task'] == 'MSMARCO' else ['test']), 'Different task split')
            entries = [v for split in source['scores'].values() for v in split]
            require(len(entries) == 1 and entries[0]['hf_subset'] == 'default', 'Different task subset')
            score = entries[0]['ndcg_at_10']
            require(math.isfinite(score) and 0 <= score <= 1, 'Invalid nDCG')
            require(score == row['ndcg_at_10'] == entries[0]['main_score'], 'Different raw/main score')
            values.append(Fraction.from_float(score))
            scores += 1
            if key in third_stage_keys:
                task_table.append({'run_id': key[0], 'step': key[1], 'task': row['task'], 'ndcg_at_10': score})
        mean = sum(values, Fraction()) / 14
        require(float(mean) == record['macro_ndcg_at_10'] and float(mean * 100) == record['macro_score_0_to_100'], 'Different reconstructed mean')
        require({'numerator': mean.numerator, 'denominator': mean.denominator} == record['exact_binary64_input_mean'], 'Different exact rational mean')
        require(len(record['original_workers']) == 14, 'Incomplete original worker set')
        worker_tasks = []
        for worker in record['original_workers']:
            started = json.loads(verify_embedded(worker['started']))
            exited = json.loads(verify_embedded(worker['exited']))
            require(exited['exit_code'] == 0 and exited['job'] == started['job'], 'Unsuccessful or mismatched native worker')
            require(tuple(exited['job'][:2]) == key and exited['pid'] == started['pid'], 'Wrong original worker identity')
            require(started['source_sha256'] == current['dispatcher_source']['sha256'] and started['authorization_sha256'] == AUTH_SHA, 'Wrong original worker source')
            worker_tasks.append(exited['job'][2])
            workers += 1
        require(worker_tasks == expected_tasks, 'Worker task coverage/order differs')
    require(scores == workers == 672 and snapshots == 768 and len(task_table) == 168, 'Unexpected evidence counts')
    require(snapshot(PRIOR)[0] == prior_bound and snapshot(args.bundle)[0] == current_bound, 'Bundle moved during reconstruction')
    flags = ('available_case_mean', 'optimizer_ranking_or_selection', 'full_primary_grid_complete', 'model_or_retrieval_recomputed', 'cuda_initialized', 'committed_source_release', 'scientific_completion')
    require(all(current[f] is False for f in flags), 'Native scope flag was widened')
    checkpoint_table = [{'run_id': k[0], 'step': k[1], 'task_count': 14, 'macro_score_0_to_100': records[k]['macro_score_0_to_100']} for k in sorted(third_stage_keys, key=order)]
    artifacts = {}
    args.output_directory.mkdir()
    for name, table in (('third-stage-checkpoint-scores.csv', checkpoint_table), ('third-stage-task-scores.csv', task_table)):
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
        artifacts[name] = write_new(args.output_directory / name, buffer.getvalue().encode())
    lines = ['# Complete step-2345 checkpoint outcomes', '', 'All rows use the complete fourteen-task macro nDCG@10, multiplied by 100.', 'All twelve declared optimizer/learning-rate configurations are retained.', 'Rows follow optimizer/learning-rate order; no selection or inference is performed.', '', '| Optimizer | Learning rate | Complete tasks | Step-2345 macro score |', '| --- | ---: | ---: | ---: |']
    for row in checkpoint_table:
        optimizer, rate = row['run_id'].removeprefix('verified-v3-').split('-', 1)
        label = {'adamw': 'AdamW', 'muon': 'Muon', 'normuon': 'NorMuon'}[optimizer]
        lines.append(f"| {label} | {rate} | 14 / 14 | {row['macro_score_0_to_100']:.4f} |")
    lines.extend(['', 'The previous 42 complete checkpoint records are unchanged. Only six original', 'pool-A step-2345 states / 84 task values are new; the twelve rows above include', 'the six previously accepted pool-B states. All twelve step-3126 states', 'remain required. Four observed stages do not establish a full trajectory,', 'convergence rate, seed robustness or mechanism. No prior endpoint inference', 'or manuscript finding is changed.', ''])
    artifacts['third-stage-checkpoint-scores.md'] = write_new(args.output_directory / 'third-stage-checkpoint-scores.md', '\n'.join(lines).encode())
    summary = {
        'scope': 'independent_complete_third_stage_raw_score_reconstruction',
        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': snapshot(__file__)[0], 'native_bundle': current_bound,
        'prior_forty_two_bundle': prior_bound, 'authorization': auth_bound, 'protocol': protocol_bound,
        'complete_checkpoint_records': 48, 'new_checkpoint_records': 6,
        'previous_forty_two_records_exactly_unchanged': True,
        'new_raw_task_values': 84, 'raw_task_scores_checked': scores,
        'native_exit_zero_workers_checked': workers, 'raw_score_and_metadata_snapshots_checked': snapshots,
        'all_original_rational_means_exact': True,
        'third_stage_all_twelve_checkpoint_scores': checkpoint_table,
        'new_pool_a_checkpoint_scores': [row for row in checkpoint_table if (row['run_id'], row['step']) in new_keys],
        'third_stage_raw_task_values': len(task_table),
        'third_stage_checkpoint_rows': len(checkpoint_table),
        'generated_artifacts': artifacts, 'all_twelve_third_stage_configurations_complete': True,
        'new_model_execution': False, 'new_statistical_inference': False, 'scientific_completion': False,
        'boundary': 'Same-host reconstruction of all twelve third-stage configurations; only six pool-A checkpoints / 84 task values are new. Not a complete five-stage trajectory, new statistical inference, portable model replay, mechanism finding or source release.'}
    raw_summary = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False).encode() + b'\n'
    summary_identity = write_new(args.output_directory / 'independent-raw-reconstruction.json', raw_summary)
    print(json.dumps({'summary': summary_identity, **summary}, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
