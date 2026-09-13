"""Independently replay a supplied complete-third-stage data snapshot using only Python stdlib.

This verifies recovered bytes and score arithmetic, not network transport, model
inference, checkpoint tensors, query-level rankings or scientific completion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path, PurePosixPath

PROTOCOL = '4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b'
AUTH = '2351e225387f42fe607d7e002a0717de79a78cba9f3c93f66652510083f0be7c'
DISPATCH = '5b8a89bed5e0eaa02a12585ee3f6c883ca3e162b550898d68322321ee41d8427'
RUNS = tuple('verified-v3-' + opt + '-' + rate for opt, rates in (
    ('adamw', ('1e-6', '3e-6', '1e-5', '3e-5')),
    ('muon', ('1e-4', '3e-4', '1e-3', '3e-3')),
    ('normuon', ('1e-4', '3e-4', '1e-3', '3e-3')),
) for rate in rates)
TASKS = ('ArguAna', 'ClimateFEVER', 'DBPedia', 'FEVER', 'FiQA2018', 'HotpotQA',
         'MSMARCO', 'NFCorpus', 'NQ', 'QuoraRetrieval', 'SCIDOCS', 'SciFact', 'TRECCOVID', 'Touche2020')
ORIGINAL_EVAL = PurePosixPath('/root/embedding-optimizer-v3-experiment/evaluations/dense-primary-v3')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_relative(name):
    require(isinstance(name, str), 'Relative path must be a string')
    path = PurePosixPath(name)
    require(bool(path.parts) and not path.is_absolute() and path.as_posix() == name
            and '..' not in path.parts and '\\' not in name, 'Unsafe relative path')
    return name


def identity(path):
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            'Expected an ordinary recovered file')
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'git_blob_sha1': hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()}


def verify(root, manifest_sha):
    require(re.fullmatch('[0-9a-f]{64}', manifest_sha) is not None, 'External manifest SHA required')
    require(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)),
            'Expected an ordinary snapshot root')
    entries = list(root.rglob('*'))
    require(not any(p.is_symlink() for p in entries), 'Symlink in snapshot refused')
    manifest_identity = identity(root / 'artifact_manifest.json')
    require(manifest_identity['sha256'] == manifest_sha, 'Manifest differs from external anchor')
    manifest = json.loads((root / 'artifact_manifest.json').read_bytes())
    require(manifest['scope'] == 'complete-primary-v3-third-stage-beir-artifact-backup'
            and manifest['primary_protocol_sha256'] == PROTOCOL
            and manifest['runs'] == list(RUNS) and manifest['checkpoint_stages'] == [2345]
            and manifest['beir_task_cells'] == 168 and manifest['native_measurement_files'] == 216
            and manifest['original_worker_records'] == 336 and manifest['source_code_included'] is False
            and manifest['cohort_selection'] == 'all_declared_optimizer_rate_configurations'
            and manifest['all_twelve_third_stage_configurations_complete'] is True
            and manifest['new_model_or_statistical_execution'] is False
            and manifest['scientific_completion'] is False, 'Wrong manifest scope')
    expected = manifest['files']
    require(len(expected) == manifest['payload_files'] == 556, 'Wrong payload population')
    actual = {p.relative_to(root).as_posix() for p in entries if p.is_file()}
    require(actual == set(expected) | {'artifact_manifest.json'} and len(actual) == 557,
            'Incomplete or extra recovered files')
    for name, want in expected.items():
        require(identity(root / safe_relative(name)) == want, 'Recovered file bytes differ')
    require(sum(i['bytes'] for i in expected.values()) == manifest['payload_bytes'], 'Wrong byte total')

    def read(name):
        require(name in expected, 'Unmanifested data reference')
        return json.loads((root / safe_relative(name)).read_bytes())

    def mapped_snapshot(item):
        old_path = PurePosixPath(item['path'])
        relative = safe_relative(old_path.relative_to(ORIGINAL_EVAL).as_posix())
        name = 'native/beir-third-stage/' + relative
        require(name in expected and all(expected[name][key] == item[key]
                                        for key in ('bytes', 'sha256')), 'Native file binding differs')
        return name

    acceptance = read('provenance/acceptance.json')
    require(acceptance['accepted_bundle_sha256'] == manifest['accepted_bundle_sha256']
            and acceptance['accepted_verification_sha256'] == manifest['accepted_verification_sha256']
            and acceptance['primary_protocol_sha256'] == PROTOCOL
            and acceptance['runs'] == list(RUNS) and acceptance['tasks'] == list(TASKS)
            and acceptance['step'] == 2345 and acceptance['task_cells'] == 168
            and acceptance['cohort_selection'] == 'all_declared_optimizer_rate_configurations'
            and acceptance['all_twelve_third_stage_configurations_complete'] is True
            and acceptance['accepted_independent_readback_sha256'] == manifest['accepted_independent_readback_sha256']
            and acceptance['full_primary_task_cells_required'] == 840
            and acceptance['source_code_included'] is False
            and acceptance['scientific_completion'] is False, 'Acceptance summary differs')
    native = {name for name in expected if name.startswith('native/beir-third-stage/')}
    workers = {name for name in expected if name.startswith('provenance/beir-workers/')}
    require(len(native) == 216 and len(workers) == 336, 'Native/worker inventory differs')
    complete_names = sorted(name for name in native if name.endswith('/all-fourteen-tasks-verified.json'))
    require(len(complete_names) == 12, 'Incomplete native checkpoint receipts')
    all_scores, means, seen_runs, used_native, used_workers = {}, {}, set(), set(), set()
    for name in complete_names:
        complete = read(name)
        plan = complete['plan']
        run = plan['checkpoint']['run_id']
        require(run in RUNS and run not in seen_runs and plan['checkpoint']['step'] == 2345
                and plan['checkpoint']['protocol_sha256'] == PROTOCOL
                and plan['protocol_sha256'] == PROTOCOL and plan['tasks'] == list(TASKS)
                and complete['complete_task_files_verified'] is True
                and complete['scientific_completion'] is False, 'Wrong native checkpoint identity')
        require(name == 'native/beir-third-stage/' + safe_relative(plan['cache_key'])
                + '/all-fourteen-tasks-verified.json', 'Wrong native cache location')
        require(plan['results_root'] == str(ORIGINAL_EVAL / plan['cache_key']), 'Wrong original root')
        require([r['task'] for r in complete['tasks']] == list(TASKS), 'Native tasks differ')
        operational_name = name.removesuffix('all-fourteen-tasks-verified.json') + 'operational_admission.json'
        operational = read(operational_name)
        require(operational['authorization_sha256'] == AUTH and operational['source_sha256'] == DISPATCH
                and operational['scientific_completion'] is False, 'Operational identity differs')
        used_native.update((name, operational_name))
        seen_runs.add(run)
        scores = []
        for task_record in complete['tasks']:
            task = task_record['task']
            require(len(task_record['files']) == 3, 'Wrong native task file population')
            mapped = [mapped_snapshot(item) for item in task_record['files']]
            used_native.update(mapped)
            score_names = [f for f in mapped if f.endswith('/' + task + 'Decontaminated.json')]
            require(len(score_names) == 1, 'Task score filename differs')
            raw = read(score_names[0])
            split = 'dev' if task == 'MSMARCO' else 'test'
            require(raw['task_name'] == task + 'Decontaminated' and len(raw['scores'][split]) == 1,
                    'Task or split population differs')
            score_row = raw['scores'][split][0]
            score = score_row['ndcg_at_10']
            require(type(score) is float and math.isfinite(score) and 0 <= score <= 1
                    and score == score_row['main_score'] == task_record['ndcg_at_10'],
                    'Raw primary score differs')
            scores.append(Fraction.from_float(score))
            all_scores[(run, task)] = score
            worker_task = task
            suffix = '/' + run + '-2345-' + worker_task + '.started.json'
            started_names = [p for p in workers if p.endswith(suffix)]
            require(len(started_names) == 1, 'Missing or duplicate original worker')
            started_name = started_names[0]
            exited_name = started_name.removesuffix('.started.json') + '.exited.json'
            started, exited = read(started_name), read(exited_name)
            require(started['job'] == exited['job'] == [run, 2345, worker_task]
                    and started['pid'] == exited['pid'] and exited['exit_code'] == 0
                    and type(started['start_ticks']) is int
                    and started['authorization_sha256'] == AUTH and started['source_sha256'] == DISPATCH
                    and started['both_gpu_lease_namespaces_inherited'] is True,
                    'Original worker identity or exit differs')
            command = started['command']
            require(command[command.index('--worker') + 1] == worker_task
                    and command[command.index('--models') + 1].endswith('/' + run + '/checkpoint-2345')
                    and command[command.index('--results_folder') + 1]
                    == plan['results_root'] + '/dense/' + run + '__checkpoint-2345', 'Worker command differs')
            used_workers.update((started_name, exited_name))
        means[run] = sum(scores, Fraction()) / 14
    require(seen_runs == set(RUNS) and used_native == native and used_workers == workers
            and len(all_scores) == 168, 'Unreconstructed native population')

    def table(name):
        require(name in expected, 'Missing data table')
        with (root / name).open(newline='') as stream:
            return list(csv.DictReader(stream))

    mean_rows = table('tables/checkpoint_means.csv')
    task_rows = table('tables/task_scores.csv')
    require(len(mean_rows) == 12 and [r['run_id'] for r in mean_rows] == list(RUNS), 'Mean rows differ')
    require(len(task_rows) == 168 and [(r['run_id'], r['task']) for r in task_rows]
            == [(r, t) for r in RUNS for t in TASKS], 'Task rows differ')
    for row in mean_rows + task_rows:
        opt, rate = row['run_id'].removeprefix('verified-v3-').split('-', 1)
        require(row['optimizer'] == opt and row['learning_rate'] == rate and row['step'] == '2345',
                'Table recipe differs')
    for row in task_rows:
        require(float(row['ndcg_at_10']) == all_scores[(row['run_id'], row['task'])], 'Task table score differs')
    for row in mean_rows:
        mean = means[row['run_id']]
        require(row['task_count'] == '14' and float(row['macro_ndcg_at_10']) == float(mean)
                and float(row['macro_score_0_to_100']) == float(mean * 100), 'Exact mean table differs')
    return {'scope': 'independent-supplied-complete-third-stage-snapshot-byte-and-score-reconstruction',
            'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest': manifest_identity,
            'files_verified': 557, 'bytes_verified': manifest['payload_bytes'] + manifest_identity['bytes'],
            'raw_task_scores_reconstructed': 168, 'native_exit_zero_workers_checked': 168,
            'complete_checkpoint_means_reconstructed': 12, 'csv_rows_reconstructed': 180,
            'third_stage_means_0_to_100': {run: float(means[run] * 100) for run in RUNS},
            'reads_only_supplied_snapshot': True, 'new_model_or_statistical_execution': False,
            'network_transport_verified_by_this_program': False, 'scientific_completion': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve existing replay output')
    result = verify(args.root, args.manifest_sha256)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()


