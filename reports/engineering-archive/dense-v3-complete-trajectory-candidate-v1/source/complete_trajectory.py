"""Reconstruct all retained retrieval trajectories from a complete native readback.

This consumes authenticated aggregate evidence, not tensors or a model runner.
It preserves the original numerical/statistical kernels and endpoint conclusions.
Formal source, whole-run, functional and manuscript acceptance remain separate.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import numpy as np

PROTOCOL = '4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b'
DISPATCH = '5b8a89bed5e0eaa02a12585ee3f6c883ca3e162b550898d68322321ee41d8427'
AUTH = '2351e225387f42fe607d7e002a0717de79a78cba9f3c93f66652510083f0be7c'
READER = '765f00465728ad6e18d299f2e064070d6cc525775a5bf1c7e71273c9497ca5b2'
PRIOR_54 = '7d12ba8f36caa05d890a75c3598176324086433408f3a322ed45c75eb66521f5'
ENDPOINT = '592c902f94b8f190f3165f4075dfddc1e8c3f52add760805da639b541d27716f'
SOURCES = {
    'configs/dense_primary_v3_protocol.json': PROTOCOL,
    'configs/dense_primary_v3_outcome_protocol.json': '914996b3423ba01329c03844419cca61e2a33452fe026665bb940d5b82e2cffe',
    'configs/dense_no_packing_outcome_protocol.json': '7f321f44b73bf18e88321204781814ff143e4072acf1f26780db17c341188b35',
    'src/embed_optim/corrected_outcome_summary.py': 'b22f5a9b14f3be0bb31c750aa2d32f220948e2f7a08d88c55617d1a6a47732d0',
    'src/embed_optim/primary_completion.py': 'a1aa61058654bd46afb41710ed606992355dab95d7c8b171a7485a8ac24d0cd5',
}
STEPS = (782, 1563, 2345, 3126, 3907)
RATE_NAMES = {'adamw': ('1e-6', '3e-6', '1e-5', '3e-5'),
              'muon': ('1e-4', '3e-4', '1e-3', '3e-3'),
              'normuon': ('1e-4', '3e-4', '1e-3', '3e-3')}
RUNS = tuple('verified-v3-' + optimizer + '-' + rate
             for optimizer, rates in RATE_NAMES.items() for rate in rates)
POOL_B = ('verified-v3-adamw-3e-6', 'verified-v3-adamw-1e-5',
          'verified-v3-muon-3e-4', 'verified-v3-muon-3e-3',
          'verified-v3-normuon-1e-3', 'verified-v3-normuon-3e-3')
TASKS = ('ArguAna', 'ClimateFEVER', 'DBPedia', 'FEVER', 'FiQA2018', 'HotpotQA',
         'MSMARCO', 'NFCorpus', 'NQ', 'QuoraRetrieval', 'SCIDOCS', 'SciFact',
         'TRECCOVID', 'Touche2020')
FALSE_FLAGS = ('available_case_mean', 'optimizer_ranking_or_selection',
               'full_primary_grid_complete', 'model_or_retrieval_recomputed',
               'cuda_initialized', 'committed_source_release', 'scientific_completion')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def strict_json(raw):
    def invalid(value):
        raise ValueError('Non-finite JSON constant: ' + value)

    return json.loads(raw, object_pairs_hook=unique_keys, parse_constant=invalid)


def mteb_json(raw):
    """Byte-input adaptation of the frozen reader's explicit unused-nAUC policy."""
    value = json.loads(raw, object_pairs_hook=unique_keys)
    undefined = []
    allowed = re.compile(
        r"nauc_(ndcg|map|recall|precision|mrr)_at_(1|3|5|10|20|100|1000)_(max|std|diff1)"
    )
    for split, rows in value.get("scores", {}).items():
        for index, row in enumerate(rows):
            for name, score in row.items():
                if allowed.fullmatch(name) and isinstance(score, float) and math.isnan(score):
                    undefined.append({"split": split, "subset_row": index, "metric": name})
                    row[name] = None  # In memory only; original bytes remain unchanged.
    json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return value, undefined


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_bound(path, expected):
    path = Path(path)
    require(path.is_absolute() and path.is_file()
            and not any(p.is_symlink() for p in (path, *path.parents)), 'Ordinary absolute input required')
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require(all(getattr(before, k) == getattr(after, k)
                for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')), 'Moving input')
    require(isinstance(expected, str) and len(expected) == 64 and sha(raw) == expected,
            'Input differs from external SHA-256 anchor')
    return raw, {'path': str(path), 'bytes': len(raw), 'sha256': expected}


def embedded(item):
    require(set(item) == {'path', 'bytes', 'sha256', 'text'}, 'Unexpected embedded snapshot schema')
    path = PurePosixPath(item['path'])
    require(path.is_absolute() and path.as_posix() == item['path']
            and '..' not in path.parts and '\\' not in item['path'], 'Invalid producer provenance path')
    raw = item['text'].encode()
    require(type(item['bytes']) is int and len(raw) == item['bytes']
            and sha(raw) == item['sha256'], 'Embedded original bytes differ')
    return raw  # Producer paths are never opened, resolved or executed.


def validate_record(record):
    run, step = record['run_id'], record['step']
    require(run in RUNS and type(step) is int and step in STEPS, 'Undeclared checkpoint')
    complete = strict_json(embedded(record['original_complete_receipt']))
    operational = strict_json(embedded(record['original_operational_receipt']))
    require(complete == record['native_complete_reread'] and complete['complete_task_files_verified'] is True
            and complete['scientific_completion'] is False, 'Original checkpoint acceptance differs')
    require(operational['source_sha256'] == DISPATCH and operational['authorization_sha256'] == AUTH
            and all(operational[k] is False for k in ('scientific_completion',
                'committed_source_release', 'native_released_contract_guard_passed')), 'Wrong operational scope')
    plan = complete['plan']
    checkpoint = plan['checkpoint']
    require(checkpoint['run_id'] == run and type(checkpoint['step']) is int and checkpoint['step'] == step
            and checkpoint['protocol_sha256'] == plan['protocol_sha256'] == PROTOCOL
            and plan['tasks'] == list(TASKS) and plan['scientific_completion'] is False,
            'Checkpoint/protocol/task identity differs')
    cache_key = sha(json.dumps({'protocol': PROTOCOL, 'checkpoint': checkpoint},
                              sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    require(plan['cache_key'] == cache_key and PurePosixPath(plan['results_root']).name == cache_key,
            'Checkpoint cache identity differs')
    require(type(record['task_count']) is int and record['task_count'] == 14
            and [r['task'] for r in complete['tasks']] == list(TASKS), 'Incomplete task coverage')
    raw_files = {}
    for item in record['raw_score_and_metadata_snapshots']:
        require(item['path'] not in raw_files, 'Duplicate embedded raw file')
        raw_files[item['path']] = (item, embedded(item))
    require(len(raw_files) == 16, 'Incomplete native score/metadata population')
    used, values = set(), []
    for row in complete['tasks']:
        require(len(row['files']) == 3, 'Wrong task file population')
        for identity in row['files']:
            require(identity['path'] in raw_files, 'Missing original raw file')
            item = raw_files[identity['path']][0]
            require(all(item[k] == identity[k] for k in ('bytes', 'sha256')), 'Raw file binding differs')
            used.add(identity['path'])
        score_files = [f for f in row['files']
                       if PurePosixPath(f['path']).name == row['task'] + 'Decontaminated.json']
        require(len(score_files) == 1, 'Wrong canonical task filename')
        raw_score, undefined = mteb_json(raw_files[score_files[0]['path']][1])
        require(undefined == row['undefined_auxiliary_metrics_not_used'],
                'Original unused-auxiliary disclosure differs')
        split = 'dev' if row['task'] == 'MSMARCO' else 'test'
        require(raw_score['task_name'] == row['task'] + 'Decontaminated'
                and list(raw_score['scores']) == [split] and len(raw_score['scores'][split]) == 1,
                'Wrong native task/split')
        score_row = raw_score['scores'][split][0]
        value = score_row['ndcg_at_10']
        require(type(value) is float and math.isfinite(value) and 0 <= value <= 1
                and score_row['hf_subset'] == 'default'
                and value == row['ndcg_at_10'] == score_row['main_score'], 'Wrong bounded raw task score')
        values.append(value)
    require(used == set(raw_files), 'Unconsumed raw score/metadata file')
    rational = sum(map(Fraction.from_float, values), Fraction()) / 14
    require(float(rational) == record['macro_ndcg_at_10']
            and float(rational * 100) == record['macro_score_0_to_100']
            and record['exact_binary64_input_mean'] == {
                'numerator': rational.numerator, 'denominator': rational.denominator}, 'Exact macro mean differs')
    require(len(record['original_workers']) == 14, 'Incomplete original worker population')
    for task, worker in zip(TASKS, record['original_workers'], strict=True):
        require(set(worker) == {'started', 'exited'}, 'Wrong original worker proof schema')
        started = strict_json(embedded(worker['started']))
        exited = strict_json(embedded(worker['exited']))
        require(started['job'] == exited['job'] == [run, step, task]
                and type(exited['exit_code']) is int and exited['exit_code'] == 0
                and type(started['pid']) is int and started['pid'] == exited['pid']
                and type(started['start_ticks']) is int and started['start_ticks'] > 0
                and started['source_sha256'] == DISPATCH and started['authorization_sha256'] == AUTH
                and started['both_gpu_lease_namespaces_inherited'] is True, 'Wrong native worker identity/exit')
        command = started['command']
        require(command[command.index('--worker') + 1] == task
                and command[command.index('--models') + 1].endswith('/' + run + '/checkpoint-' + str(step))
                and command[command.index('--results_folder') + 1]
                == plan['results_root'] + '/dense/' + run + '__checkpoint-' + str(step), 'Wrong worker command')
    optimizer, rate = run.removeprefix('verified-v3-').split('-', 1)
    stage = STEPS.index(step) + 1
    return [{'run_id': run, 'model_family': 'dense', 'optimizer': optimizer,
             'learning_rate': float(rate), 'stage': stage, 'step': step,
             'fraction': stage / 5, 'task': task, 'ndcg_at_10': value}
            for task, value in zip(TASKS, values, strict=True)]


def complete_rows(current, prior):
    require(current['scope'] == prior['scope'] == 'descriptive_complete_fourteen_task_checkpoint_readback'
            and current['protocol_sha256'] == prior['protocol_sha256'] == PROTOCOL
            and all(current[k] is False and prior[k] is False for k in FALSE_FLAGS), 'Wrong native reader scope')
    require(sha(embedded(current['source'])) == READER and sha(embedded(prior['source'])) == READER
            and current['dispatcher_source'] == prior['dispatcher_source']
            and current['assembly'] == prior['assembly'], 'Original reader/source assembly differs')
    wanted = {(run, step) for run in RUNS for step in STEPS}
    old_wanted = {(run, step) for run in RUNS for step in STEPS if step != 3126}
    old_wanted |= {(run, 3126) for run in POOL_B}
    old = {(r['run_id'], r['step']): r for r in prior['checkpoints']}
    records = {(r['run_id'], r['step']): r for r in current['checkpoints']}
    require(len(old) == len(prior['checkpoints']) == 54 and set(old) == old_wanted, 'Wrong accepted prior cohort')
    require(len(records) == len(current['checkpoints']) == 60 and set(records) == wanted,
            'Require all 60 checkpoints and 840 task cells; missing states are not imputed')
    require(all(records[key] == value for key, value in old.items()), 'A prior complete checkpoint changed')
    rows = []
    for run in RUNS:
        identity = old[(run, 3907)]['native_complete_reread']['plan']['checkpoint']['run_identity_sha256']
        for step in STEPS:
            record = records[(run, step)]
            require(record['native_complete_reread']['plan']['checkpoint']['run_identity_sha256'] == identity,
                    'A checkpoint belongs to a different trained run')
            rows.extend(validate_record(record))
    require(len(rows) == 840, 'Wrong complete task population')
    return rows


def load_kernel(repository):
    contents, bindings = {}, {}
    for name, expected in SOURCES.items():
        contents[name], bindings[name] = read_bound(repository / name, expected)
    primary = strict_json(contents['configs/dense_primary_v3_protocol.json'])
    parent = strict_json(contents['configs/dense_no_packing_outcome_protocol.json'])
    outcome = strict_json(contents['configs/dense_primary_v3_outcome_protocol.json'])
    require(primary['evaluation']['tasks'] == list(TASKS) and primary['checkpoint_steps'] == list(STEPS),
            'Different task order or checkpoint schedule')
    for key in ('inference', 'dynamics'):
        require(outcome['scientific_rules'][key] == parent[key], 'Frozen scientific rule changed')
    namespace = {'np': np, 'math': math, 'statistics': statistics, 'defaultdict': defaultdict,
                 'Any': Any, 'RunConfig': object, 'DECONTAMINATED_TASK_NAMES': TASKS}
    source = contents['src/embed_optim/corrected_outcome_summary.py'].decode()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('OPTIMIZERS', 'CONTRASTS'):
                namespace[node.targets[0].id] = ast.literal_eval(node.value)
    functions = {}
    for name in ('_validate_matrix', '_index_score_rows', '_task_effects',
                 'paired_max_t_intervals', 'summarize_score_rows'):
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        require(len(nodes) == 1 and not nodes[0].decorator_list, 'Ambiguous original pure function')
        node = nodes[0]
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(repository / 'src/embed_optim/corrected_outcome_summary.py'), 'exec'), namespace)
        functions[name] = sha(ast.get_source_segment(source, node).encode())
    return namespace['summarize_score_rows'], bindings, functions, parent


def verify_dynamics(rows, tables):
    indexed = {(r['run_id'], r['stage'], r['task']): Fraction.from_float(r['ndcg_at_10']) for r in rows}
    means = {(run, stage): sum((indexed[(run, stage, task)] for task in TASKS), Fraction()) / 14
             for run in RUNS for stage in range(1, 6)}
    for row in tables['run_stage_scores']:
        require(abs(row['mean_ndcg_at_10'] - float(means[(row['run_id'], row['stage'])])) <= 1e-15,
                'Independent run-stage mean differs')
    for row in tables['optimizer_stage_scores']:
        members = [run for run in RUNS if run.startswith('verified-v3-' + row['optimizer'] + '-')]
        expected = sum((means[(run, row['stage'])] for run in members), Fraction()) / 4
        require(abs(row['mean_ndcg_at_10_across_rates'] - float(expected)) <= 1e-15,
                'Independent optimizer-stage mean differs')
    for row in tables['run_observed_auc']:
        run = row['run_id']
        area = sum(((means[(run, stage)] + means[(run, stage + 1)]) / 10
                    for stage in range(1, 5)), Fraction())
        require(abs(row['observed_auc_20_to_100'] - float(area)) <= 1e-15
                and abs(row['observed_mean_20_to_100'] - float(area * Fraction(5, 4))) <= 1e-15,
                'Independent exact-rational observed-range area differs')
    return {'run_stage_means': 60, 'optimizer_stage_means': 15, 'run_areas_and_normalized_means': 24,
            'absolute_tolerance_ndcg_units': 1e-15, 'initialization_imputed': False}


def build(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Require CPU-only readout')
    require(args.output.is_absolute() and not args.output.exists()
            and not any(p.is_symlink() for p in args.output.parents), 'Preserve outputs; use a new ordinary directory')
    current_raw, current_id = read_bound(args.bundle, args.bundle_sha256)
    prior_raw, prior_id = read_bound(args.prior_bundle, PRIOR_54)
    endpoint_raw, endpoint_id = read_bound(args.endpoint_readout, ENDPOINT)
    current, prior, endpoint = map(strict_json, (current_raw, prior_raw, endpoint_raw))
    rows = complete_rows(current, prior)  # No output, curve or figure before complete acceptance.
    summarize, source_ids, functions, protocol = load_kernel(args.repository)
    require(endpoint['manifest_sha256'] == 'bf443d5623c1312f8661a918ab93f9aea5f119058307733d477c8eac0c6da74f'
            and endpoint['task_cells'] == 168 and endpoint['model_execution'] is False
            and endpoint['scientific_completion'] is False and endpoint['numpy_version'] == np.__version__,
            'Different accepted endpoint parent or numerical environment')
    # Metadata-only views for the unchanged pure summarizer, not model admission objects.
    configs = [SimpleNamespace(run_id=run, model_family='dense', dense_can_flatten_inputs=False,
                              checkpoint_fractions=(.2, .4, .6, .8, 1.),
                              optimizer=SimpleNamespace(name=optimizer, lr=float(rate)))
               for optimizer, rates in RATE_NAMES.items() for rate in rates
               for run in ['verified-v3-' + optimizer + '-' + rate]]
    tables = summarize(rows, configs, endpoint['selected'],
                       bootstrap_samples=protocol['inference']['bootstrap_samples'],
                       bootstrap_seed=protocol['inference']['bootstrap_seed'])
    for family in ('primary', 'secondary'):
        require(tables[family + '_task_effects'] == endpoint['tables'][family + '_task_effects.csv'],
                'Previously accepted endpoint task effects changed')
        for new, old in zip(tables[family + '_summary'], endpoint['tables'][family + '_summary.csv'], strict=True):
            require({key: new[key] for key in old} == old, 'Previously accepted endpoint interval/decision changed')
    counts = {'primary_task_effects': 14, 'primary_summary': 3, 'secondary_task_effects': 14,
              'secondary_summary': 3, 'run_stage_scores': 60, 'optimizer_stage_scores': 15,
              'run_observed_auc': 12}
    require(set(tables) == set(counts) and all(len(tables[k]) == n for k, n in counts.items()),
            'Incomplete original outcome table population')
    checks = verify_dynamics(rows, tables)
    tables['all_task_scores'] = rows
    args.output.mkdir()
    outputs = {}
    for name, table in tables.items():
        path = args.output / (name + '.csv')
        with path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(table[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(table)
        raw = path.read_bytes()
        outputs[path.name] = {'bytes': len(raw), 'sha256': sha(raw), 'rows': len(table)}
    report = {'scope': 'complete_retained_retrieval_trajectory_readout',
              'observed_at_utc': datetime.now(timezone.utc).isoformat(),
              'source': {'bytes': Path(__file__).stat().st_size, 'sha256': sha(Path(__file__).read_bytes())},
              'native_bundle': current_id, 'prior_54_bundle': prior_id, 'accepted_endpoint': endpoint_id,
              'frozen_sources': source_ids, 'unchanged_function_sha256': functions,
              'numpy_version': np.__version__, 'selected': endpoint['selected'],
              'checkpoint_count': 60, 'task_score_count': 840, 'prior_54_records_exactly_unchanged': True,
              'all_six_endpoint_contrasts_unchanged': True, 'independent_dynamics_checks': checks,
              'dynamics_rule': protocol['dynamics'], 'outputs': outputs,
              'model_or_retrieval_recomputed': False, 'new_inferential_rule': False,
              'original_whole_grid_consumer_called': False, 'original_whole_grid_admission_passed': False,
              'functional_or_causal_acceptance': False, 'source_release': False, 'scientific_completion': False,
              'boundary': 'Complete observed retrieval trajectories only. Task-level endpoint inference remains unchanged. Not a new training seed, tensor audit, functional explanation, formal source/runtime admission or completed manuscript.'}
    with (args.output / 'readout.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repository', 'bundle', 'prior-bundle', 'endpoint-readout', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--bundle-sha256', required=True)
    report = build(parser.parse_args())
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
