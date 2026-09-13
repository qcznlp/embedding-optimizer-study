"""Current native-to-W&B read-only tracking audit, no remote mutations."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import threading
from typing import Any

HERE = Path(__file__).resolve().parent
STORY = Path('/root/embedding-optimizer-story-refactor')
EXP = Path('/root/embedding-optimizer-v3-experiment')
PRIMARY = STORY / 'reports/engineering-archive/dense-v3-closed-primary-paper-replay-v1/closed-primary'
TRAINING_SOURCE = Path('/root/embedding-optimizer-primary-v3')
NATIVE = STORY / 'reports/engineering-archive/dense-v3-current-publication-consumer-v1/actual/native-primary/evidence.json'
TRAIN = EXP / 'launch/factorial-training-v1'
TRAIN_AUTH = '00c52f0bec979fb163a734ad0f9eadcd6ea09f7393133fd35c65bc62878ce11f'
KEYS = ['train/global_step', 'train/loss', 'train/grad_norm', 'train/learning_rate']
CONFIG_FIELDS = ('run_id', 'model_family', 'optimizer', 'model_name', 'model_revision', 'dataset_path',
    'output_root', 'seed', 'epochs', 'global_batch_size', 'micro_batch_size', 'temperature',
    'max_length', 'warmup_ratio', 'max_grad_norm', 'dataloader_workers', 'gradient_checkpointing',
    'flash_attention', 'dense_can_flatten_inputs', 'wandb_project', 'wandb_entity',
    'checkpoint_fractions', 'numerical_policy', 'learning_rate', 'num_train_epochs',
    'gradient_accumulation_steps', 'per_device_train_batch_size', 'run_name', 'report_to',
    'logging_first_step', 'logging_steps', 'warmup_steps', 'bf16', 'fp16', 'tf32', 'data_seed',
    'optim', 'weight_decay', 'max_steps', 'save_strategy', 'dataloader_drop_last')
LOCAL_BINDINGS = {}
THREAD = threading.local()


def need(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Ordinary input required')
    before = path.stat()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need(all(getattr(before, k) == getattr(after, k) for k in
             ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')), 'Input raced')
    return {'bytes': after.st_size, 'sha256': sha}


def read(path, expected=None):
    actual = identity(path)
    if expected is not None:
        need(actual['sha256'] == expected if isinstance(expected, str)
             else actual == {k: expected[k] for k in actual}, 'Native input binding differs: ' + str(path))
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite native input')
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=invalid)
    need(identity(path) == actual, 'Input changed')
    LOCAL_BINDINGS[str(path)] = actual
    return value


def write(path, value):
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    patterns = (r'wandb_v1_[A-Za-z0-9_-]{40,}', r'\bhf_[A-Za-z0-9]{30,}',
                r'\bgh[pousr]_[A-Za-z0-9]{30,}', r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
    need(not any(re.search(p, raw) for p in patterns), 'Credential-shaped payload refused')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(raw)


def inputs():
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('PYTHONPATH') == str(TRAINING_SOURCE / 'src'),
         'Use the original primary training configuration namespace and hide CUDA')
    from embed_optim.config import RunConfig, source_wandb_run_id
    from embed_optim.primary_contract import digest
    import embed_optim.config as native_config
    need(Path(native_config.__file__).resolve() == TRAINING_SOURCE / 'src/embed_optim/config.py', 'Wrong original configuration source')
    original_path = STORY / 'src/embed_optim/corrected_wandb_audit.py'
    names = {'_normalized', '_expected_remote_config', '_summary_value', '_finite_number', 'audit_run'}
    nodes = [n for n in ast.parse(original_path.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    need(len(nodes) == len(names) and {n.name for n in nodes} == names, 'Original pure tracking checks missing')
    namespace = dict(json=json, math=math, Any=Any, RunConfig=RunConfig, source_wandb_run_id=source_wandb_run_id)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(original_path), 'exec'), namespace)
    original_audit_run = namespace['audit_run']
    for path in (TRAINING_SOURCE / 'src/embed_optim/config.py', original_path,
                 TRAINING_SOURCE / 'src/embed_optim/primary_contract.py'):
        LOCAL_BINDINGS[str(path)] = identity(path)
    manifest = read(PRIMARY / 'manifest.json', '88861e52218f56b860851489106cd30e35dd79bb038eb9082967e3376db6457d')
    catalog = read(PRIMARY / 'inputs/recipe-catalog.json', manifest['files']['inputs/recipe-catalog.json'])
    native = read(NATIVE, '4915423f782a5ce952bc6c253f85568d691e435908542b93d2bf0154b4d26317')
    need(native['current_native_primary_admission_complete'] is True
         and native['original_single_selection_guard_passed'] is False, 'Native primary scope differs')
    primary_rows = {r['run_id']: r for r in native['grid']['runs']}
    need(len(primary_rows) == 12 and set(primary_rows) == set(catalog['expected_identities']), 'Incomplete primary native population')
    result = []
    for run_id, proof in sorted(primary_rows.items()):
        root = EXP / catalog['payload']['output_root'] / 'dense' / run_id
        expected = catalog['expected_identities'][run_id]
        need(proof['whole_run_artifacts_verified'] is True and proof['run_identity_sha256'] == digest(expected),
             'Incomplete primary native run identity')
        config_dict = read(root / 'run_config.json', proof['metadata']['run_config.json'])
        state = read(root / 'trainer_state_final.json', proof['metadata']['trainer_state_final.json'])
        contract = read(root / 'dense_run_contract.json', proof['metadata']['dense_run_contract.json'])
        need(contract == expected, 'Current run differs from accepted native identity')
        config = RunConfig.from_dict(config_dict)
        result.append({'family': 'primary', 'run_id': run_id, 'remote_id': source_wandb_run_id(config),
             'entity': config.wandb_entity, 'project': config.wandb_project, 'max_steps': 3907,
             'local_state': state, 'local_recipe': config_dict, 'config_object': config,
             'native_run_identity_sha256': proof['run_identity_sha256'],
             'native_whole_run_verified': proof['whole_run_artifacts_verified'],
             'primary_auditor': original_audit_run, 'expected_arguments': {}, 'native_factory': None})
    auth = read(TRAIN / 'authorization.json', TRAIN_AUTH)
    need(set(auth['queues']) == {'a', 'b'} and all(len(r) == 6 for r in auth['queues'].values()), 'Wrong continuation queues')
    for pool, runs in sorted(auth['queues'].items()):
        for run_id in runs:
            req = auth['requests'][run_id]
            job = TRAIN / 'run' / ('pool-' + pool) / run_id
            completed = read(job / 'completed.json')
            need(completed['authorization_sha256'] == TRAIN_AUTH and completed['source_sha256'] == auth['source_sha256']
                 and completed['run_id'] == run_id and completed['actual_rank_exits'] == [0, 0, 0, 0]
                 and completed['actual_fresh_reader_exit'] == 0
                 and completed['all_five_checkpoints_verified'] is True, 'Incomplete continuation production')
            worker = read(Path(req['record_root']) / 'worker-complete.json', completed['worker_completion'])
            fresh = read(job / 'fresh-native-readback.json', completed['native_readback_binding'])
            need(worker['request'] == req and worker['status'] == 'training_and_native_readback_complete', 'Wrong original worker')
            factory = read(Path(req['record_root']) / 'factory.json', worker['factory_record'])
            run_identity = worker['native_readback']['run_identity']
            need(factory['creation']['run_identity_sha256'] == digest(run_identity), 'Factory/native identity differs')
            native_check = [v for v in worker['native_readback']['artifacts']['checkpoints'] if v['step'] == 391]
            need(len(native_check) == 1, 'Missing final native checkpoint')
            checkpoint = Path(req['run_root']) / 'checkpoint-391'
            component = read(checkpoint / 'factorial_trainer_component.json', native_check[0]['component_binding']['sha256'])
            record = [r for r in component['files'] if r['path'] == 'trainer_state.json']
            need(len(record) == 1, 'Missing bound final Trainer state')
            state = read(checkpoint / 'trainer_state.json', record[0])
            recipe = factory['creation']['recipe']
            result.append({'family': 'continuation', 'run_id': run_id, 'remote_id': run_id,
                'entity': req['entity'], 'project': req['project'], 'max_steps': 391, 'local_state': state,
                'local_recipe': recipe, 'native_run_identity_sha256': digest(run_identity),
                'native_whole_run_verified': worker['native_readback']['artifacts']['whole_run_artifacts_verified'],
                'config_object': None, 'primary_auditor': None,
                'expected_arguments': factory['creation']['requested_arguments'],
                'native_factory': factory['creation']})
    need(len(result) == 24 and len({r['remote_id'] for r in result}) == 24, 'Incomplete tracking scope')
    return result


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def ulps(a, b):
    if a == b:
        return 0
    def ordered(value):
        bits = struct.unpack('>Q', struct.pack('>d', float(value)))[0]
        return (~bits & 0xffffffffffffffff) if bits & 0x8000000000000000 else bits | 0x8000000000000000
    return abs(ordered(a) - ordered(b))


def metric_comparison(local, online, total, base_lr):
    expected_steps = [1, *range(10, total + 1, 10)]
    local_rows = [r for r in local['log_history'] if 'loss' in r]
    need(local['global_step'] == local['max_steps'] == total and local['epoch'] == 1, 'Native endpoint differs')
    need([r['step'] for r in local_rows] == expected_steps, 'Native log cadence differs')
    observed_steps = [r.get('train/global_step') for r in online]
    need(observed_steps == expected_steps and all(type(s) is int for s in observed_steps), 'Online log cadence differs')
    differences, schedules = [], []
    for left, right in zip(local_rows, online):
        for name in ('loss', 'grad_norm', 'learning_rate'):
            a, b = left[name], right['train/' + name]
            need(finite(a) and finite(b), 'Nonfinite log metric')
            if a != b:
                differences.append({'step': left['step'], 'metric': name, 'native': a, 'online': b,
                                    'absolute_difference': abs(a - b), 'ulp_distance': ulps(a, b)})
        previous = left['step'] - 1
        warmup = math.ceil(total * .1)
        scale = previous / max(1, warmup) if previous < warmup else max(0, (total - previous) / max(1, total - warmup))
        expected = base_lr * scale
        if left['learning_rate'] != expected:
            schedules.append({'step': left['step'], 'recorded': left['learning_rate'], 'expected': expected,
                              'absolute_difference': abs(left['learning_rate'] - expected)})
    return {'complete_cadence': True, 'finite_metrics': True, 'metric_rows': len(local_rows),
            'exact_native_online_match': not differences, 'numeric_differences': differences,
            'max_online_ulp_distance': max((d['ulp_distance'] for d in differences), default=0),
            'native_lr_schedule_exact': not schedules, 'native_lr_schedule_differences': schedules,
            'no_tolerance_used_to_relabel_equality': True}


def remote_read(spec, output):
    import wandb
    if not hasattr(THREAD, 'api'):
        THREAD.api = wandb.Api(timeout=30)
    remote = THREAD.api.run('/'.join((spec['entity'], spec['project'], spec['remote_id'])))
    observed = dict(remote.config or {})
    online = list(remote.scan_history(keys=KEYS, page_size=1000))
    metrics = metric_comparison(spec['local_state'], online, spec['max_steps'], spec['local_recipe']['optimizer']['lr'])
    summary = {k: remote.summary.get(k) for k in ('train/global_step', 'train/epoch')}
    need(str(remote.id) == spec['remote_id'] and remote.state == 'finished'
         and summary == {'train/global_step': spec['max_steps'], 'train/epoch': 1}, 'Remote endpoint differs')
    primary_check = None
    if spec['family'] == 'primary':
        primary_check = spec['primary_auditor'](spec['config_object'], remote,
            local_complete=spec['native_whole_run_verified'], expected_steps=spec['max_steps'], world_size=4)
    args = spec['expected_arguments']
    compared = [key for key in args if key in observed]
    missing = sorted(set(args) - set(observed))
    mismatches = {key: {'native': args[key], 'online': observed[key]} for key in compared if args[key] != observed[key]}
    config = {key: observed[key] for key in CONFIG_FIELDS if key in observed}
    result = {'scope': 'read-only-current-native-wandb-tracking', 'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'family': spec['family'], 'run_id': spec['run_id'], 'remote_id': str(remote.id),
        'url': remote.url, 'name': remote.name, 'state': remote.state, 'group': remote.group,
        'tags': sorted(remote.tags or []), 'summary': summary, 'online_config': config,
        'native_recipe': spec['local_recipe'], 'native_run_identity_sha256': spec['native_run_identity_sha256'],
        'native_factory': spec['native_factory'], 'native_whole_run_verified_upstream': spec['native_whole_run_verified'],
        'primary_original_per_run_audit': primary_check, 'argument_fields_compared': sorted(compared),
        'argument_fields_absent_online': missing, 'argument_exact_mismatches': mismatches,
        'research_recipe_fields_absent_online': sorted(set(spec['local_recipe']) - set(observed)),
        'metrics': metrics, 'online_history': online, 'native_history': spec['local_state']['log_history'],
        'online_mutation': False, 'fresh_native_model_read': False,
        'old_historical_whole_entry_guard_claimed': False, 'scientific_completion': False}
    write(output / (spec['run_id'] + '.json'), result)
    return {'family': spec['family'], 'run_id': spec['run_id'], 'remote_id': spec['remote_id'],
            'state': remote.state, 'history_rows': len(online), 'endpoint_step': spec['max_steps'],
            'primary_config_terminal_valid': primary_check['status'] == 'valid' if primary_check else None,
            'exact_history_match': metrics['exact_native_online_match'],
            'metric_differences': len(metrics['numeric_differences']), 'max_ulp': metrics['max_online_ulp_distance'],
            'native_schedule_exact': metrics['native_lr_schedule_exact'],
            'missing_research_fields_online': len(result['research_recipe_fields_absent_online']),
            'argument_mismatches': list(mismatches), 'record': identity(output / (spec['run_id'] + '.json'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(args.output.is_absolute() and not args.output.exists(), 'Use a new absolute tracking output')
    specs = inputs()
    args.output.mkdir(exist_ok=False)
    write(args.output / 'input-bindings.json', {'files': LOCAL_BINDINGS, 'source': identity(__file__),
          'scope': 'actual-native-records-reused-for-read-only-tracking', 'expected_runs': 24})
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(remote_read, spec, args.output / 'runs'): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                row = future.result()
                rows.append(row)
                print(json.dumps(row), flush=True)
            except Exception as error:
                failures.append({'run_id': spec['run_id'], 'exception_type': type(error).__name__})
                print(json.dumps({'read_only_query_failed': failures[-1]}), flush=True)
    for path, value in LOCAL_BINDINGS.items():
        need(identity(path) == value, 'Native evidence changed during read-only API audit')
    import wandb
    result = {'scope': 'complete-current-primary-and-continuation-read-only-tracking',
        'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'source': identity(__file__),
        'wandb_version': wandb.__version__, 'runs': sorted(rows, key=lambda r: (r['family'], r['run_id'])),
        'query_failures': failures, 'remote_finished_runs': len(rows),
        'all_24_native_online_histories_covered': len(rows) == 24 and not failures,
        'primary_config_and_terminal_passes': sum(r['primary_config_terminal_valid'] is True for r in rows),
        'exact_native_online_history_matches': sum(r['exact_history_match'] for r in rows),
        'total_metric_rows': sum(r['history_rows'] for r in rows),
        'all_native_schedules_exact': len(rows) == 24 and all(r['native_schedule_exact'] for r in rows),
        'input_bindings': identity(args.output / 'input-bindings.json'),
        'full_goal_complete': False, 'source_release': False, 'online_mutations': False,
        'old_original_whole_entry_gate_changed': False}
    write(args.output / 'readout.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('runs',)}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
