"""CPU-only continuation from the completed 61-state raw-vector matrix.

The failed original feature attempt stays untouched. Computation, serialization,
fresh raw-vector recomputation and full table reading are unchanged. This new
entry prepares required nested parents and writes a separate feature namespace.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
EXPERIMENT = Path('/root/embedding-optimizer-v3-experiment')
DEPLOY = EXPERIMENT / 'launch/functional-features-recovery-v2'
OUTPUT = EXPERIMENT / 'analyses/dense-primary-v3-functional-features-recovery-v2'
PRIOR = EXPERIMENT / 'launch/functional-dimensions-recovery-v1'
VECTOR_ROOT = EXPERIMENT / 'analyses/dense-primary-v3-functional-dimensions-recovery-v1/vectors'
RUN = HERE / 'run'
SCOPE = 'owner_authorized_cpu_functional_features_recovery_20260912'
OWNER = '你有权做一切事情，目标是尽快完成任务'
PINS = {
    PRIOR / 'recovery.py': 'f0d732610f715b6463e92a9acc1c6c3e9fc414d981c5384ae7ab0aca3e491040',
    PRIOR / 'recovery_support.py': '8c4644a691b09898714a718816d0c93d0af1e7a6f798a92d6380eeb43a8ebd85',
    PRIOR / 'authorization.json': '50414c3541ea3031c4388c9b21274b33faf0676ce74abd44d3ccd1bfea398122',
    PRIOR / 'run/vectors.completed.json': 'b1f1dd989848ef0e98619e13ac466ec0ce06de6e6d2766342530d6efd98f5f23',
    PRIOR / 'run/features.exited.json': '4eadbb5ef765e93223e7896f3e95b68770d96647e5548fc4269aa9fb6c8f70ab',
    PRIOR / 'run/failed.json': '6ec5eaced1f3effcfed3d8ab666933a16b92f2f7e937f78d7d77f801779d94ae',
    VECTOR_ROOT / 'manifest.json': '9bf90d652d54ee6b9b571bf9dc88cd919da8c78bb70db0d476e307733e8c23e7',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Nonordinary input')
    first = path.stat()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    last = path.stat()
    need(all(getattr(first, key) == getattr(last, key) for key in
         ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')), 'Input raced')
    return {'bytes': last.st_size, 'sha256': sha}


def read(path, sha=None):
    before = identity(path)
    need(sha is None or before['sha256'] == sha, 'External source/data binding differs')
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
        parse_constant=lambda _: need(False, 'Nonfinite JSON'))
    need(identity(path) == before, 'JSON raced')
    return value


def require_inputs():
    for path, sha in PINS.items():
        need(identity(path)['sha256'] == sha, 'Completed vectors or preserved prior attempt changed')
    need(read(PRIOR / 'run/features.exited.json')['exit_code'] == 1, 'Prior feature process is not terminal failure')
    raw = read(PRIOR / 'run/vectors.completed.json')
    need(raw['all_states_encoded_and_native_readback_verified'] == 61
         and raw['authorization_sha256'] == PINS[PRIOR / 'authorization.json']
         and raw['manifest']['path'] == str(VECTOR_ROOT / 'manifest.json')
         and raw['manifest']['sha256'] == PINS[VECTOR_ROOT / 'manifest.json'],
         'Require the actual complete raw-vector matrix')
    return raw


def original():
    require_inputs()
    # Load only the exact known support/entry, never an old main/coordinator.
    for name, filename in (('recovery_support', 'recovery_support.py'),
                            ('completed_functional_vector_parent', 'recovery.py')):
        spec = importlib.util.spec_from_file_location(name, PRIOR / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def authenticate(args):
    need(HERE == DEPLOY and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'New explicit CPU-only feature entry required')
    need(not any(p.is_symlink() for p in (HERE, *HERE.parents, OUTPUT, *OUTPUT.parents)), 'Symlinked new namespace')
    need(identity(__file__)['sha256'] == args.source_sha, 'New feature entry changed')
    approval = read(HERE / 'owner-approval.json', args.approval_sha)
    need(approval == {'scope': SCOPE, 'source_sha256': args.source_sha, 'owner_message': OWNER,
        'automatic_continuation': False, 'cpu_only': True, 'protected_helper_access': False,
        'old_controller_transition': False, 'preserve_original_vectors_and_failed_features': True,
        'native_numerical_sources_changed': False, 'new_feature_output': str(OUTPUT),
        'scientific_completion': False}, 'Wrong owner-scoped feature repair approval')
    tests = read(HERE / 'tests.json', args.tests_sha)
    need(tests['source_sha256'] == args.source_sha and tests['tests_run'] >= 8
         and tests['errors'] == tests['failures'] == tests['skipped'] == 0
         and tests['test_source'] == identity(HERE / 'test_features.py'), 'Untested new feature boundary')
    prior = original()
    entry, context, authority = prior.authenticate(SimpleNamespace(
        source_sha=PINS[PRIOR / 'recovery.py'], authorization_sha=PINS[PRIOR / 'authorization.json']))
    return prior, entry, context, authority


def prepare_parent(location, root):
    location, root = Path(location), Path(root)
    need(location.is_relative_to(root) and location != root
         and not any(p.is_symlink() for p in (location, *location.parents)), 'Unsafe new state/record parent')
    need(not location.exists(), 'Preserve any existing state/record')
    location.parent.mkdir(parents=True, exist_ok=True)


def compute(args):
    prior, entry, context, authority = authenticate(args)
    os.nice(prior.SETTINGS['cpu_feature_nice'])
    from embed_optim.primary_contract import require_same
    from embed_optim.primary_v3_dimension_contract import TABLE_COUNTS
    from embed_optim.primary_v3_outcomes import csv_bytes
    raw = require_inputs()
    vectors, output = VECTOR_ROOT, OUTPUT
    need(not output.exists() and not output.is_symlink(), 'Preserve partial/prior feature matrix')
    manifest = context.exports._manifest(vectors, context.admitted, raw['manifest']['sha256'])
    plan = context.features.feature_plan(context.contract, context.admitted, raw['manifest']['sha256'])
    (output / 'states').mkdir(parents=True, exist_ok=False)
    context.geometry.write_new(output / 'admission.json', plan)
    tables = {key: [] for key in TABLE_COUNTS}
    state_receipts = {}
    tasks = sorted({row['source'] for row in context.identities})
    for job, arrays, _ in context.exports.iter_vectors(vectors, manifest, context.jobs, context.identities):
        state = job['plan']['state']
        cell = state['cell']
        result = context.kernels.compute_state(state['meta'], arrays, context.contract.scientific)
        context.features.check_result(result, state, tasks)
        current = context.features.state_plan(plan, job, manifest['states'][cell])
        location = output / 'states' / cell
        prepare_parent(location, output / 'states')
        context.feature_io.save_state(location, current, result)
        del arrays, result
        fresh, _ = context.vectors.inspect_vectors(vectors / 'states' / cell, job['plan'],
            context.identities, job['checkpoint'],
            expected_manifest_sha256=manifest['states'][cell]['sha256'])
        replay = context.kernels.compute_state(state['meta'], fresh, context.contract.scientific)
        context.features.check_result(replay, state, tasks)
        context.feature_io.inspect_state(location, current, replay)
        state_receipts[cell] = {'path': f'states/{cell}/manifest.json',
                                **context.geometry.identity(location / 'manifest.json')}
        for key, rows in replay['tables'].items():
            tables[key].extend(rows)
        receipt_path = RUN / 'jobs' / f'{cell}.features-verified.json'
        prepare_parent(receipt_path, RUN / 'jobs')
        context.geometry.write_new(receipt_path, {
            'cell': cell, 'verified_at_utc': context.geometry.now(),
            'manifest': state_receipts[cell], 'fresh_raw_vectors_fully_recomputed': True,
            'scientific_completion': False})
        print(json.dumps({'event': 'functional_features_verified', 'cell': cell,
                          'verified_states': len(state_receipts)}), flush=True)
        del fresh, replay
    require_same({key: len(rows) for key, rows in tables.items()}, TABLE_COUNTS)
    after, _ = entry.authenticate(prior.FUNCTIONAL_SHA, prior.INPUTS_SHA)
    require_same(after.admitted, context.admitted)
    context.exports._manifest(vectors, context.admitted, raw['manifest']['sha256'])
    context.exports.recheck_contract(context.contract)
    records = {}
    for key, rows in tables.items():
        path = output / (key + '.csv')
        with path.open('xb') as stream:
            stream.write(csv_bytes(rows))
        records[key] = {'path': path.name, 'rows': len(rows), **context.geometry.identity(path)}
    expected = {'status': 'complete', 'plan': plan, 'states': state_receipts, 'tables': records}
    context.geometry.write_new(output / 'manifest.json', expected)
    require_same(context.features._feature_manifest(output, plan, context.jobs), expected)
    require_inputs()
    result = {'scope': SCOPE, 'completed_at_utc': context.geometry.now(),
        'source_sha256': args.source_sha, 'owner_approval': identity(HERE / 'owner-approval.json'),
        'tests': identity(HERE / 'tests.json'), 'prior_vector_authorization_sha256': PINS[PRIOR / 'authorization.json'],
        'input_vectors': raw['manifest'], 'states': len(state_receipts),
        'table_counts': TABLE_COUNTS, 'fresh_raw_vector_recomputation': True,
        'manifest': {'path': str(output / 'manifest.json'), **context.geometry.identity(output / 'manifest.json')},
        'scientific_completion': False}
    context.geometry.write_new(RUN / 'features.completed.json', result)
    return result


def coordinate(args):
    prior, entry, context, _ = authenticate(args)
    need(not RUN.exists() and not OUTPUT.exists(), 'Never restart an existing feature attempt')
    (RUN / 'jobs').mkdir(parents=True)
    context.geometry.write_new(RUN / 'coordinator.started.json', {
        **prior.process_identity(os.getpid()), 'started_at_utc': context.geometry.now(),
        'source_sha256': args.source_sha, 'approval_sha256': args.approval_sha,
        'tests_sha256': args.tests_sha, 'cuda_hidden': True, 'scope': SCOPE})
    argv = ['/usr/bin/python', '-B', str(Path(__file__).resolve()), '--source-sha', args.source_sha,
        '--approval-sha', args.approval_sha, '--tests-sha', args.tests_sha, '--compute']
    env = prior.environment(context, entry)
    child = None
    try:
        with (RUN / 'features.log').open('xb') as log:
            child = subprocess.Popen(argv, cwd=entry.STORY, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, close_fds=True)
            context.geometry.write_new(RUN / 'features.started.json', {
                **prior.process_identity(child.pid), 'command': argv,
                'started_at_utc': context.geometry.now(), 'source_sha256': args.source_sha,
                'approval_sha256': args.approval_sha, 'tests_sha256': args.tests_sha, 'cuda_hidden': True})
            code = child.wait()
        context.geometry.write_new(RUN / 'features.exited.json', {
            'exit_code': code, 'exited_at_utc': context.geometry.now(), 'pid': child.pid,
            'source_sha256': args.source_sha, 'approval_sha256': args.approval_sha})
        need(code == 0, 'CPU feature computation failed; preserve outputs')
        completed = read(RUN / 'features.completed.json')
        need(completed['states'] == 61 and completed['source_sha256'] == args.source_sha,
             'Incomplete feature matrix')
        context.geometry.write_new(RUN / 'coordinator.completed.json', {
            'completed_at_utc': context.geometry.now(), 'source_sha256': args.source_sha,
            'actual_feature_worker_exit': code, 'feature_completion': identity(RUN / 'features.completed.json'),
            'all_61_states_twice_computed_and_native_verified': True, 'scientific_completion': False})
        return completed
    except BaseException as exc:
        if child is not None and child.poll() is None:
            # This is the exact direct child we created, not any external process.
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        context.geometry.write_new(RUN / 'failed.json', {'failed_at_utc': context.geometry.now(),
            'exception_type': type(exc).__name__, 'automatic_retry': False,
            'preserve_partial_outputs': True, 'scientific_completion': False})
        raise


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-sha', 'approval-sha', 'tests-sha'):
        parser.add_argument('--' + name, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--coordinate', action='store_true')
    mode.add_argument('--compute', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse()
    print(json.dumps((coordinate if args.coordinate else compute)(args), sort_keys=True), flush=True)
