"""Owner-approved recovery of the 61-state functional measurement campaign.

Validation has resource priority. One GPU worker at a time inherits both existing
leases. Native numerical functions and old release guards are never modified.
This is operational primary measurement, not a committed/publication release.
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
import time
import traceback
import recovery_support as recovery

HERE = Path(__file__).resolve().parent
ORIGINAL = Path('/root/embedding-optimizer-v3-experiment/launch/functional-dimensions')
FUNCTIONAL = ORIGINAL / 'functional.py'
FUNCTIONAL_SHA = 'c55d3fa101fa681070509a95d987023b7ed442d89ce385a8885f940ea3cc05b8'
INPUTS_SHA = 'be6d92fda2f7f0ba3e86f2c2866fa9dca530f066d29ce92d0887fdad91684067'
SUPPORT_SHA = '8c4644a691b09898714a718816d0c93d0af1e7a6f798a92d6380eeb43a8ebd85'
RUN = HERE / 'run'
TOKENS = tuple(str(i) for i in range(8))
SETTINGS = {
    'validation_resource_priority': 'all_twelve_full_validations_before_first_encoding',
    'maximum_simultaneous_gpu_workers': 1,
    'both_existing_gpu_lease_namespaces_inherited': True,
    'native_encoding_and_feature_kernels_unchanged': True,
    'complete_states': 61,
    'score_based_state_selection': False,
    'overwrite_or_adopt_partial_results': False,
    'cpu_threads': 4,
    'cpu_feature_nice': 10,
    'feature_readback': 'fresh_raw_vector_reload_and_full_native_recomputation_each_state',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_entry():
    require(Path(recovery.__file__).resolve() == HERE / 'recovery_support.py'
            and recovery.identity(recovery.__file__)['sha256'] == SUPPORT_SHA,
            'Recovery support source changed')
    require(FUNCTIONAL.is_file() and not FUNCTIONAL.is_symlink()
            and hashlib.sha256(FUNCTIONAL.read_bytes()).hexdigest() == FUNCTIONAL_SHA,
            'Functional input source changed')
    spec = importlib.util.spec_from_file_location('actual_primary_functional_inputs', FUNCTIONAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_guard(entry, source_sha):
    recovery.namespace(HERE)
    geometry = entry.geometry_parent()
    require(geometry.identity(__file__)['sha256'] == source_sha,
            'Functional dispatcher source changed')
    require(geometry.identity(ORIGINAL / 'inputs.json')['sha256'] == INPUTS_SHA,
            'Functional input record changed')
    return geometry


def prepare(args):
    entry = load_entry()
    geometry = source_guard(entry, args.source_sha)
    return recovery.prepare(args, entry, geometry, HERE, SETTINGS, FUNCTIONAL_SHA, INPUTS_SHA)


def authenticate(args, *, gpu_worker=False):
    entry = load_entry()
    geometry = source_guard(entry, args.source_sha)
    require(geometry.identity(HERE / 'authorization.json')['sha256'] == args.authorization_sha,
            'Functional execution authority changed')
    authority, _ = recovery.authority(HERE, args.source_sha, args.authorization_sha, SETTINGS)
    require(authority['scope'] == entry.SCOPE
            and authority['source_sha256'] == args.source_sha
            and authority['functional_source_sha256'] == FUNCTIONAL_SHA
            and authority['inputs_sha256'] == INPUTS_SHA
            and authority['settings'] == SETTINGS
            and authority['execution_authorized'] is True
            and authority['native_guards_modified'] is False
            and authority['committed_source_release'] is False
            and authority['scientific_completion'] is False,
            'Wrong operational functional authority')
    for record in [authority['tests'], authority['test_source']]:
        require(geometry.identity(record['path']) == {k: record[k] for k in ('bytes', 'sha256')},
                'Bound test evidence changed')
    for path, expected in authority['sources'].items():
        require(geometry.identity(path) == expected, 'Bound functional dependency changed')
    if gpu_worker:
        require(args.gpu_token in TOKENS and os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu_token,
                'Worker needs exactly its admitted GPU token')
        # Authenticate the original stdlib-only lease checker before importing model code.
        validation = entry.import_path('original_functional_worker_lease_check', entry.VALIDATION)
        parent = entry.import_path('original_functional_worker_lease_paths', entry.PARENT)
        validation.require_inherited_leases(args.lease_fd,
            [root / f'gpu-{args.gpu_token}.lock' for root in parent.LEASE_ROOTS])
    else:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU coordinator/features must hide CUDA')
    context, inputs = entry.authenticate(FUNCTIONAL_SHA, INPUTS_SHA)
    require(authority['sources'] == inputs['sources']
            and authority['primary_protocol_sha256'] == context.primary.sha256
            and authority['dimension_protocol_sha256'] == context.contract.sha256
            and authority['state_order'] == [job['plan']['state']['cell'] for job in context.jobs],
            'Authority differs from actual complete inputs')
    import torch
    require(not torch.cuda.is_initialized(), 'Authentication must not initialize CUDA')
    args.run_root = RUN
    return recovery.output_facade(entry), context, authority


def validation_priority(root, run_ids, identity):
    """Resource gate only: no numerical validation/BEIR score is consumed."""
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'Missing original validation attempt')
    require(not (root / 'failed.json').exists(), 'Original validation reported failure')
    path = root / 'completed.json'
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    require(record['all_twelve_validations_verified'] is True
            and record['scientific_completion'] is False, 'Incomplete original validation receipt')
    actual = sorted(p.name.removesuffix('.scored.json') for p in (root / 'jobs').glob('*.scored.json'))
    require(actual == sorted(run_ids) and len(actual) == 12,
            'Resource priority requires every declared full validation')
    return {'all_twelve_full_validations_finished': True,
            'completion_receipt': {'path': str(path), **identity(path)},
            'scored_run_ids': actual, 'scores_used_for_state_selection': False}


def priority(context, entry):
    return validation_priority(entry.VALIDATION.parent / 'run',
                               list(context.admitted['complete_runs']), context.geometry.identity)


def job_for(context, cell, plan_sha):
    jobs = [job for job in context.jobs if job['plan']['state']['cell'] == cell]
    require(len(jobs) == 1 and context.digest(jobs[0]['plan']) == plan_sha,
            'Unknown, duplicated or changed functional state plan')
    return jobs[0]


def environment(context, entry, token=None):
    env = context.parent.environment([token] if token is not None else None)
    env.update(PYTHONPATH=f'{entry.STORY / "src"}:{entry.STORY}', WANDB_MODE='disabled',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    return env


def worker_command(args, job, context, token, descriptors):
    require(token in TOKENS and len(descriptors) == len(set(descriptors)) == 2,
            'Require one declared GPU and both distinct inherited descriptors')
    command = ['/usr/bin/python', '-B', str(Path(__file__).resolve()),
               '--source-sha', args.source_sha, '--authorization-sha', args.authorization_sha,
               '--worker', job['plan']['state']['cell'],
               '--plan-sha', context.digest(job['plan']), '--gpu-token', token]
    for descriptor in descriptors:
        command.extend(['--lease-fd', str(descriptor)])
    return command


def process_identity(pid, command=None):
    """Only self or a child just created by this source-authorized coordinator."""
    root = Path('/proc') / str(pid)
    raw = (root / 'stat').read_text()
    before = raw[raw.rfind(')') + 2:].split()
    start, parent = int(before[19]), int(before[1])
    actual = [p.decode() for p in (root / 'cmdline').read_bytes().split(b'\0') if p]
    if command is not None:
        require(actual == command and parent == os.getpid(), 'New child identity differs')
    raw = (root / 'stat').read_text()
    after = raw[raw.rfind(')') + 2:].split()
    require((int(after[19]), int(after[1])) == (start, parent), 'Owned process identity raced')
    return {'pid': pid, 'ppid': parent, 'start_ticks': start, 'command': actual}


def worker(args):
    entry, context, _ = authenticate(args, gpu_worker=True)
    job = job_for(context, args.worker, args.plan_sha)
    require(priority(context, entry) is not None, 'Full validation still has priority')
    cell = job['plan']['state']['cell']
    root = entry.OUTPUT / 'vectors' / 'states' / cell
    require(not root.exists() and not root.is_symlink(), 'Never adopt or overwrite a vector state')
    admission = json.loads((RUN / 'jobs' / f'{cell}.admission.json').read_text())
    require(admission['plan_sha256'] == args.plan_sha
            and admission['authorization_sha256'] == args.authorization_sha
            and admission['gpu_token'] == args.gpu_token
            and admission['checkpoint'] == str(job['checkpoint']), 'Worker admission differs')
    entry.verify_state_inputs(context, job)
    started = time.monotonic()
    arrays, observed = context.vectors.encode_state(job['checkpoint'], context.dataset, context.identities)
    encoding_seconds = time.monotonic() - started
    entry.verify_state_inputs(context, job)
    saved = context.vectors.save_vectors(root, job['plan'], arrays, observed,
                                         context.identities, job['checkpoint'])
    del arrays
    entry.verify_state_inputs(context, job)
    receipt = {'scope': entry.SCOPE, 'cell': cell, 'plan_sha256': args.plan_sha,
               'authorization_sha256': args.authorization_sha, 'source_sha256': args.source_sha,
               'gpu_token': args.gpu_token, 'completed_at_utc': context.geometry.now(),
               'encoding_seconds': encoding_seconds, 'saved': saved,
               'both_lease_namespaces_inherited': True, 'scientific_completion': False}
    context.geometry.write_new(RUN / 'jobs' / f'{cell}.encoded.json', receipt)
    return {'event': 'functional_state_encoded', 'cell': cell,
            'encoding_seconds': encoding_seconds, 'saved_manifest': saved['manifest']}


def encode_one(args, entry, context, job):
    cell = job['plan']['state']['cell']
    while True:
        require(priority(context, entry) is not None, 'Original validation completion changed')
        context.parent.handoff()
        for token in TOKENS:
            lease = context.parent.leases([token])
            try:
                descriptors = lease.__enter__()
            except BlockingIOError:
                continue
            try:
                command = worker_command(args, job, context, token, descriptors)
                context.geometry.write_new(RUN / 'jobs' / f'{cell}.admission.json', {
                    'scope': entry.SCOPE, 'cell': cell, 'plan_sha256': context.digest(job['plan']),
                    'checkpoint': str(job['checkpoint']), 'gpu_token': token,
                    'authorization_sha256': args.authorization_sha, 'source_sha256': args.source_sha,
                    'validation_resource_priority': priority(context, entry),
                    'both_lease_namespaces_inherited': True, 'scientific_completion': False})
                with (RUN / 'jobs' / f'{cell}.log').open('xb') as log:
                    child = subprocess.Popen(command, cwd=entry.STORY,
                        env=environment(context, entry, token), stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, pass_fds=descriptors)
                    handle = process_identity(child.pid, command)
                    context.geometry.write_new(RUN / 'jobs' / f'{cell}.started.json', {
                        **handle, 'started_at_utc': context.geometry.now(), 'gpu_token': token,
                        'cell': cell, 'authorization_sha256': args.authorization_sha,
                        'source_sha256': args.source_sha, 'both_lease_namespaces_inherited': True})
                    print(json.dumps({'event': 'functional_worker_started', **handle,
                                      'cell': cell, 'gpu_token': token}), flush=True)
                    code = child.wait()
                context.geometry.write_new(RUN / 'jobs' / f'{cell}.exited.json', {
                    **handle, 'exit_code': code, 'exited_at_utc': context.geometry.now(),
                    'cell': cell, 'authorization_sha256': args.authorization_sha})
                require(code == 0, f'Functional worker failed for {cell}; preserve outputs, no retry')
                path = RUN / 'jobs' / f'{cell}.encoded.json'
                result = json.loads(path.read_text())
                require(result['cell'] == cell and result['authorization_sha256'] == args.authorization_sha
                        and result['source_sha256'] == args.source_sha
                        and result['plan_sha256'] == context.digest(job['plan']),
                        'Actual worker completion differs')
                root = entry.OUTPUT / 'vectors' / 'states' / cell
                arrays, checked = context.vectors.inspect_vectors(root, job['plan'], context.identities,
                    job['checkpoint'], expected_manifest_sha256=result['saved']['manifest']['sha256'])
                require(checked == result['saved'], 'Native saved vector receipt differs')
                del arrays
                context.geometry.write_new(RUN / 'jobs' / f'{cell}.verified.json', {
                    'cell': cell, 'verified_at_utc': context.geometry.now(),
                    'actual_exit_code': code, 'worker_receipt': {'path': str(path), **context.geometry.identity(path)},
                    'saved': checked, 'native_readback_passed': True, 'scientific_completion': False})
                return {'path': f'states/{cell}/manifest.json',
                        **context.geometry.identity(root / 'manifest.json')}
            finally:
                # Close only: the inherited child keeps the same descriptions on parent death.
                lease.__exit__(None, None, None)
        time.sleep(10)


def finalize_vectors(args, entry, context, states):
    require(list(states) == [j['plan']['state']['cell'] for j in context.jobs]
            and len(states) == 61, 'Incomplete or reordered vector population')
    current, _ = entry.authenticate(FUNCTIONAL_SHA, INPUTS_SHA)
    require(current.admitted == context.admitted, 'Final input admission changed')
    root = entry.OUTPUT / 'vectors'
    context.geometry.write_new(root / 'manifest.json', {
        'status': 'complete', 'admission': {'path': 'admission.json',
                                         **context.geometry.identity(root / 'admission.json')},
        'states': states, 'scientific_completion': False})
    trusted_sha = context.geometry.identity(root / 'manifest.json')['sha256']
    manifest = context.exports._manifest(root, context.admitted, trusted_sha)
    count = 0
    for _, arrays, _ in context.exports.iter_vectors(root, manifest, context.jobs, context.identities):
        count += 1
        del arrays
    require(count == 61, 'Native complete-vector readback omitted states')
    context.exports._manifest(root, context.admitted, trusted_sha)
    result = {'scope': entry.SCOPE, 'completed_at_utc': context.geometry.now(),
              'authorization_sha256': args.authorization_sha,
              'manifest': {'path': str(root / 'manifest.json'), **context.geometry.identity(root / 'manifest.json')},
              'all_states_encoded_and_native_readback_verified': 61,
              'model_encoding_repeated_by_readback': False, 'scientific_completion': False}
    context.geometry.write_new(RUN / 'vectors.completed.json', result)
    return result


def feature_worker(args):
    entry, context, _ = authenticate(args)
    os.nice(SETTINGS['cpu_feature_nice'])
    from embed_optim.primary_contract import require_same
    from embed_optim.primary_v3_dimension_contract import TABLE_COUNTS
    from embed_optim.primary_v3_outcomes import csv_bytes
    raw = json.loads((RUN / 'vectors.completed.json').read_text())
    require(raw['authorization_sha256'] == args.authorization_sha
            and raw['all_states_encoded_and_native_readback_verified'] == 61,
            'Feature input lacks complete actual vector production')
    vectors, output = entry.OUTPUT / 'vectors', entry.OUTPUT / 'features'
    require(raw['manifest']['path'] == str(vectors / 'manifest.json')
            and context.geometry.identity(vectors / 'manifest.json')
            == {k: raw['manifest'][k] for k in ('bytes', 'sha256')}, 'Vector production anchor changed')
    manifest = context.exports._manifest(vectors, context.admitted, raw['manifest']['sha256'])
    plan = context.features.feature_plan(context.contract, context.admitted, raw['manifest']['sha256'])
    require(not output.exists() and not output.is_symlink(), 'Preserve partial/prior feature matrix')
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
        context.geometry.write_new(RUN / 'jobs' / f'{cell}.features-verified.json', {
            'cell': cell, 'verified_at_utc': context.geometry.now(),
            'manifest': state_receipts[cell], 'fresh_raw_vectors_fully_recomputed': True,
            'scientific_completion': False})
        print(json.dumps({'event': 'functional_features_verified', 'cell': cell,
                          'verified_states': len(state_receipts)}), flush=True)
        del fresh, replay
    require_same({key: len(rows) for key, rows in tables.items()}, TABLE_COUNTS)
    after, _ = entry.authenticate(FUNCTIONAL_SHA, INPUTS_SHA)
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
    result = {'scope': entry.SCOPE, 'completed_at_utc': context.geometry.now(),
              'authorization_sha256': args.authorization_sha, 'states': len(state_receipts),
              'table_counts': TABLE_COUNTS, 'fresh_raw_vector_recomputation': True,
              'manifest': {'path': str(output / 'manifest.json'), **context.geometry.identity(output / 'manifest.json')},
              'scientific_completion': False}
    context.geometry.write_new(RUN / 'features.completed.json', result)
    return result


def coordinate(args):
    entry, context, authority = authenticate(args)
    require(not RUN.exists() and not entry.OUTPUT.exists(), 'Never duplicate an existing campaign')
    (RUN / 'jobs').mkdir(parents=True, exist_ok=False)
    layout, flow = recovery.modules(HERE)
    layout.prepare_record_parents(RUN / 'jobs', [j['plan']['state']['cell'] for j in context.jobs])
    context.geometry.write_new(RUN / 'coordinator.started.json', {
        **process_identity(os.getpid()), 'started_at_utc': context.geometry.now(),
        'scope': entry.SCOPE, 'source_sha256': args.source_sha,
        'authorization_sha256': args.authorization_sha, 'settings': SETTINGS})
    try:
        if priority(context, entry) is None:
            context.geometry.write_new(RUN / 'waiting-for-validation.json', {
                'observed_at_utc': context.geometry.now(), 'gpu_jobs_launched': 0,
                'reason': SETTINGS['validation_resource_priority'], 'scientific_completion': False})
            print(json.dumps({'event': 'waiting_for_all_twelve_full_validations',
                              'gpu_jobs_launched': 0}), flush=True)
        while priority(context, entry) is None:
            context.parent.handoff()
            time.sleep(10)
        (entry.OUTPUT / 'vectors').mkdir(parents=True, exist_ok=False)
        context.geometry.write_new(entry.OUTPUT / 'vectors/admission.json', context.admitted)
        states = {}
        for job in context.jobs:
            states[job['plan']['state']['cell']] = (
                flow.reuse_pretrained(args, entry, context, job, authority['pretrained_origin'])
                if job['plan']['state']['cell'] == 'pretrained'
                else encode_one(args, entry, context, job))
        raw = finalize_vectors(args, entry, context, states)
        command = ['/usr/bin/python', '-B', str(Path(__file__).resolve()),
                   '--source-sha', args.source_sha, '--authorization-sha', args.authorization_sha,
                   '--features']
        with (RUN / 'features.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=entry.STORY, env=environment(context, entry),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
            handle = process_identity(child.pid, command)
            context.geometry.write_new(RUN / 'features.started.json', {
                **handle, 'started_at_utc': context.geometry.now(),
                'authorization_sha256': args.authorization_sha, 'cuda_hidden': True})
            code = child.wait()
        context.geometry.write_new(RUN / 'features.exited.json', {
            **handle, 'exit_code': code, 'exited_at_utc': context.geometry.now(),
            'authorization_sha256': args.authorization_sha})
        require(code == 0, 'Feature computation failed; preserve all outputs, no automatic retry')
        features = json.loads((RUN / 'features.completed.json').read_text())
        require(features['states'] == 61 and features['authorization_sha256'] == args.authorization_sha,
                'Incomplete feature worker result')
        result = {'scope': entry.SCOPE, 'completed_at_utc': context.geometry.now(),
                  'authorization_sha256': args.authorization_sha, 'vectors': raw,
                  'features': features, 'scientific_completion': False}
        context.geometry.write_new(RUN / 'completed.json', result)
        return {'event': 'functional_campaign_completed', 'vector_states': 61,
                'feature_states': 61, 'scientific_completion': False}
    except BaseException as error:
        context.geometry.write_new(RUN / 'failed.json', {
            'failed_at_utc': context.geometry.now(), 'scope': entry.SCOPE,
            'source_sha256': args.source_sha, 'authorization_sha256': args.authorization_sha,
            'error_type': type(error).__name__, 'error': str(error),
            'traceback': traceback.format_exc(), 'preserve_partial_outputs': True,
            'automatic_retry_authorized': False, 'scientific_completion': False})
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--authorization-sha')
    parser.add_argument('--tests-sha')
    parser.add_argument('--proposal-sha')
    parser.add_argument('--approval-sha')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--coordinate', action='store_true')
    mode.add_argument('--worker')
    mode.add_argument('--features', action='store_true')
    parser.add_argument('--plan-sha')
    parser.add_argument('--gpu-token')
    parser.add_argument('--lease-fd', type=int, action='append', default=[])
    args = parser.parse_args(argv)
    if args.prepare:
        require(args.tests_sha is not None and args.authorization_sha is None
                and args.proposal_sha is not None and args.approval_sha is not None,
                'Preparation needs tests, proposal and explicit owner approval')
    else:
        require(args.authorization_sha is not None and args.tests_sha is None
                and args.proposal_sha is None and args.approval_sha is None,
                'Execution needs the exact prepared authority')
    if args.worker is not None:
        require(args.worker and args.plan_sha is not None and args.gpu_token in TOKENS
                and len(args.lease_fd) == len(set(args.lease_fd)) == 2,
                'GPU worker requires cell, plan, token and both distinct descriptors')
    else:
        require(args.plan_sha is None and args.gpu_token is None and not args.lease_fd,
                'CPU modes may not inherit GPU worker arguments')
    return args


def main():
    args = parse_args()
    function = prepare if args.prepare else coordinate if args.coordinate else worker if args.worker else feature_worker
    print(json.dumps(function(args), sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
