"""Actual CPU functional readout using unchanged, externally bound kernels.

This owner-scoped entry reconstructs every feature from genuine vectors before
inference. It does not call or alter the historical formal-execution/publication
entry points. Independent verification is a separate invocation with an external
result digest. No model encoding, GPU scheduling or manuscript write occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
STORY = Path('/root/embedding-optimizer-story-refactor')
EXP = Path('/root/embedding-optimizer-v3-experiment')
FEATURE_ENTRY = EXP / 'launch/functional-features-recovery-v3'
WEIGHTS = STORY / 'reports/dense-v3-weight-retrieval-v1'
SCOPE = 'owner_authorized_actual_functional_inference_20260912'
OWNER = '你有权做一切事情，目标是尽快完成任务'
PINS = {
    FEATURE_ENTRY / 'features.py': '694069c2d0e2866bba1126ea891cc9360ff03923691afa26ae04724165d35d9f',
    FEATURE_ENTRY / 'owner-approval.json': '4c151939a90937f27488620291f64d58105dc1f1a8e962eaa7e3949379113514',
    FEATURE_ENTRY / 'tests.json': 'e3083be2dad6970c24b728efe626dd18be7182d90d498a12fb5a0ac89332ffd1',
    STORY / 'configs/dense_primary_v3_dimension_inference_protocol.json': 'd66274878d90e463b74a03cf6e09518ae01dcddfe76584542e3bece435713545',
    STORY / 'scripts/dimension_inference_reference.py': '8aa430b0c0a759d697fef50b03e688b7757ea607c6c00f18cf10bd43eb13b19d',
    WEIGHTS / 'source/readout.py': 'b2cd61b867943916b6ed99982864ac228a6e982f996296c3250778a9cde80792',
    WEIGHTS / 'actual/readout.json': '6891e82a46df11a6d142e79fdf495afa329f40912569337f4b1d6da33d5e8046',
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
    need(all(getattr(first, k) == getattr(last, k) for k in
             ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')), 'Input raced')
    return {'bytes': last.st_size, 'sha256': sha}


def read(path, expected=None):
    before = identity(path)
    if expected is not None:
        need(before['sha256'] == expected if isinstance(expected, str) else before == expected,
             'External binding differs')
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


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def module(name, path):
    need(identity(path)['sha256'] == PINS[Path(path)], 'Source changed before import')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def source_admission(args):
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only readout')
    need(identity(__file__)['sha256'] == args.source_sha, 'Changed new readout source')
    approval = read(HERE / 'owner-approval.json', args.approval_sha)
    need(approval == {'scope': SCOPE, 'source_sha256': args.source_sha,
        'owner_message': OWNER, 'automatic_continuation': False, 'cpu_only': True,
        'scientific_kernels_changed': False, 'protected_helper_access': False,
        'old_controller_transition': False, 'manuscript_installation': False,
        'source_publication': False, 'scientific_completion': False}, 'Wrong bounded owner approval')
    for path, sha in PINS.items():
        need(identity(path)['sha256'] == sha, 'Changed protected source/evidence')
    return {str(path): identity(path) for path in PINS}


def actual_context():
    features = module('_actual_completed_features', FEATURE_ENTRY / 'features.py')
    prior, entry, context, _ = features.authenticate(SimpleNamespace(
        source_sha=PINS[FEATURE_ENTRY / 'features.py'],
        approval_sha=PINS[FEATURE_ENTRY / 'owner-approval.json'],
        tests_sha=PINS[FEATURE_ENTRY / 'tests.json']))
    from embed_optim.primary_v3_dimension_inference import FunctionalInferenceContract
    contract = FunctionalInferenceContract.load(
        STORY / 'configs/dense_primary_v3_dimension_inference_protocol.json', context.primary)
    need(contract.dimensions.sha256 == context.contract.sha256, 'Different functional definitions')
    return features, prior, entry, context, contract


def feature_completion(features, expected_sha):
    completed = read(FEATURE_ENTRY / 'run/coordinator.completed.json', expected_sha)
    need(completed['source_sha256'] == PINS[FEATURE_ENTRY / 'features.py']
         and completed['actual_feature_worker_exit'] == 0
         and completed['all_61_states_twice_computed_and_native_verified'] is True,
         'Feature producer not completely successful')
    raw = read(FEATURE_ENTRY / 'run/features.completed.json', completed['feature_completion'])
    need(raw['states'] == 61 and raw['fresh_raw_vector_recomputation'] is True
         and raw['source_sha256'] == PINS[FEATURE_ENTRY / 'features.py'], 'Incomplete actual features')
    need(raw['owner_approval'] == identity(FEATURE_ENTRY / 'owner-approval.json')
         and raw['tests'] == identity(FEATURE_ENTRY / 'tests.json'), 'Feature production authority differs')
    need(raw['manifest']['path'] == str(features.OUTPUT / 'manifest.json'), 'Wrong actual feature root')
    read(features.OUTPUT / 'manifest.json', {k: raw['manifest'][k] for k in ('bytes', 'sha256')})
    exited = read(FEATURE_ENTRY / 'run/features.exited.json')
    need(exited['exit_code'] == 0 and exited['source_sha256'] == PINS[FEATURE_ENTRY / 'features.py'],
         'Require actual exit-zero feature process')
    return raw


def join_checkpoints(jobs, joined):
    observed = {(r['run_id'], r['step']): r for r in joined}
    need(len(observed) == len(joined) == 60, 'Require 60 distinct original bridge identities')
    seen = set()
    for job in jobs:
        state, model = job['plan']['state'], job['plan']['model']
        if state['stage'] == 0:
            continue
        key = state['meta']['run_id'], state['meta']['step']
        need(key in observed and key not in seen, 'Functional/weight/retrieval grid differs')
        seen.add(key)
        row = observed[key]
        ckpt = model['checkpoint']
        files = {f['path']: f for f in ckpt['files']}
        need(model['run_identity_sha256'] == ckpt['run_identity_sha256'] == row['run_identity_sha256']
             and ckpt['checkpoint_seal'] == row['checkpoint_seal']
             and {k: files['model.safetensors'][k] for k in ('bytes', 'sha256')} == row['model_safetensors']
             and ckpt['step'] == row['step'] and state['stage'] == row['stage'],
             'Functional, geometry and retrieval refer to different saved weights')
    need(seen == set(observed), 'Missing shared checkpoint')


def original_panel(context):
    old = module('_actual_original_weight_readout', WEIGHTS / 'source/readout.py')
    kernels = old.pure_kernels(WEIGHTS / 'source-original')
    inputs = old.inspect_inputs(SimpleNamespace(weight_root=WEIGHTS / 'inputs/weights',
        outcome_root=WEIGHTS / 'inputs/outcomes', index=WEIGHTS / 'inputs/raw-score-index.json'), kernels)
    join_checkpoints(context.jobs, inputs['joined'])
    a = inputs['tables']['approximate']
    rows = kernels['join']['assemble_bridge_rows'](
        a['checkpoints'], a['pairs'], inputs['scores'], inputs['configs'])
    # The complete old nine-feature panel is retained; no old OLS fits are repeated.
    return rows, inputs['bindings'], inputs['joined']


def fresh_reconstruction(features, entry, prior, context, raw, output):
    from embed_optim.primary_contract import require_same
    from embed_optim.primary_v3_dimension_contract import TABLE_COUNTS
    from embed_optim.primary_v3_outcomes import csv_bytes
    vector_sha = raw['input_vectors']['sha256']
    manifest = context.exports._manifest(features.VECTOR_ROOT, context.admitted, vector_sha)
    plan = context.features.feature_plan(context.contract, context.admitted, vector_sha)
    accepted = context.features._feature_manifest(features.OUTPUT, plan, context.jobs)
    tables = {key: [] for key in TABLE_COUNTS}
    tasks = sorted({row['source'] for row in context.identities})
    checked = {}
    for job, arrays, _ in context.exports.iter_vectors(
            features.VECTOR_ROOT, manifest, context.jobs, context.identities):
        state = job['plan']['state']
        cell = state['cell']
        result = context.kernels.compute_state(state['meta'], arrays, context.contract.scientific)
        context.features.check_result(result, state, tasks)
        current = context.features.state_plan(plan, job, manifest['states'][cell])
        context.feature_io.inspect_state(features.OUTPUT / 'states' / cell, current, result)
        for key, rows in result['tables'].items():
            tables[key].extend(rows)
        checked[cell] = accepted['states'][cell]
        destination = output / 'states' / (cell + '.json')
        destination.parent.mkdir(parents=True, exist_ok=True)
        write(destination, {'cell': cell, 'feature_manifest': checked[cell],
            'fresh_raw_reconstruction_matches_every_csv_and_attribution_array': True,
            'checked_at_utc': context.geometry.now(), 'scientific_completion': False})
        print(json.dumps({'event': 'fresh_functional_readback', 'states': len(checked), 'cell': cell}), flush=True)
        del arrays, result
    require_same({key: len(rows) for key, rows in tables.items()}, TABLE_COUNTS)
    require_same(checked, accepted['states'])
    (output / 'features').mkdir()
    bindings = {}
    for key, rows in tables.items():
        data = csv_bytes(rows)
        path = features.OUTPUT / (key + '.csv')
        need(path.read_bytes() == data, 'Full aggregate differs from raw-vector reconstruction')
        with (output / 'features' / path.name).open('xb') as stream:
            stream.write(data)
        bindings[str(path)] = identity(path)
    after, _ = entry.authenticate(prior.FUNCTIONAL_SHA, prior.INPUTS_SHA)
    require_same(after.admitted, context.admitted)
    context.exports._manifest(features.VECTOR_ROOT, context.admitted, vector_sha)
    context.exports.recheck_contract(context.contract)
    require_same(context.features._feature_manifest(features.OUTPUT, plan, context.jobs), accepted)
    return tables, tasks, bindings


def calculate(args):
    bindings = source_admission(args)
    features, prior, entry, context, contract = actual_context()
    raw = feature_completion(features, args.feature_completion_sha)
    output = args.output
    need(output.is_absolute() and output.parent == HERE and not output.exists()
         and not any(p.is_symlink() for p in (output, *output.parents)), 'Use a new explicit local result directory')
    panel, original_bindings, joined = original_panel(context)
    bindings.update(original_bindings)
    output.mkdir()
    write(output / 'started.json', {**prior.process_identity(os.getpid()),
        'started_at_utc': context.geometry.now(), 'scope': SCOPE,
        'source_sha256': args.source_sha, 'feature_completion_sha256': args.feature_completion_sha,
        'cuda_hidden': True, 'scientific_completion': False})
    write(output / 'original_panel.json', panel)
    write(output / 'shared_checkpoint_identities.json', joined)
    tables, tasks, fresh_bindings = fresh_reconstruction(features, entry, prior, context, raw, output)
    bindings.update(fresh_bindings)
    from embed_optim.dimension_inference import summarize, TABLE_COUNTS
    from embed_optim.dimension_inference_render import render
    from embed_optim.primary_v3_outcomes import csv_bytes
    calculated, decisions = summarize(context.primary, tables, panel, context.contract.scientific, tasks)
    need({k: len(v) for k, v in calculated.items()} == TABLE_COUNTS, 'Incomplete inference outputs')
    (output / 'tables').mkdir()
    for key, rows in calculated.items():
        with (output / 'tables' / (key + '.csv')).open('xb') as stream:
            stream.write(csv_bytes(rows))
    write(output / 'tables.json', calculated)
    write(output / 'decisions.json', decisions)
    write(output / 'scientific_protocol.json', context.contract.scientific)
    with (output / 'results.tex').open('x') as stream:
        stream.write(render(calculated, decisions))
    contract.recheck()
    source_admission(args)
    feature_completion(features, args.feature_completion_sha)
    for path, bound in bindings.items():
        need(identity(path) == bound, 'Input changed during actual inference')
    for name, record in contract.payload['sources'].items():
        bindings[str(STORY / name)] = identity(STORY / name)
    payloads = {str(p.relative_to(output)): identity(p) for p in sorted(output.rglob('*')) if p.is_file()}
    result = {'scope': SCOPE, 'completed_at_utc': context.geometry.now(),
        'source_sha256': args.source_sha, 'approval_sha256': args.approval_sha,
        'feature_completion_sha256': args.feature_completion_sha,
        'feature_producer': raw, 'raw_vector_states_freshly_reconstructed': 61,
        'input_table_counts': {k: len(v) for k, v in tables.items()},
        'output_table_counts': TABLE_COUNTS, 'shared_saved_checkpoints': len(joined),
        'unchanged_original_geometry_columns_retained': True, 'source_bindings': bindings,
        'payloads': payloads, 'old_nine_predictor_fits_repeated': False,
        'formal_primary_consumer': False, 'independent_inference_verification': False,
        'manuscript_installed': False, 'scientific_completion': False}
    write(output / 'readout.json', result)
    return {k: result[k] for k in ('completed_at_utc', 'raw_vector_states_freshly_reconstructed', 'output_table_counts')}


def verify(args):
    source_admission(args)
    need(args.result_sha is not None, 'Independent verifier requires external result digest')
    result = read(args.output / 'readout.json', args.result_sha)
    need(result['source_sha256'] == args.source_sha and result['feature_completion_sha256'] == args.feature_completion_sha
         and result['raw_vector_states_freshly_reconstructed'] == 61, 'Wrong actual readout')
    for path, bound in result['source_bindings'].items():
        need(identity(path) == bound, 'Changed source/data before independent inference')
    for name, bound in result['payloads'].items():
        need(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Escaping result path')
        need(identity(args.output / name) == bound, 'Changed calculated table')
    from embed_optim.primary_v3_exact_bridge import typed_csv
    tables = read(args.output / 'tables.json')
    decisions = read(args.output / 'decisions.json')
    for name in tables:
        need(typed_csv(args.output / 'tables' / (name + '.csv')) == tables[name], 'JSON/CSV results differ')
    reference = module('_actual_functional_independent_reference', STORY / 'scripts/dimension_inference_reference.py')
    inputs = {'protocol': read(args.output / 'scientific_protocol.json'),
        'tables': {name: typed_csv(args.output / 'features' / (name + '.csv')) for name in result['input_table_counts']}}
    task = reference.task_inference(inputs, tables, decisions)
    predictions = reference.exact_predictions(tables)
    source_admission(args)
    for name, bound in result['payloads'].items():
        need(identity(args.output / name) == bound, 'Calculated result changed during verification')
    verified = {'scope': SCOPE, 'producer_readout': {'path': str(args.output / 'readout.json'),
        **identity(args.output / 'readout.json')}, 'separate_process_id': os.getpid(),
        'task_inference': task, 'exact_predictions': predictions, 'all_checks_passed': True,
        'manuscript_installed': False, 'scientific_completion': False}
    write(args.output / 'independent_verification.json', verified)
    return {'all_checks_passed': True, 'exact_predictions': predictions['exact_predictions'],
        'task_contrasts': len(task['contrasts']), 'rotations': len(task['rotations'])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('calculate', 'verify', 'admit'))
    for name in ('source-sha', 'approval-sha'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--feature-completion-sha')
    p.add_argument('--output', type=Path)
    p.add_argument('--result-sha')
    args = p.parse_args()
    if args.mode == 'admit':
        source_admission(args)
        _, _, _, context, contract = actual_context()
        panel, _, joined = original_panel(context)
        value = {'actual_native_input_admission': True, 'shared_checkpoints': len(joined),
                 'original_panel_rows': len(panel), 'inference_contract_sha256': contract.sha256,
                 'numerical_reconstruction_or_inference_executed': False, 'scientific_completion': False}
    else:
        need(args.feature_completion_sha is not None and args.output is not None, 'Require actual feature/result targets')
        value = (calculate if args.mode == 'calculate' else verify)(args)
    print(json.dumps(value, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
