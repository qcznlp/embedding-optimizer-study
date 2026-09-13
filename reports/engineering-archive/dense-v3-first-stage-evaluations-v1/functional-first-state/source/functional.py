"""Actual 61-state functional inputs; original numerical functions stay unchanged.

This explicitly versioned operational admission uses genuine current full-run
proofs. It does not modify or pass the original DRAFT/single-selection guards.
It preserves the original vector/feature plans and every declared state.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
LAUNCH = HERE.parent
EXPERIMENT = LAUNCH.parent
PRIMARY = Path('/root/embedding-optimizer-primary-v3')
STORY = Path('/root/embedding-optimizer-story-refactor')
GEOMETRY = LAUNCH / 'weight-geometry/run_geometry.py'
GEOMETRY_SHA = '5ab87568a1a9050e423c2e778a4fda8b53f906eec2ee2f5a824e2522c09c0ec6'
GEOMETRY_AUTH_SHA = '44aad4192ed0871ff9bf0be370ba7f1f73e7f3e789f2c65e92481bd7d884bff4'
PROTOCOL = 'configs/dense_primary_v3_dimensions_protocol.json'
PROTOCOL_SHA = '9246f7d15f141ba270eb4c839948c852ccc503de282f09090f58a0efdafe09f2'
PARENT = LAUNCH / 'train_matrix.py'
PARENT_SHA = 'dd823408de44d0696d0cac3a9d865ffc41936abdf2542ae815c3f5eda8dffcf3'
VALIDATION = LAUNCH / 'validation-handoff/validation.py'
VALIDATION_SHA = 'd2fc55c67670ea5b42cacb055f48379ef278a465ce86d95808231ab86049d913'
REFERENCE = Path('/tmp/dense-weight-entry-reference.Um8TxW')
OLD_PROBE = Path('/root/embedding-optimizer-study/data/probes/decontaminated-beir-224-seed4242')
PROBE = EXPERIMENT / 'data/probes/decontaminated-beir-224-seed4242'
OUTPUT = EXPERIMENT / 'analyses/dense-primary-v3-functional-dimensions'
SCOPE = 'owner_authorized_content_bound_primary_functional_dimensions_20260910'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def geometry_parent():
    import hashlib
    require(GEOMETRY.is_file() and not GEOMETRY.is_symlink()
            and hashlib.sha256(GEOMETRY.read_bytes()).hexdigest() == GEOMETRY_SHA,
            'Operational complete-run parent changed')
    geometry = import_path('original_geometry_completion_parent_for_dimensions', GEOMETRY)
    require(geometry.identity(GEOMETRY.parent / 'authorization.json')['sha256'] == GEOMETRY_AUTH_SHA,
            'Original all-twelve admission authority changed')
    return geometry


def load_inputs(probe_root):
    geometry = geometry_parent()
    from embed_optim.primary_contract import digest, require_same
    from embed_optim.primary_v3_contract import PrimaryV3Contract
    from embed_optim.primary_v3_dimension_contract import DimensionContract
    from embed_optim import primary_v3_dimension_exports as exports
    from embed_optim import primary_v3_dimension_vector_io as vectors
    from embed_optim import primary_v3_dimensions as features
    from embed_optim import dimension_interventions as kernels
    from embed_optim import dimension_intervention_io as feature_io
    from embed_optim import probe_export
    from embed_optim.runtime import verify_runtime_spec
    from embed_optim.primary_weight_entries import reference_identity
    require(geometry.identity(STORY / PROTOCOL)['sha256'] == PROTOCOL_SHA,
            'Functional measurement protocol changed')
    primary = PrimaryV3Contract.load(STORY / 'configs/dense_primary_v3_protocol.json', STORY, PRIMARY)
    contract = DimensionContract.load(STORY / PROTOCOL, primary)
    actual, child = geometry.read_primary_admission(GEOMETRY_SHA)
    baseline = json.loads((GEOMETRY.parent / 'authorization.json').read_text())['completion_admission']
    require_same(actual['rows'], baseline['rows'])
    accepted = geometry.validate_admissions(primary, actual)
    reference = reference_identity(primary, REFERENCE)
    dataset, probe = contract.probe(probe_root)
    runs = {expected['recipe']['run_id']: {'root': root, 'checked': checked}
            for root, expected, checked in accepted}
    jobs, admitted = build_plans(primary, contract, runs, reference, probe)
    runtime = verify_runtime_spec(PRIMARY / 'configs/formal_runtime.json')
    for path, sha in [(PARENT, PARENT_SHA), (VALIDATION, VALIDATION_SHA)]:
        require(geometry.identity(path)['sha256'] == sha, 'Original operational helper changed')
    parent = import_path('original_primary_leases_for_dimensions', PARENT)
    validation = import_path('original_validation_lease_check_for_dimensions', VALIDATION)
    parent.handoff()
    for name, module in tuple(sys.modules.items()):
        if name == 'embed_optim' or name.startswith('embed_optim.'):
            require(Path(module.__file__).resolve().is_relative_to(STORY / 'src/embed_optim'),
                    f'Foreign functional package dependency: {name}')
    return SimpleNamespace(geometry=geometry, primary=primary, contract=contract,
                           exports=exports, vectors=vectors, features=features, kernels=kernels,
                           feature_io=feature_io, probe_export=probe_export, dataset=dataset,
                           probe=probe, identities=probe['row_identities'], jobs=jobs,
                           admitted=admitted, actual_completion=actual, actual_child=child,
                           runtime=runtime, reference=reference, parent=parent,
                           validation=validation, digest=digest)


def build_plans(primary, contract, runs, reference, probe):
    """Original plan construction, with separately authenticated completion inputs.

No replacement of contract.admit_runs/require_execution or other native method.
The operational proof objects remain verbatim, including their false old gates.
"""
    from embed_optim.primary_contract import digest, require_same
    from embed_optim.primary_v3_dimension_contract import planned_states, SCOPE as NATIVE_SCOPE
    from embed_optim.primary_v3_dimension_contract import ENCODING, INPUT_EXECUTION
    expected_runs = {row['run_id'] for row in primary.inputs['runs']}
    require(set(runs) == expected_runs and len(runs) == 12, 'Require the complete twelve-run population')
    states = planned_states(primary)
    require_same(states, contract.payload['states'])
    checked_runs = {}
    for run_id, item in sorted(runs.items()):
        checked = item['checked']
        require(checked['whole_run_artifacts_verified'] is True
                and checked['run_identity_sha256'] == digest(primary.expected_identity(run_id)),
                'Incomplete or mismatched actual primary admission')
        require_same(checked['steps'], primary.payload['checkpoint_steps'])
        require_same([r['step'] for r in checked['checkpoints']], primary.payload['checkpoint_steps'])
        require(item['root'] == EXPERIMENT / primary.payload['output_root'] / 'dense' / run_id,
                'Changed model root')
        checked_runs[run_id] = checked
    common = {'scope': NATIVE_SCOPE, 'dimension_protocol_sha256': contract.sha256,
              'primary_protocol_sha256': primary.sha256, 'probe': probe,
              'encoding': ENCODING, 'input_execution': INPUT_EXECUTION,
              'scientific_completion': False}
    jobs = []
    for state in states:
        if state['cell'] == 'pretrained':
            checkpoint = REFERENCE
            model = {'kind': 'immutable_pretrained', 'reference': reference}
        else:
            item = runs[state['meta']['run_id']]
            checkpoint = item['root'] / f"checkpoint-{state['meta']['step']}"
            checked = item['checked']
            model = {'kind': 'complete_primary_checkpoint',
                     'run_identity_sha256': checked['run_identity_sha256'],
                     'complete_run_sha256': digest(checked),
                     'checkpoint': checked['checkpoints'][state['stage'] - 1]}
        jobs.append({'checkpoint': checkpoint, 'plan': {**common, 'state': state, 'model': model}})
    require(len(jobs) == len({job['plan']['state']['cell'] for job in jobs}) == 61,
            'Incomplete pretrained-plus-sixty state population')
    return jobs, {**common, 'reference': reference, 'complete_runs': checked_runs,
                  'states': [job['plan'] for job in jobs]}


def inventory(context):
    paths = {Path(__file__).resolve(), GEOMETRY, GEOMETRY.parent / 'authorization.json',
             STORY / PROTOCOL, PARENT, VALIDATION, VALIDATION.parent / 'authorization.json'}
    for name, module in tuple(sys.modules.items()):
        if name == 'embed_optim' or name.startswith('embed_optim.'):
            paths.add(Path(module.__file__).resolve())
    return {str(path): context.geometry.identity(path) for path in sorted(paths)}


def prepare(source_sha):
    geometry = geometry_parent()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Preparation must hide CUDA')
    require(geometry.identity(__file__)['sha256'] == source_sha, 'Functional entry source changed')
    require(not (HERE / 'inputs.json').exists() and not OUTPUT.exists(),
            'Use a fresh actual functional campaign')
    context = load_inputs(OLD_PROBE)
    if PROBE.exists():
        dataset, copied = context.contract.probe(PROBE)
        require(len(dataset) == 224 and copied == context.probe, 'Existing probe copy differs')
    else:
        PROBE.mkdir(parents=True, exist_ok=False)
        for row in context.probe['files']:
            source, target = OLD_PROBE / row['path'], PROBE / row['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target, follow_symlinks=False)
        dataset, copied = context.contract.probe(PROBE)
        require(len(dataset) == 224 and copied == context.probe, 'Relocated probe differs')
    sources = inventory(context)
    record = {'scope': SCOPE, 'prepared_at_utc': geometry.now(), 'source_sha256': source_sha,
              'primary_protocol_sha256': context.primary.sha256,
              'dimension_protocol_sha256': context.contract.sha256,
              'probe_root': str(PROBE), 'reference_root': str(REFERENCE), 'output_root': str(OUTPUT),
              'probe': context.probe, 'runtime': context.runtime, 'sources': sources,
              'actual_completion': context.actual_completion, 'actual_admission_child': context.actual_child,
              'admitted': context.admitted, 'admitted_sha256': context.digest(context.admitted),
              'jobs': [{'checkpoint': str(job['checkpoint']), 'plan': job['plan'],
                        'plan_sha256': context.digest(job['plan'])} for job in context.jobs],
              'states': 61, 'gpu_jobs_launched': 0, 'model_encoding_started': False,
              'selection_by_beir_or_validation': False, 'native_guards_modified': False,
              'committed_source_release': False, 'scientific_completion': False,
              'boundary': 'Actual all-twelve operational completion and exact fixed-probe preparation only. Original encoding, vector IO, full-feature kernels and native release guards are unchanged. GPU dispatch needs its own tested source-bound authority and both inherited lease namespaces.'}
    for path, bound in sources.items():
        require(geometry.identity(path) == bound, 'Source changed during preparation')
    geometry.write_new(HERE / 'inputs.json', record)
    return {'scope': SCOPE, 'inputs': geometry.identity(HERE / 'inputs.json'),
            'source_files': len(sources), 'states': 61, 'query_rows': 224,
            'candidate_columns': 8, 'probe_files': 5, 'gpu_jobs_launched': 0,
            'model_encoding_started': False, 'scientific_completion': False}


def authenticate(source_sha, inputs_sha):
    geometry = geometry_parent()
    require(geometry.identity(__file__)['sha256'] == source_sha, 'Functional source changed')
    require(geometry.identity(HERE / 'inputs.json')['sha256'] == inputs_sha, 'Prepared inputs changed')
    expected = json.loads((HERE / 'inputs.json').read_text())
    require(expected['scope'] == SCOPE and expected['source_sha256'] == source_sha
            and expected['probe_root'] == str(PROBE) and expected['reference_root'] == str(REFERENCE)
            and expected['output_root'] == str(OUTPUT)
            and expected['native_guards_modified'] is False
            and expected['committed_source_release'] is False
            and expected['scientific_completion'] is False, 'Wrong functional input authority')
    for path, bound in expected['sources'].items():
        require(geometry.identity(path) == bound, f'Bound source changed: {path}')
    context = load_inputs(PROBE)
    from embed_optim.primary_contract import require_same
    require_same(context.actual_completion['rows'], expected['actual_completion']['rows'])
    require_same(context.admitted, expected['admitted'])
    require_same(context.digest(context.admitted), expected['admitted_sha256'])
    require_same(context.probe, expected['probe'])
    require_same(context.runtime, expected['runtime'])
    require_same([{'checkpoint': str(job['checkpoint']), 'plan': job['plan'],
                   'plan_sha256': context.digest(job['plan'])} for job in context.jobs], expected['jobs'])
    require_same(inventory(context), expected['sources'])
    return context, expected


def verify_state_inputs(context, job):
    """Authenticate every input payload for this state before/after GPU encoding."""
    from embed_optim.primary_contract import require_same, verify_file
    state = job['plan']['state']
    if state['cell'] == 'pretrained':
        for row in context.reference['files']:
            verify_file(REFERENCE / row['path'], row)
    else:
        actual = context.primary.checkpoint(job['checkpoint'], state['meta']['run_id'], state['meta']['step'])
        require_same({key: actual[key] for key in job['plan']['model']['checkpoint']},
                     job['plan']['model']['checkpoint'])
    _, probe = context.contract.probe(PROBE)
    require_same(probe, context.probe)
    context.exports.recheck_contract(context.contract)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--prepare', action='store_true', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_sha), sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
