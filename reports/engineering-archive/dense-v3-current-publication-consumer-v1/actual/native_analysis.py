"""Join immutable completed native measurement chains without rerunning GPU work.

Full model SVD and coordinate computations were already natively replayed. This
consumer authenticates those exact completed payloads, original source admission,
and all checkpoint joins, then reconstructs the complete aggregate tables. It
does not relabel a historical failed outer invocation or an old release guard.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from native_primary import identity, need, write_new, event

STORY = Path('/root/embedding-optimizer-story-refactor')
EXP = Path('/root/embedding-optimizer-v3-experiment')
GEOMETRY = EXP / 'launch/weight-geometry'
GEOMETRY_SHA = '5ab87568a1a9050e423c2e778a4fda8b53f906eec2ee2f5a824e2522c09c0ec6'
GEOMETRY_AUTH = '44aad4192ed0871ff9bf0be370ba7f1f73e7f3e789f2c65e92481bd7d884bff4'
GEOMETRY_COMPLETE = {
    'approximate': '6b2eab14aa4add03254ad278ded3764745e4a622230da641fd35b9ba2cb9b3e6',
    'exact': '30868950a910502b2f5997e32388415c2ae864441f640da85862684de9321f9e',
}
FUNCTIONAL = Path('/tmp/dense-v3-functional-inference-launch.0JaLetXA')
RECONCILE_SHA = 'acafb3189dc7d46d38b0ece7f4b98dd6738e025c40924ee506df386e96959d59'
RECONCILE_APPROVAL = '0fbfcceed375a279ed3daf0212e11bcbea4a01f990090be739fad3f84a59cbe0'
FUNCTIONAL_READOUT = '608965f5297fc196adcce6ca7417bd40e7f35d05e26133f20c0811dd1fd7027e'
HELPER_SHA = '98d2cfaf76d212792906f4701f5c6014a4da69fec40fe41eedd096a717fbb6e7'


def imported(name, path, sha):
    need(identity(path)['sha256'] == sha, 'Original native source changed')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def geometry(output):
    old = imported('original_completed_geometry_consumer', GEOMETRY / 'run_geometry.py', GEOMETRY_SHA)
    authority, primary, contracts, modules, admitted, reference, child = old.authenticate(GEOMETRY_SHA, GEOMETRY_AUTH)
    from embed_optim.primary_contract import read_json, require_same, verify_file
    from embed_optim.primary_v3_geometry_io import descriptor
    from embed_optim.primary_v3_outcomes import csv_bytes
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4)
    all_tables, bindings, run_proofs = {}, {}, {}
    for kind in ('approximate', 'exact'):
        module, contract = modules[kind], contracts[kind]
        root = old.OUTPUT / kind
        completed_path = GEOMETRY / 'receipts' / f'{kind}.completed.json'
        need(identity(completed_path)['sha256'] == GEOMETRY_COMPLETE[kind], 'Native completion changed')
        completed = read_json(completed_path)
        need(completed['source_sha256'] == GEOMETRY_SHA
             and completed['authorization_sha256'] == GEOMETRY_AUTH
             and completed['native_numeric_readback_passed'] is True
             and completed['complete_runs'] == 12 and completed['checkpoints'] == 60,
             'Incomplete original native geometry replay')
        need(not (GEOMETRY / 'receipts' / f'{kind}.failed.json').exists(), 'Original geometry failed')
        bindings[str(completed_path)] = identity(completed_path)
        for name, bound in completed['outputs'].items():
            verify_file(root / name, bound)
            bindings[str(root / name)] = identity(root / name)
        summary = read_json(root / 'summary.json')
        require_same(summary['table_counts'], contract.payload['table_counts'])
        need(len(summary['raw_bindings']) == 132, 'Incomplete raw geometry family')
        for bound in summary['raw_bindings']:
            verify_file(root / bound['path'], bound)
            bindings[str(root / bound['path'])] = identity(root / bound['path'])
        need({p.name for p in (root / 'runs').iterdir()} == set(authority['run_order']), 'Geometry run population differs')
        checked_rows = []
        for run, expected, proof in admitted:
            run_id = expected['recipe']['run_id']
            target = root / 'runs' / run_id
            manifest = read_json(target / 'manifest.json')
            require_same(manifest['plan'], descriptor(expected, proof, reference,
                         contract.payload['settings'], module.SCOPE, contract.sha256))
            require_same([row['step'] for row in manifest['outputs']], proof['steps'])
            receipt_path = GEOMETRY / 'receipts' / f'{kind}-{run_id}.verified.json'
            receipt = read_json(receipt_path)
            need(receipt['all_five_stages_verified'] is True and receipt['run_id'] == run_id
                 and receipt['source_sha256'] == GEOMETRY_SHA
                 and receipt['authorization_sha256'] == GEOMETRY_AUTH,
                 'Missing native per-run numeric replay')
            require_same(receipt['original_completion_binding'], next(row['binding'] for row in authority['completion_admission']['rows'] if row['run_id'] == run_id))
            require_same(receipt['native_numeric_readback']['manifest'], identity(target / 'manifest.json'))
            native = {'manifest': identity(target / 'manifest.json'), 'checkpoint_rows': [],
                      ('matrix_rows' if kind == 'exact' else 'subspace_health'): []}
            names = {'manifest.json'}
            for stage in manifest['outputs']:
                for key in ('records', 'arrays' if kind == 'exact' else 'bases'):
                    verify_file(target / stage[key]['path'], stage[key])
                    names.add(stage[key]['path'])
                record = read_json(target / stage['records']['path'])
                native['checkpoint_rows'].append(record['checkpoint_row'])
                native['matrix_rows' if kind == 'exact' else 'subspace_health'].extend(
                    record['records' if kind == 'exact' else 'subspace_health'])
            need({p.name for p in target.iterdir()} == names, 'Native geometry payload inventory differs')
            require_same(native, receipt['native_numeric_readback'])
            checked_rows.append(native)
            bindings[str(receipt_path)] = identity(receipt_path)
        # Reconstruct all pair/projector summaries from the unchanged native
        # bases. This does not claim to repeat the already verified full SVDs.
        with threadpool_limits(limits=4):
            args = (admitted, root / 'runs', reference, checked_rows)
            tables = (module.tables_from_verified_runs(*args, contract.payload['settings'])
                      if kind == 'exact' else module.tables_from_verified_runs(*args))
        require_same({name: len(rows) for name, rows in tables.items()}, contract.payload['table_counts'])
        for name, rows in tables.items():
            need((root / f'{name}.csv').read_bytes() == csv_bytes(rows), 'Geometry aggregate differs from original native records/bases')
        all_tables[kind] = tables
        run_proofs[kind] = checked_rows
        event('complete_geometry_aggregate_reconstructed', kind=kind, runs=12, checkpoints=60)
    old.verify_inventory(authority['sources'])
    for path, bound in bindings.items():
        need(identity(path) == bound, 'Geometry evidence changed during reconstruction')
    need(not torch.cuda.is_initialized(), 'Geometry consumer initialized CUDA')
    write_new(output / 'tables.json', all_tables)
    write_new(output / 'evidence.json', {'scope': 'current_primary_geometry_native_chain_v1',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'original_authority': identity(GEOMETRY / 'authorization.json'),
        'source_bindings': authority['sources'], 'input_bindings': bindings,
        'original_primary_admission': authority['completion_admission'],
        'original_native_replays': run_proofs, 'actual_admission_child': child,
        'all_264_raw_files_rehashed': True, 'all_24_native_full_run_replays_reused': True,
        'all_eight_aggregate_tables_reconstructed': True, 'full_model_svds_repeated': False,
        'original_guards_modified': False, 'scientific_completion': False})


def functional(output):
    old = imported('original_functional_reconciliation', FUNCTIONAL / 'reconcile.py', RECONCILE_SHA)
    p, prefix, bindings = old.admission(SimpleNamespace(source_sha=RECONCILE_SHA, approval_sha=RECONCILE_APPROVAL))
    features = p.module('current_native_feature_context', p.FEATURE_ENTRY / 'features.py')
    prior, entry, context, _ = features.authenticate(SimpleNamespace(
        source_sha=p.PINS[p.FEATURE_ENTRY / 'features.py'],
        approval_sha=p.PINS[p.FEATURE_ENTRY / 'owner-approval.json'],
        tests_sha=p.PINS[p.FEATURE_ENTRY / 'tests.json']))
    raw = p.feature_completion(features, prefix['feature_completion_sha256'])
    panel, original_bindings, joined = p.original_panel(context)
    bindings.update(original_bindings)
    accepted = p.read(STORY / 'reports/dense-v3-functional-inference-v1/actual/readout.json', FUNCTIONAL_READOUT)
    bindings.update(old.reconstructed_aggregates(p, prefix, context, features, raw, output / 'features'))
    after, _ = entry.authenticate(prior.FUNCTIONAL_SHA, prior.INPUTS_SHA)
    from embed_optim.primary_contract import require_same
    require_same(after.admitted, context.admitted)
    vector_manifest = context.exports._manifest(features.VECTOR_ROOT, context.admitted, raw['input_vectors']['sha256'])
    event('native_functional_chain_reconstructed', states=61, aggregate_families=4)
    # Import statistical consumers only after the original exact input closure
    # has completed, as in the accepted reconciliation source.
    from embed_optim.primary_v3_dimension_inference import FunctionalInferenceContract
    from embed_optim.primary_v3_exact_bridge import typed_csv
    from embed_optim.dimension_inference import summarize, TABLE_COUNTS
    from embed_optim.dimension_inference_render import render
    contract = FunctionalInferenceContract.load(STORY / 'configs/dense_primary_v3_dimension_inference_protocol.json', context.primary)
    tables = {name: typed_csv(output / 'features' / (name + '.csv')) for name in raw['table_counts']}
    calculated, decisions = summarize(context.primary, tables, panel, context.contract.scientific,
                                      sorted({row['source'] for row in context.identities}))
    require_same({k: len(v) for k, v in calculated.items()}, TABLE_COUNTS)
    actual = STORY / 'reports/dense-v3-functional-inference-v1/actual'
    for name in ('tables.json', 'decisions.json', 'results.tex', 'original_panel.json', 'shared_checkpoint_identities.json'):
        need(identity(actual / name) == accepted['payloads'][name], 'Prior functional output changed')
    require_same(calculated, p.read(actual / 'tables.json'))
    require_same(decisions, p.read(actual / 'decisions.json'))
    require_same(panel, p.read(actual / 'original_panel.json'))
    require_same(joined, p.read(actual / 'shared_checkpoint_identities.json'))
    rendered = render(calculated, decisions)
    need(rendered.encode() == (actual / 'results.tex').read_bytes(), 'Original functional rendering differs')
    contract.recheck()
    old.admission(SimpleNamespace(source_sha=RECONCILE_SHA, approval_sha=RECONCILE_APPROVAL))
    for path, bound in bindings.items():
        need(identity(path) == bound, 'Functional evidence changed during reconstruction')
    for name, value in (('tables.json', calculated), ('decisions.json', decisions),
                        ('original_panel.json', panel), ('shared_checkpoint_identities.json', joined)):
        write_new(output / name, value)
    with (output / 'results.tex').open('x') as stream:
        stream.write(rendered)
    write_new(output / 'evidence.json', {'scope': 'current_primary_functional_native_chain_v1',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'input_bindings': bindings, 'original_functional_readout_sha256': FUNCTIONAL_READOUT,
        'admitted': context.admitted, 'vector_manifest': vector_manifest,
        'checkpoint_joins': joined, 'original_context_admission_before_and_after': True,
        'all_61_native_state_reconstructions_reused': True,
        'all_raw_vector_and_feature_payloads_rehashed': True,
        'all_four_feature_aggregates_reconstructed': True,
        'all_nine_inference_tables_and_decisions_recomputed_exactly': True,
        'original_generated_tex_byte_exact': True, 'raw_coordinate_computation_repeated': False,
        'original_failed_outer_exit_preserved': 1, 'original_guards_modified': False,
        'scientific_completion': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('geometry', 'functional'))
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only consumer')
    need(identity(__file__)['sha256'] == args.source_sha, 'Current consumer changed')
    need(identity(Path(__file__).with_name('native_primary.py'))['sha256'] == HELPER_SHA,
         'Current read-only helper changed')
    need(args.output.is_absolute() and not args.output.exists()
         and not any(p.is_symlink() for p in (args.output, *args.output.parents)), 'Use a fresh ordinary output')
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / 'started.json', {'kind': args.kind, 'pid': os.getpid(),
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'source': identity(__file__),
        'helper': identity(Path(__file__).with_name('native_primary.py')), 'scientific_completion': False})
    try:
        (geometry if args.kind == 'geometry' else functional)(args.output)
        need(identity(__file__)['sha256'] == args.source_sha, 'Current consumer raced')
        need(identity(Path(__file__).with_name('native_primary.py'))['sha256'] == HELPER_SHA,
             'Current read-only helper raced')
        write_new(args.output / 'completed.json', {'kind': args.kind,
            'evidence': identity(args.output / 'evidence.json'), 'tables': identity(args.output / 'tables.json'),
            'source': identity(__file__), 'scientific_completion': False})
        event('current_native_analysis_chain_complete', kind=args.kind, **identity(args.output / 'evidence.json'))
    except Exception as error:
        write_new(args.output / 'failed.json', {'exception_type': type(error).__name__,
            'message': str(error), 'original_guards_modified': False, 'scientific_completion': False})
        raise
