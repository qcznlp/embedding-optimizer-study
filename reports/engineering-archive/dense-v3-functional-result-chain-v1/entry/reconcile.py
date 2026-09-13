"""Finish inference from 61 genuine native state reads, with fresh source admission.

The previous invocation's outer import-inventory rejection is preserved. All 61
raw-vector recomputations had already passed native per-state CSV/NPZ comparison.
This reader authenticates that exact complete prefix, independently reloads the
original context without statistical imports, reconstructs every aggregate CSV
from those bound state payloads, and only then imports the unchanged estimators.
It does not repeat encoding, replace an old guard, or claim the old process passed.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
OLD_SHA='84e6aaf4dcb573ac04b74c6e97bf88b44fe4e7c410842eb80228239555168254'
OLD_APPROVAL='9c165cfb5247d4a14eb13022009c4dc8d592ca18a2a932c2d3b52cf29b5ffe91'
PREFIX_SHA='fe718a0f5ac608553b932f39d27f1695fd2b667fba8fabdc0e519fd828eb54f1'
SCOPE='actual_functional_inference_from_complete_native_state_readbacks_v2'


def parent():
    import hashlib
    path=HERE/'inference.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=OLD_SHA:raise ValueError('Original executed reader changed')
    spec=importlib.util.spec_from_file_location('functional_readback_parent',path)
    p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)
    return p


def admission(args):
    p=parent()
    p.need(p.identity(__file__)['sha256']==args.source_sha,'Reconciliation source changed')
    approved=p.read(HERE/'reconciliation-approval.json',args.approval_sha)
    p.need(approved=={'scope':SCOPE,'source_sha256':args.source_sha,
        'owner_message':p.OWNER,'automatic_continuation':False,'cpu_only':True,
        'preserved_old_exit_code':1,'complete_native_state_reads':61,
        'reconciliation_inputs_sha256':PREFIX_SHA,'old_guard_changed':False,
        'fresh_original_input_admission_required':True,'scientific_estimators_changed':False,
        'manuscript_installed':False,'scientific_completion':False},'Wrong scoped completion approval')
    bindings=p.source_admission(SimpleNamespace(source_sha=OLD_SHA,approval_sha=OLD_APPROVAL))
    prefix=p.read(HERE/'reconciliation-inputs.json',PREFIX_SHA)
    p.need(prefix['prior_process_exit']==1 and len(prefix['states'])==61
        and prefix['requires_fresh_unmodified_original_context_and_complete_aggregate_reconstruction'] is True,
        'Not the complete real per-state prefix')
    for path,bound in prefix['bindings'].items():p.need(p.identity(path)==bound,'Changed real state-read evidence')
    bindings.update(prefix['bindings'])
    bindings[str(HERE/'reconciliation-inputs.json')]=p.identity(HERE/'reconciliation-inputs.json')
    return p,prefix,bindings


def reconstructed_aggregates(p,prefix,context,features,raw,output):
    from embed_optim.primary_contract import require_same
    from embed_optim.primary_v3_outcomes import csv_bytes
    from embed_optim.primary_v3_dimension_contract import TABLE_COUNTS
    manifest=context.exports._manifest(features.VECTOR_ROOT,context.admitted,raw['input_vectors']['sha256'])
    plan=context.features.feature_plan(context.contract,context.admitted,raw['input_vectors']['sha256'])
    accepted=context.features._feature_manifest(features.OUTPUT,plan,context.jobs)
    combined={name:[] for name in TABLE_COUNTS}
    bindings={}
    for job in context.jobs:
        cell=job['plan']['state']['cell'];bound=prefix['states'][cell]
        record=p.read(bound['path'],{k:bound[k] for k in ('bytes','sha256')})
        require_same(record['feature_manifest'],accepted['states'][cell])
        p.need(record['cell']==cell and record['fresh_raw_reconstruction_matches_every_csv_and_attribution_array'] is True,
            'Not a genuine successful state computation/readback')
        path=features.OUTPUT/'states'/cell
        saved=p.read(path/'manifest.json',{k:accepted['states'][cell][k] for k in ('bytes','sha256')})
        require_same(saved['plan'],context.features.state_plan(plan,job,manifest['states'][cell]))
        for name in TABLE_COUNTS:
            source=path/(name+'.csv')
            combined[name].extend(csv.DictReader(io.StringIO(source.read_text(),newline='')))
            bindings[str(source)]=p.identity(source)
    require_same({key:len(rows) for key,rows in combined.items()},TABLE_COUNTS)
    output.mkdir(exist_ok=False)
    for name,rows in combined.items():
        expected=csv_bytes(rows)
        original=features.OUTPUT/(name+'.csv')
        p.need(original.read_bytes()==expected,'Native aggregate differs from the complete raw-recomputed per-state tables')
        with (output/(name+'.csv')).open('xb') as stream:stream.write(expected)
        bindings[str(original)]=p.identity(original)
    require_same(context.features._feature_manifest(features.OUTPUT,plan,context.jobs),accepted)
    return bindings


def calculate(args):
    p,prefix,bindings=admission(args)
    # Original context is imported and rechecked before any statistical module.
    features=p.module('actual_feature_native_context',p.FEATURE_ENTRY/'features.py')
    prior,entry,context,_=features.authenticate(SimpleNamespace(
        source_sha=p.PINS[p.FEATURE_ENTRY/'features.py'],approval_sha=p.PINS[p.FEATURE_ENTRY/'owner-approval.json'],
        tests_sha=p.PINS[p.FEATURE_ENTRY/'tests.json']))
    raw=p.feature_completion(features,prefix['feature_completion_sha256'])
    original,original_bindings,joined=p.original_panel(context)
    p.need(original==p.read(HERE/'actual/original_panel.json')
        and joined==p.read(HERE/'actual/shared_checkpoint_identities.json'),'Actual weight/outcome panel changed')
    bindings.update(original_bindings)
    output=HERE/'reconciled'
    p.need(not output.exists() and not output.is_symlink(),'Preserve all earlier result attempts')
    output.mkdir()
    p.write(output/'started.json',{**prior.process_identity(os.getpid()),'started_at_utc':context.geometry.now(),
        'source_sha256':args.source_sha,'scope':SCOPE,'scientific_completion':False})
    bindings.update(reconstructed_aggregates(p,prefix,context,features,raw,output/'features'))
    # This unchanged original call must genuinely pass in the clean input namespace.
    after,_=entry.authenticate(prior.FUNCTIONAL_SHA,prior.INPUTS_SHA)
    from embed_optim.primary_contract import require_same
    require_same(after.admitted,context.admitted)
    context.exports._manifest(features.VECTOR_ROOT,context.admitted,raw['input_vectors']['sha256'])
    p.write(output/'native_input_readback.json',{'completed_at_utc':context.geometry.now(),
        'original_admission_before_and_after_passed':True,'unchanged_original_source_files':66,
        'genuine_per_state_raw_recomputations_reconciled':61,'complete_aggregate_families':4,
        'prior_outer_invocation_exit_code':1,'old_guard_changed':False,'scientific_completion':False})
    # Statistical imports are deliberately after that complete original admission.
    from embed_optim.primary_v3_dimension_inference import FunctionalInferenceContract
    from embed_optim.primary_v3_exact_bridge import typed_csv
    from embed_optim.dimension_inference import summarize,TABLE_COUNTS
    from embed_optim.dimension_inference_render import render
    from embed_optim.primary_v3_outcomes import csv_bytes
    contract=FunctionalInferenceContract.load(p.STORY/'configs/dense_primary_v3_dimension_inference_protocol.json',context.primary)
    tables={name:typed_csv(output/'features'/(name+'.csv')) for name in raw['table_counts']}
    tasks=sorted({row['source'] for row in context.identities})
    calculated,decisions=summarize(context.primary,tables,original,context.contract.scientific,tasks)
    require_same({name:len(rows) for name,rows in calculated.items()},TABLE_COUNTS)
    (output/'tables').mkdir()
    for name,rows in calculated.items():
        with (output/'tables'/(name+'.csv')).open('xb') as stream:stream.write(csv_bytes(rows))
    for name,value in [('tables.json',calculated),('decisions.json',decisions),
            ('scientific_protocol.json',context.contract.scientific),('original_panel.json',original),
            ('shared_checkpoint_identities.json',joined)]:p.write(output/name,value)
    with (output/'results.tex').open('x') as stream:stream.write(render(calculated,decisions))
    contract.recheck();admission(args)
    for path,bound in bindings.items():p.need(p.identity(path)==bound,'Bound input changed')
    for name in contract.payload['sources']:bindings[str(p.STORY/name)]=p.identity(p.STORY/name)
    result={'scope':SCOPE,'completed_at_utc':context.geometry.now(),'source_sha256':args.source_sha,
        'approval_sha256':args.approval_sha,'reconciliation_inputs_sha256':PREFIX_SHA,
        'feature_completion_sha256':prefix['feature_completion_sha256'],
        'feature_manifest_sha256':prefix['feature_manifest_sha256'],
        'genuine_raw_vector_states_recomputed_by_prior_invocation':61,'prior_invocation_exit_code':1,
        'fresh_original_context_admission_passed':True,'every_aggregate_reconstructed_from_bound_state_payloads':True,
        'input_table_counts':raw['table_counts'],'output_table_counts':TABLE_COUNTS,
        'source_bindings':bindings,'payloads':{str(path.relative_to(output)):p.identity(path)
            for path in sorted(output.rglob('*')) if path.is_file()},
        'raw_encoding_or_coordinate_computation_repeated_by_this_successor':False,
        'old_guards_modified':False,'formal_primary_consumer':False,'manuscript_installed':False,'scientific_completion':False}
    p.write(output/'readout.json',result)
    return {'result':p.identity(output/'readout.json'),'table_counts':TABLE_COUNTS}


def verify(args):
    p,_,_=admission(args);output=HERE/'reconciled'
    p.need(args.result_sha is not None,'Require external actual readout digest')
    result=p.read(output/'readout.json',args.result_sha)
    p.need(result['source_sha256']==args.source_sha and result['scope']==SCOPE
        and result['fresh_original_context_admission_passed'] is True,'Wrong reconciled numerical result')
    for path,bound in result['source_bindings'].items():p.need(p.identity(path)==bound,'Changed actual source/input')
    for name,bound in result['payloads'].items():p.need(p.identity(output/name)==bound,'Changed actual output')
    from embed_optim.primary_v3_exact_bridge import typed_csv
    tables=p.read(output/'tables.json');decisions=p.read(output/'decisions.json')
    for name in tables:p.need(typed_csv(output/'tables'/(name+'.csv'))==tables[name],'JSON/CSV results differ')
    reference=p.module('actual_independent_functional_inference',p.STORY/'scripts/dimension_inference_reference.py')
    inputs={'protocol':p.read(output/'scientific_protocol.json'),
        'tables':{name:typed_csv(output/'features'/(name+'.csv')) for name in result['input_table_counts']}}
    task=reference.task_inference(inputs,tables,decisions)
    predicted=reference.exact_predictions(tables)
    admission(args)
    for name,bound in result['payloads'].items():p.need(p.identity(output/name)==bound,'Results changed during independent verification')
    from datetime import datetime,timezone
    checked={'scope':SCOPE,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'readout':p.identity(output/'readout.json'),'separate_process_id':os.getpid(),
        'independent_task_inference':task,'independent_exact_predictions':predicted,
        'all_checks_passed':True,'same_data_not_independent_experiment':True,'scientific_completion':False}
    p.write(output/'independent_verification.json',checked)
    return {'all_checks_passed':True,'contrasts':len(task['contrasts']),
        'rotation_contrasts':len(task['rotations']),'exact_held_dose_predictions':predicted['exact_predictions']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('calculate','verify'))
    for name in ('source-sha','approval-sha'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--result-sha');args=parser.parse_args()
    print(json.dumps((calculate if args.mode=='calculate' else verify)(args),sort_keys=True),flush=True)
