"""Freeze the 61 real successful state reads, retaining the failed outer call."""
import importlib.util
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prior_readout',ROOT/'inference.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
p.need(p.identity(ROOT/'inference.py')['sha256']=='84e6aaf4dcb573ac04b74c6e97bf88b44fe4e7c410842eb80228239555168254','Changed actual reader')
inputs=p.read(p.EXP/'launch/functional-dimensions/inputs.json','be6d92fda2f7f0ba3e86f2c2866fa9dca530f066d29ce92d0887fdad91684067')
features=p.EXP/'analyses/dense-primary-v3-functional-features-recovery-v3'
manifest=p.read(features/'manifest.json','a5ef6021dc9ab535c5888e753f0a1397406a614f229894db28bd6c802f7353f5')
started=p.read(ROOT/'actual/started.json')
p.need(started['pid']==785676 and started['start_ticks']==319933766
       and started['source_sha256']==p.identity(ROOT/'inference.py')['sha256'],'Wrong executed reader')
terminal=p.read(ROOT/'terminal.json')
p.need(terminal['exit_code']==1 and terminal['per_state_fresh_raw_reconstruction_and_native_readbacks_returned']==61,'Changed actual terminal observation')
paths=[ROOT/'inference.py',ROOT/'test_inference.py',ROOT/'tests.json',ROOT/'owner-approval.json',ROOT/'preparation.json',
       ROOT/'terminal.json',ROOT/'actual/started.json',ROOT/'actual/original_panel.json',ROOT/'actual/shared_checkpoint_identities.json',
       features/'manifest.json',p.EXP/'launch/functional-dimensions/inputs.json']
states={}
for job in inputs['jobs']:
    cell=job['plan']['state']['cell'];path=ROOT/'actual/states'/(cell+'.json')
    value=p.read(path)
    p.need(value['cell']==cell and value['feature_manifest']==manifest['states'][cell]
           and value['fresh_raw_reconstruction_matches_every_csv_and_attribution_array'] is True,'Incomplete genuine state reconstruction')
    states[cell]={'path':str(path),**p.identity(path)};paths.append(path)
p.need(len(states)==61 and {str(x) for x in (ROOT/'actual/states').rglob('*.json')}=={v['path'] for v in states.values()},'Incomplete/extended readback state family')
p.need(not (ROOT/'actual/readout.json').exists() and not (ROOT/'actual/tables.json').exists(),'Do not relabel a completed inference')
record={'scope':'complete_per_state_native_reads_from_failed_outer_source_inventory_check',
        'prior_process_exit':1,'states':states,'bindings':{str(path):p.identity(path) for path in paths},
        'feature_manifest_sha256':p.identity(features/'manifest.json')['sha256'],
        'feature_completion_sha256':'29ad21b3cb6b7b90283b6da2f162472d9ac402298debcd6a17fef4e6a6750304',
        'requires_fresh_unmodified_original_context_and_complete_aggregate_reconstruction':True,
        'new_inference_executed':False,'scientific_completion':False}
p.write(ROOT/'reconciliation-inputs.json',record)
print(json.dumps({'states':len(states),'binding':p.identity(ROOT/'reconciliation-inputs.json')}))
