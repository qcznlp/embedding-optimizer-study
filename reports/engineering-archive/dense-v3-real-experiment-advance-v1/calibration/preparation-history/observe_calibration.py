"""Observe only the externally anchored v2 coordinator and its exact created children."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path('/root/embedding-optimizer-v3-experiment/launch/factorial-calibration-v2')
SOURCE='3436597f8ac85b1dde1e53d4fa2fef007cf19fbbc97d168b439a61cd2ef6b598'
AUTH='f7d29c908fc251a2affcf8c2205992f4f44d61b4b97201a60569e79e5e78081d'
STARTED='ce97c0e82fc049146f8d24128127736b5aefe51788f960858a41d765070ecbc4'
sys.path.insert(0,str(ROOT))
import calibration_dispatch as entry

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
auth,_=entry.authenticate(SimpleNamespace(source_sha=SOURCE,authorization_sha=AUTH))
started=entry.read(ROOT/'run/coordinator.started.json',STARTED)
prefix=['/usr/bin/python','-B',str(ROOT/'calibration_dispatch.py'),'--source-sha',SOURCE,'--authorization-sha',AUTH]
entry.need(started['command']==prefix+['--coordinate'] and started['source_sha256']==SOURCE
           and started['authorization_sha256']==AUTH and started['pid']==773704,
           'Wrong actual coordinator')
old=entry.EXPERIMENT/'launch/functional-dimensions/observe.py'
entry.need(entry.identity(old)['sha256']=='c314e4d540305bb67e0bcdcb6ad921d22926e4cbd40f59e00e84415be18cd078','Process reader changed')
spec=importlib.util.spec_from_file_location('unchanged_exact_process_reader',old)
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
workers={}
for state in entry.STATES:
    workers[state]={}
    for suffix in ('', '.verify'):
        name=state+suffix;start=ROOT/'run'/f'{name}.started.json'
        if not start.exists():continue
        s=entry.read(start);argv=s['command']
        entry.need(s['ppid']==started['pid'] and s['authorization_sha256']==AUTH,'Wrong exact child owner')
        if not suffix:
            expected=prefix+['--worker',state,'--gpu-token',entry.GPU[state]]
            entry.need(argv[:-4]==expected and argv[-4]=='--lease-fd' and argv[-2]=='--lease-fd'
                       and len({int(argv[-3]),int(argv[-1])})==2 and s['source_sha256']==SOURCE,'Wrong GPU child command')
        else:
            produced=entry.read(ROOT/'run'/f'{state}.produced.json')
            entry.need(argv==prefix+['--verify',state,'--calibration-sha',produced['calibration_binding']['sha256']],
                       'Wrong CPU readback command')
        end=ROOT/'run'/f'{name}.exited.json'
        if end.exists():
            e=entry.read(end)
            entry.need(all(e[k]==s[k] for k in ('pid','ppid','start_ticks','command','authorization_sha256')),'Wrong exact terminal child')
            value={'terminal':True,'exit_code':e['exit_code']}
        else:value={'terminal':False,'observed':reader.process(s)}
        workers[state]['fresh_reader' if suffix else 'gpu_worker']=value
    for name in ('produced','verified'):
        path=ROOT/'run'/f'{state}.{name}.json'
        if path.exists():
            v=entry.read(path)
            entry.need(v['state']==state and v['source_sha256']==SOURCE and v['authorization_sha256']==AUTH,
                       'State result authority differs')
            entry.need(entry.identity(entry.OUTPUT/state/'directions/calibration.json')==v['calibration_binding'],
                       'Native calibration anchor changed')
            workers[state][name]={'calibration_binding':v['calibration_binding'],'calibration':v['calibration']}
terminal={}
for name in ('completed','failed'):
    path=ROOT/'run'/f'{name}.json'
    if path.exists():
        v=entry.read(path)
        entry.need(v['source_sha256']==SOURCE and v['authorization_sha256']==AUTH and v['scientific_completion'] is False,
                   'Wrong coordinator terminal authority')
        terminal[name]=v
entry.need(len(terminal)<=1,'Conflicting coordinator terminals')
if 'completed' in terminal:
    for state in entry.STATES:
        w=workers[state]
        entry.need(w['gpu_worker']==w['fresh_reader']=={'terminal':True,'exit_code':0}
                   and w['produced']==w['verified'],'Incomplete accepted calibration chain')
    entry.need(terminal['completed']['fresh_process_native_readbacks']==2
               and terminal['completed']['actual_gpu_worker_exits']=={s:0 for s in entry.STATES},'Incomplete native completion')
result={'observed_at_utc':datetime.now(timezone.utc).isoformat(),
        'coordinator':{'terminal_record':next(iter(terminal))} if terminal else reader.process(started),
        'workers':workers,'terminal_records':terminal,
        'writes_only_this_observation':True,'broad_process_or_gpu_enumeration':False,'scientific_completion':False}
with args.output.open('x') as stream:stream.write(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(result,sort_keys=True),flush=True)
