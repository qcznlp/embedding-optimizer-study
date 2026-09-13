"""New-only immutable backup of the twelve actual crossed continuation runs.

Waits for real four-rank exits plus native whole-run reading. Uses the original
native worker reader and remote checksum comparator without changing them. No
GPU inspection/lease, training transition, source upload, deletion or overwrite.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
TRAIN = Path('/root/embedding-optimizer-v3-experiment/launch/factorial-training-v1')
TRAIN_SHA = 'd11d3281523b1573ce7fcd650e10c538cfa84022b5db92117559d47213e57e53'
AUTH_SHA = '00c52f0bec979fb163a734ad0f9eadcd6ea09f7393133fd35c65bc62878ce11f'
REPO = 'qcz/embedding-optimizer-study-checkpoints'
PREFIX = 'dense-v3-state-operator-v1/' + AUTH_SHA
SCOPE = 'actual_v3_crossed_continuation_new_only_durability_20260912'
OWNER = '你有权做一切事情，目标是尽快完成任务'


def need(ok, message):
    if not ok: raise ValueError(message)


def identity(path):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Nonordinary artifact')
    first = path.stat()
    with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    last = path.stat()
    need(all(getattr(first, k) == getattr(last, k) for k in
        ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')), 'Artifact raced')
    return {'bytes': last.st_size, 'sha256': digest}


def read(path, bound=None):
    actual = identity(path)
    if bound is not None:
        need(actual['sha256'] == bound if isinstance(bound,str) else actual == bound, 'External binding differs')
    def pairs(values):
        result = {}
        for key,value in values:
            need(key not in result, 'Duplicate JSON key'); result[key] = value
        return result
    result = json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
        parse_constant=lambda _: need(False,'Nonfinite JSON'))
    need(identity(path) == actual, 'Receipt raced')
    return result


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def admit(args):
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU/network only')
    need(identity(__file__)['sha256'] == args.source_sha, 'Changed backup source')
    approval = read(HERE/'owner-approval.json',args.approval_sha)
    need(approval == {'scope':SCOPE,'source_sha256':args.source_sha,'owner_message':OWNER,
        'automatic_continuation':False,'repository':REPO,'new_prefix':PREFIX,
        'all_twelve_runs_all_five_checkpoints':True,'new_only':True,'source_upload':False,
        'delete_or_overwrite':False,'gpu_access':False,'old_controller_transition':False,
        'scientific_completion':False}, 'Wrong scoped backup approval')
    need(identity(TRAIN/'factorial_dispatch.py')['sha256']==TRAIN_SHA, 'Actual training source changed')
    if str(TRAIN) not in sys.path: sys.path.insert(0,str(TRAIN))
    spec=importlib.util.spec_from_file_location('actual_factorial_backup_training_parent',TRAIN/'factorial_dispatch.py')
    entry=importlib.util.module_from_spec(spec);sys.modules[spec.name]=entry;spec.loader.exec_module(entry)
    auth,_=entry.authenticate(SimpleNamespace(source_sha=TRAIN_SHA,authorization_sha=AUTH_SHA))
    _,native,locations=entry.context(auth['source_files'])
    import torch
    need(not torch.cuda.is_initialized(),'Backup cannot initialize CUDA')
    return entry,auth,native,locations


def complete_record(entry, auth, run_id, pool):
    path=TRAIN/'run'/('pool-'+pool)/run_id
    result=read(path/'completed.json')
    need(result['run_id']==run_id and result['source_sha256']==TRAIN_SHA
        and result['authorization_sha256']==AUTH_SHA and result['actual_rank_exits']==[0,0,0,0]
        and result['actual_fresh_reader_exit']==0
        and result['full_horizon_391_steps_verified'] is True
        and result['all_five_checkpoints_verified'] is True, 'Not a complete actual branch')
    native=read(path/'fresh-native-readback.json',result['native_readback_binding'])
    need(native['worker_completion']==result['worker_completion']
        and native['fresh_process_native_readback'] is True,'Native completion chain differs')
    exits=read(path/'ranks.exited.json')
    need(exits['actual_rank_exits']=={str(i):0 for i in range(4)}
        and read(path/'reader.exited.json')['exit_code']==0,'Actual process exits differ')
    return path,result


def data_files(root):
    root=Path(root)
    files=[]
    for path in sorted(root.rglob('*')):
        need(not path.is_symlink() and (path.is_dir() or path.is_file()),'Nonordinary payload tree')
        if not path.is_file(): continue
        name=path.relative_to(root).as_posix()
        need(path.suffix in {'.json','.md','.safetensors','.pt','.pth','.bin'},'Unexpected source/secret payload type')
        if path.suffix in {'.json','.md'}:
            text=path.read_bytes()
            need(not re.search(rb'wandb_v1_[A-Za-z0-9_\-]+|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}',text),
                 'Credential-like text refused; no payload content is printed')
        files.append({'path':name,**identity(path)})
    need(bool(files),'Empty payload')
    return files


def backup_one(args, entry, auth, native, locations, api, run_id, pool):
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.errors import RemoteEntryNotFoundError
    from embed_optim.primary_v3_io import remote_inventory,compare_remote
    job,completed=complete_record(entry,auth,run_id,pool)
    declaration=auth['requests'][run_id]
    checked=native.read_execution(locations,declaration,completed['worker_completion'])
    need(checked['worker_and_native_artifacts_verified'] is True,'Native complete run rejected')
    run_root=Path(declaration['run_root'])
    need({p.name for p in run_root.iterdir()}=={
        'checkpoint_schedule.json','factorial_run_identity.json','factorial_run_complete.json',
        'trainer_state_final.json','final',*(f'checkpoint-{s}' for s in (79,157,235,313,391))},
        'Unexpected complete run root inventory')
    sources={'run':(run_root,data_files(run_root)),
        'native-worker':(Path(declaration['record_root']),data_files(declaration['record_root']))}
    proof_names=['completed.json','fresh-native-readback.json','ranks.exited.json','reader.exited.json',
        'admission.json',*[f'rank-{i}.returned.json' for i in range(4)]]
    sources['actual-dispatch']=(job,[{'path':name,**identity(job/name)} for name in proof_names])
    # Dispatch proof text is also screened without exposing its contents.
    for row in sources['actual-dispatch'][1]:
        text=(job/row['path']).read_bytes()
        need(not re.search(rb'wandb_v1_|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}',text),
             'Credential-like dispatch text refused')
    prefix=PREFIX+'/'+run_id
    receipt_root=HERE/'run'/run_id
    need(not receipt_root.exists(),'Preserve every backup attempt')
    receipt_root.mkdir()
    manifest={'scope':SCOPE,'run_id':run_id,'training_authorization_sha256':AUTH_SHA,
        'actual_dispatch_completion':identity(job/'completed.json'),'native_readback':checked,
        'files':{name+'/'+row['path']:row for name,(_,rows) in sources.items() for row in rows},
        'source_release':False,'scientific_completion':False}
    write(receipt_root/'artifact_manifest.json',manifest)
    head=api.repo_info(REPO,repo_type='model').sha
    need(re.fullmatch(r'[0-9a-f]{40}',head or ''),'Require immutable remote parent')
    try: existing=remote_inventory(api,REPO,prefix,head)
    except RemoteEntryNotFoundError: existing={}
    need(not existing,'Remote prefix exists; never overwrite or blindly retry')
    operations=[CommitOperationAdd(path_in_repo=prefix+'/'+name+'/'+row['path'],
        path_or_fileobj=str(root/row['path'])) for name,(root,rows) in sources.items() for row in rows]
    operations.append(CommitOperationAdd(path_in_repo=prefix+'/artifact_manifest.json',
        path_or_fileobj=str(receipt_root/'artifact_manifest.json')))
    write(receipt_root/'upload.started.json',{'run_id':run_id,'parent_commit':head,'prefix':prefix,
        'files':len(operations),'bytes':sum(row['bytes'] for _,rows in sources.values() for row in rows),
        'started_at_utc':entry.now(),'source_sha256':args.source_sha,'scientific_completion':False})
    print(json.dumps({'event':'new_run_backup_started','run_id':run_id,'files':len(operations)}),flush=True)
    commit=api.create_commit(REPO,repo_type='model',operations=operations,parent_commit=head,
        commit_message='Back up genuine crossed Dense continuation '+run_id)
    need(re.fullmatch(r'[0-9a-f]{40}',getattr(commit,'oid','') or ''),'No immutable upload commit')
    write(receipt_root/'upload.returned.json',{'repo_id':REPO,'prefix':prefix,'commit_oid':commit.oid,
        'returned_at_utc':entry.now(),'durability_verified':False})
    # Fresh anonymous remote reads; no mutable-HEAD substitution.
    from huggingface_hub import HfApi
    remote=remote_inventory(HfApi(token=False),REPO,prefix,commit.oid)
    need(set(remote)==set(manifest['files'])|{'artifact_manifest.json'},'Complete remote inventory differs')
    for name,(root,rows) in sources.items():
        compare_remote(root,rows,{k[len(name)+1:]:v for k,v in remote.items() if k.startswith(name+'/')})
        for row in rows: need(identity(root/row['path'])=={k:row[k] for k in ('bytes','sha256')},'Local payload changed')
    compare_remote(receipt_root,[{'path':'artifact_manifest.json',**identity(receipt_root/'artifact_manifest.json')}],
        {'artifact_manifest.json':remote['artifact_manifest.json']})
    need(native.read_execution(locations,declaration,completed['worker_completion'])==checked,'Native read changed after upload')
    result={'scope':SCOPE,'run_id':run_id,'repo_id':REPO,'repo_type':'model','prefix':prefix,
        'commit_oid':commit.oid,'verified_at_utc':entry.now(),'remote_inventory':remote,
        'artifact_manifest':identity(receipt_root/'artifact_manifest.json'),'all_five_checkpoints':True,
        'anonymous_remote_metadata_verified':True,'all_payloads_downloaded_again':False,
        'cross_host_native_resume_claimed':False,'scientific_completion':False}
    write(receipt_root/'verified.json',result)
    print(json.dumps({'event':'run_backup_verified','run_id':run_id,'commit_oid':commit.oid}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-sha','approval-sha'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--inspect',action='store_true')
    args=parser.parse_args()
    entry,auth,native,locations=admit(args)
    units=[(run,pool) for pool,runs in auth['queues'].items() for run in runs]
    need(len(units)==12 and len(set(run for run,_ in units))==12,'Wrong fixed branch population')
    if args.inspect:
        print(json.dumps({'new_prefix':PREFIX,'runs':len(units),'checkpoints':60,
            'source_admitted':True,'external_writes':False,'gpu_access':False}));return
    (HERE/'run').mkdir(exist_ok=False)
    write(HERE/'run/started.json',{'pid':os.getpid(),'started_at_utc':entry.now(),
        'source_sha256':args.source_sha,'approval_sha256':args.approval_sha,'scientific_completion':False})
    from huggingface_hub import HfApi
    api=HfApi();need(api.whoami()['name']=='qcz','Unexpected artifact account')
    done,stable={},{}
    while len(done)<12:
        entry,auth,native,locations=admit(args)
        for run,pool in units:
            if run in done: continue
            path=TRAIN/'run'/('pool-'+pool)/run/'completed.json'
            if not path.is_file(): continue
            bound=identity(path)
            previous=stable.get(run);stable[run]=bound
            if previous!=bound: continue
            done[run]=backup_one(args,entry,auth,native,locations,api,run,pool)
        if len(done)<12: time.sleep(30)
    write(HERE/'run/completed.json',{'completed_at_utc':entry.now(),'verified_runs':12,
        'verified_checkpoints':60,'receipts':{k:identity(HERE/'run'/k/'verified.json') for k in done},
        'scientific_completion':False})


if __name__=='__main__': main()
