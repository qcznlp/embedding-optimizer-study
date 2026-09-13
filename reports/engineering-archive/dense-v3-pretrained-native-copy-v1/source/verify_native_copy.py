"""Actual CPU checkpoint-backed inspection of existing pretrained vectors after copying."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime,timezone

WORK=Path(__file__).parents[1]
STORY=Path('/root/embedding-optimizer-story-refactor')
OLD=Path('/root/embedding-optimizer-v3-experiment/launch/functional-dimensions')
INPUTS_SHA='be6d92fda2f7f0ba3e86f2c2866fa9dca530f066d29ce92d0887fdad91684067'
AUTH_SHA='d72527595e2c963eae7fd46b0a1de2bd83d1ea15a863ae467a1065cfc5b2f336'
ENCODED_SHA='d668916f56dfd7c4f8ee35f6f5f189a4f397e7c08f09456e0bca4e4d477d32c5'
VERIFIED_SHA='8867643a9309af31b73e9b4f6bd9b3b8329389bb687b1e967d5043482cdd1dcb'
MANIFEST_SHA='90d9f044b53c05065b00d16a53a31a5c82a37d86e33552d797f6c38364e64fac'
RAW_SHA='e57057107312363619ebb1ba19b55892fe539568d9545726f3c9b9277325112f'
NATIVE_SHA='ea33869f8cf677ad0c15e8747b6d91045ba55a39301a8bf2c52f09bf642ed858'


def need(condition,message):
    if not condition:raise ValueError(message)


def identity(path):
    path=Path(path)
    need(path.name!='gpu.py','Protected helper is out of scope')
    need(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)),'Nonordinary input')
    before=path.stat()
    with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    after=path.stat()
    need(all(getattr(before,k)==getattr(after,k) for k in
             ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'File changed during read')
    return {'bytes':after.st_size,'sha256':sha}


def bound(path,expected):
    got=identity(path)
    expected={'sha256':expected} if isinstance(expected,str) else expected
    need(all(got[k]==v for k,v in expected.items()),'Bound input differs: '+str(path))
    return got


def read(path,sha=None):
    if sha:bound(path,sha)
    before=identity(path);value=json.loads(Path(path).read_bytes())
    need(identity(path)==before,'JSON changed during read');return value


def dump(path,value):
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CUDA must remain hidden')
    need(WORK.parent==Path('/tmp') and WORK.name.startswith('dense-v3-pretrained-native-copy.'),
         'This one-shot actual call needs its fresh preparation directory')
    target=WORK/'copied-vectors'
    need(not target.exists() and not (WORK/'readout.json').exists(),'Preserve prior copy/readout')
    authority=read(OLD/'authorization.json',AUTH_SHA)
    inputs=read(OLD/'inputs.json',INPUTS_SHA)
    preceding=read(STORY/'reports/engineering-archive/dense-v3-functional-flow-candidate-v1/actual/tests-third.json',
                   '6b704be5ad2377437f42778deee8ff97b79053b27f03ae1c8b3963d4d09dfdc3')
    protected=preceding['protected_files']
    for name,item in protected.items():bound(name,item)
    for name,item in authority['sources'].items():bound(name,item)
    job=inputs['jobs'][0];checkpoint=Path(job['checkpoint']);plan=job['plan']
    need(plan['state']['cell']=='pretrained' and plan['state']['stage']==0
         and str(checkpoint)==inputs['reference_root'],'Wrong immutable pretrained plan')
    model_files={r['path']:{k:r[k] for k in ('bytes','sha256')} for r in inputs['admitted']['reference']['files']}
    need(len(model_files)==11,'Wrong pretrained input inventory')
    for name,item in model_files.items():bound(checkpoint/name,item)
    original=Path(inputs['output_root'])/'vectors/states/pretrained'
    bound(original/'manifest.json',MANIFEST_SHA);bound(original/'vectors.npz',RAW_SHA)
    encoded=read(OLD/'run/jobs/pretrained.encoded.json',ENCODED_SHA)
    verified=read(OLD/'run/jobs/pretrained.verified.json',VERIFIED_SHA)
    need(verified['actual_exit_code']==0 and verified['native_readback_passed'] is True
         and encoded['saved']==verified['saved'] and encoded['plan_sha256']==job['plan_sha256']
         and encoded['authorization_sha256']==AUTH_SHA,'Prior accepted worker evidence differs')
    bound(STORY/'src/embed_optim/primary_v3_dimension_vector_io.py',NATIVE_SHA)
    import torch
    import numpy as np
    from embed_optim import primary_v3_dimension_vector_io as native
    need(Path(native.__file__).resolve()==STORY/'src/embed_optim/primary_v3_dimension_vector_io.py',
         'Foreign native reader')
    need(not torch.cuda.is_initialized(),'Import initialized CUDA unexpectedly')
    torch.set_num_threads(4)
    dependencies={}
    for name,module in tuple(sys.modules.items()):
        if name=='embed_optim' or name.startswith('embed_optim.'):
            source=Path(module.__file__).resolve()
            need(str(source) in authority['sources'],'Unbound native package import: '+name)
            dependencies[str(source)]=bound(source,authority['sources'][str(source)])
    identities=inputs['probe']['row_identities']
    arrays,actual_original=native.inspect_vectors(original,plan,identities,checkpoint,
                                                  expected_manifest_sha256=MANIFEST_SHA)
    need(actual_original==encoded['saved'],'Actual original native read differs from accepted receipt')
    target.mkdir(exist_ok=False)
    for name,sha in (('manifest.json',MANIFEST_SHA),('vectors.npz',RAW_SHA)):
        with (original/name).open('rb') as src,(target/name).open('xb') as dst:shutil.copyfileobj(src,dst)
        bound(target/name,sha)
    copied,actual_copy=native.inspect_vectors(target,plan,identities,checkpoint,
                                             expected_manifest_sha256=MANIFEST_SHA)
    expected=copy.deepcopy(actual_original);expected['manifest']['path']=str(target/'manifest.json')
    need(actual_copy==expected,'Copied native receipt differs beyond its explicit absolute location')
    array_results={}
    need(set(arrays)==set(copied)==native.ARRAYS,'Different array population')
    for name in sorted(arrays):
        a,b=arrays[name],copied[name]
        need(a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
             and a.tobytes()==b.tobytes(),'Array values or bytes differ')
        array_results[name]={'shape':list(a.shape),'dtype':str(a.dtype),'values':int(a.size),
                             'sha256':hashlib.sha256(a.tobytes()).hexdigest(),'bitwise_equal':True}
    controls=[]
    wrong=copy.deepcopy(plan);wrong['state']['meta']['step']=1
    for label,test_plan,sha in (('wrong_external_manifest_anchor',plan,'0'*64),('different_state_plan',wrong,MANIFEST_SHA)):
        try:native.inspect_vectors(target,test_plan,identities,checkpoint,expected_manifest_sha256=sha)
        except ValueError as error:controls.append({'case':label,'refused':True,'error_type':type(error).__name__})
        else:raise ValueError('Native reader accepted wrong input: '+label)
    raw_manifest=read(target/'manifest.json',MANIFEST_SHA)
    loaded=raw_manifest['observed_loading']
    need(loaded['before']==loaded['after'] and loaded['parameters_unchanged'] is True
         and len(loaded['before']['state_tensors'])==134,'Wrong immutable loaded-state observation')
    for name,item in protected.items():bound(name,item)
    for name,item in model_files.items():bound(checkpoint/name,item)
    for name,sha in (('manifest.json',MANIFEST_SHA),('vectors.npz',RAW_SHA)):
        bound(original/name,sha);bound(target/name,sha)
    need(not torch.cuda.is_initialized(),'Native compatibility inspection initialized CUDA')
    dump(WORK/'original-native-readout.json',actual_original)
    dump(WORK/'copied-native-readout.json',actual_copy)
    result={'scope':'actual-checkpoint-backed-pretrained-vector-copy-compatibility',
        'completed_at_utc':datetime.now(timezone.utc).isoformat(),'source':identity(Path(__file__)),
        'old_authority_sha256':AUTH_SHA,'inputs_sha256':INPUTS_SHA,'original_output_root':str(original),
        'copied_output_root':str(target),'checkpoint_root':str(checkpoint),'plan_sha256':job['plan_sha256'],
        'native_reader_sha256':NATIVE_SHA,'bound_loaded_native_source_files':dependencies,
        'original_bound_files_unchanged':protected,'pretrained_checkpoint_files_rechecked':model_files,
        'original_accepted_worker_unchanged':True,'native_reads':2,'saved_state_tensor_fingerprints_checked_each_read':134,
        'original_and_copied_arrays':array_results,'loaded_state_tensors_sha256':loaded['before']['state_tensors_sha256'],
        'original_native_receipt':identity(WORK/'original-native-readout.json'),
        'copied_native_receipt':identity(WORK/'copied-native-readout.json'),'negative_controls':controls,
        'only_returned_manifest_absolute_location_changed':True,'raw_vector_files_byte_identical':True,
        'actual_model_forward_or_encoding':False,'cuda_initialized':False,'new_worker_or_lease':False,
        'new_authority_created':False,'recovery_launched':False,'feature_computation':False,
        'scientific_completion':False,'physical_second_host':False}
    dump(WORK/'readout.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in
         ('original_bound_files_unchanged','pretrained_checkpoint_files_rechecked','bound_loaded_native_source_files')},sort_keys=True))


if __name__=='__main__':main()
