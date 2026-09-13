"""Independent NumPy-only read of genuine copied arrays and preserved native receipts."""
import copy
import hashlib
import json
import re
from datetime import datetime,timezone
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).parents[1]
STORY=Path('/root/embedding-optimizer-story-refactor')
WORK=Path('/tmp/dense-v3-pretrained-native-copy.t5yq4jKO')


def need(v,m):
    if not v:raise ValueError(m)


def identity(p):
    p=Path(p)
    need(p.name!='gpu.py' and p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)),'Nonordinary or out-of-scope input')
    before=p.stat()
    with p.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    after=p.stat()
    need(all(getattr(before,k)==getattr(after,k) for k in
             ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'Input changed during read')
    return {'bytes':after.st_size,'sha256':sha}


def read(p):return json.loads(p.read_bytes())


def main():
    output=ROOT/'verification.json';need(not output.exists(),'Preserve verification')
    need(identity(ROOT/'actual/readout.json')['sha256']=='773c184b474c0508ff22a549db5303abefe922c95272bd539c32ed5c0b503195','Actual receipt differs')
    record=read(ROOT/'actual/readout.json')
    need(record['source']==identity(ROOT/'source/verify_native_copy.py')==identity(WORK/'source/verify_native_copy.py'),'Source differs')
    for key in ('actual_model_forward_or_encoding','cuda_initialized','new_worker_or_lease','new_authority_created',
                'recovery_launched','feature_computation','scientific_completion','physical_second_host'):
        need(record[key] is False,'Actual read scope differs')
    need(record['native_reads']==2 and record['saved_state_tensor_fingerprints_checked_each_read']==134,'Wrong native coverage')
    need(record['negative_controls']==[{'case':'wrong_external_manifest_anchor','refused':True,'error_type':'ValueError'},
        {'case':'different_state_plan','refused':True,'error_type':'ValueError'}],'Native refusal checks differ')
    first=read(ROOT/'actual/original-native-readout.json');second=read(ROOT/'actual/copied-native-readout.json')
    need(identity(ROOT/'actual/original-native-readout.json')==record['original_native_receipt']
         and identity(ROOT/'actual/copied-native-readout.json')==record['copied_native_receipt'],'Native receipt bytes differ')
    expected=copy.deepcopy(first);expected['manifest']['path']=str(Path(record['copied_output_root'])/'manifest.json')
    need(second==expected,'Copied native receipt differs beyond location')
    pair=ROOT/'actual/copied-vectors';original=Path(record['original_output_root'])
    for name,item in [('manifest.json',first['manifest']),('vectors.npz',first['output'])]:
        expected_id={k:item[k] for k in ('bytes','sha256')}
        for base in (pair,original,Path(record['copied_output_root'])):
            need(identity(base/name)==expected_id,'Original/copied vector file changed')
    arrays={}
    with np.load(pair/'vectors.npz',allow_pickle=False) as current, np.load(original/'vectors.npz',allow_pickle=False) as previous:
        need(set(current.files)==set(previous.files)==set(record['original_and_copied_arrays']),'Array population differs')
        for name,expected in record['original_and_copied_arrays'].items():
            a,b=current[name],previous[name]
            actual={'shape':list(a.shape),'dtype':str(a.dtype),'values':int(a.size),
                    'sha256':hashlib.sha256(a.tobytes()).hexdigest(),'bitwise_equal':a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()}
            need(actual==expected and actual['bitwise_equal'] is True,'Array byte equality differs')
            arrays[name]=actual
    manifest=read(pair/'manifest.json')
    observed=manifest['observed_loading']
    need(observed['before']==observed['after'] and len(observed['before']['state_tensors'])==134
         and observed['before']['state_tensors_sha256']==record['loaded_state_tensors_sha256'],'Loaded-state observation differs')
    for name,item in record['original_bound_files_unchanged'].items():need(identity(name)==item,'Original bound input changed')
    for name,item in record['pretrained_checkpoint_files_rechecked'].items():
        need(identity(Path(record['checkpoint_root'])/name)==item,'Pretrained checkpoint changed')
    need(len(record['original_bound_files_unchanged'])==66 and len(record['pretrained_checkpoint_files_rechecked'])==11,'Wrong source/checkpoint inventory')
    need(identity(ROOT/'source-original/primary_v3_dimension_vector_io.py')['sha256']==record['native_reader_sha256'],'Original native source differs')
    need('torch' not in sys.modules and 'embed_optim' not in sys.modules,'Archive verifier imported model code')
    need(identity(STORY/'paper/main.tex')['sha256']=='45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e','Manuscript changed')
    payloads={}
    secret=re.compile(rb'(?:wandb_v1_[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})')
    for p in sorted(ROOT.rglob('*')):
        need(not p.is_symlink(),'Symlinked archive')
        if p.is_file():
            need(secret.search(p.read_bytes()) is None,'Credential-like content; suppressed')
            payloads[p.relative_to(ROOT).as_posix()]=identity(p)
    links=0
    for target in re.findall(r'\]\(([^)]+)\)',(ROOT/'README.md').read_text()):
        resolved=(ROOT/target.split('#',1)[0]).resolve();need(resolved.exists() or resolved==output,'Broken archive link');links+=1
    result={'scope':'independent-genuine-pretrained-native-copy-archive-verification',
        'verified_at_utc':datetime.now(timezone.utc).isoformat(),'actual_native_reads':2,'tensors_checked_per_native_read':134,
        'arrays_independently_reopened':arrays,'original_bound_files_unchanged':66,'pretrained_checkpoint_files_unchanged':11,
        'bound_native_imports':len(record['bound_loaded_native_source_files']),'negative_native_controls':2,
        'native_reader_modified':False,'new_worker_or_lease':False,'recovery_authorized_or_launched':False,
        'model_encoding_or_feature_computation':False,'scientific_completion':False,'manuscript_modified':False,
        'remote_publication':False,'local_links_checked':links,'credential_findings':0,'payloads':payloads}
    with output.open('x') as stream:stream.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='payloads'},sort_keys=True))


if __name__=='__main__':main()
