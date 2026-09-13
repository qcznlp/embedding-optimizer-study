"""Archive both terminal paired attempts without copying large gradient arrays."""
import hashlib
import json
from pathlib import Path
import re
import shutil

here = Path(__file__).resolve().parent
root = here.parents[2]
work = Path('/tmp/dense-v3-paired-backward-fixed.JXOJBril')
failed = Path('/tmp/dense-v3-paired-backward.lNjcKc0t')

def identity(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes':path.stat().st_size, 'sha256':digest}

def copy(src, dst):
    assert not src.is_symlink()
    if re.search(rb'(?:wandb_v1_|ghp_|hf_)[A-Za-z0-9_-]{24,}',src.read_bytes()):
        raise ValueError('Credential-shaped content; value not emitted')
    dst.parent.mkdir(parents=True,exist_ok=True)
    with dst.open('xb') as out, src.open('rb') as original:
        shutil.copyfileobj(original,out)
    assert identity(src)==identity(dst)

assert identity(work/'probe.py')['sha256']=='52d44883225e27ee1b0db9cd862ff949b14ec060f58f291d13dad1f8320cc34a'
assert identity(work/'authorization.json')['sha256']=='b43f2ba1a49e0e53b5049d2f27234fde818cecbe90ffc73e1950cd36aa9bab52'
assert identity(failed/'probe.py')['sha256']=='28d706e46c7a6ae549532fbcce7a951553e38a06d3e731d6eaa96149c7425774'
assert identity(failed/'authorization.json')['sha256']=='38d6add0e12786e722312fd40628ff94fe3c0408ed243c9bada6521da3473ef6'
auth=json.loads((work/'authorization.json').read_bytes())
for name,key in (('trace.py','trace'),('paired.py','paired'),('test_trace.py','test_source'),
                 ('test_paired.py','paired_test'),('tests.xml','tests')):
    assert identity(work/name)==auth[key]
for name in ('probe.py','trace.py','paired.py','test_trace.py','test_paired.py','tests.xml',
             'authorization.json','RUNNING.md','compare.py','comparison.json'):
    copy(work/name,here/'executed'/name)
for path in sorted(failed.iterdir()):
    if path.is_file() and path.suffix in ('.py','.json','.xml','.md'):
        copy(path,here/'failed-first'/path.name)
gradients=[]
for attempt,destination in ((work,here/'actual'),(failed,here/'failed-first'/'run')):
    for pool in ('a','b'):
        job=attempt/'run'/('pool-'+pool)
        for path in sorted(job.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.log'):
                copy(path,destination/path.relative_to(attempt/'run'))
        for rank in range(4):
            d=job/f'rank-{rank}'
            for receipt,names in (('backward.json',(('ddp-preclip.safetensors','preclip'),('local-leaf-sum.safetensors','local_sum'))),
                                  ('paired.json',(('warm-ddp-preclip.safetensors','warm_preclip'),('warm-local-leaf-sum.safetensors','warm_local_sum')))):
                if not (d/receipt).exists():
                    continue
                value=json.loads((d/receipt).read_bytes())
                for name,key in names:
                    path=d/name
                    assert identity(path)==value[key]
                    gradients.append({'path':str(path),**value[key]})
for name in ('AGENTS.md','CURRENT_EXPERIMENT.md','PROJECT_STATUS.md'):
    copy(root/name,here/'after'/name)
with (here/'gradient-bindings.json').open('x') as out:
    json.dump({'files':gradients,'large_arrays_copied_into_repo':False,'HF_upload_performed':False},out,indent=2,sort_keys=True)
with (here/'terminal-tools.json').open('x') as out:
    json.dump({'first_tests':[73922,'2e57d9',0],'first_prepare':[95598,'1cf94c',0],
               'first_a':[17406,'663091',1],'first_b':[39489,'33677f',1],
               'tests':[11434,'0de2c0',0],'prepare':[30104,'f2d912',0],
               'a':[32954,'c06ef0',0],'b':[85232,'6ee966',0],
               'cpu_comparison':{'artifacts_complete':True,'session':None,'exit':None,
                                 'reason':'Original launch output lost; no artifact-to-exit inference'}},out,indent=2)
rows={p.relative_to(here).as_posix():identity(p) for p in sorted(here.rglob('*')) if p.is_file()}
with (here/'manifest.json').open('x') as out:
    json.dump({'files':rows},out,indent=2,sort_keys=True)
print(json.dumps({'files':len(rows),'bytes':sum(v['bytes'] for v in rows.values()),
                  'gradient_bytes_retained':sum(v['bytes'] for v in gradients),'manifest':identity(here/'manifest.json')}),flush=True)
