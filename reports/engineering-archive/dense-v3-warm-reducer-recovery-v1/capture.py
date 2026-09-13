"""Seal terminal warm-reducer evidence; large diagnostic saves remain in place."""
import hashlib
import json
from pathlib import Path
import re
import shutil

here=Path(__file__).resolve().parent
root=here.parents[2]
work=Path('/tmp/dense-v3-warm-reducer-recovery.rP4NVV4i')
def identity(path):
    with path.open('rb') as stream:
        sha=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'bytes':path.stat().st_size,'sha256':sha}
def copy(src,dst):
    if src.is_symlink() or re.search(rb'(?:wandb_v1_|ghp_|hf_)[A-Za-z0-9_-]{24,}',src.read_bytes()):
        raise ValueError('Refuse symlink or credential-shaped content; no value emitted')
    dst.parent.mkdir(parents=True,exist_ok=True)
    with src.open('rb') as source,dst.open('xb') as output:
        shutil.copyfileobj(source,output)
    assert identity(src)==identity(dst)

assert identity(work/'probe.py')['sha256']=='3edaa4ea6ed39006bcf178cf611bee46811d0319e7d2608857eb250df7242fbb'
assert identity(work/'authorization.json')['sha256']=='49de01d10a5bca26436db205dd0159df49813daf73abe68dd03ee23f52022b08'
auth=json.loads((work/'authorization.json').read_bytes())
for name,key in (('trace.py','trace'),('paired.py','paired'),('gate.py','gate'),
                 ('test_trace.py','test_source'),('test_paired.py','paired_test'),
                 ('test_gate.py','gate_test'),('tests-final.xml','tests')):
    assert identity(work/name)==auth[key]
tools=json.loads((here/'terminal-tools.json').read_bytes())
for pool in ('a','b'):
    job=work/'run'/('pool-'+pool)
    assert tools[pool]['exit'] in (0,1)
    assert json.loads((job/'ranks.exited.json').read_bytes())['actual_rank_exits']==[0]*4
    reader=json.loads((job/'comparison.exited.json').read_bytes())['exit_code']
    assert reader==tools[pool]['exit']
    assert (job/('completed.json' if reader==0 else 'failed.json')).is_file()
    for path in sorted(job.rglob('*')):
        if path.is_file() and path.suffix in ('.json','.log'):
            copy(path,here/'actual'/path.relative_to(work/'run'))
for path in sorted(work.iterdir()):
    if path.is_file() and path.suffix in ('.py','.xml','.json','.md'):
        copy(path,here/'executed'/path.name)
for name in ('AGENTS.md','CURRENT_EXPERIMENT.md','PROJECT_STATUS.md','README.md','docs/completion-gates.md','docs/current-training-source.md'):
    copy(root/name,here/'after'/name)
rows={p.relative_to(here).as_posix():identity(p) for p in sorted(here.rglob('*')) if p.is_file()}
with (here/'manifest.json').open('x') as stream:
    json.dump({'files':rows,'large_diagnostic_saved_states_copied':False,'HF_upload':False},stream,indent=2,sort_keys=True)
print(json.dumps({'files':len(rows),'bytes':sum(v['bytes'] for v in rows.values()),'manifest':identity(here/'manifest.json')}),flush=True)
