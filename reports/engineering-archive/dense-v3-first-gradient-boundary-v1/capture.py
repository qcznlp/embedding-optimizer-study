"""Preserve terminal first-backward evidence without copying large gradient arrays."""
import hashlib
import json
from pathlib import Path
import re
import shutil

here = Path(__file__).resolve().parent
root = here.parents[2]
work = Path('/tmp/dense-v3-first-gradient.FDK2kcqs')

def identity(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}

def copy(src, dst):
    raw = src.read_bytes()
    if re.search(rb'(?:wandb_v1_|ghp_|hf_)[A-Za-z0-9_-]{24,}', raw):
        raise ValueError('Credential-shaped content; do not publish the value')
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open('xb') as out, src.open('rb') as original:
        shutil.copyfileobj(original, out)
    assert identity(src) == identity(dst)

assert identity(work/'probe.py')['sha256'] == '17c76f5519b49f305be4c22353e52262fa25d2e5804ec73e34459f127f311d5a'
assert identity(work/'authorization.json')['sha256'] == '6bbe4aea5db03a43ff8b01f1662673c495893d33b60cbe564528eb82c2868f49'
auth = json.loads((work/'authorization.json').read_bytes())
assert identity(work/'trace.py') == auth['trace']
assert identity(work/'test_trace.py') == auth['test_source']
assert identity(work/'tests.xml') == auth['tests']
training_auth = json.loads((Path('/root/embedding-optimizer-v3-experiment/launch/factorial-training-v1')/'authorization.json').read_bytes())
assert training_auth['source_files'] == auth['source_files']
for name, binding in auth['source_files'].items():
    assert identity(Path(training_auth['source_root'])/name) == binding
for name in ('probe.py','trace.py','test_trace.py','tests.xml','authorization.json',
             'compare_reduction.py','reduction-comparison.json','RUNNING.md'):
    copy(work/name, here/'executed'/name)
gradients = []
for pool in ('a','b'):
    job = work/'run'/('pool-'+pool)
    assert json.loads((job/'completed.json').read_bytes())['actual_rank_exits'] == [0]*4
    for path in sorted(job.rglob('*')):
        if path.is_file() and path.suffix in ('.json','.log'):
            copy(path, here/'actual'/path.relative_to(work/'run'))
    for rank in range(4):
        rank_dir = job/f'rank-{rank}'
        backward = json.loads((rank_dir/'backward.json').read_bytes())
        for name, key in (('ddp-preclip.safetensors','preclip'),('local-leaf-sum.safetensors','local_sum')):
            path = rank_dir/name
            assert identity(path) == backward[key]
            gradients.append({'path': str(path), **backward[key]})
for name in ('AGENTS.md','CURRENT_EXPERIMENT.md','PROJECT_STATUS.md'):
    copy(root/name, here/'after'/name)
with (here/'gradient-bindings.json').open('x') as out:
    json.dump({'files': gradients, 'large_arrays_copied_into_repo': False,
               'HF_upload_performed': False}, out, indent=2, sort_keys=True)
with (here/'terminal-tools.json').open('x') as out:
    json.dump({'trace_tests': {'session':26660,'terminal':'ccdcac','exit':0},
               'prepare': {'session':76884,'terminal':'bf1572','exit':0},
               'a': {'session':50164,'terminal':'84aa35','exit':0},
               'b': {'session':69733,'terminal':'4a2137','exit':0},
               'reduction': {'session':86535,'terminal':'bc1ffd','exit':0}}, out, indent=2)
rows = {p.relative_to(here).as_posix(): identity(p) for p in sorted(here.rglob('*')) if p.is_file()}
with (here/'manifest.json').open('x') as out:
    json.dump({'files': rows}, out, indent=2, sort_keys=True)
print(json.dumps({'files':len(rows),'bytes':sum(v['bytes'] for v in rows.values()),
                  'gradient_bytes_retained_in_original_directory':sum(v['bytes'] for v in gradients),
                  'manifest':identity(here/'manifest.json')}))
