"""Verify every new namespace source/resource in wheel and sdist, then execute cold."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import zipfile

here=Path(__file__).resolve().parent
root=Path('/root/embedding-optimizer-story-refactor')
dist=here/'dist'
wheel=next(dist.glob('*.whl'))
sdist=next(dist.glob('*.tar.gz'))
package=root/'src/embed_optim_restore'
expected={p.relative_to(root/'src').as_posix():p.read_bytes()
          for p in package.rglob('*') if p.is_file() and p.suffix in ('.py','.json')}
assert len(expected)==25
with zipfile.ZipFile(wheel) as archive:
    names={n for n in archive.namelist() if n.startswith('embed_optim_restore/') and not n.endswith('/')}
    assert names==set(expected)
    for name,raw in expected.items():
        assert archive.read(name)==raw
    for name in archive.namelist():
        path=Path(name)
        assert not path.is_absolute() and '..' not in path.parts
    archive.extractall(here/'wheel')
with tarfile.open(sdist) as archive:
    for name,raw in expected.items():
        member=archive.getmember('embedding_optimizer_study-0.1.0/src/'+name)
        assert member.isfile() and archive.extractfile(member).read()==raw
(here/'wheel-tests').mkdir()
shutil.copyfile(root/'tests/test_restore_entry.py',here/'wheel-tests/test_restore_entry.py')
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1',
     'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
     'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','WANDB_MODE':'disabled'}
with (here/'cold.log').open('xb') as log:
    result=subprocess.run(['/usr/bin/python','-I','-B',str(here/'cold.py')],cwd=here,env=env,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
record={'namespace_files':{n:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for n,v in expected.items()},
        'actual_cold_exit':result.returncode,'actual_original_audit_not_replaced':True}
with (here/'wheel-check.json').open('x') as stream:
    json.dump(record,stream,indent=2,sort_keys=True)
print(json.dumps({'namespace_files':len(expected),'actual_cold_exit':result.returncode}),flush=True)
raise SystemExit(result.returncode)
