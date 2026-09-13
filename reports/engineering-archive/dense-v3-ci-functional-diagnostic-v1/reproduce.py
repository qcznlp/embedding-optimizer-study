"""Reproduce the hosted numerical failure in a separate genuine-package environment."""
import json
import os
from pathlib import Path
import subprocess

work=Path(__file__).resolve().parent
root=Path('/root/embedding-optimizer-story-refactor')
venv=work/'venv'
env={k:os.environ[k] for k in ('PATH','LANG','LC_ALL','LD_LIBRARY_PATH') if k in os.environ}
env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root/'src')+os.pathsep+str(root),
           PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
records=[]
def run(name,argv):
    with (work/(name+'.log')).open('xb') as stream:
        result=subprocess.run(argv,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
    records.append({'name':name,'argv':argv,'returncode':result.returncode})
    (work/'commands.json').write_text(json.dumps(records,indent=2))
    print(name,result.returncode,flush=True)
    result.check_returncode()
run('create',['uv','venv','--system-site-packages','--python','/usr/bin/python',str(venv)])
base=['uv','pip','install','--python',str(venv/'bin/python'),'--no-config','--require-hashes']
run('formal-base',base+['--torch-backend','cu129','--overrides',str(root/'requirements-formal.lock'),'-r',str(root/'requirements-formal.lock')])
run('actual-primary-hub',base+['--no-deps','-r',str(root/'requirements-primary-replay.txt')])
run('authentic-flash-binary',base+['--no-deps','--reinstall-package','flash-attn','-r',str(root/'requirements-ci-flash-wheel.txt')])
run('runtime',[str(venv/'bin/python'),'-B','-m','embed_optim.runtime','--spec',str(root/'configs/formal_runtime.json')])
run('full-numerical-paper',['make','-C','paper','release','PYTHON='+str(venv/'bin/python'),
    'NUMERICAL_BUNDLE='+str(root/'reports/engineering-archive/dense-v3-complete-replay-entry-v1/closed'),
    'RELEASE_OUTPUT='+str(work/'complete-paper')])
