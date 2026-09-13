"""Reproduce the single hosted failure with genuine packages in an isolated venv."""
import json
import os
from pathlib import Path
import subprocess

work = Path(__file__).resolve().parent / 'hub-role-check'
work.mkdir(exist_ok=False)
root = Path('/root/embedding-optimizer-story-refactor')
env = {k:os.environ[k] for k in ('PATH','LANG','LC_ALL','LD_LIBRARY_PATH') if k in os.environ}
env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(root/'src')+os.pathsep+str(root),
           PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
receipts=[]
def run(name, argv, expected=0):
    with (work/(name+'.log')).open('xb') as stream:
        result=subprocess.run(argv,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
    receipt={'name':name,'argv':argv,'returncode':result.returncode,'expected_returncode':expected}
    receipts.append(receipt)
    (work/'commands.json').write_text(json.dumps(receipts,indent=2))
    assert result.returncode==expected, receipt
venv=work/'venv'
run('create', ['uv','venv','--system-site-packages','--python','/usr/bin/python',str(venv)])
lines=(root/'requirements-formal.lock').read_text().splitlines()
index=next(i for i,s in enumerate(lines) if s.startswith('huggingface-hub=='))
record=[lines[index]]
while record[-1].endswith('\\'):
    index+=1
    record.append(lines[index])
mismatch=work/'original-lock-hub.txt'
mismatch.write_text('\n'.join(record)+'\n')
base=['uv','pip','install','--python',str(venv/'bin/python'),'--no-config','--no-deps','--require-hashes']
run('install-original-1.29',base+['-r',str(mismatch)])
test=[str(venv/'bin/python'),'-B','-m','pytest','-q','tests/test_current_primary_source.py']
run('original-source-refuses-1.29',test+['--junitxml='+str(work/'mismatch.xml')],1)
run('install-recorded-1.28',base+['-r',str(root/'requirements-primary-replay.txt')])
run('original-source-accepts-1.28',test+['--junitxml='+str(work/'matched.xml')])
run('original-runtime',[str(venv/'bin/python'),'-B','-m','embed_optim.runtime','--spec',str(root/'configs/formal_runtime.json')])
print(json.dumps({'complete':True,'expected_failure_reproduced':True,'unchanged_actual_source_checks_pass':True,'system_installation':False,'commands':receipts}),flush=True)
