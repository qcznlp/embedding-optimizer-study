"""Capture original build/audit and new genuine portable reader outcomes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

here = Path(__file__).resolve().parent
root = here.parents[2]
work = Path('/tmp/dense-v3-portable-factorial.KUvkU0gR')
actual = here / 'actual-corrected'
actual.mkdir()
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
       'PYTHONPATH': str(root / 'src') + os.pathsep + str(root),
       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
def run(label, command):
    with (actual / (label + '.stdout')).open('xb') as out, (actual / (label + '.stderr')).open('xb') as err:
        child = subprocess.run(command, cwd=work, env=env, stdout=out, stderr=err)
    (actual / (label + '.exit.json')).write_text(json.dumps({'command': command, 'actual_exit_code': child.returncode}, indent=2))
    print(json.dumps({'stage': label, 'actual_exit_code': child.returncode}), flush=True)
    assert child.returncode == 0
run('build', ['/usr/bin/python', '-m', 'build', '--no-isolation', '--outdir', str(work / 'dist-corrected'), str(root)])
run('audit', ['/usr/bin/python', '-m', 'embed_optim.distribution_audit', '--repo-root', str(root), '--dist-dir', str(work / 'dist-corrected')])
audit = json.loads((actual / 'audit.stdout').read_bytes())
assert audit['complete'] is True and audit['problems'] == []
wheel = next((work / 'dist-corrected').glob('*.whl'))
extracted = work / 'wheel-corrected'
extracted.mkdir()
with zipfile.ZipFile(wheel) as archive:
    archive.extractall(extracted)
runtime = extracted / 'embedding_optimizer_study-0.1.0.data/data/share/embedding-optimizer-study/configs/formal_runtime.json'
for case in ('a', 'b'):
    output = work / ('read-corrected-' + case + '.json')
    run('case-' + case, ['/usr/bin/python', '-B', '-I', str(here / 'actual_read.py'),
        str(extracted), str(here / 'original-download.json'), str(runtime), str(output), case])
    shutil.copyfile(output, actual / output.name)
