"""Build the unmodified official CUDA extension with bounded CPU parallelism."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

work = Path(__file__).resolve().parent
prepared = json.loads((work / 'prepared.json').read_text())
source = Path(prepared['source_directory'])
for name, expected in prepared['source_files'].items():
    raw = (work / 'source' / name).read_bytes()
    assert len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256']
dist = work / 'dist'
dist.mkdir(exist_ok=False)
(work / 'tmp').mkdir(exist_ok=False)
env = {k: os.environ[k] for k in ('PATH', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH') if k in os.environ}
env.update(CUDA_VISIBLE_DEVICES='', CUDA_HOME='/usr/local/cuda-12.9', FLASH_ATTENTION_FORCE_BUILD='TRUE', FLASH_ATTN_CUDA_ARCHS='80;90', MAX_JOBS='32', NVCC_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(work / 'tmp'))
env['PATH'] = str(work / 'build-venv/bin') + os.pathsep + env['PATH']
argv = [prepared['build_python'], '-B', 'setup.py', 'bdist_wheel', '--dist-dir', str(dist)]
started = time.time()
with (work / 'build-command.json').open('x') as stream:
    json.dump({'argv': argv, 'cwd': str(source), 'environment': {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'FLASH_ATTENTION_FORCE_BUILD', 'FLASH_ATTN_CUDA_ARCHS', 'MAX_JOBS', 'NVCC_THREADS', 'TMPDIR')}, 'started_unix': started, 'source_sha256': prepared['source_sha256']}, stream, indent=2, sort_keys=True)
with (work / 'build.log').open('xb') as stream:
    result = subprocess.run(argv, cwd=source, env=env, stdout=stream, stderr=subprocess.STDOUT)
receipt = {'argv': argv, 'returncode': result.returncode, 'elapsed_seconds': time.time() - started, 'genuine_source_compilation': True, 'gpu_execution_requested': False, 'system_installation_requested': False, 'wheels': {}}
for path in dist.glob('*.whl'):
    raw = path.read_bytes()
    receipt['wheels'][path.name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
with (work / 'build-result.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
print(json.dumps(receipt), flush=True)
result.check_returncode()
assert len(receipt['wheels']) == 1
