"""Prepare a source-authenticated CPU wheel build without changing the study runtime."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import urllib.request

root = Path('/root/embedding-optimizer-story-refactor')
work = Path(__file__).resolve().parent
source_sha = 'f03485c9a49a4d68d0733acdcad80ab0e72afa025a777fdc2966ceccf9d51765'
base_sha = '0c88c4b3c24b1b1b3b92216c41d23bd0dc0510acd98e734e9ab7e75244b18fa9'
base = (root / 'requirements-formal.lock').read_bytes()
assert hashlib.sha256(base).hexdigest() == base_sha
assert source_sha in (root / 'requirements-formal-flash.txt').read_text()
env = {k: os.environ[k] for k in ('PATH', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH') if k in os.environ}
env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(root / 'src'), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
def execute(name, args):
    with (work / (name + '.log')).open('xb') as stream:
        result = subprocess.run(args, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT)
    with (work / (name + '-exit.json')).open('x') as stream:
        json.dump({'argv': args, 'returncode': result.returncode}, stream, indent=2)
    result.check_returncode()
execute('system-runtime-before', ['/usr/bin/python', '-B', '-m', 'embed_optim.runtime', '--spec', str(root / 'configs/formal_runtime.json')])
with urllib.request.urlopen('https://pypi.org/pypi/flash-attn/2.7.4.post1/json', timeout=60) as response:
    release = json.load(response)
sdists = [x for x in release['urls'] if x['packagetype'] == 'sdist']
assert len(sdists) == 1 and sdists[0]['digests']['sha256'] == source_sha
source_url = sdists[0]['url']
assert source_url.startswith('https://files.pythonhosted.org/')
with urllib.request.urlopen(source_url, timeout=60) as response:
    raw = response.read()
assert len(raw) == sdists[0]['size'] and hashlib.sha256(raw).hexdigest() == source_sha
archive_path = work / sdists[0]['filename']
with archive_path.open('xb') as stream:
    stream.write(raw)
source = work / 'source'
source.mkdir()
members = {}
with tarfile.open(archive_path, 'r:gz') as archive:
    for member in archive.getmembers():
        assert member.name.startswith('flash_attn-2.7.4.post1/') or member.name == 'flash_attn-2.7.4.post1'
        assert not member.issym() and not member.islnk()
        if member.isfile():
            payload = archive.extractfile(member).read()
            members[member.name] = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
    archive.extractall(source, filter='data')
for name, expected in members.items():
    payload = (source / name).read_bytes()
    assert len(payload) == expected['bytes'] and hashlib.sha256(payload).hexdigest() == expected['sha256']
selected = {'ninja': '1.11.1.4', 'packaging': '26.3', 'psutil': '7.2.2', 'setuptools': '84.0.0'}
blocks = re.split(r'(?=^[a-zA-Z0-9_.-]+==)', base.decode(), flags=re.MULTILINE)
records = []
for block in blocks:
    match = re.match(r'([a-zA-Z0-9_.-]+)==([^\s\\]+)', block)
    if match and match[1] in selected:
        assert match[2] == selected[match[1]]
        records.append(block)
assert len(records) == len(selected)
lock = work / 'build-tools-from-formal.lock'
with lock.open('x') as stream:
    stream.write('# Exact selected records from the original formal lock.\n' + '\n'.join(records))
venv = work / 'build-venv'
execute('create-build-venv', ['uv', 'venv', '--no-config', '--system-site-packages', '--python', '/usr/bin/python', str(venv)])
execute('install-isolated-build-tools', ['uv', 'pip', 'install', '--no-config', '--python', str(venv / 'bin/python'), '--no-deps', '--require-hashes', '-r', str(lock)])
execute('isolated-runtime', [str(venv / 'bin/python'), '-B', '-m', 'embed_optim.runtime', '--spec', str(root / 'configs/formal_runtime.json')])
execute('isolated-build-tool-versions', [str(venv / 'bin/python'), '-B', '-c', 'import importlib.metadata as m; import json; print(json.dumps({p:m.version(p) for p in ["ninja", "packaging", "psutil", "setuptools", "wheel", "torch", "flash-attn"]}, sort_keys=True))'])
execute('compiler', ['/usr/local/cuda-12.9/bin/nvcc', '--version'])
execute('host-compiler', ['g++', '--version'])
receipt = {'source_url': source_url, 'source_sha256': source_sha, 'formal_base_sha256': base_sha, 'source_files': members, 'source_directory': str(source / 'flash_attn-2.7.4.post1'), 'build_python': str(venv / 'bin/python'), 'compiled': False, 'gpu_execution': False, 'system_runtime_installation': False}
with (work / 'prepared.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
print(json.dumps({k: v for k, v in receipt.items() if k != 'source_files'}), flush=True)
