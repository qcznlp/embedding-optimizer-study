"""Prepare an allowlisted authentic runtime binary distribution, never negative controls."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import zipfile

work = Path(__file__).resolve().parent
root = Path('/root/embedding-optimizer-story-refactor')
sys.path.insert(0, str(root / 'src'))
from embed_optim.distribution_audit import _credential_findings

verified = json.loads((work / 'verified.json').read_text())
assert verified['complete'] is True
out = work / 'publication'
out.mkdir(exist_ok=False)
names = [
    'prepare.py', 'build.py', 'verify.py', 'prepared.json', 'build-command.json',
    'build-result.json', 'verified.json', 'build-tools-from-formal.lock',
    'compiler.log', 'compiler-exit.json', 'host-compiler.log', 'host-compiler-exit.json',
    'system-runtime-before.log', 'system-runtime-before-exit.json',
    'isolated-runtime.log', 'isolated-runtime-exit.json',
    'isolated-build-tool-versions.log', 'isolated-build-tool-versions-exit.json',
    'create-build-venv.log', 'create-build-venv-exit.json',
    'install-isolated-build-tools.log', 'install-isolated-build-tools-exit.json',
    'build.log', 'fresh-wheel-import.log', 'fresh-wheel-import-exit.json',
    'system-runtime-after.log', 'flash_attn-2.7.4.post1.tar.gz',
]
for name in names:
    shutil.copyfile(work / name, out / name)
wheel_name = verified['wheel']['name']
shutil.copyfile(work / 'dist' / wheel_name, out / wheel_name)
source = work / 'source/flash_attn-2.7.4.post1'
shutil.copyfile(source / 'LICENSE', out / 'FLASH_ATTENTION_LICENSE')
shutil.copyfile(source / 'csrc/cutlass/include/cutlass/cutlass.h', out / 'CUTLASS_LICENSE_HEADER.h')
shutil.copyfile('/usr/local/lib/python3.12/dist-packages/torch-2.9.1+cu129.dist-info/licenses/LICENSE', out / 'TORCH_LICENSE')
shutil.copyfile(work / 'PUBLIC_README.md', out / 'README.md')
for name in ['requirements-formal.lock', 'requirements-formal-flash.txt', 'configs/formal_runtime.json']:
    shutil.copyfile(root / name, out / Path(name).name)
findings = []
for path in out.iterdir():
    if path.suffix == '.whl':
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                findings.extend(_credential_findings('wheel', member, archive.read(member)))
    elif path.name.endswith('.tar.gz'):
        with tarfile.open(path) as archive:
            for member in archive:
                if member.isfile():
                    findings.extend(_credential_findings('source', member.name, archive.extractfile(member).read()))
    else:
        findings.extend(_credential_findings('receipt', path.name, path.read_bytes()))
assert not findings, '\n'.join(findings)
manifest = {'schema_version': 1, 'scope': 'authentic-flash-attention-cpu-ci-runtime',
            'source_commit': '1fe41cf77ec79228ec12d12990c327e9100b188b',
            'gpu_validation': False, 'second_host_validation': False, 'credential_scan_findings': 0,
            'files': {}}
for path in sorted(out.iterdir()):
    manifest['files'][path.name] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
with (out / 'manifest.json').open('x') as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
print(json.dumps({'files': len(manifest['files']), 'manifest_sha256': hashlib.sha256((out / 'manifest.json').read_bytes()).hexdigest(), 'wheel': verified['wheel']}))
