"""Verify a genuine built wheel and its fresh CPU import, without system installation."""
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

work = Path(__file__).resolve().parent
root = Path('/root/embedding-optimizer-story-refactor')
prepared = json.loads((work / 'prepared.json').read_text())
built = json.loads((work / 'build-result.json').read_text())
assert built['returncode'] == 0 and len(built['wheels']) == 1
name, expected = next(iter(built['wheels'].items()))
wheel = work / 'dist' / name
assert wheel.stat().st_size == expected['bytes']
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == expected['sha256']

def inspect_wheel(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), 'duplicate wheel member'
        for member in names:
            parts = PurePosixPath(member)
            assert not parts.is_absolute() and '..' not in parts.parts, 'unsafe wheel member'
        record_names = [p for p in names if p.endswith('.dist-info/RECORD')]
        assert len(record_names) == 1, 'missing wheel RECORD'
        record = record_names[0]
        rows = list(csv.reader(io.StringIO(archive.read(record).decode())))
        assert len(rows) == len(names) and {r[0] for r in rows} == set(names), 'RECORD coverage'
        for member, digest, size in rows:
            if member == record:
                assert digest == '' and size == '', 'RECORD self binding'
            else:
                raw = archive.read(member)
                encoded = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')
                assert digest == 'sha256=' + encoded and size == str(len(raw)), 'RECORD digest mismatch'
        extension_names = [p for p in names if p.startswith('flash_attn_2_cuda.') and p.endswith('.so')]
        assert len(extension_names) == 1, 'missing genuine CUDA extension'
        assert archive.read(extension_names[0]).startswith(b'\x7fELF'), 'not an ELF extension'
        python_files = [p for p in names if p.startswith('flash_attn/') and p.endswith('.py')]
        assert python_files, 'missing Python package'
        for member in python_files:
            raw = archive.read(member)
            binding = prepared['source_files']['flash_attn-2.7.4.post1/' + member]
            assert len(raw) == binding['bytes'] and hashlib.sha256(raw).hexdigest() == binding['sha256'], 'upstream Python source mismatch'
        return {'members': len(names), 'python_files': len(python_files), 'extension': extension_names[0]}

inspection = inspect_wheel(wheel)
controls = work / 'negative-controls'
controls.mkdir(exist_ok=False)
with zipfile.ZipFile(wheel) as archive:
    bad = controls / 'changed-python.whl'
    with zipfile.ZipFile(bad, 'w', compression=zipfile.ZIP_DEFLATED) as target:
        for member in archive.namelist():
            raw = archive.read(member)
            if member == 'flash_attn/__init__.py':
                raw += b'\n# synthetic negative control\n'
            target.writestr(member, raw)
assert hashlib.sha256(bad.read_bytes()).hexdigest() != expected['sha256']
try:
    inspect_wheel(bad)
except AssertionError as error:
    assert str(error) == 'RECORD digest mismatch'
else:
    raise AssertionError('corrupted wheel was accepted')

extracted = work / 'wheel-import-root'
extracted.mkdir(exist_ok=False)
with zipfile.ZipFile(wheel) as archive:
    archive.extractall(extracted)
env = {k: os.environ[k] for k in ('PATH', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH') if k in os.environ}
env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(extracted) + os.pathsep + str(root / 'src'), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
snippet = '''import importlib.metadata as m
import json
import platform
from pathlib import Path
import sys
import sysconfig
import torch
import flash_attn
import flash_attn_2_cuda
from embed_optim.runtime import verify_runtime_spec
prefix = Path(sys.argv[1]).resolve()
assert Path(flash_attn.__file__).resolve().is_relative_to(prefix)
assert Path(flash_attn_2_cuda.__file__).resolve().is_relative_to(prefix)
assert Path(m.distribution("flash-attn").locate_file("flash_attn")).resolve() == prefix / "flash_attn"
assert flash_attn.__version__ == "2.7.4.post1"
assert torch.version.cuda == "12.9"
exports = ["fwd", "varlen_fwd", "bwd", "varlen_bwd", "fwd_kvcache"]
assert all(callable(getattr(flash_attn_2_cuda, name)) for name in exports)
print(json.dumps({"runtime": verify_runtime_spec(sys.argv[2]), "imports": {"flash_attn": flash_attn.__file__, "flash_attn_2_cuda": flash_attn_2_cuda.__file__, "torch": torch.__file__}, "native_exports": exports, "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI), "libc": platform.libc_ver(), "machine": platform.machine(), "soabi": sysconfig.get_config_var("SOABI"), "gpu_kernels_executed": False}, sort_keys=True))
'''
argv = ['/usr/bin/python', '-B', '-c', snippet, str(extracted), str(root / 'configs/formal_runtime.json')]
with (work / 'fresh-wheel-import.log').open('xb') as stream:
    imported = subprocess.run(argv, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT)
with (work / 'fresh-wheel-import-exit.json').open('x') as stream:
    json.dump({'returncode': imported.returncode, 'argv': argv}, stream, indent=2)
imported.check_returncode()
env['PYTHONPATH'] = str(root / 'src')
with (work / 'system-runtime-after.log').open('xb') as stream:
    after = subprocess.run(['/usr/bin/python', '-B', '-m', 'embed_optim.runtime', '--spec', str(root / 'configs/formal_runtime.json')], cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT)
after.check_returncode()
assert (work / 'system-runtime-before.log').read_bytes() == (work / 'system-runtime-after.log').read_bytes(), 'system runtime changed'
changed = []
for member, binding in prepared['source_files'].items():
    path = work / 'source' / member
    raw = path.read_bytes()
    if len(raw) != binding['bytes'] or hashlib.sha256(raw).hexdigest() != binding['sha256']:
        assert '.egg-info/' in member or member.endswith('/PKG-INFO'), 'computational source changed'
        changed.append(member)
receipt = {'complete': True, 'wheel': {'name': name, **expected}, 'inspection': inspection, 'fresh_import_returncode': imported.returncode, 'system_runtime_after_returncode': after.returncode, 'system_runtime_before_after_equal': True, 'corrupted_wheel_refused': True, 'generated_metadata_changes': changed, 'computational_source_unchanged': True, 'gpu_execution': False, 'second_host_execution': False, 'remote_publication': False}
with (work / 'verified.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
print(json.dumps(receipt), flush=True)
