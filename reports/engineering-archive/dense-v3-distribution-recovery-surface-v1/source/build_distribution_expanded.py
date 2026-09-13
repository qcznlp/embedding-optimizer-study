"""Build the expanded current-input distribution; preserve the original baseline."""
from pathlib import Path
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime, timezone

HERE = Path(__file__).parent
ROOT = Path('/root/embedding-optimizer-story-refactor')
STAGE = HERE / 'distribution-expanded'
OUTPUT = HERE / 'built-expanded'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise ValueError('CPU-only packaging requires CUDA hidden')
if STAGE.exists() or OUTPUT.exists():
    raise ValueError('Preserve any existing packaging attempt')
spec = tomllib.loads((ROOT / 'pyproject.toml').read_text())
names = {'pyproject.toml', 'MANIFEST.in', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md'}
data_files = {}
for destination, patterns in spec['tool']['setuptools']['data-files'].items():
    for pattern in patterns:
        matches = [p for p in ROOT.glob(pattern) if p.is_file()] if glob.has_magic(pattern) else [ROOT / pattern]
        for path in matches:
            if not path.is_file() or path.is_symlink():
                raise ValueError('Missing or symlinked declared data input')
            relative = path.relative_to(ROOT).as_posix()
            names.add(relative)
            data_files[relative] = destination + '/' + path.name
for directory in ('src', 'tests', 'scripts'):
    for path in (ROOT / directory).rglob('*'):
        if path.name == 'gpu.py':
            raise ValueError('Protected filename is outside this packaging task')
        if '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        if path.is_symlink():
            raise ValueError('Unexpected source symlink')
        if path.is_file():
            names.add(path.relative_to(ROOT).as_posix())
inputs = {}
for name in sorted(names):
    path = ROOT / name
    if path.name == 'gpu.py':
        raise ValueError('Protected filename excluded')
    inputs[name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
STAGE.mkdir()
for name, binding in inputs.items():
    dst = STAGE / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / name, dst)
    if sha(dst) != binding['sha256'] or sha(ROOT / name) != binding['sha256']:
        raise ValueError('Original input or staged copy changed')
v3 = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / 'configs').glob('*v3*') if p.is_file())
write(HERE / 'distribution-expanded-inputs.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
      'scope': 'expanded_current_input_distribution_not_final_scientific_release',
      'files': inputs, 'declared_data_files': data_files,
      'existing_v3_config_candidates_not_declared': [name for name in v3 if name not in data_files],
      'installed_environment_modified': False, 'original_tree_modified': False})
command = [sys.executable, '-B', '-m', 'build', '--no-isolation', '--wheel', '--sdist', '--outdir', str(OUTPUT)]
result = subprocess.run(command, cwd=STAGE, capture_output=True, text=True, check=False)
with (HERE / 'distribution-expanded-build.log').open('x') as stream:
    stream.write(result.stdout)
    stream.write(result.stderr)
for name, binding in inputs.items():
    if sha(ROOT / name) != binding['sha256']:
        raise ValueError('Original tree changed during build')
artifacts = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in OUTPUT.glob('*') if p.is_file()} if OUTPUT.exists() else {}
write(HERE / 'distribution-expanded-build.json', {'completed_at_utc': datetime.now(timezone.utc).isoformat(),
      'command': command, 'exit_code': result.returncode, 'source_sha256': sha(Path(__file__)),
      'artifact_files': artifacts, 'input_files': len(inputs), 'declared_data_files': len(data_files),
      'original_tree_unchanged': True, 'final_release': False, 'scientific_completion': False})
print(json.dumps({'build_exit_code': result.returncode, 'artifact_files': artifacts,
                  'input_files': len(inputs), 'declared_data_files': len(data_files),
                  'final_release': False, 'scientific_completion': False}))
sys.exit(result.returncode)
