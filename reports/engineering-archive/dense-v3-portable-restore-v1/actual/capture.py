"""Capture exact local restoration delivery artifacts, without modifying sources."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path('/root/embedding-optimizer-story-refactor')
work = Path(__file__).resolve().parent
report = root / 'reports/engineering-archive/dense-v3-portable-restore-v1'
assert not (report / 'manifest.json').exists()

def copy(source, target):
    assert source.is_file() and not source.is_symlink()
    assert not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert source.read_bytes() == target.read_bytes()

for source in sorted((root / 'src/embed_optim_restore').rglob('*')):
    if source.is_file() and source.suffix in {'.py', '.json'}:
        copy(source, report / 'source' / source.relative_to(root))
for name in ('entry', 'gate', 'paired', 'trace', 'step_devices'):
    source = root / f'tests/test_restore_{name}.py'
    copy(source, report / 'tests' / source.name)
for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md',
             'pyproject.toml', 'docs/training-restoration.md'):
    copy(root / name, report / 'after' / name)
for name in ('cold.log', 'cold.py', 'cold-result.json', 'wheel-check.json',
             'wheel_check.py', 'wheel-tests.xml', 'tests-first.xml',
             'tests-expanded.xml', 'tests-final.xml', 'capture.py'):
    copy(work / name, report / 'actual' / name)
for folder in ('format-first', 'dist'):
    for source in sorted((work / folder).rglob('*')):
        if source.is_file():
            copy(source, report / 'actual' / source.relative_to(work))
files = {}
for path in sorted(report.rglob('*')):
    if path.is_file():
        raw = path.read_bytes()
        files[path.relative_to(report).as_posix()] = {
            'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
manifest = {'schema': 'dense-v3-portable-restore-archive-v1', 'files': files,
            'new_gpu_execution': False, 'source_publication_complete': False}
with (report / 'manifest.json').open('x') as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps({'files': len(files), 'bytes': sum(x['bytes'] for x in files.values()),
                  'manifest_sha256': hashlib.sha256((report / 'manifest.json').read_bytes()).hexdigest()}))
