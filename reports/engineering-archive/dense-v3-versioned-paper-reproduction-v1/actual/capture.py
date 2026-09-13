"""Preserve actual composed-reproduction evidence, excluding duplicate primary tables/caches."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path('/root/embedding-optimizer-story-refactor')
work = Path(__file__).resolve().parent
report = root / 'reports/engineering-archive/dense-v3-versioned-paper-reproduction-v1'
assert not (report / 'manifest.json').exists()

def copy(source, target):
    assert source.is_file() and not source.is_symlink() and not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert source.read_bytes() == target.read_bytes()

for name in ('src/embed_optim/paper_reproduction.py', 'tests/test_paper_reproduction.py',
             'docs/versioned-paper-reproduction.md', 'pyproject.toml'):
    copy(root / name, report / 'source' / name)
for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md', 'docs/completion-gates.md'):
    copy(root / name, report / 'after' / name)
for name in ('launch.py', 'launch.log', 'RUNNING.md', 'tests-first.xml', 'tests-integration.xml',
             'build.log', 'distribution-audit.json', 'actual-wheel.json', 'capture.py',
             'actual/complete.json', 'actual/numerical-exit.json', 'actual/numerical.log',
             'actual/numerical/complete.json', 'actual/numerical/io-boundary.json',
             'actual/numerical/primary/io-boundary.json',
             'actual/numerical/primary/reconstructed/complete.json',
             'actual/numerical/primary-exited.json', 'actual/numerical/primary-replay.log',
             'actual/numerical/document/document.json'):
    copy(work / name, report / 'actual' / name)
for folder in ('dist', 'actual/reviewed', 'actual/numerical/factorial'):
    for path in sorted((work / folder).rglob('*')):
        if path.is_file():
            copy(path, report / 'actual' / path.relative_to(work))
files = {}
for path in sorted(report.rglob('*')):
    if path.is_file():
        raw = path.read_bytes()
        files[path.relative_to(report).as_posix()] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
value = {'schema': 'dense-v3-versioned-paper-reproduction-archive-v1', 'files': files,
         'actual_owned_tool_exit': 0, 'new_gpu_execution': False, 'source_publication_complete': False}
with (report / 'manifest.json').open('x') as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps({'files': len(files), 'bytes': sum(x['bytes'] for x in files.values()),
                  'manifest_sha256': hashlib.sha256((report / 'manifest.json').read_bytes()).hexdigest()}))
