"""Preserve this source integration and actual package/document evidence."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WORK = Path('/tmp/dense-v3-current-paper-entry.e2ViDMaN')

def identity(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def copy(source, target):
    assert source.is_file() and not source.is_symlink() and not target.exists()
    before = identity(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert identity(source) == identity(target) == before

for name in ('prepare.py', 'package_check.py', 'packaged_child.py', 'package-check.json',
             'current_paper.before-format.py', 'test_current_paper.before-format.py',
             'first-tests.xml', 'integration-tests-first.xml', 'integration-tests-final.xml',
             'reviewed-pdf-text.txt', 'build-final.stdout', 'build-final.stderr',
             'build-final.exit.json', 'audit-final.stdout', 'audit-final.stderr',
             'audit-final.exit.json', 'packaged-child.stdout', 'packaged-child.stderr',
             'packaged-child.exit.json'):
    copy(WORK / name, HERE / 'actual' / name)
for folder in ('dist', 'dist-final', 'actual', 'packaged-document', 'before'):
    for path in sorted((WORK / folder).rglob('*')):
        if path.is_file():
            copy(path, HERE / 'actual' / folder / path.relative_to(WORK / folder))
for name in ('src/embed_optim/current_paper.py', 'src/embed_optim/complete_paper_document.py',
             'tests/test_current_paper.py', 'pyproject.toml', 'paper/Makefile', 'paper/README.md',
             'docs/continuation-outcomes-restoration.md', 'AGENTS.md', 'CURRENT_EXPERIMENT.md',
             'PROJECT_STATUS.md', 'README.md'):
    copy(ROOT / name, HERE / 'source-current' / name)
for path in sorted((ROOT / 'paper/current').rglob('*')):
    if path.is_file():
        copy(path, HERE / 'source-current/paper/current' / path.relative_to(ROOT / 'paper/current'))
for name in ('pyproject.toml', 'paper/Makefile', 'paper/README.md'):
    before = (WORK / 'before' / name).read_text()
    after = (ROOT / name).read_text()
    path = HERE / 'task-diffs' / (name + '.diff')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.writelines(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
            fromfile='before/' + name, tofile='after/' + name))
before = (WORK / 'before/paper/Makefile').read_text()
after = (ROOT / 'paper/Makefile').read_text()
for target in ('all', 'release'):
    begin = '\n' + target + ':'
    assert before.split(begin, 1)[1].split('\n\n', 1)[0] == after.split(begin, 1)[1].split('\n\n', 1)[0]
assert identity(ROOT / 'paper/main.tex')['sha256'] == '45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e'
assert identity(ROOT / 'src/embed_optim/complete_paper_document.py')['sha256'] == 'dbed8e2d4d418ce89d5d56d75da4e433e9b9b0888194a9c44ca053dd03d2684b'
with (HERE / 'actual/terminal-tools.json').open('x') as stream:
    json.dump({'69537': ['65b595', 0], '4778': ['50428f', 0], '12056': ['5593d7', 0],
               '49464': ['583863', 1], '30859': ['66d481', 0], '81331': ['59d2ab', 0],
               'first-full-audit': ['f06da2', 1], 'initial-format-check': ['2c8f5b', 1],
               'format-fix': ['41ef15', 0], 'final-lint-format-diff': ['77d86f', 0]}, stream, indent=2)
rows = {p.relative_to(HERE).as_posix(): identity(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
with (HERE / 'archive-manifest.json').open('x') as stream:
    json.dump({'files': rows, 'files_excluding_manifest': len(rows),
               'bytes_excluding_manifest': sum(r['bytes'] for r in rows.values()),
               'scope': 'current_document_source_integration_not_full_scientific_release'}, stream, indent=2, sort_keys=True)
print(json.dumps({'files': len(rows), 'bytes': sum(r['bytes'] for r in rows.values()),
                  'manifest_sha256': identity(HERE / 'archive-manifest.json')['sha256']}))
