"""Preserve completed source integration without rerunning training or statistics."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

here = Path(__file__).resolve().parent
root = here.parents[2]
work = Path('/tmp/dense-v3-numerical-integration.DUnt6osB')
def identity(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as out, source.open('rb') as input_file:
        shutil.copyfileobj(input_file, out)
    assert identity(source) == identity(destination)

assembly = json.loads((here / 'primary-source-assembly.json').read_bytes())
for relative, row in assembly['files'].items():
    assert identity(root / relative) == row['identity']
for name in ('config.py', 'optimizers.py', 'short_branch.py', 'train.py',
             'dense_numerical_contract.py', 'dense_run_contract.py'):
    copy(root / 'src/embed_optim' / name, here / 'after/src/embed_optim' / name)
for name in ('test_dense_numerical_contract.py', 'test_dense_run_contract.py', 'test_current_primary_source.py'):
    copy(root / 'tests' / name, here / 'after/tests' / name)
for name in ('configs/dense_correctness_candidate.yaml', 'docs/current-training-source.md',
             'AGENTS.md', 'PROJECT_STATUS.md', 'CURRENT_EXPERIMENT.md', 'README.md', 'pyproject.toml'):
    copy(root / name, here / 'after' / name)
copy(work / 'current-primary.xml', here / 'actual-final/current-primary.xml')
for name in ('dist', 'dist-final'):
    for path in sorted((work / name).iterdir()):
        copy(path, here / name / path.name)
wheel = next((work / 'dist-final').glob('*.whl'))
with zipfile.ZipFile(wheel) as archive:
    for relative, row in assembly['files'].items():
        if relative.startswith('src/embed_optim/'):
            raw = archive.read(relative.removeprefix('src/'))
            assert {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == row['identity']
scanner = root / 'src/embed_optim/distribution_audit.py'
assert identity(scanner) == identity(here.parent / 'dense-v3-portable-handoff-v1/original/distribution_audit.py')
old_readme = (here / 'before/README.md').read_text()
new_readme = (root / 'README.md').read_text()
marker = r'<!-- FINAL-CONCLUSION:BEGIN -->.*?<!-- FINAL-CONCLUSION:END -->'
assert re.search(marker, old_readme, re.S).group() == re.search(marker, new_readme, re.S).group()
links = []
for path in [*(root / name for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md')),
             root / 'docs/current-training-source.md']:
    for link in re.findall(r'\]\(([^)]+)\)', path.read_text()):
        if '://' not in link and not link.startswith('#'):
            assert (path.parent / link.split('#')[0]).exists(), (path, link)
            links.append([str(path.relative_to(root)), link])
from embed_optim.current_paper import read_snapshot
read_snapshot(root / 'paper/current')
checks = {'original_scanner_unchanged': identity(scanner), 'current_paper_snapshot_unchanged': True,
    'historical_readme_release_marker_unchanged': True, 'local_links': links,
    'all_56_primary_bindings_unchanged': True, 'all_33_primary_modules_equal_in_wheel': True,
    'full_source_release': False, 'factorial_compatibility_transition_complete': False}
(here / 'checks.json').write_text(json.dumps(checks, indent=2))
rows = {path.relative_to(here).as_posix(): identity(path)
        for path in sorted(here.rglob('*')) if path.is_file()}
with (here / 'manifest.json').open('x') as out:
    json.dump({'files': rows}, out, indent=2, sort_keys=True)
print(json.dumps({'files': len(rows), 'bytes': sum(row['bytes'] for row in rows.values()),
                  'manifest': identity(here / 'manifest.json')}))
