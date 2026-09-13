"""Preserve reader evidence once, without rerunning completed scientific work."""
import hashlib
import json
from pathlib import Path
import re
import shutil

here = Path(__file__).resolve().parent
root = here.parents[2]
work = Path('/tmp/dense-v3-portable-factorial.KUvkU0gR')

def identity(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as out, source.open('rb') as original:
        shutil.copyfileobj(original, out)
    assert identity(source) == identity(destination)

for name in ('src/embed_optim/saved_factorial_checkpoint.py',
             'tests/test_saved_factorial_checkpoint.py', 'docs/saved-factorial-checkpoints.md',
             'AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md', 'pyproject.toml'):
    copy(root / name, here / 'after' / name)
for name in ('unit.xml', 'unit-final.xml'):
    copy(work / name, here / 'earlier-units' / name)
for name in ('dist', 'dist-corrected', 'dist-final'):
    for path in sorted((work / name).iterdir()):
        copy(path, here / name / path.name)
for name in ('factorial_v3_run_contract.py', 'factorial_v3_checkpoint.py'):
    assert identity(root / 'src/embed_optim' / name) == identity(here / 'before' / name)
assembly = json.loads((here.parent / 'dense-v3-numerical-source-integration-v1/primary-source-assembly.json').read_bytes())
for relative, row in assembly['files'].items():
    assert identity(root / relative) == row['identity']
scanner = root / 'src/embed_optim/distribution_audit.py'
assert identity(scanner) == identity(here.parent / 'dense-v3-portable-handoff-v1/original/distribution_audit.py')
marker = r'<!-- FINAL-CONCLUSION:BEGIN -->.*?<!-- FINAL-CONCLUSION:END -->'
assert re.search(marker, (here / 'before/README.md').read_text(), re.S).group() == re.search(
    marker, (root / 'README.md').read_text(), re.S).group()
links = []
for path in [*(root / name for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md')),
             root / 'docs/saved-factorial-checkpoints.md', here / 'README.md']:
    for link in re.findall(r'\]\(([^)]+)\)', path.read_text()):
        if '://' not in link and not link.startswith('#'):
            assert (path.parent / link.split('#')[0]).exists(), (path, link)
            links.append([str(path.relative_to(root)), link])
from embed_optim.current_paper import read_snapshot
read_snapshot(root / 'paper/current')
(here / 'checks.json').write_text(json.dumps({
    'original_scanner_unchanged': identity(scanner), 'original_validators_unchanged': True,
    'all_56_primary_bindings_unchanged': True, 'current_paper_snapshot_unchanged': True,
    'historical_readme_release_marker_unchanged': True, 'local_links': links,
    'expanded_unit_original_tool_exit': None,
    'fresh_run_source_admission': False, 'gpu_resume_equivalence': False,
    'scientific_completion_changed': False, 'source_release': False}, indent=2))
rows = {path.relative_to(here).as_posix(): identity(path)
        for path in sorted(here.rglob('*')) if path.is_file()}
with (here / 'manifest.json').open('x') as out:
    json.dump({'files': rows}, out, indent=2, sort_keys=True)
print(json.dumps({'files': len(rows), 'bytes': sum(row['bytes'] for row in rows.values()),
                  'manifest': identity(here / 'manifest.json')}))
