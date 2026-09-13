"""Freeze this task's exact source, test, documentation and distribution evidence."""
import hashlib
import json
from pathlib import Path
import re
import shutil

here = Path(__file__).resolve().parent
root = here.parents[2]
work = Path('/tmp/dense-v3-portable-input-roles.eOsTxin7')
source_names = (
    'src/embed_optim/factorial_v3_inputs.py', 'tests/test_factorial_v3_input_roles.py',
    'tests/test_dense_evaluation_pause.py', 'tests/test_factorial_v3_calibration.py',
    'tests/test_factorial_v3_run_contract.py', 'tests/test_primary_v3_outcome_reconstruction.py',
    'AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md', 'pyproject.toml',
)
def identity(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as out, source.open('rb') as input_file:
        shutil.copyfileobj(input_file, out)
    assert identity(source) == identity(destination)

for name in source_names:
    copy(root / name, here / 'after' / name)
for name in ('roles-first.xml', 'cold-cli.xml'):
    copy(work / name, here / 'actual' / name)
for namespace in ('dist', 'dist-clean', 'dist-final', 'dist-handoff'):
    for path in sorted((work / namespace).iterdir()):
        copy(path, here / namespace / path.name)
scanner = root / 'src/embed_optim/distribution_audit.py'
previous = root / 'reports/engineering-archive/dense-v3-portable-handoff-v1/original/distribution_audit.py'
assert identity(scanner) == identity(previous)
assert identity(root / 'pyproject.toml') == identity(here.parent / 'dense-v3-portable-handoff-v1/before/pyproject.toml')
links = []
for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md'):
    for link in re.findall(r'\]\(([^)]+)\)', (root / name).read_text()):
        if '://' not in link and not link.startswith('#'):
            assert (root / link.split('#')[0]).exists(), (name, link)
            links.append([name, link])
from embed_optim.current_paper import read_snapshot
read_snapshot(root / 'paper/current')
checks = {'original_scanner_unchanged': identity(scanner), 'packaging_declarations_unchanged': True,
          'current_paper_snapshot_unchanged': True, 'current_local_links': links,
          'source_release': False, 'scientific_results_changed': False,
          'cold_cli_tool': {'session': 23255, 'terminal': 'cb01fc', 'actual_exit': 0},
          'component_driver_tool': {'session': 71969, 'terminal': '2a8f07', 'actual_exit': 0}}
(here / 'checks.json').write_text(json.dumps(checks, indent=2))
rows = {path.relative_to(here).as_posix(): identity(path)
        for path in sorted(here.rglob('*')) if path.is_file()}
with (here / 'manifest.json').open('x') as out:
    json.dump({'files': rows}, out, indent=2, sort_keys=True)
print(json.dumps({'files': len(rows), 'bytes': sum(row['bytes'] for row in rows.values()),
                  'manifest': identity(here / 'manifest.json')}))
