"""Seal bounded release-transition evidence after actual complete test-role success."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path('/root/embedding-optimizer-story-refactor')
work = Path(__file__).resolve().parent
result = json.loads((work / 'complete-source-test-matrix.json').read_text())
assert result['complete'] is True and result['cases'] == 3800
assert json.loads((work / 'distribution-final.json').read_text())['complete'] is True
assert json.loads((work / 'release-paper/complete.json').read_text())
archive = root / 'reports/engineering-archive/dense-v3-release-transition-v1'
assert not archive.exists()
selected = set()
for pattern in ('*.py', '*.json', '*.xml', '*.log', '*.md', '*.Makefile'):
    selected.update(path for path in work.glob(pattern) if path.is_file())
selected.update(path for path in (work / 'before').rglob('*') if path.is_file())
selected.update(path for path in (work / 'dist-final').iterdir() if path.is_file())
for directory in ('full-source-tests-v1', 'current-role-final'):
    for pattern in ('*.json', '*.xml', '*.log'):
        selected.update((work / directory).glob(pattern))
for label in (
    'default-paper/current-paper.json',
    'release-paper/complete.json',
    'release-paper/numerical.log',
    'release-paper/numerical/complete.json',
    'release-paper/numerical/numerical-exit.json',
    'release-paper/reviewed/current-paper.json',
    'release-paper/reviewed/paper/build/main.pdf',
):
    path = work / label
    if path.is_file():
        selected.add(path)
archive.mkdir(parents=True, exist_ok=False)
files = {}
for source in sorted(selected):
    assert not source.is_symlink() and source.name != 'gpu.py'
    label = 'actual/' + str(source.relative_to(work))
    target = archive / label
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert target.read_bytes() == source.read_bytes()
    files[label] = {'bytes': target.stat().st_size, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
for label in ('scripts/test_source_roles.py', 'configs/source_test_roles.json', 'tests/test_source_roles.py', 'tests/test_successor_parent_publication.py', 'paper/Makefile', 'paper/legacy.Makefile'):
    source = root / label
    target_label = 'source/' + label
    target = archive / target_label
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    files[target_label] = {'bytes': target.stat().st_size, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
readme = '''# Reviewed default/release and complete source-version tests

Scientific computation is unchanged. Actual default Make and complete numerical-to-reviewed-paper
Make release both pass. The original Makefile is byte-preserved as `paper/legacy.Makefile`.
README's conclusion is generated from authenticated numerical outputs, not a hand-edited verdict.

All **3,800 tests** pass in their explicit source versions: current 2,873; original analysis
733; original factorial 194. No failed, errored or skipped case is accepted. The first complete
matrix's current-role failure remains a failure; its five old-Make routing errors were corrected
and all 2,873 current-role cases rerun. The independent historical processes were not restarted.
The joined receipt verifies all executed test bytes against the final checkout and preserves
all source inventories. This is not a passing bare mixed-source pytest run or fresh-worker admission.

The earlier mixed-source baseline (3,332 passes, 239 failures, 210 errors), first role-assembly
collection failure, intermediate analysis failures and frontend failures are retained.
Historical source identities and every scientific assertion/tolerance remain unchanged.
The 47-command regression uses the original four-slot path projection before exact vector comparison.
Synthetic layout/old guard tests now explicitly use their byte-identical legacy Make rules.

The complete distribution audit, portable evidence audit, style checks and isolated CFF validation
pass. The initial credential scan is preserved with its ten findings; separate exact recomputation
proves each is a wheel RECORD SHA-256 substring, not a credential. No scan pattern was weakened.
See actual/complete-source-test-matrix.json, distribution-final.json, release-paper/complete.json,
and the full logs/XML. Tool exits and scope limits are in actual/RUNNING.md.

This archive does not claim remote source publication, general new-host training admission,
a new GPU experiment or removal of previously denied HF paths. Tiny synthetic engineering
checkpoints remain test evidence, not the study's scientific checkpoints.
'''
(archive / 'README.md').write_text(readme)
files['README.md'] = {'bytes': len(readme.encode()), 'sha256': hashlib.sha256(readme.encode()).hexdigest()}
manifest = {'schema_version': 1, 'status': 'sealed-release-transition-evidence', 'scientific_results_changed': False,
            'remote_publication': False, 'files': files}
(archive / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print(json.dumps({'archive': str(archive), 'files': len(files), 'bytes': sum(item['bytes'] for item in files.values()), 'manifest_sha256': hashlib.sha256((archive / 'manifest.json').read_bytes()).hexdigest()}), flush=True)
