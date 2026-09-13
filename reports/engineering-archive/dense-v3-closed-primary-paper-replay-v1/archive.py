"""Preserve actual closed numerical replay and separate controls; no producer edits."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
STORY = Path('/root/embedding-optimizer-story-refactor')
REPORT = STORY / 'reports/engineering-archive/dense-v3-closed-primary-paper-replay-v1'


def identity(path):
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Ordinary archive source required')
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': sha}


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


assert identity(HERE / 'closed-primary/manifest.json')['sha256'] == \
    '88861e52218f56b860851489106cd30e35dd79bb038eb9082967e3376db6457d'
assert identity(HERE / 'actual-primary/reconstructed/complete.json')['sha256'] == \
    'cc980ed44ed7a3681a35d8d40f84d8d14bf0c42a17996fae75c0e060a773e41a'
io = read(HERE / 'actual-primary/io-boundary.json')
assert io['failure'] is None and io['producer_reads_refused'] == [] and io['network_refused'] == 0
tests = read(HERE / 'controls-result.json')
assert tests['tests_run'] == 11 and tests['failures'] == tests['errors'] == tests['skips'] == 0
protected = {
    STORY / 'paper/main.tex': '45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e',
    Path('/tmp/dense-v3-factorial-evaluation.IOV7MK93/evaluate.py'): '45577537257e1dff2b7c3a26ee12f7564fbb476daa256e7b0f09300305b68cba',
    Path('/tmp/dense-v3-factorial-summary.BQ08HjeP/summarize.py'): '4be265ce12cc649a595afaec9e5ef8b68a50eaa347ccb583d4d32ba3e100c5a1',
    Path('/tmp/dense-v3-document-integration.Xz1qvTME/author.py'): 'cac9486b6e92acedb190bad67e34a5fd647d1c10a691071483fbf001406b1f9d',
    Path('/tmp/dense-v3-resume-verification.wcsmQnDm/resume.py'): '1746b8a382bb45fefc2ab17beaf7706984fe91802c528bec48c88a2daa038bb2',
}
for path, sha in protected.items():
    assert identity(path)['sha256'] == sha
REPORT.mkdir(exist_ok=False)
files, links = {}, {}
for original in sorted(HERE.rglob('*')):
    relative = original.relative_to(HERE)
    if '__pycache__' in relative.parts or relative.parts[:2] == ('actual-primary', 'runtime'):
        continue
    if original.is_symlink():
        # Preserve the intentionally invalid synthetic symlink as metadata only.
        assert relative.parts[0] == 'controls'
        links[relative.as_posix()] = {'target': str(original.readlink()), 'synthetic_control': True}
        continue
    if not original.is_file():
        continue
    target = REPORT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    before = identity(original)
    shutil.copyfile(original, target)
    assert identity(target) == identity(original) == before
    files[relative.as_posix()] = {**before, 'origin': str(original)}
for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md'):
    original, target = STORY / name, REPORT / 'before' / name
    target.parent.mkdir(exist_ok=True)
    before = identity(original)
    shutil.copyfile(original, target)
    assert identity(target) == identity(original) == before
    files['before/' + name] = {**before, 'origin': str(original)}
write(REPORT / 'synthetic-symlink-records.json', links)
files['synthetic-symlink-records.json'] = identity(REPORT / 'synthetic-symlink-records.json')
patterns = [rb'wandb_v1_[A-Za-z0-9_-]{40,}', rb'\bhf_[A-Za-z0-9]{30,}',
            rb'\bgh[pousr]_[A-Za-z0-9]{30,}', rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
findings = []
text_files = 0
for name in files:
    path = REPORT / name
    if path.suffix not in ('.py', '.json', '.jsonl', '.md', '.txt', '.tex', '.csv', '.bib', '.sty', '.bst'):
        continue
    text_files += 1
    raw = path.read_bytes()
    if any(re.search(pattern, raw) for pattern in patterns):
        findings.append(name)
assert not findings, 'Credential-shaped content: preserve locally and do not release'
result = {'scope': 'actual-closed-current-primary-numerical-graph-preservation',
    'archived_at_utc': datetime.now(timezone.utc).isoformat(), 'files': files,
    'copied_files': len(files), 'copied_bytes': sum(row['bytes'] for row in files.values()),
    'copy_hashes_verified': True, 'credential_screened_text_files': text_files,
    'credential_shaped_findings': findings, 'synthetic_symlink_not_followed': len(links),
    'primary_bundle_manifest': identity(REPORT / 'closed-primary/manifest.json'),
    'actual_complete_replay': identity(REPORT / 'actual-primary/reconstructed/complete.json'),
    'actual_io_boundary': identity(REPORT / 'actual-primary/io-boundary.json'),
    'original_source_files_unchanged': {str(path): sha for path, sha in protected.items()},
    'goal_turn_classification': 'PROGRESS', 'full_goal_complete': False,
    'actual_factorial_and_complete_pdf_reconstruction_pending': True,
    'upstream_model_svd_coordinate_measurements_not_repeated': True,
    'gpu_access_or_new_training': False, 'source_publication': False}
write(REPORT / 'archive-manifest.json', result)
print(json.dumps({'report': str(REPORT), 'manifest': identity(REPORT / 'archive-manifest.json'),
    'copied_files': result['copied_files'], 'copied_bytes': result['copied_bytes'],
    'credential_screened_text_files': text_files, 'synthetic_symlinks_recorded': len(links),
    'full_goal_complete': False}))
