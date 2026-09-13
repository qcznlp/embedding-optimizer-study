"""Preserve actual paper revision and terminal recovery diagnostics, without reruns."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
STORY = Path('/root/embedding-optimizer-story-refactor')
PAPER = STORY / 'reports/paper-review/dense-v3-complete-manuscript-revision-v1'
DIAG = STORY / 'reports/engineering-archive/dense-v3-resume-endpoint-diagnosis-v1'
WORK = Path('/tmp/dense-v3-resume-device-recovery.uQ0ynb0k')
OLD = Path('/tmp/dense-v3-document-integration.Xz1qvTME/actual-complete-paper')

def identity(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def copy(source, target):
    assert source.is_file() and not source.is_symlink() and not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert identity(source) == identity(target)

assert not (PAPER / 'actual').exists() and not (DIAG / 'actual').exists()
for path in sorted((HERE / 'candidate-v1').rglob('*')):
    if path.is_file():
        copy(path, PAPER / 'actual' / path.relative_to(HERE / 'candidate-v1'))
for name in ('prepare.py', 'build.py', 'archive.py', 'original-inputs.json', 'original-document.json'):
    copy(HERE / name, PAPER / 'provenance' / name)
copy(OLD.parent / 'document_component.py', PAPER / 'provenance/document_component.py')
for name in ('main.tex', 'references.bib'):
    before = OLD / 'paper' / name
    after = HERE / 'candidate-v1/paper' / name
    copy(before, PAPER / 'before' / name)
    with (PAPER / 'provenance' / (name + '.diff')).open('x') as stream:
        stream.writelines(difflib.unified_diff(before.read_text().splitlines(True),
            after.read_text().splitlines(True), fromfile='completed-native/' + name,
            tofile='reviewed-candidate/' + name))
for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md', 'paper/README.md'):
    copy(STORY / name, PAPER / 'ops-before' / name)
for pool in ('a', 'b'):
    root = WORK / f'run/pool-{pool}'
    for path in sorted(root.iterdir()):
        if path.is_file():
            copy(path, DIAG / 'actual' / f'pool-{pool}' / path.name)
for name in ('diagnose.py', 'differences.json'):
    copy(Path('/tmp/dense-v3-resume-endpoint-diagnosis.e3vIic9A') / name, DIAG / 'actual' / name)
with (DIAG / 'actual/tool-exits.json').open('x') as stream:
    json.dump({'39663': {'terminal': '8ceca3', 'exit_code': 1},
               '75985': {'terminal': '1672c5', 'exit_code': 1},
               '39062': {'terminal': 'b7f32d', 'exit_code': 0,
                         'scope': 'diagnostic_only_not_equivalence_acceptance'}}, stream, indent=2)
with (PAPER / 'provenance/tool-exits.json').open('x') as stream:
    json.dump({'prepare': {'terminal': '7e59f3', 'exit_code': 0},
               '28526': {'terminal': '4ef483', 'exit_code': 0},
               '3568': {'terminal': '2b0bf1', 'exit_code': 0},
               'visual_inspection': {'pages': list(range(1, 14)), 'all_pages_viewed': True}}, stream, indent=2)
for report in (PAPER, DIAG):
    rows = {p.relative_to(report).as_posix(): identity(p) for p in sorted(report.rglob('*')) if p.is_file()}
    with (report / 'capture.json').open('x') as stream:
        json.dump(rows, stream, indent=2, sort_keys=True)
    print(json.dumps({'report': str(report), 'copied_and_authored_files': len(rows),
                      'bytes': sum(r['bytes'] for r in rows.values())}))
