"""Check and seal this completed review/diagnostic handoff without rerunning work."""
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
STORY = Path('/root/embedding-optimizer-story-refactor')
PAPER = STORY / 'reports/paper-review/dense-v3-complete-manuscript-revision-v1'
DIAG = STORY / 'reports/engineering-archive/dense-v3-resume-endpoint-diagnosis-v1'

def identity(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

for root in (PAPER, DIAG):
    for name, binding in json.loads((root / 'capture.json').read_bytes()).items():
        assert identity(root / name) == binding, name

before = json.loads((PAPER / 'provenance/original-inputs.json').read_bytes())
for name, binding in before.items():
    if name not in ('main.tex', 'references.bib'):
        assert identity(PAPER / 'actual/paper' / name) == binding
assert identity(STORY / 'paper/main.tex')['sha256'] == '45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e'
assert identity(STORY / 'paper/references.bib')['sha256'] == 'fca6bf5762a2cecab795a1a60c1338ab4eb999acb5d78c437b9df0e5f3553218'

for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md', 'paper/README.md'):
    target = PAPER / 'ops-after' / name
    assert not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(STORY / name, target)
    assert identity(STORY / name) == identity(target)

before_readme = (PAPER / 'ops-before/README.md').read_text()
after_readme = (STORY / 'README.md').read_text()
pattern = r'<!-- FINAL-CONCLUSION:BEGIN -->.*?<!-- FINAL-CONCLUSION:END -->'
assert re.search(pattern, before_readme, re.S).group() == re.search(pattern, after_readme, re.S).group()
assert before_readme.split('-->', 1)[0] == after_readme.split('-->', 1)[0]
for path in (PAPER / 'README.md', DIAG / 'README.md', STORY / 'paper/README.md'):
    for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
        if '://' not in target and not target.startswith('#'):
            assert (path.parent / target.split('#')[0]).exists(), (path, target)

shutil.copyfile(HERE / 'finalize.py', PAPER / 'provenance/finalize.py')
for root in (PAPER, DIAG):
    paths = sorted(p for p in root.rglob('*') if p.is_file())
    rows = {p.relative_to(root).as_posix(): identity(p) for p in paths}
    manifest = {'scope': 'review_and_diagnosis_archive_not_scientific_release',
                'files': rows, 'files_excluding_manifest': len(rows),
                'bytes_excluding_manifest': sum(r['bytes'] for r in rows.values())}
    target = root / 'archive-manifest.json'
    with target.open('x') as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
    print(json.dumps({'root': str(root), 'files': len(rows),
                      'bytes': manifest['bytes_excluding_manifest'],
                      'manifest_sha256': identity(target)['sha256']}))
