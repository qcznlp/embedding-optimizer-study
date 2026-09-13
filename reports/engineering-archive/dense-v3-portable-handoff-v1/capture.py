"""Freeze completed handoff evidence; does not rerun any verification."""
import hashlib
import json
from pathlib import Path
import shutil

here = Path(__file__).resolve().parent
root = here.parents[2]
work = Path('/tmp/dense-v3-portable-handoff.u0r0Q1W0')
for source, destination in (
    *((root / name, here / 'after' / name) for name in
      ('AGENTS.md', 'PROJECT_STATUS.md', 'CURRENT_EXPERIMENT.md', 'README.md', 'pyproject.toml')),
    *((source, here / 'dist' / source.name) for source in sorted((work / 'dist').iterdir())),
    (root / 'src/embed_optim/distribution_audit.py', here / 'original/distribution_audit.py'),
):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as output, source.open('rb') as input_file:
        shutil.copyfileobj(input_file, output)
rows = {}
for path in sorted(here.rglob('*')):
    if path.is_file():
        raw = path.read_bytes()
        rows[path.relative_to(here).as_posix()] = {
            'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
with (here / 'manifest.json').open('x') as output:
    json.dump({'files': rows, 'outer_verification_exit': None,
               'full_distribution_admission': False}, output, indent=2, sort_keys=True)
print(json.dumps({'files': len(rows), 'bytes': sum(row['bytes'] for row in rows.values()),
                  'manifest_sha256': hashlib.sha256((here / 'manifest.json').read_bytes()).hexdigest()}))
