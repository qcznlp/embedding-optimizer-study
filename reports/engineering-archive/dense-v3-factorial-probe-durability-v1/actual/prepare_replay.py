"""Copy only original numerical sources and the standalone snapshot verifier."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
STORY = Path('/root/embedding-optimizer-story-refactor')
root = HERE / 'numerical-replay/source'
root.mkdir(parents=True, exist_ok=False)
sources = {
    'representation_geometry.py': (STORY / 'src/embed_optim/representation_geometry.py',
        '2910a084079e2c53a6b4152fcb1a32f4ab30387975a1dbbca336db9e546469a2'),
    'state_operator_factorial_probe.py': (STORY / 'src/embed_optim/state_operator_factorial_probe.py',
        '7a40bafe3b8d43122cf0559bb83e1d3e6dd169da3fbf4ece8f70f5ccc720784c'),
    'restore_factorial_probes.py': (STORY / 'scripts/restore_factorial_probes.py', None),
}
records = {}
for name, (source, expected) in sources.items():
    if source.is_symlink() or any(p.is_symlink() for p in source.parents):
        raise ValueError('Nonordinary source')
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('Original numerical source changed')
    with (root / name).open('xb') as stream:
        stream.write(raw)
    if source.read_bytes() != raw or (root / name).read_bytes() != raw:
        raise ValueError('Source copy changed')
    records[name] = {'bytes': len(raw), 'sha256': digest, 'origin': str(source)}
with (HERE / 'numerical-replay/source-bindings.json').open('x') as stream:
    json.dump(records, stream, indent=2, sort_keys=True)
    stream.write('\n')
print({'copied_numerical_sources': len(records), 'sha256': hashlib.sha256(
    (HERE / 'numerical-replay/source-bindings.json').read_bytes()).hexdigest()})
