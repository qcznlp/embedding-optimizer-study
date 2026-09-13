"""Preserve actual functional sensitivity and portable unchanged numerical parents."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
STORY = Path('/root/embedding-optimizer-story-refactor')
DEST = STORY / 'reports/dense-v3-functional-sensitivity-v1'
EXPECTED = {
    HERE / 'functional_sensitivity.py': '0e98fbf4c768ef10bcae2cf6c660e37c583c88863055229ccdd65c0109661013',
    HERE / 'verify_functional_sensitivity.py': '82428fadc227312a30e2ae4620089ea81ccf48c3a0d2d8fccd9b604c054a1994',
    HERE / 'actual/readout.json': '8cb9b93da12a245e214429ffcf61f6922803c72c9c9ea496e572d24ecdc3e231',
    HERE / 'actual/independent_verification.json': '0b7f2a1b18405de7ab798d46edc6175f08d0e7ecb2b27e71a7c0458992cc246e',
    STORY / 'reports/dense-v3-functional-inference-v1/actual/readout.json': '608965f5297fc196adcce6ca7417bd40e7f35d05e26133f20c0811dd1fd7027e',
    STORY / 'reports/dense-v3-predictor-sensitivity-v1/source/sensitivity.py': '6a87b441917c95d1fb0355c340beb8715acee89e66c102b51ecdfce6265d9141',
    STORY / 'reports/dense-v3-weight-retrieval-v1/source-original/src/embed_optim/bridge_exact_arithmetic.py': '0d66ce6e99aa42b8d93c8295cb0f5f3015ea7b7907a9f10fb095d95be2d646b4',
    STORY / 'reports/dense-v3-weight-retrieval-v1/source/verify_predictions.py': '176ce4c6134ac728351f9099c5529ef54f84f59abe12efc111c0e7808c0f7100',
}


def identity(path):
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Nonordinary payload')
    raw = path.read_bytes()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


for path, digest in EXPECTED.items():
    if identity(path)['sha256'] != digest:
        raise ValueError('Actual source or evidence changed')
receipt = json.loads((HERE / 'actual/readout.json').read_text())
for name, binding in receipt['outputs'].items():
    if identity(HERE / 'actual' / name) != binding:
        raise ValueError('Actual calculation changed')
parent = STORY / 'reports/dense-v3-functional-inference-v1/actual'
pr = json.loads((parent / 'readout.json').read_text())
if identity(parent / 'tables.json') != pr['payloads']['tables.json']:
    raise ValueError('Parent data changed')
selected = {f'actual/{p.name}': p for p in (HERE / 'actual').iterdir() if p.is_file()}
selected.update({f'source/{name}': HERE / name for name in
                 ('functional_sensitivity.py', 'verify_functional_sensitivity.py', 'archive.py')})
selected.update({f'inputs/functional/{name}': parent / name for name in ('readout.json', 'tables.json')})
for path in EXPECTED:
    if path.name in ('sensitivity.py', 'bridge_exact_arithmetic.py', 'verify_predictions.py'):
        selected['source-parent/' + path.name] = path
payloads = {}
for name, source in sorted(selected.items()):
    target = DEST / name
    if target.exists():
        raise ValueError('Preserve existing archive')
    binding = identity(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if identity(target) != binding or identity(source) != binding:
        raise ValueError('Archive copy differs')
    payloads[name] = binding
record = dict(scope='complete-actual-functional-predictor-sensitivity-local-preservation',
    completed_at_utc=datetime.now(timezone.utc).isoformat(), copied_files=len(payloads), payloads=payloads,
    source_parent_bindings={str(p): s for p, s in EXPECTED.items()},
    producer_session=60841, producer_exit=0, producer_terminal='7b792a',
    independent_session=88839, independent_exit=0, independent_terminal='d89a44',
    remote_source_publication=False, scientific_completion=False)
with (DEST / 'verification.json').open('x') as stream:
    stream.write(json.dumps(record, indent=2, sort_keys=True) + '\n')
print(json.dumps(dict(copied_files=len(payloads), verification=identity(DEST / 'verification.json'))))
