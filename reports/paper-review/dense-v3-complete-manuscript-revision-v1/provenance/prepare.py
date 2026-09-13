"""Copy completed, authenticated paper inputs for a separately reviewed prose revision."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
OLD = Path('/tmp/dense-v3-document-integration.Xz1qvTME/actual-complete-paper')
receipt_bytes = (OLD / 'document.json').read_bytes()
assert hashlib.sha256(receipt_bytes).hexdigest() == 'fd477eda08d870a6a9be5bbc66a9e1551ffebf114142f84c99dbc05629bb9ccb'
receipt = json.loads(receipt_bytes)
copied = {}
for name, binding in receipt['source_inspection']['inputs'].items():
    source = OLD / 'paper' / name
    raw = source.read_bytes()
    assert {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == binding
    target = HERE / 'candidate-v1/paper' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(raw)
    copied[name] = binding
with (HERE / 'original-inputs.json').open('x') as stream:
    json.dump(copied, stream, indent=2, sort_keys=True)
shutil.copyfile(OLD / 'document.json', HERE / 'original-document.json')
print(json.dumps({'copied_inputs': len(copied), 'old_inputs_unchanged': True}))
