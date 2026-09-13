"""Retain the failed two-file assembly; add the original trainer in a fresh role."""
import hashlib
import json
from pathlib import Path
import shutil

work = Path(__file__).resolve().parent
root = Path('/root/embedding-optimizer-story-refactor')
value = json.loads((work / 'test-roles.json').read_text())
first = value['roles']['original-analysis']
target_root = work / 'original-analysis-v2'
target_root.mkdir(exist_ok=False)
trainer = root / 'reports/engineering-archive/dense-v3-numerical-source-integration-v1/before/src/embed_optim/train.py'
expected_trainer = 'e52cfcb5857aa64d4fb826c1f0b12eabe547506de25241a93b88ef1969d1720d'
assert hashlib.sha256(trainer.read_bytes()).hexdigest() == expected_trainer
changes = {**first['changes'], 'src/embed_optim/train.py': {'origin': str(trainer), 'sha256': expected_trainer}}
for name, binding in value['base_files'].items():
    source = trainer if name == 'src/embed_optim/train.py' else Path(first['root']) / name
    expected = changes[name]['sha256'] if name in changes else binding['sha256']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    target = target_root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == expected
record = {**first, 'root': str(target_root), 'changes': changes, 'prior_collection_failure_preserved': True}
with (work / 'analysis-v2-role.json').open('x') as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
print(json.dumps({'root': str(target_root), 'source_delta': changes, 'test_modules': len(record['tests'])}), flush=True)
