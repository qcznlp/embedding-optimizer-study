"""Create explicit original-source test roles; never alter contracts or assertions."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root = Path('/root/embedding-optimizer-story-refactor')
work = Path(__file__).resolve().parent
roles = {
    'original-analysis': {
        'src/embed_optim/config.py': (
            root / 'reports/engineering-archive/dense-v3-complete-replay-entry-v1/closed/primary/src/embed_optim/config.py',
            '8f9d0ab7251b4051c06a5b9a30bea5d8fa47b51513eae78bc205c953facab91a'),
        'src/embed_optim/optimizers.py': (
            root / 'reports/engineering-archive/dense-v3-complete-replay-entry-v1/closed/primary/src/embed_optim/optimizers.py',
            '12158eee448b4a7cc4ae5d4dd9e2b3492f52e452b0cb7ed4b2a96e996e71d3a6'),
    },
    'original-factorial': {
        'src/embed_optim/factorial_v3_inputs.py': (
            root / 'reports/engineering-archive/dense-v3-factorial-worker-v1/actual/source-final/src/embed_optim/factorial_v3_inputs.py',
            None),
    },
}
summary = json.loads((work / 'tests-summary.json').read_text())
factorial = {'tests.test_factorial_v3_factory', 'tests.test_factorial_v3_run_contract', 'tests.test_factorial_v3_worker'}
other = {'tests.test_paper_only_surface', 'tests.test_state_operator_claim_wording', 'tests.test_wandb_dense_provenance_audit'}
tests = {
    'original-analysis': sorted(set(summary['groups']) - factorial - other),
    'original-factorial': sorted(factorial),
}
# The factorial delta must be read from the originally executed, independently
# inventoried 70-file source closure. Record its exact bytes without inventing
# a new historical contract or changing any expected digest.
factorial_origin = roles['original-factorial']['src/embed_optim/factorial_v3_inputs.py'][0]
assert factorial_origin.is_file() and not factorial_origin.is_symlink()
factorial_sha = hashlib.sha256(factorial_origin.read_bytes()).hexdigest()
roles['original-factorial']['src/embed_optim/factorial_v3_inputs.py'] = (factorial_origin, factorial_sha)

names = set(subprocess.check_output(['git', 'ls-files', '-c', '-o', '--exclude-standard', '-z'], cwd=root).decode().split('\0')) - {''}
inventory = {}
for name in sorted(names):
    relative = Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    if relative.name == 'gpu.py':
        continue
    path = root / relative
    if not path.exists():
        continue  # Tracked rename/deletion; the current counterpart is separate.
    assert not path.is_symlink(), 'Unreviewed symlink: ' + name
    if not path.is_file():
        continue
    raw = path.read_bytes()
    inventory[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

records = {}
for role, changes in roles.items():
    destination = work / role
    destination.mkdir(exist_ok=False)
    for name, binding in inventory.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        original, expected = changes.get(name, (root / name, binding['sha256']))
        assert hashlib.sha256(original.read_bytes()).hexdigest() == expected
        shutil.copyfile(original, target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == expected
    # Closed inventories are copied file-for-file if ignore rules omitted a
    # bound input. Only named files from the authenticated numerical manifest
    # are eligible, not arbitrary producer data or live artifacts.
    base = root / 'reports/engineering-archive/dense-v3-complete-replay-entry-v1/closed'
    manifest = json.loads((base / 'manifest.json').read_text())
    assert hashlib.sha256((base / 'manifest.json').read_bytes()).hexdigest() == '746a545d3e59d86aab8d4ff6579abca3be9884f1ea824a702e4fa28bc85956b7'
    for name in [*manifest['files'], 'manifest.json']:
        source = base / name
        target = destination / source.relative_to(root)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        assert source.read_bytes() == target.read_bytes()
    records[role] = {'root': str(destination), 'changes': {n: {'origin': str(p), 'sha256': s} for n, (p, s) in changes.items()},
                     'tests': ['tests/' + n.removeprefix('tests.') + '.py' for n in tests[role]],
                     'current_mixed_source_admission_claimed': False, 'tests_or_contracts_modified': False}
with (work / 'test-roles.json').open('x') as stream:
    json.dump({'base_files': inventory, 'roles': records}, stream, indent=2, sort_keys=True)
print(json.dumps({'base_files': len(inventory), 'base_bytes': sum(v['bytes'] for v in inventory.values()),
                  'roles': {k: {'test_modules': len(v['tests']), 'changes': v['changes']} for k, v in records.items()}}), flush=True)
