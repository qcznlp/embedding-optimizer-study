"""Validate source consolidation locally without training or changing old locks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
root = here.parents[2]
primary = Path('/root/embedding-optimizer-primary-v3')
actual = here / 'actual'
actual.mkdir()
assembly = json.loads((here / 'primary-source-assembly.json').read_bytes())
verified = {}
for path in sorted((primary / 'src/embed_optim').glob('*.py')):
    relative = path.relative_to(primary).as_posix()
    expected = assembly['files'][relative]['identity']
    raw = (root / relative).read_bytes()
    observed = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    assert observed == expected and raw == path.read_bytes(), relative
    verified[relative] = observed
assert len(verified) == 33
for name in ('test_dense_numerical_contract.py', 'test_dense_run_contract.py'):
    assert (root / 'tests' / name).read_bytes() == (primary / 'tests' / name).read_bytes()
(actual / 'source-equality.json').write_text(json.dumps({'primary_modules': verified,
    'new_tests_byte_identical': True, 'old_source_locks_changed': False,
    'training_launched': False, 'source_release': False}, indent=2))
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
       'PYTHONPATH': str(root / 'src') + os.pathsep + str(root), 'OMP_NUM_THREADS': '1',
       'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
groups = {
    'primary-core': ['tests/test_collators.py', 'tests/test_dense_numerical_contract.py',
        'tests/test_dense_run_contract.py', 'tests/test_losses.py', 'tests/test_optimizers.py',
        'tests/test_runtime.py', 'tests/test_config.py', 'tests/test_train.py', 'tests/test_short_branch.py'],
    'factorial-core': [str(path.relative_to(root)) for path in sorted((root / 'tests').glob('test_factorial_v3_*.py'))],
}
results = {}
for label, tests in groups.items():
    command = ['/usr/bin/python', '-B', '-m', 'pytest', *tests, '-q', '--junitxml=' + str(actual / (label + '.xml'))]
    with (actual / (label + '.stdout')).open('xb') as out, (actual / (label + '.stderr')).open('xb') as err:
        child = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err)
    results[label] = child.returncode
    (actual / (label + '.exit.json')).write_text(json.dumps({'command': command, 'actual_exit_code': child.returncode}, indent=2))
    print(json.dumps({'group': label, 'actual_exit_code': child.returncode}), flush=True)
(actual / 'test-exits.json').write_text(json.dumps(results, indent=2))
raise SystemExit(0 if all(code == 0 for code in results.values()) else 1)
