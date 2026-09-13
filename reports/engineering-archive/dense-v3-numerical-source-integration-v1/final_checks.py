"""Retain original failures and verify the completed primary integration."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
root = here.parents[2]
actual = here / 'actual-final'
actual.mkdir()
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
       'PYTHONPATH': str(root / 'src') + os.pathsep + str(root),
       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
tests = ['tests/test_collators.py', 'tests/test_dense_numerical_contract.py',
    'tests/test_dense_run_contract.py', 'tests/test_losses.py', 'tests/test_optimizers.py',
    'tests/test_runtime.py', 'tests/test_config.py', 'tests/test_train.py', 'tests/test_short_branch.py',
    'tests/test_factorial_v3_calibration.py', 'tests/test_factorial_v3_input_roles.py',
    'tests/test_current_paper.py', 'tests/test_current_distribution_surface.py', 'tests/test_distribution.py']
command = ['/usr/bin/python', '-B', '-m', 'pytest', *tests, '-q', '--junitxml=' + str(actual / 'tests.xml')]
with (actual / 'tests.stdout').open('xb') as out, (actual / 'tests.stderr').open('xb') as err:
    child = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err)
(actual / 'tests.exit.json').write_text(json.dumps({'command': command, 'actual_exit_code': child.returncode}, indent=2))
print(json.dumps({'focused_tests_exit': child.returncode}), flush=True)
assert child.returncode == 0
from embed_optim import factorial_v3_run_contract as factorial
from embed_optim.primary_contract import file_identity
accepted = json.loads((root / 'reports/engineering-archive/dense-v3-factorial-calibration-v1/actual/actual-inputs-third.json').read_bytes())
differences = {name: {'historical': old, 'current': file_identity(root / name)}
               for name, old in accepted['source_files'].items() if file_identity(root / name) != old}
assert list(differences) == ['src/embed_optim/factorial_v3_inputs.py']
try:
    factorial.source_identity(root, root)
except ValueError as error:
    refusal = str(error)
else:
    raise AssertionError('Historical factorial source lock was improperly replaced')
(actual / 'factorial-parent-boundary.json').write_text(json.dumps({
    'differences_from_original_66_file_parent': differences,
    'original_refusal': refusal, 'factorial_source_integration_complete': False,
    'source_release': False}, indent=2))
print('PRIMARY_INTEGRATED; FACTORIAL_ORIGINAL_SOURCE_REFUSAL_RETAINED', flush=True)
