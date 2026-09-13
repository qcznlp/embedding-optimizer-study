"""Capture actual subprocess outcomes without changing any historical parent guard."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

here = Path(__file__).resolve().parent
root = here.parents[2]
work = Path('/tmp/dense-v3-portable-input-roles.eOsTxin7')
actual = here / 'actual'
actual.mkdir()
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
       'PYTHONPATH': str(root / 'src') + os.pathsep + str(root)}
def run(label, args, environment=env):
    with (actual / (label + '.stdout')).open('xb') as out, (actual / (label + '.stderr')).open('xb') as err:
        result = subprocess.run(args, cwd=root, env=environment, stdout=out, stderr=err)
    (actual / (label + '.exit.json')).write_text(json.dumps({'command': args, 'actual_exit_code': result.returncode}, indent=2))
    print(json.dumps({'label': label, 'actual_exit_code': result.returncode}), flush=True)
    return result.returncode

assert run('native', ['/usr/bin/python', '-B', str(here / 'native_check.py')]) == 0
assert run('local-tests', ['/usr/bin/python', '-m', 'pytest', 'tests/test_factorial_v3_input_roles.py',
    'tests/test_dense_evaluation_pause.py', 'tests/test_distribution.py', '-q',
    '--junitxml=' + str(actual / 'local-tests.xml')]) == 0
# This is a diagnostic assembly, not a rewrite of any accepted 66-file parent.
stage = work / 'diagnostic-source/src/embed_optim'
stage.parent.mkdir(parents=True)
shutil.copytree(root / 'src/embed_optim', stage, ignore=shutil.ignore_patterns('__pycache__'))
accepted = Path('/tmp/dense-v3-factorial-calibration.U5gTyB/source-third/src/embed_optim')
for source in accepted.glob('*.py'):
    shutil.copyfile(source, stage / source.name)
shutil.copyfile(root / 'src/embed_optim/factorial_v3_inputs.py', stage / 'factorial_v3_inputs.py')
sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.glob('*.py'))}
(actual / 'diagnostic-sources.json').write_text(json.dumps(sources, indent=2))
stage_env = {**env, 'PYTHONPATH': str(stage.parent) + os.pathsep + str(root)}
stage_code = run('calibration-tests', ['/usr/bin/python', '-m', 'pytest',
    'tests/test_factorial_v3_calibration.py',
    'tests/test_factorial_v3_run_contract.py::test_missing_genuine_calibrations_cannot_prepare_a_formal_run',
    '-q', '--junitxml=' + str(actual / 'calibration-tests.xml')], stage_env)
assert stage_code == 0
assert run('build', ['/usr/bin/python', '-m', 'build', '--no-isolation', '--outdir', str(work / 'dist')]) == 0
assert run('audit', ['/usr/bin/python', '-m', 'embed_optim.distribution_audit', '--repo-root', str(root),
                    '--dist-dir', str(work / 'dist')]) == 1
audit = json.loads((actual / 'audit.stdout').read_bytes())
assert audit['complete'] is False
assert audit['problems'] == ['sdist producer checkout path: embedding_optimizer_study-0.1.0/tests/test_primary_v3_outcome_reconstruction.py']
(actual / 'complete.json').write_text(json.dumps({'source_release': False, 'remaining_distribution_findings': audit['problems'],
    'genuine_input_readback_exact': True, 'diagnostic_calibration_tests_exit': stage_code}, indent=2))
