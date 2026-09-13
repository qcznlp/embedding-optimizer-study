"""Execute an explicitly assembled historical test role, preserving its actual exit."""
import json
import os
from pathlib import Path
import subprocess
import sys

work = Path(__file__).resolve().parent
role = sys.argv[1]
value = json.loads((work / 'test-roles.json').read_text())['roles'][role]
root = Path(value['root'])
command = [sys.executable, '-B', '-m', 'pytest', '-q', *value['tests'], '--junitxml=' + str(work / (role + '.xml'))]
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(root / 'src') + ':' + str(root),
       'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
       'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'WANDB_MODE': 'disabled'}
with (work / (role + '.log')).open('xb') as log:
    child = subprocess.run(command, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
with (work / (role + '-exit.json')).open('x') as stream:
    json.dump({'role': role, 'command': command, 'cwd': str(root), 'exit_code': child.returncode,
               'tests_or_historical_contracts_modified': False, 'current_mixed_source_admission_claimed': False}, stream, indent=2)
print(json.dumps({'role': role, 'exit_code': child.returncode}), flush=True)
raise SystemExit(child.returncode)
