"""Execute the complete original-analysis tests with its three original source files."""
import json
import os
from pathlib import Path
import subprocess
import sys

work = Path(__file__).resolve().parent
role = json.loads((work / 'analysis-v2-role.json').read_text())
root = Path(role['root'])
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(root / 'src') + ':' + str(root),
       'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
       'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'WANDB_MODE': 'disabled'}
command = [sys.executable, '-B', '-m', 'pytest', '-q', *role['tests'], '--junitxml=' + str(work / 'original-analysis-v2.xml')]
with (work / 'original-analysis-v2.log').open('xb') as stream:
    child = subprocess.run(command, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
with (work / 'original-analysis-v2-exit.json').open('x') as stream:
    json.dump({'command': command, 'cwd': str(root), 'exit_code': child.returncode, 'tests_or_contracts_modified': False}, stream, indent=2)
print(json.dumps({'exit_code': child.returncode}), flush=True)
raise SystemExit(child.returncode)
