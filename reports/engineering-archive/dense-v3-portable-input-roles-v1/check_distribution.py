"""Run unchanged full audit on a fresh build with explicit output namespaces."""
import json
import os
from pathlib import Path
import subprocess
import sys

here = Path(__file__).resolve().parent
root = here.parents[2]
actual = here / sys.argv[1]
actual.mkdir()
dist = Path('/tmp/dense-v3-portable-input-roles.eOsTxin7') / sys.argv[2]
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
       'PYTHONPATH': str(root / 'src')}
for label, command in (
    ('build', ['/usr/bin/python', '-m', 'build', '--no-isolation', '--outdir', str(dist)]),
    ('audit', ['/usr/bin/python', '-m', 'embed_optim.distribution_audit', '--repo-root', str(root), '--dist-dir', str(dist)]),
):
    with (actual / (label + '.stdout')).open('xb') as out, (actual / (label + '.stderr')).open('xb') as err:
        child = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err)
    (actual / (label + '.exit.json')).write_text(json.dumps({'command': command, 'actual_exit_code': child.returncode}, indent=2))
    print(json.dumps({'stage': label, 'actual_exit_code': child.returncode}), flush=True)
    assert child.returncode == 0
audit = json.loads((actual / 'audit.stdout').read_bytes())
assert audit['complete'] is True and audit['problems'] == []
