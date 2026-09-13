"""Check the changed documentation distribution and preserve observed exits."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

here = Path(__file__).resolve().parent
root = here.parents[2]
work = Path('/tmp/dense-v3-complete-entry-dist.r1edF6fb')
actual = here / 'actual'
actual.mkdir()
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
       'PYTHONPATH': str(root / 'src') + os.pathsep + str(root),
       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}

def identity(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open('rb') as incoming, dst.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    assert identity(src) == identity(dst)

def run(name, command):
    with (actual / (name + '.stdout')).open('xb') as out, (actual / (name + '.stderr')).open('xb') as err:
        result = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err)
    (actual / (name + '.exit.json')).write_text(json.dumps({'command': command, 'actual_exit_code': result.returncode}, indent=2))
    print(json.dumps({'stage': name, 'actual_exit_code': result.returncode}), flush=True)
    assert result.returncode == 0

run('docs-tests', ['/usr/bin/python', '-B', '-m', 'pytest',
    'tests/test_current_distribution_surface.py', 'tests/test_distribution.py',
    '-q', '--junitxml=' + str(actual / 'unit.xml')])
run('build', ['/usr/bin/python', '-m', 'build', '--no-isolation', '--outdir', str(work / 'dist'), str(root)])
run('audit', ['/usr/bin/python', '-m', 'embed_optim.distribution_audit',
    '--repo-root', str(root), '--dist-dir', str(work / 'dist')])
audit = json.loads((actual / 'audit.stdout').read_bytes())
assert audit['complete'] is True and audit['problems'] == []
paths = ['docs/paper-results-reproduction.md', 'AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md']
for name in paths:
    copy(root / name, here / 'after' / name)
    for link in re.findall(r'\]\(([^)]+)\)', (root / name).read_text()):
        if '://' not in link and not link.startswith('#'):
            assert (root / name).parent.joinpath(link.split('#')[0]).exists(), (name, link)
with zipfile.ZipFile(next((work / 'dist').glob('*.whl'))) as archive:
    prefix = 'embedding_optimizer_study-0.1.0.data/data/share/embedding-optimizer-study/'
    for name in paths[:-1]:
        assert archive.read(prefix + name) == (root / name).read_bytes()
    assert archive.read('embed_optim/saved_factorial_checkpoint.py') == (root / 'src/embed_optim/saved_factorial_checkpoint.py').read_bytes()
for path in (work / 'dist').iterdir():
    copy(path, here / 'dist' / path.name)
assert identity(root / 'src/embed_optim/distribution_audit.py') == identity(
    here.parent / 'dense-v3-portable-handoff-v1/original/distribution_audit.py')
rows = {path.relative_to(here).as_posix(): identity(path)
        for path in sorted(here.rglob('*')) if path.is_file()}
with (here / 'manifest.json').open('x') as out:
    json.dump({'files': rows}, out, indent=2, sort_keys=True)
print(json.dumps({'archive_files': len(rows), 'archive_bytes': sum(v['bytes'] for v in rows.values()),
                  'manifest': identity(here / 'manifest.json')}))
