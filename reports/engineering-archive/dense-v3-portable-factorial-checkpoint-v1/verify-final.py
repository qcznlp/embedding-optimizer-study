"""Verify new CLI controls and final documentation/package surface, without GPU work."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

here = Path(__file__).resolve().parent
root = here.parents[2]
work = Path('/tmp/dense-v3-portable-factorial.KUvkU0gR')
actual = here / 'actual-final'
actual.mkdir()
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
       'PYTHONPATH': str(root / 'src') + os.pathsep + str(root),
       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}

def run(label, command):
    with (actual / (label + '.stdout')).open('xb') as out, (actual / (label + '.stderr')).open('xb') as err:
        child = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err)
    (actual / (label + '.exit.json')).write_text(json.dumps({
        'command': command, 'actual_exit_code': child.returncode}, indent=2))
    print(json.dumps({'stage': label, 'actual_exit_code': child.returncode}), flush=True)
    assert child.returncode == 0

run('unit', ['/usr/bin/python', '-B', '-m', 'pytest',
    'tests/test_saved_factorial_checkpoint.py', 'tests/test_current_primary_source.py',
    'tests/test_current_distribution_surface.py', 'tests/test_distribution.py', '-q',
    '--junitxml=' + str(actual / 'unit.xml')])
run('style', ['/usr/bin/python', '-m', 'ruff', 'check',
    'src/embed_optim/saved_factorial_checkpoint.py', 'tests/test_saved_factorial_checkpoint.py'])
run('build', ['/usr/bin/python', '-m', 'build', '--no-isolation', '--outdir', str(work / 'dist-final'), str(root)])
run('audit', ['/usr/bin/python', '-m', 'embed_optim.distribution_audit',
    '--repo-root', str(root), '--dist-dir', str(work / 'dist-final')])
audit = json.loads((actual / 'audit.stdout').read_bytes())
assert audit['complete'] is True and audit['problems'] == []
for name in ('factorial_v3_run_contract.py', 'factorial_v3_checkpoint.py'):
    assert (root / 'src/embed_optim' / name).read_bytes() == (here / 'before' / name).read_bytes()
with zipfile.ZipFile(next((work / 'dist-final').glob('*.whl'))) as archive:
    source = archive.read('embed_optim/saved_factorial_checkpoint.py')
    assert source == (work / 'wheel-corrected/embed_optim/saved_factorial_checkpoint.py').read_bytes()
    assert source == (root / 'src/embed_optim/saved_factorial_checkpoint.py').read_bytes()
    guide = archive.read('embedding_optimizer_study-0.1.0.data/data/share/embedding-optimizer-study/docs/saved-factorial-checkpoints.md')
    assert guide == (root / 'docs/saved-factorial-checkpoints.md').read_bytes()
    entries = archive.read('embedding_optimizer_study-0.1.0.dist-info/entry_points.txt').decode()
    assert 'embed-optim-inspect-factorial-checkpoint = embed_optim.saved_factorial_checkpoint:main' in entries
(actual / 'source-and-guide.json').write_text(json.dumps({
    'original_run_and_checkpoint_validators_unchanged': True,
    'reader_identical_to_genuine_tested_wheel': True,
    'reader_sha256': hashlib.sha256(source).hexdigest(),
    'guide_packaged_exactly': True, 'console_entry_declared': True,
    'gpu_resume_equivalence': False, 'source_release': False}, indent=2))
