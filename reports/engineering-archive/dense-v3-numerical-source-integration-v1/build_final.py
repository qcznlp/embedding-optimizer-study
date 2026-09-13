"""Fresh package audit and actual wheel-local primary source import check."""
import json
import os
from pathlib import Path
import subprocess
import zipfile

here = Path(__file__).resolve().parent
root = here.parents[2]
work = Path('/tmp/dense-v3-numerical-integration.DUnt6osB')
actual = here / 'distribution-final'
actual.mkdir()
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
       'PYTHONPATH': str(root / 'src'), 'OMP_NUM_THREADS': '1'}
def run(label, command):
    with (actual / (label + '.stdout')).open('xb') as out, (actual / (label + '.stderr')).open('xb') as err:
        child = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err)
    (actual / (label + '.exit.json')).write_text(json.dumps({'command': command, 'actual_exit_code': child.returncode}, indent=2))
    print(json.dumps({'stage': label, 'actual_exit_code': child.returncode}), flush=True)
    assert child.returncode == 0
run('build', ['/usr/bin/python', '-m', 'build', '--no-isolation', '--outdir', str(work / 'dist-final')])
run('audit', ['/usr/bin/python', '-m', 'embed_optim.distribution_audit', '--repo-root', str(root), '--dist-dir', str(work / 'dist-final')])
audit = json.loads((actual / 'audit.stdout').read_bytes())
assert audit['complete'] is True and audit['problems'] == []
wheel = next((work / 'dist-final').glob('*.whl'))
extracted = work / 'wheel-final'
extracted.mkdir()
with zipfile.ZipFile(wheel) as archive:
    archive.extractall(extracted)
code = '''
import json, sys
from pathlib import Path
wheel, reference, output = map(Path, sys.argv[1:])
sys.path.insert(0, str(wheel))
from embed_optim import dense_run_contract, dense_numerical_contract, train
dense_numerical_contract.verify_stack(train.OptimizerTrainer)
source = dense_run_contract.source_identity()
expected = json.loads(reference.read_bytes())
assert len(expected['runs']) == 12
assert all(identity['source'] == source for identity in expected['runs'].values())
modules = {name: module.__file__ for name,module in sys.modules.items()
           if (name == 'embed_optim' or name.startswith('embed_optim.')) and getattr(module,'__file__',None)}
assert modules and all(Path(path).is_relative_to(wheel) for path in modules.values())
assert not train.torch.cuda.is_initialized()
with output.open('x') as stream:
    json.dump({'all_twelve_primary_source_identities_equal': True, 'loaded_modules': modules,
               'training_executed': False, 'source_release': False}, stream, indent=2)
print('WHEEL_LOCAL_PRIMARY_SOURCES_MATCH_ALL_TWELVE_RUNS')
'''
run('wheel-import', ['/usr/bin/python', '-B', '-I', '-c', code, str(extracted),
                    str(here / 'actual/primary-one-tree.json'), str(actual / 'wheel-source-readback.json')])
