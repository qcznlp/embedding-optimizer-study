"""Actual complete installed-wheel source-composition execution, not a mock."""
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

work = Path(__file__).resolve().parent
repo = Path('/root/embedding-optimizer-story-refactor')
wheel_path = next((work / 'dist').glob('*.whl'))
wheel = work / 'wheel'
assert not wheel.exists()
with zipfile.ZipFile(wheel_path) as archive:
    for name in archive.namelist():
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
    archive.extractall(wheel)
sys.path.insert(0, str(wheel))
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''

def no_network(event, args):
    if event == 'socket.connect':
        raise PermissionError('No network in versioned reproduction integration')

sys.addaudithook(no_network)
from embed_optim import paper_reproduction
result = paper_reproduction.reproduce(
    repo / 'reports/engineering-archive/dense-v3-complete-replay-entry-v1/closed',
    wheel / 'embedding_optimizer_study-0.1.0.data/data/share/embedding-optimizer-study/paper/current',
    work / 'actual',
)
loaded = {}
for name, module in list(sys.modules.items()):
    if name == 'embed_optim' or name.startswith('embed_optim.'):
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(wheel), (name, path)
        raw = path.read_bytes()
        loaded[name] = {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
assert result['complete'] is True and result['repository_publication_complete'] is False
assert not __import__('torch').cuda.is_initialized()
with (work / 'actual-wheel.json').open('x') as stream:
    json.dump({'complete': True, 'loaded_modules': loaded, 'wheel_sha256': hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
               'new_gpu_work': False, 'remote_source_publication': False}, stream, indent=2, sort_keys=True)
print(json.dumps({'complete': True, 'current_wheel_modules': len(loaded), 'pdf': result['reviewed_document']['pdf']}), flush=True)
