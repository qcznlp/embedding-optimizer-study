"""Read-only exact endpoint differences, not a replacement acceptance test."""
import hashlib
import json
import os
from pathlib import Path
import struct
import numpy as np
import torch
from safetensors.torch import load_file

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
WORK = Path('/tmp/dense-v3-resume-device-recovery.uQ0ynb0k')
auth_bytes = (WORK / 'authorization.json').read_bytes()
assert hashlib.sha256(auth_bytes).hexdigest() == '1e7fbcbbde7021ece3a18e5608f8cb909991860911097d0bb54e7b7a7043989a'
auth = json.loads(auth_bytes)

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def authenticated(binding):
    root = Path(binding['path'])
    manifest = root / 'factorial_trainer_component.json'
    assert sha(manifest) == binding['sha256']
    value = json.loads(manifest.read_bytes())
    for row in value['files']:
        path = root / row['path']
        assert path.is_relative_to(root) and path.stat().st_size == row['bytes']
        assert sha(path) == row['sha256']
    return root

def compare(a, b, name, rows):
    assert type(a) is type(b), name
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape, name
        equal = torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))
        row = {'name': name, 'kind': 'tensor', 'elements': a.numel(), 'exact': equal}
        if not equal:
            diff = a.to(torch.float64) - b.to(torch.float64)
            row.update(changed_elements=int(torch.count_nonzero(a != b)), max_abs=float(diff.abs().max()),
                       difference_l2=float(torch.linalg.vector_norm(diff)), original_l2=float(torch.linalg.vector_norm(a.to(torch.float64))))
        rows.append(row)
    elif isinstance(a, np.ndarray):
        rows.append({'name': name, 'kind': 'array', 'exact': a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()})
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), name
        for k in a:
            compare(a[k], b[k], name + '/' + str(k), rows)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), name
        for k, (left, right) in enumerate(zip(a, b)):
            compare(left, right, name + '/' + str(k), rows)
    else:
        exact = struct.pack('!d', a) == struct.pack('!d', b) if isinstance(a, float) else a == b
        row = {'name': name, 'kind': 'scalar', 'exact': bool(exact)}
        if not exact:
            row.update(original=a, resumed=b)
        rows.append(row)

result = {'scope': 'diagnosis_only_no_relaxed_acceptance_no_new_training', 'cases': {}}
for pool in ('a', 'b'):
    case = auth['cases'][pool]
    exits = json.loads((WORK / f'run/pool-{pool}/ranks.exited.json').read_bytes())
    assert exits['actual_rank_exits'] == {str(i): 0 for i in range(4)}
    returned = json.loads((WORK / f'run/pool-{pool}/rank-0.returned.json').read_bytes())
    left = authenticated(case['original_checkpoints']['391'])
    right = authenticated(returned['checkpoint'])
    rows = []
    compare(load_file(left / 'model.safetensors'), load_file(right / 'model.safetensors'), 'model', rows)
    for name in ('optimizer.pt', 'scheduler.pt', *(f'rng_state_{r}.pth' for r in range(4))):
        # Complete native inventories checked above before trusted RNG deserialization.
        kw = {'map_location': 'cpu', 'weights_only': not name.startswith('rng_state_')}
        compare(torch.load(left / name, **kw), torch.load(right / name, **kw), name, rows)
    original_state = json.loads((left / 'trainer_state.json').read_bytes())
    resumed_state = json.loads((right / 'trainer_state.json').read_bytes())
    for field in ('global_step', 'max_steps', 'num_train_epochs', 'train_batch_size', 'epoch'):
        compare(original_state[field], resumed_state[field], 'trainer/' + field, rows)
    summary = {}
    for component in ('model', 'optimizer.pt', 'scheduler.pt', *(f'rng_state_{r}.pth' for r in range(4)), 'trainer'):
        selected = [r for r in rows if r['name'].startswith(component + '/')]
        summary[component] = {'leaves': len(selected), 'unequal': sum(not r['exact'] for r in selected)}
    result['cases'][pool] = {'summary': summary, 'rows': rows,
        'original_new_history': [r for r in original_state['log_history'] if r['step'] > 313],
        'resumed_new_history': [r for r in resumed_state['log_history'] if r['step'] > 313]}
    print(json.dumps({'pool': pool, 'summary': summary}), flush=True)
assert not torch.cuda.is_initialized()
with (HERE / 'differences.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
