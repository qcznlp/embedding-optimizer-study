"""Use wheel-local readers on already anonymously restored genuine checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import sys

wheel, reference, runtime, output = map(Path, sys.argv[1:5])
case_name = sys.argv[5]
raw = reference.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '3c48d5c7d10434fe48925ec223e55c85df59e23ee6767a9d4bffbf1f5901f332'
case = json.loads(raw)['cases'][case_name]
sys.path.insert(0, str(wheel))
from embed_optim import saved_factorial_checkpoint as saved

forbidden = tuple(Path(path) for path in (
    '/root/embedding-optimizer-study', '/root/embedding-optimizer-story-refactor',
    '/root/embedding-optimizer-primary-v3', '/root/embedding-optimizer-v3-experiment'))
# Also forbid the precise original runtime-source directory stored in the save.
manifest = Path(case['binding']['path']) / saved.checkpoints.NAME
component = json.loads(manifest.read_bytes())['identity']
forbidden += (Path(component['bound_factorial_run']['runtime_spec']['path']).parents[1],)
denials = []
def guard(event, args):
    if event.startswith('socket.') or event in {'subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn'}:
        denials.append(event)
        raise PermissionError('No network or spawned process during saved-state reading')
    if event in {'open', 'os.listdir', 'os.scandir'} and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.path.abspath(os.fsdecode(args[0])))
        if any(path.is_relative_to(old) for old in forbidden):
            denials.append({'event': event, 'path': str(path)})
            raise PermissionError('Original source/model/data location forbidden')
sys.addaudithook(guard)
for old in forbidden:
    try:
        (old / 'must-not-be-opened').open('rb')
    except PermissionError:
        pass
    else:
        raise AssertionError('Original location was not refused before opening')
control_denials = list(denials)
denials.clear()
result = saved.read_checkpoint(case['binding']['path'], case['binding']['sha256'], runtime)
assert result['native'] == case['native']
assert result['saved_identities_unchanged'] is True and result['scientific_completion'] is False
assert result['gpu_resume_equivalence'] is False
modules = {name: module.__file__ for name, module in sys.modules.items()
           if (name == 'embed_optim' or name.startswith('embed_optim.')) and getattr(module, '__file__', None)}
assert modules and all(Path(path).is_relative_to(wheel) for path in modules.values())
assert denials == []
import torch
assert not torch.cuda.is_initialized()
with output.open('x') as stream:
    json.dump({'case': case_name, 'result': result, 'original_native_readback_exact': True,
               'separate_control_denials': control_denials, 'reader_denials': denials,
               'loaded_modules': modules, 'gpu_initialized': False,
               'physical_second_host': False, 'source_release': False}, stream, indent=2)
print('GENUINE_SAVED_CHECKPOINT_READ_EXACT: ' + case_name, flush=True)
