"""Recheck changed location mapping against the already relocated genuine payload."""
import json
import os
from pathlib import Path
import sys

from embed_optim import factorial_v3_inputs as native
from embed_optim.primary_contract import digest, file_identity

here = Path(__file__).resolve().parent
root = here.parents[2]
old_path = root / 'reports/engineering-archive/dense-v3-packaged-input-relocation-v1/actual/native-readback.json'
assert file_identity(old_path)['sha256'] == 'da57593de89f5405ce7c1d31f16c3d203f5285dbe619652abca705ee13f8d227'
previous = json.loads(old_path.read_bytes())
locations = native.Locations(**{role: Path(path) for role, path in previous['role_locations'].items()})
roots = native._recorded_roots(native._fixed_audit(locations.evidence, 'data-first.json'),
                              native._fixed_audit(locations.evidence, 'states-first.json'))
forbidden = (*roots.values(), roots['data_store'].parent)
source = file_identity(native.__file__)
output = here / 'actual/native-readback.json'
denied = []
def guard(event, args):
    if event.startswith('socket.') or event in {'subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn'}:
        denied.append(event)
        raise PermissionError('No network or child process')
    if event in {'open', 'os.listdir', 'os.scandir'} and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.path.abspath(os.fsdecode(args[0])))
        if any(path.is_relative_to(old) for old in forbidden) and path != output:
            denied.append(event)
            raise PermissionError('No original producer input reads')
sys.addaudithook(guard)
result = native.load_inputs(locations)
assert digest(result) == previous['readback_digest_path_dependent']
assert denied == []
with output.open('x') as stream:
    json.dump({'input_digest': digest(result), 'equals_accepted_original_result': True,
               'source': source, 'role_locations': previous['role_locations'],
               'denied_operations': denied, 'gpu_access': False,
               'scientific_admission': result['scientific_admission'],
               'execution_authorized': result['execution_authorized'],
               'scope': 'Changed mapping with genuine relocated inputs; no tensor deserialization, numerical replay, OS sandbox or source release'}, stream, indent=2)
print('GENUINE_INPUT_RESULT_EXACTLY_UNCHANGED', flush=True)
