"""Actual one-tree primary contract inspection; no device setup or training."""
import json
from pathlib import Path

from embed_optim import dense_run_contract as runtime
from embed_optim import train
from embed_optim.primary_contract import file_identity, require_same
from embed_optim.primary_v3_contract import PrimaryV3Contract

here = Path(__file__).resolve().parent
root = here.parents[2]
source = json.loads((here / 'primary-source-assembly.json').read_bytes())
for relative, row in source['files'].items():
    require_same(file_identity(root / relative), row['identity'])
contract = PrimaryV3Contract.load(root / 'configs/dense_primary_v3_protocol.json', root, root)
assert contract.repository == contract.training_root == root
actual_source = runtime.source_identity()
runs = {}
for run in contract.inputs['runs']:
    identity = contract.expected_identity(run['run_id'])
    require_same(identity['source'], actual_source)
    runs[run['run_id']] = identity
assert len(runs) == 12
try:
    contract.require_execution()
except ValueError as error:
    release_refusal = str(error)
else:
    raise AssertionError('Source consolidation must not release a historical draft')
with (here / 'actual/primary-one-tree.json').open('x') as stream:
    json.dump({'all_primary_assembly_files': len(source['files']),
        'all_twelve_run_sources_match_current_package': True,
        'loaded_train_source': str(Path(train.__file__).resolve()),
        'runs': runs, 'original_release_refusal': release_refusal,
        'gpu_work': False, 'training_executed': False, 'source_release': False}, stream, indent=2)
print('ONE_TREE_PRIMARY_CONTRACT_AND_ALL_TWELVE_SOURCES_MATCH')
