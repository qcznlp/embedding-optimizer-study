"""Fresh CPU native read of the first two genuine four-GPU step-79 saves."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path('/root/embedding-optimizer-v3-experiment/launch/factorial-training-v1')
SOURCE = 'd11d3281523b1573ce7fcd650e10c538cfa84022b5db92117559d47213e57e53'
AUTH = '00c52f0bec979fb163a734ad0f9eadcd6ea09f7393133fd35c65bc62878ce11f'
PINS = {
    'factorial-v3-adamw_state-adamw-seed314159': {
        'seal': '805a71c2217cda5f7fc98ea7d014ebbe9d9fbf6f20143b78badf07dede31b9f0',
        'factory': '781bc951889c2692dab07e9a30c6c651b985f7d0c37a0efd1a39a32246845eca'},
    'factorial-v3-adamw_state-muon-seed314159': {
        'seal': '4b2a5d224c21d6b7ff47d96b21be74e40ec5c002f3bae4fd1a310e6b024f509e',
        'factory': '2dd833cc1611909795d749f222072aba823140eca8308de0d9abaa772f4ee2ee'},
}
sys.path.insert(0, str(ROOT))
import factorial_dispatch as entry

parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
auth, _ = entry.authenticate(SimpleNamespace(source_sha=SOURCE, authorization_sha=AUTH))
_, native, locations = entry.context(auth['source_files'])
from embed_optim import factorial_v3_run_contract as contract
import torch
entry.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only reader required')
rows = []
for run_id, pins in PINS.items():
    declared = auth['requests'][run_id]
    factory = entry.read(Path(declared['record_root']) / 'factory.json', pins['factory'])
    expected = native._require_factory_record(factory['creation'], factory['component_identity'],
                                              declared, auth['source_files'])
    actual, _ = contract.prepare_run(locations, auth['calibrations'], declared['state'],
        declared['operator'], declared['seed'], entry.SOURCE_ROOT)
    entry.need(expected == actual, 'Actual native calibrated data/run identity differs')
    path = Path(declared['run_root']) / 'checkpoint-79'
    binding = {'path': str(path), 'sha256': pins['seal']}
    inspected = contract.inspect_checkpoint(binding, actual, factory['component_identity'])
    entry.need(inspected['step'] == 79 and inspected['model_tensors'] == 134
               and inspected['rank_rng_states'] == 4, 'Incomplete genuine checkpoint')
    try:
        contract.inspect_checkpoint({**binding, 'sha256': '0' * 64}, actual, factory['component_identity'])
    except ValueError:
        refused = True
    else:
        raise ValueError('Native reader accepted the wrong external checkpoint anchor')
    rows.append({'run_id': run_id, 'inspected': inspected, 'wrong_external_anchor_refused': refused,
        'factory_binding': entry.identity(Path(declared['record_root']) / 'factory.json')})
    print(json.dumps({'run_id': run_id, 'native_step_79_readback': True}), flush=True)
entry.c.check_imports(auth['source_files'])
entry.need(entry.source_files() == auth['source_files'] and not torch.cuda.is_initialized(), 'Reader changed source or entered CUDA')
result = {'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'rows': rows,
    'scope': 'actual_first_four_gpu_saved_checkpoint_native_cpu_readback',
    'source_sha256': SOURCE, 'authorization_sha256': AUTH,
    'new_training_executed_by_reader': False, 'actual_gpu_resume_equivalence': False,
    'whole_branches_complete': False, 'scientific_completion': False}
with args.output.open('x') as stream: stream.write(json.dumps(result, sort_keys=True, indent=2) + '\n')
