"""Check the new typed reader against all previously accepted real diagnostic records."""
from pathlib import Path

import torch

from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_geometry_primitives import stage_records
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff

root = Path('/root/embedding-optimizer-story-refactor')
work = Path(__file__).parent
receipt = root / 'reports/engineering-archive/dense-v3-geometry-chain-v1/rehearsal-v2.json'
assert file_identity(receipt)['sha256'] == '0bd4d9290fd672229ee2d3a5cd4c891854c3e951fc47286bf8e8424190c5c665'
old = read_json(receipt)
bound = {row['path']: row for row in old['artifacts']}
paths = [root / 'src/embed_optim/primary_v3_geometry_primitives.py', Path(__file__), receipt]
sources = [{'path': str(path), **file_identity(path)} for path in paths]
before = handoff()
torch.set_num_threads(4)
checks, inputs = [], []
for path in sorted(bound):
    path = Path(path)
    if path.name != 'manifest.json' or '/runs/' not in str(path):
        continue
    verify_file(path, bound[str(path)])
    inputs.append(bound[str(path)])
    manifest = read_json(path)
    plan = manifest['plan']
    assert plan['scope'] == 'engineering_dense_v3_geometry_diagnostic'
    require_same(plan['steps'], [1, 2, 3])
    expected, checked = {'recipe': plan['recipe']}, {'steps': plan['steps']}
    for stage, row in enumerate(manifest['outputs'], 1):
        original = path.parent / row['records']['path']
        verify_file(original, bound[str(original)])
        inputs.append(bound[str(original)])
        value = read_json(original)
        fresh, ranks = stage_records(value, expected, checked, plan['reference'], plan['settings'], stage)
        require_same(fresh, value['checkpoint_row'])
        assert len(value['records']) == len(value['entry_records']) == 88
        checks.append({'run_id': plan['recipe']['run_id'], 'step': row['step'],
                       'records': 88, 'health': len(ranks), 'defined': sum(bool(v) for v in ranks.values()),
                       'exact_checkpoint_row_match': True})
assert len(checks) == 9
assert sum(row['health'] for row in checks) == 1584
for row in sources + inputs:
    verify_file(row['path'], row)
require_same(handoff(), before)
write_new(work / 'result.json', {'scope': 'engineering_real_record_geometry_schema_readback',
    'passed': True, 'sources': sources, 'inputs': inputs, 'checks': checks,
    'raw_weight_metrics_recomputed': False, 'spectral_health_remeasured': False,
    'primary_admission': False, 'scientific_completion': False, 'post_execution_dispatchers': before})
print({'passed': True, 'real_checkpoint_records': 9, 'hidden_matrix_records': 792,
       'health_records': 1584, 'defined': sum(row['defined'] for row in checks),
       'scientific_completion': False, **file_identity(work / 'result.json')}, flush=True)
