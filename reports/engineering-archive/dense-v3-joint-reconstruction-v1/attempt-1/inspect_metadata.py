"""Read only: compare typed feature CSV identities with all declared vector states."""
from pathlib import Path
from embed_optim.primary_contract import read_json
from embed_optim.primary_v3_exact_bridge import typed_csv
from embed_optim.primary_v3_validation_io import write_new

work = Path(__file__).parent
root = work / 'fixture/producer'
admitted = read_json(root / 'vectors/admission.json')
metadata = {(s['state']['meta']['run_id'], s['state']['meta']['step']): s['state']['meta'] for s in admitted['states']}
result = {}
for name in ('checkpoint_summary', 'task_summary', 'random_removal', 'rotation_summary'):
    rows = typed_csv(root / 'features' / (name + '.csv'))
    mismatches = []
    for row in rows:
        expected = metadata[row['run_id'], row['step']]
        observed = {key: row.get(key) for key in expected}
        if observed != expected:
            mismatches.append({'observed': observed, 'expected': expected})
    result[name] = {'rows': len(rows), 'typed_mismatches': len(mismatches), 'first_mismatch': next(iter(mismatches), None)}
write_new(work / 'typed-metadata-check.json', result)
print(result, flush=True)
