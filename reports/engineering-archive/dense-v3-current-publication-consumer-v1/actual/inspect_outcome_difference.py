"""Read-only localization of a new cross-reader comparison failure."""
import json
from pathlib import Path
from assemble_publication import read, NATIVE, STORY, PRIMARY, HERE
from native_primary import write_new, identity
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_outcomes import outcome_tables
from embed_optim.primary_v3_exact_bridge import typed_csv

primary = PrimaryV3Contract.load(STORY / 'configs/dense_primary_v3_protocol.json', STORY, PRIMARY)
native = read(HERE / 'native-primary/evidence.json', NATIVE['primary'][1])
fresh = outcome_tables(primary, native['grid']['score_rows'], native['validation_selection'])
results = {}
for name in ('all_task_scores', 'primary_task_effects', 'primary_summary', 'secondary_task_effects',
             'secondary_summary', 'run_stage_scores', 'optimizer_stage_scores', 'run_observed_auc'):
    path = STORY / 'reports/dense-v3-complete-trajectories-v1/tables' / (name + '.csv')
    previous = typed_csv(path)
    differences = []
    for index, (a, b) in enumerate(zip(fresh[name], previous, strict=True)):
        if a != b:
            differences.append({'index': index,
                'fields': {k: {'fresh': a.get(k), 'previous': b.get(k),
                              'fresh_type': type(a.get(k)).__name__, 'previous_type': type(b.get(k)).__name__}
                           for k in set(a) | set(b) if a.get(k) != b.get(k)}})
    results[name] = {'fresh_rows': len(fresh[name]), 'previous_rows': len(previous),
                     'differing_rows': len(differences), 'examples': differences[:2],
                     'original': {'path': str(path), **identity(path)}}
write_new(HERE / 'outcome-comparison-difference.json', results)
print(json.dumps(results, indent=2, sort_keys=True))
