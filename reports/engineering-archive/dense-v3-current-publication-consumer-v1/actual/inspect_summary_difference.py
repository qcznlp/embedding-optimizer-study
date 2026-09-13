"""Localize display-summary schema differences without changing any inference."""
import json
from assemble_publication import read, NATIVE, STORY, PRIMARY, HERE, FRAGMENTS
from native_primary import write_new
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_outcomes import outcome_tables, system_rows
from embed_optim.primary_v3_publication_tables import summarize

primary = PrimaryV3Contract.load(STORY / 'configs/dense_primary_v3_protocol.json', STORY, PRIMARY)
native = read(HERE / 'native-primary/evidence.json', NATIVE['primary'][1])
outcomes = outcome_tables(primary, native['grid']['score_rows'], native['validation_selection'])
outcomes['system_metrics'] = system_rows(primary, native['grid']['runs'])
geometry = read(HERE / 'native-geometry/tables.json')['approximate']
bridge = read(STORY / 'reports/dense-v3-weight-retrieval-v1/actual/tables.json')['original']
fresh = summarize(primary, outcomes, geometry['checkpoint_geometry'], geometry['run_pair_subspace_overlap'],
                  bridge, native['validation_selection'], native['grid']['runs'])
previous = read(FRAGMENTS / 'actual/primary-summary.json')
differences = []
def compare(a, b, path):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                differences.append({'path': path + '/' + key, 'missing': 'fresh' if key not in a else 'previous'})
            else:
                compare(a[key], b[key], path + '/' + key)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path + '/' + str(i))
    elif type(a) is not type(b) or a != b:
        differences.append({'path': path, 'fresh': a, 'previous': b,
                             'fresh_type': type(a).__name__, 'previous_type': type(b).__name__})
compare({k: fresh[k] for k in previous}, previous, '')
write_new(HERE / 'summary-comparison-difference.json', differences)
print(json.dumps({'count': len(differences), 'examples': differences[:12]}, indent=2))
