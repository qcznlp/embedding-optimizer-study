"""Print selected nonsecret schema/identity fields for the final tracking audit."""
import json
from pathlib import Path

story = Path('/root/embedding-optimizer-story-refactor')
catalog = json.loads((story / 'reports/engineering-archive/dense-v3-closed-primary-paper-replay-v1/closed-primary/inputs/recipe-catalog.json').read_bytes())
print(json.dumps({'primary_catalog_keys': sorted(catalog), 'input_keys': sorted(catalog['inputs']),
                  'run_row_keys': sorted(catalog['inputs']['runs'][0]),
                  'first_run_row': catalog['inputs']['runs'][0],
                  'expected_identity_keys': sorted(next(iter(catalog['expected_identities'].values()))),
                  'wandb_target': catalog['payload'].get('wandb')}))
native = json.loads((story / 'reports/engineering-archive/dense-v3-current-publication-consumer-v1/actual/native-primary/evidence.json').read_bytes())
print(json.dumps({'native_keys': sorted(native)}))
train = Path('/root/embedding-optimizer-v3-experiment/launch/factorial-training-v1')
auth = json.loads((train / 'authorization.json').read_bytes())
request = next(iter(auth['requests'].values()))
print(json.dumps({'training_authority_keys': sorted(auth), 'request_keys': sorted(request),
                  'request_run_root': request['run_root'], 'request_record_root': request['record_root'],
                  'logging': {k: request.get(k) for k in ('project', 'entity', 'state', 'operator', 'seed')}}))
record_root = Path(request['record_root'])
print(json.dumps({'factorial_record_files': [p.name for p in record_root.iterdir() if p.is_file()]}))
factory = json.loads((record_root / 'factory.json').read_bytes())
print(json.dumps({'factory_keys': sorted(factory), 'factory_child_keys':
                  {k: sorted(v) for k, v in factory.items() if isinstance(v, dict) and k != 'component_identity'}}))
print(json.dumps({'native_completions_type': type(native['original_completions']).__name__,
                  'native_completion_keys': sorted(native['original_completions']) if isinstance(native['original_completions'], dict)
                      else sorted(native['original_completions'][0])}))
primary_root = Path('/root/embedding-optimizer-v3-experiment') / catalog['payload']['output_root'] / 'dense/verified-v3-adamw-3e-5'
print(json.dumps({'primary_run_root': str(primary_root), 'primary_run_files': [p.name for p in primary_root.iterdir() if p.is_file()]}))
