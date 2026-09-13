"""Preserve completed synthetic raw branches; no numerical/primary acceptance."""
from pathlib import Path
from embed_optim import reconstruction_files as files
from embed_optim.primary_contract import file_identity, read_json, verify_file
from embed_optim.primary_v3_validation_io import write_new

work = Path(__file__).parent
failure = work / 'failed.json'
verify_file(failure, {'bytes': failure.stat().st_size, 'sha256': 'ea3e6b17bab4fb04bad7fb1bca7585d71c0ca203dce7679f9368c6d431b6ce10'})
value = read_json(failure)
assert value['passed'] is False
assert 'different primary identity' in value['traceback']
copies = []
for row in value['sources']:
    relative = Path(row['path']).relative_to('/root/embedding-optimizer-story-refactor')
    copied = work / 'source' / relative
    verify_file(copied, row)
    copies.append({'path': str(copied), **file_identity(copied)})
producer = work / 'fixture/producer'
parent = work / 'fixture/outcome-input/archive'
original = files.inspect(parent, file_identity(parent / 'manifest.json')['sha256'])
selected = {}
for role in ('validation', 'beir', 'outcomes', 'geometry', 'vectors', 'features'):
    for name in files.inventory(producer / role):
        path = producer / role / name
        files.select(selected, role, name, path, file_identity(path))
metadata = {
    'scope': 'engineering_complete_synthetic_joint_raw_components',
    'outcome_parent_metadata': original['metadata'],
    'initial_failed_attempt': {'path': str(failure), **file_identity(failure)},
    'initial_source_copies': copies,
    'all_primary_admissions_and_measurements_simulated': True,
    'one_complete_run_population': True, 'vector_states_computed': 61,
    'raw_vectors_are_full_768': True,
    'inference_and_publication_accepted': False,
    'scientific_completion': False,
}
result = files.write(work / 'raw-components', selected, metadata)
write_new(work / 'raw-components.json', {
    'scope': metadata['scope'], 'payload': str(work / 'raw-components'),
    **result, 'component_transport_verified': True,
    'joint_numerical_reconstruction_verified': False,
})
print(result, flush=True)
