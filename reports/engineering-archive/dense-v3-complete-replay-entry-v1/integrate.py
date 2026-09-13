"""Expose the already-completed original closed replay, without recomputing it."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

here = Path(__file__).resolve().parent
root = here.parents[2]
source = Path('/tmp/dense-v3-combined-paper-replay.URO4uk9L/closed')
manifest_sha = '746a545d3e59d86aab8d4ff6579abca3be9884f1ea824a702e4fa28bc85956b7'
entry_sha = '1a9c50dba2f58cc68b110749ccbc40d427445d53f0dc6c994b8f78c02096d566'

def identity(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open('rb') as incoming, dst.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    assert identity(src) == identity(dst)

assert identity(source / 'replay_complete.py')['sha256'] == entry_sha
spec = importlib.util.spec_from_file_location('_unchanged_combined_inventory', source / 'replay_complete.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
manifest = driver.inventory(source, manifest_sha, driver.SCOPE)
destination = here / 'closed'
destination.mkdir()
for name in [*manifest['files'], 'manifest.json']:
    copy(source / name, destination / name)
driver.inventory(source, manifest_sha, driver.SCOPE)
driver.inventory(destination, manifest_sha, driver.SCOPE)
assert identity(destination / 'replay_complete.py')['sha256'] == entry_sha
generated = ['results.tex', 'generated/optimizer-primary.tex',
             'generated/dimension-utilization.tex', 'generated/recipe-sensitivity.tex',
             'generated/state-operator-factorial.tex', 'figures/optimizer-weight-dimension-map.pdf',
             'figures/weight-to-retrieval-map.pdf', 'figures/full-rate-retrieval-trajectories.pdf']
for name in generated:
    assert identity(destination / 'expected-paper' / name) == identity(root / 'paper/current' / name)
actual = Path('/tmp/dense-v3-combined-replay-cwd.iU9436g5/actual')
assert identity(actual / 'complete.json')['sha256'] == '6a0a8a3657d88a20529db8d45d32c31c5b74be783c0abb9d57994b0009d3c275'
for name in ('complete.json', 'io-boundary.json', 'primary/io-boundary.json'):
    copy(actual / name, here / 'original-completed' / name)
for name in ('io-boundary.json', 'primary/io-boundary.json'):
    io = json.loads((here / 'original-completed' / name).read_bytes())
    assert io['failure'] is None and io['producer_reads_refused'] == [] and io['network_refused'] == 0
(here / 'integration.json').write_text(json.dumps({
    'input_manifest': identity(destination / 'manifest.json'), 'entry': identity(destination / 'replay_complete.py'),
    'bound_files': len(manifest['files']), 'bound_bytes': sum(x['bytes'] for x in manifest['files'].values()),
    'original_source_and_copied_inventory_exact': True,
    'generated_current_paper_inputs_identical': generated,
    'actual_completed_replay_reused': identity(actual / 'complete.json'),
    'numerical_recomputation_repeated': False, 'gpu_work': False,
    'physical_second_host': False, 'source_release': False}, indent=2))
print(json.dumps({'copied_files': len(manifest['files']) + 1, 'original_entry_unchanged': True,
                  'reviewed_generated_inputs_equal': len(generated), 'science_repeated': False}))
