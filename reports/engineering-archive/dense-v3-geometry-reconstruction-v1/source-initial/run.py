import json
from pathlib import Path

from embed_optim.primary_contract import file_identity, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_dimension_inference import FunctionalInferenceContract
from embed_optim.primary_v3_geometry_reconstruction import reconstruct
from embed_optim.primary_v3_reconstruction_inputs import ReconstructionInput
from embed_optim.primary_v3_reconstruction_sources import INFERENCE_PROTOCOL
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.geometry_reconstruction_fixture import make

root = Path('/root/embedding-optimizer-story-refactor')
work = Path(__file__).resolve().parent
names = ('src/embed_optim/primary_v3_geometry_primitives.py',
         'src/embed_optim/primary_v3_geometry_reconstruction.py',
         'scripts/geometry_reconstruction_fixture.py')
sources = [{'path': str(root / n), **file_identity(root / n)} for n in names]
sources.append({'path': str(Path(__file__)), **file_identity(__file__)})
before = handoff()
primary = PrimaryV3Contract.load(root / 'configs/dense_primary_v3_protocol.json', root,
    root / 'reports/engineering-archive/dense-full-identity-v1/candidate-source')
contract = FunctionalInferenceContract.load(root / INFERENCE_PROTOCOL, primary)
archive, anchor = make(work / 'fixture', contract, progress=lambda x: print(x, flush=True))
write_new(work / 'fixture.json', {'payload': str(archive), 'manifest_sha256': anchor,
    'upstream_primary_admission_simulated': True, 'scientific_completion': False})
print({'fixture_ready': True, 'anchor': anchor}, flush=True)
result = reconstruct(ReconstructionInput.load(archive, anchor), work / 'recomputed')
for row in sources:
    verify_file(row['path'], row)
require_same(handoff(), before)
write_new(work / 'result.json', {'scope': 'engineering_complete_geometry_same_process_smoke',
    'passed': True, 'sources': sources, 'fixture': file_identity(work / 'fixture.json'),
    'reconstruction': result, 'post_execution_dispatchers': before,
    'cold_reconstruction_verified': False, 'scientific_completion': False})
print(json.dumps({'passed': True, **file_identity(work / 'result.json')}), flush=True)
