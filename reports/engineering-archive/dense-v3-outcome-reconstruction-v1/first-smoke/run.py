"""First complete synthetic outcome-reader smoke; never primary evidence."""
from pathlib import Path

from embed_optim.primary_contract import file_identity, require_same
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_dimension_inference import FunctionalInferenceContract
from embed_optim.primary_v3_outcome_reconstruction import reconstruct
from embed_optim.primary_v3_reconstruction_inputs import ReconstructionInput
from embed_optim.primary_v3_reconstruction_sources import INFERENCE_PROTOCOL
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.outcome_reconstruction_fixture import make

root = Path('/root/embedding-optimizer-story-refactor')
work = Path(__file__).parent
paths = [root / name for name in (
    'src/embed_optim/primary_v3_outcome_primitives.py',
    'src/embed_optim/primary_v3_outcome_reconstruction.py',
    'scripts/outcome_reconstruction_fixture.py',
)] + [Path(__file__)]
sources = [{'path': str(p), **file_identity(p)} for p in paths]
before = handoff()
primary = PrimaryV3Contract.load(root / 'configs/dense_primary_v3_protocol.json', root,
    root / 'reports/engineering-archive/dense-full-identity-v1/candidate-source')
contract = FunctionalInferenceContract.load(root / INFERENCE_PROTOCOL, primary)
payload, anchor = make(work / 'fixture', contract)
write_new(work / 'fixture.json', {
    'scope': 'engineering_synthetic_outcome_fixture', 'payload': str(payload),
    'manifest_sha256': anchor, 'upstream_admission_simulated': True,
    'scientific_completion': False, 'sources': sources,
})
print({'fixture_ready': True, 'upstream_admission_simulated': True}, flush=True)
result = reconstruct(ReconstructionInput.load(payload, anchor), work / 'recomputed')
require_same(sources, [{'path': str(p), **file_identity(p)} for p in paths])
require_same(handoff(), before)
write_new(work / 'result.json', {
    'scope': 'engineering_synthetic_outcome_reader_smoke', 'passed': True,
    'sources': sources, 'reconstruction': result,
    'upstream_admission_simulated': True, 'scientific_completion': False,
    'post_execution_dispatchers': handoff(),
})
print({'passed': True, 'scientific_completion': False, **file_identity(work / 'result.json')}, flush=True)
