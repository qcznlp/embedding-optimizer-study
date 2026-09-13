"""First complete synthetic joint authoring/reconstruction; preserve every attempt."""
import json
import shutil
import time
import traceback
from pathlib import Path

from embed_optim.primary_contract import file_identity, require_same
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_dimension_inference import FunctionalInferenceContract
from embed_optim.primary_v3_joint_reconstruction import reconstruct
from embed_optim.primary_v3_reconstruction_inputs import ReconstructionInput
from embed_optim.primary_v3_reconstruction_sources import INFERENCE_PROTOCOL
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.joint_reconstruction_fixture import make

root = Path('/root/embedding-optimizer-story-refactor')
work = Path(__file__).parent
names = [
    'src/embed_optim/primary_v3_joint_reconstruction.py',
    'scripts/joint_reconstruction_fixture.py',
    'scripts/joint_reconstruction_vectors.py',
    'scripts/joint_reconstruction_geometry.py',
]
sources = [{'path': str(root / n), **file_identity(root / n)} for n in names]
for name in names:
    target = work / 'source' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / name, target)
before = handoff()
started = time.monotonic()
primary = PrimaryV3Contract.load(root / 'configs/dense_primary_v3_protocol.json', root,
    root / 'reports/engineering-archive/dense-full-identity-v1/candidate-source')
contract = FunctionalInferenceContract.load(root / INFERENCE_PROTOCOL, primary)
try:
    archive, anchor = make(work / 'fixture', contract, progress=lambda row: print(row, flush=True))
    write_new(work / 'fixture.json', {
        'archive': str(archive), 'anchor': anchor, 'sources': sources,
        'scope': 'engineering_full_synthetic_joint_fixture', 'scientific_completion': False,
    })
    result = reconstruct(ReconstructionInput.load(archive, anchor), work / 'recomputed',
        progress=lambda row: print(row, flush=True))
    require_same(sources, [{'path': str(root / n), **file_identity(root / n)} for n in names])
    after = handoff()
    write_new(work / 'result.json', {
        'passed': True, 'sources': sources, 'fixture': file_identity(work / 'fixture.json'),
        'receipt': file_identity(work / 'recomputed/reconstruction.json'),
        'elapsed_seconds': time.monotonic() - started, 'before': before, 'after': after,
        'scientific_completion': False,
    })
    print(json.dumps({'passed': True, 'elapsed_seconds': time.monotonic()-started}), flush=True)
except Exception:
    write_new(work / 'failed.json', {
        'passed': False, 'traceback': traceback.format_exc(), 'sources': sources,
        'elapsed_seconds': time.monotonic()-started, 'after': handoff(),
        'scientific_completion': False,
    })
    raise
