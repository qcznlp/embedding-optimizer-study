"""Read the genuine existing v3 publication contracts without altering any gate."""
import json
import os
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
REPOSITORY = Path('/root/embedding-optimizer-story-refactor')
TRAINING = Path('/root/embedding-optimizer-primary-v3')

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise ValueError('Publication context inspection is CPU-only')
try:
    from embed_optim.primary_v3_contract import PrimaryV3Contract
    from embed_optim.primary_v3_publication_contract import PublicationContract
    from embed_optim.primary_v3_exact_publication_contract import ExactPublicationContract
    from embed_optim.primary_contract import file_identity
    primary = PrimaryV3Contract.load(REPOSITORY / 'configs/dense_primary_v3_protocol.json', REPOSITORY, TRAINING)
    publication = PublicationContract.load(REPOSITORY / 'configs/dense_primary_v3_publication_protocol.json', primary)
    exact = ExactPublicationContract.load(REPOSITORY / 'configs/dense_primary_v3_exact_publication_protocol.json', primary)
    import torch
    if torch.cuda.is_initialized():
        raise ValueError('CPU context inspection initialized CUDA')
    value = dict(inspected_at_utc=datetime.now(timezone.utc).isoformat(), primary_protocol_sha256=primary.sha256,
        publication_protocol_sha256=publication.sha256, exact_publication_protocol_sha256=exact.sha256,
        actual_primary_class=type(primary).__name__, primary_run_count=len(primary.inputs['runs']),
        original_contracts_loaded=True, model_payloads_read=False, full_gather_called=False,
        manuscript_installed=False, scientific_completion=False, source=file_identity(__file__))
    with (HERE / 'original-context.json').open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True); f.write('\n')
    print(json.dumps(value))
except Exception as error:
    with (HERE / 'original-context.failed.json').open('x') as f:
        json.dump(dict(exception_type=type(error).__name__, message=str(error), gates_changed=False), f, indent=2)
    raise
