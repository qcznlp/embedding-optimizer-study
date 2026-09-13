"""Read-only source-integration boundary while independent GPU checks run."""
import json
import os
from pathlib import Path

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise ValueError('CPU-only source-context read')
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_publication_contract import PublicationContract
from embed_optim.primary_v3_exact_publication_contract import ExactPublicationContract
from embed_optim.primary_contract import file_identity
from embed_optim import factorial_v3_run_contract as factorial
import torch

root = Path('/root/embedding-optimizer-story-refactor')
out = Path(__file__).parent/'source-context.json'
primary = PrimaryV3Contract.load(root/'configs/dense_primary_v3_protocol.json',root,root)
publication = PublicationContract.load(root/'configs/dense_primary_v3_publication_protocol.json',primary)
exact = ExactPublicationContract.load(root/'configs/dense_primary_v3_exact_publication_protocol.json',primary)
value = {'repository_and_primary_source_are_same_checkout':True,
    'primary':primary.sha256,'publication':publication.sha256,'exact_publication':exact.sha256,
    'source':file_identity(__file__),'model_or_dataset_loaded':False,
    'full_native_gather_performed':False,'source_release':False}
try:
    factorial.source_identity(root,root)
except ValueError as error:
    value['factorial_old_source_rejection'] = {'exception':type(error).__name__,'message':str(error)}
else:
    raise ValueError('Historical factorial rejection unexpectedly disappeared')
if torch.cuda.is_initialized():
    raise ValueError('Source context initialized CUDA')
with out.open('x') as stream:
    json.dump(value,stream,indent=2,sort_keys=True)
print(json.dumps(value),flush=True)
