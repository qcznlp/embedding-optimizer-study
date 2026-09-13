import hashlib,json
from pathlib import Path
from datetime import UTC,datetime
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile
from embed_optim.incremental_checkpoint_backup import local_checkpoint_inventory,stat_signature,compare_checkpoint_inventories
root=Path('/root/embedding-optimizer-study'); api=HfApi(); records=[]
for rate in ('1e-4','3e-4'):
 path=root/f'reports/dense-no-packing/incremental-checkpoint-backup/padded-normuon-{rate}-checkpoint-3126.json'
 raw=path.read_bytes(); receipt=json.loads(raw)
 assert receipt['status']=='complete' and receipt['checkpoint_step']==3126 and receipt['scientific_completion'] is False
 local_root=root/receipt['local_root']; before=stat_signature(local_root)
 local=local_checkpoint_inventory(local_root); remote={}
 for e in api.list_repo_tree(receipt['repo_id'],repo_type='model',revision=receipt['commit_oid'],path_in_repo=receipt['remote_prefix'],recursive=True,expand=True):
  if isinstance(e,RepoFile):
   remote[str(Path(e.path).relative_to(receipt['remote_prefix']))]={'size':e.size,'digest_kind':'sha256' if e.lfs else 'git_blob_sha1','digest':e.lfs.sha256 if e.lfs else e.blob_id}
 assert stat_signature(local_root)==before and path.read_bytes()==raw
 comparison=compare_checkpoint_inventories(local,remote)
 assert comparison['complete'],comparison
 records.append({'run_id':receipt['run_id'],'checkpoint_step':3126,'repo_id':receipt['repo_id'],'revision':receipt['commit_oid'],'prefix':receipt['remote_prefix'],'original_receipt_sha256':hashlib.sha256(raw).hexdigest(),'local_inventory':local,'remote_inventory':remote,'comparison':comparison})
print(json.dumps({'schema_version':1,'scope':'independent_immutable_intermediate_checkpoint_digest_audit','observed_at_utc':datetime.now(UTC).isoformat(),'complete_for_audited_checkpoints':True,'scientific_completion':False,'files':sum(r['comparison']['local_files'] for r in records),'bytes':sum(r['comparison']['local_bytes'] for r in records),'mutations':False,'records':records},indent=2))
