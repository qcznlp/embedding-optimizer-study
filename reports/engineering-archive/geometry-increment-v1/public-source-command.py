import json
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import HfApi
root=Path('/root/embedding-optimizer-study')
preflight=json.loads(Path('/root/embedding-optimizer-story-refactor/reports/experiment-integrity/primary-geometry-eight-run-preflight.json').read_bytes())
api=HfApi(token=False)
rows=[]
for run in ['padded-muon-1e-4','padded-muon-3e-4','padded-muon-1e-3','padded-muon-3e-3']:
 receipt=json.loads((root/'reports/dense-no-packing/checkpoint-backup'/f'{run}.json').read_bytes())
 paths=[f"{receipt['remote_prefix']}/checkpoint-{step}/model.safetensors" for step in [782,1563,2345,3126,3907]]
 entries=api.get_paths_info(receipt['repo_id'],paths=paths,revision=receipt['commit_oid'],repo_type='model')
 if {e.path for e in entries} != set(paths): raise ValueError('Public source weights are missing')
 for entry in entries:
  suffix=Path(entry.path).relative_to(receipt['remote_prefix'])
  local=str(root/'outputs/dense-no-packing-v1/dense'/run/suffix)
  source=preflight['model_input_inventory'][local]
  if entry.lfs is None or entry.lfs.sha256 != source['sha256'] or entry.size != source['size']: raise ValueError('Public checkpoint differs from geometry source')
  rows.append({'run_id':run,'path':entry.path,'revision':receipt['commit_oid'],'size':entry.size,'sha256':entry.lfs.sha256})
status=[]
for repo,kind in [('qcz/embedding-optimizer-study-checkpoints','model'),('qcz/embedding-optimizer-study-analysis-artifacts','dataset')]:
 info=api.repo_info(repo,repo_type=kind)
 if info.private: raise ValueError('Repository is not public')
 status.append({'repo':repo,'repo_type':kind,'private':info.private,'revision':info.sha})
print(json.dumps({'observed_at_utc':datetime.now(timezone.utc).isoformat(),'access':'anonymous_read_only','repositories':status,'public_checkpoint_sources':rows,'public_weights_verified':len(rows),'new_geometry_files':len(preflight['inventory']['missing']),'new_geometry_bytes':preflight['inventory']['local_bytes']-preflight['inventory']['remote_bytes'],'remote_upload_performed':False},indent=2))
