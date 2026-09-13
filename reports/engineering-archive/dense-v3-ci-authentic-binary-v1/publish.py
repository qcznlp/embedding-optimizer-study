"""Append a verified, content-addressed positive runtime bundle to the existing dataset."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

work = Path(__file__).resolve().parent
out = work / 'publication'
raw = (out / 'manifest.json').read_bytes()
manifest = json.loads(raw)
digest = hashlib.sha256(raw).hexdigest()
repo = 'qcz/embedding-optimizer-study-analysis-artifacts'
prefix = 'runtime/flash-attention-2.7.4.post1-cu129-cp312-v1/' + digest
api = HfApi()
assert api.whoami()['name'] == 'qcz'
info = api.repo_info(repo, repo_type='dataset')
assert info.private is False
files = api.list_repo_files(repo, repo_type='dataset', revision=info.sha)
assert not any(p == prefix or p.startswith(prefix + '/') for p in files), 'existing target must not be overwritten'
assert {p.name for p in out.iterdir()} == set(manifest['files']) | {'manifest.json'}
for name, expected in manifest['files'].items():
    path = out / name
    assert path.stat().st_size == expected['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256']
result = api.create_commit(repo_id=repo, repo_type='dataset', parent_commit=info.sha,
    operations=[CommitOperationAdd(path_in_repo=prefix + '/' + p.name, path_or_fileobj=p) for p in sorted(out.iterdir())],
    commit_message='Add authentic source-built FlashAttention runtime for CPU CI')
receipt = {'repo_id': repo, 'repo_type': 'dataset', 'revision': result.oid,
           'previous_revision': info.sha, 'prefix': prefix, 'manifest_sha256': digest,
           'files': len(manifest['files']) + 1, 'deleted_files': 0}
with (work / 'published.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
print(json.dumps(receipt), flush=True)
