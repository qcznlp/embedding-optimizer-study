"""Read-only current inputs/client check; full outcome selection remains pending."""
import inspect
import json
from pathlib import Path
import outcomes as o
from huggingface_hub import _commit_api, hf_api, HfApi

t = o.transport()
clients = {
    Path(inspect.getsourcefile(_commit_api)): '250ed0e5a5a39383974cab5baae08a964e60f3f40c394c0d13a52e119e1f5f39',
    Path(inspect.getsourcefile(hf_api)): '659636025aa3a7efefa69ca7f16741d8d6cc12f9301beac8c69f9b3d51f81cd4',
}
for path, digest in clients.items():
    o.need(o.sha(path) == digest, 'Original HF client changed')
run = 'factorial-v3-adamw_state-adamw-seed314159'
job = o.EVAL / 'run/pool-a/jobs' / (run + '-391-SciFact')
paths = [path for path in o.PINS if path.suffix == '.json']
paths.extend(o.HERE / name for name in ('related-artifacts.json', 'ARTIFACT_README.md'))
paths.extend(job.with_suffix(suffix) for suffix in ('.started.json', '.exited.json', '.task-verified.json'))
paths.append(o.EVAL / 'run/pool-a/native' / (run + '.json'))
accepted = t.read(job.with_suffix('.task-verified.json'))
for record in accepted['files']:
    path = Path(record['path'])
    paths.append(path)
    # Shared settings remain provisional until the whole fourteen-task run finishes.
    if path.name != 'run_settings.jsonl':
        t.compare_file(path, {k: record[k] for k in ('bytes', 'sha256')})
for path in paths:
    t.scan_text(path)
info = HfApi().repo_info(o.REPO, repo_type='dataset', token=False)
o.need(info.sha == o.PARENT and info.private is False, 'Original public dataset parent changed')
value = {'scope': 'actual-partial-input-and-transport-preflight-not-complete-outcome-admission',
         'observed_at_utc': t.stamp(), 'files_screened': len(paths), 'client_sources_unchanged': {str(k): v for k, v in clients.items()},
         'current_parent_revision': info.sha, 'repository_public': True,
         'full_outcome_selection_available': o.ready(), 'current_shared_settings_provisional': True,
         'source': t.file_identity(o.HERE / 'outcomes.py'), 'source_code_uploaded': False,
         'remote_mutations': False, 'scientific_completion': False}
t.write_new(o.HERE / 'actual-preflight.json', value)
print(json.dumps(value), flush=True)
