"""Preserve the queued complete-paper replay handoff without changing live code."""
import json
import re
from pathlib import Path
import combined as c

report = c.STORY / 'reports/engineering-archive/dense-v3-combined-paper-replay-handoff-v1'
c.bound(c.HERE / 'combined.py', '1a9c50dba2f58cc68b110749ccbc40d427445d53f0dc6c994b8f78c02096d566')
c.bound(c.HERE / 'authorization.json', 'c66a9e434789d9d428e7fbe273df4db2fc2f383aabfb2145e97d8473dfcb7d41')
c.parent_inputs()
report.mkdir(exist_ok=False)
names = ['PLAN.md', 'combined.py', 'test_combined.py', 'observe.py', 'tests.json', 'tests-final.json',
         'authorization.json', 'observed-initial.json', 'evaluation-final.json', 'summary-final.json',
         'document-final.json', 'resume-final.json', 'combined-final.json', 'tool-responses.json',
         'run/started.json', 'archive.py', 'first-passing-controls/combined.py',
         'first-passing-controls/test_combined.py', 'first-passing-controls/tests.json']
files = {}
for name in names:
    source = c.HERE / name
    files[name] = {**c.copy(source, report / name, c.identity(source)), 'origin': str(source)}
for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md', 'docs/paper-results-reproduction.md'):
    source = c.STORY / name
    files['before/' + name] = {**c.copy(source, report / 'before' / name, c.identity(source)), 'origin': str(source)}
patterns = [rb'wandb_v1_[A-Za-z0-9_-]{40,}', rb'\bhf_[A-Za-z0-9]{30,}',
            rb'\bgh[pousr]_[A-Za-z0-9]{30,}', rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
for name in files:
    c.need(not any(re.search(p, (report / name).read_bytes()) for p in patterns), 'Credential-shaped content')
c.write(report / 'archive-manifest.json', {'scope': 'queued-complete-paper-portability-handoff-preservation',
        'archived_at_utc': c.now(), 'files': files, 'copied_files': len(files),
        'copied_bytes': sum(v['bytes'] for v in files.values()), 'copy_hashes_verified': True,
        'credential_shaped_findings': [], 'goal_turn_classification': 'PROGRESS',
        'actual_full_paper_replay_complete': False, 'full_goal_complete': False})
print(json.dumps({'report': str(report), 'manifest': c.identity(report / 'archive-manifest.json'),
                   'copied_files': len(files), 'full_goal_complete': False}))
