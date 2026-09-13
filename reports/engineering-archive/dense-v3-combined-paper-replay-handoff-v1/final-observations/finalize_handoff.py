"""Record final handoff documentation; preserve the original queued source/archive."""
import json
from pathlib import Path
import re
import combined as c

report = c.STORY / 'reports/engineering-archive/dense-v3-combined-paper-replay-handoff-v1'
manifest_path = report / 'archive-manifest.json'
old = c.read(manifest_path, '052ab737ca2458ca6d29e1fc5fe7c3e939a89120d13e93c60bcb3de3065b7a74')
for name, value in old['files'].items():
    c.bound(report / name, value)
c.parent_inputs()
c.bound(c.STORY / 'paper/main.tex', '45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e')
c.bound(c.HERE / 'combined.py', '1a9c50dba2f58cc68b110749ccbc40d427445d53f0dc6c994b8f78c02096d566')
c.bound(c.HERE / 'authorization.json', 'c66a9e434789d9d428e7fbe273df4db2fc2f383aabfb2145e97d8473dfcb7d41')
final = report / 'final-observations'
final.mkdir(exist_ok=False)
files = {}
for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md', 'docs/paper-results-reproduction.md'):
    source = c.STORY / name
    target = final / 'after' / name
    files[target.relative_to(report).as_posix()] = c.copy(source, target, c.identity(source))
files['final-observations/finalize_handoff.py'] = c.copy(Path(__file__), final / 'finalize_handoff.py', c.identity(Path(__file__)))
files['README.md'] = c.identity(report / 'README.md')
patterns = [rb'wandb_v1_[A-Za-z0-9_-]{40,}', rb'\bhf_[A-Za-z0-9]{30,}',
            rb'\bgh[pousr]_[A-Za-z0-9]{30,}', rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
for name in files:
    c.need(not any(re.search(p, (report / name).read_bytes()) for p in patterns), 'Credential-shaped content')
c.write(report / 'final-handoff.json', {'scope': 'complete-paper-replay-waiter-final-documentation',
        'observed_at_utc': c.now(), 'original_archive': c.identity(manifest_path),
        'additional_files': files, 'original_archive_unchanged': True,
        'actual_combined_replay_complete': False, 'authoritative_manuscript_unchanged': True,
        'new_waiter_session': 56449, 'new_waiter_pid': 865702, 'new_waiter_start_ticks': 322276697,
        'source_release': False, 'full_goal_complete': False, 'goal_turn_classification': 'PROGRESS'})
print(json.dumps({'handoff': c.identity(report / 'final-handoff.json'),
                  'additional_files': len(files), 'full_goal_complete': False}))
