"""Add final dated handoff copies without changing the frozen waiter or original archive."""
import json
from pathlib import Path
import re
import outcomes as o
from archive_handoff import binding, DEST

t = o.transport()
manifest_path = DEST / 'copy-inventory.json'
o.need(o.sha(manifest_path) == 'a80491092a436d025c3a3ef4e5ea47d554f71964f262871292685bef39d26e12', 'Original inventory changed')
manifest = json.loads(manifest_path.read_bytes())
for item in manifest['files']:
    o.need(binding(DEST / item['path']) == {k: item[k] for k in ('bytes', 'sha256')}, 'Original archive copy changed')
o.need(o.sha(o.STORY / 'paper/main.tex') == '45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e', 'Authoritative paper changed')
o.authenticate(t, 'a27fbbd07c3b6bbe1b8d2675be7c4ad58ec13c6ca74d8ed66c97d993aa7c4183',
               '4b48d60de683ea3af68808987a29499f32369dabdb5cf35356ba5e5e269c0fba')
files = [(o.STORY / name, 'after/' + name) for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md')]
files.extend((o.HERE / name, 'observations/' + name) for name in
             ('evaluation-latest.json', 'summary-latest.json', 'document-latest.json', 'combined-latest.json',
              'resume-latest.json', 'observation-latest.json'))
files.append((Path(__file__), 'finish_handoff.py'))
root = DEST / 'final-observations'
root.mkdir(exist_ok=False)
rows = []
for source, name in files:
    data = source.read_bytes()
    o.need(not re.search(rb'wandb_v1_[A-Za-z0-9_-]{40,}|\bhf_[A-Za-z0-9]{30,}|\bgh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', data),
           'Credential-shaped copy refused')
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(data)
    o.need(binding(source) == binding(target), 'Final copy differs')
    rows.append({'source': str(source), 'path': target.relative_to(DEST).as_posix(), **binding(target)})
value = {'scope': 'actual-live-outcome-backup-handoff', 'observed_at_utc': t.stamp(), 'goal_turn': 'PROGRESS',
         'previous_goal_turn': 'PROGRESS', 'files': rows,
         'original_manifest_sha256': o.sha(manifest_path), 'original_copies_verified': len(manifest['files']),
         'original_pinned_sources_verified': len(o.PINS),
         'tool_results': {'synthetic_tests': {'terminal': '6764d4', 'exit_code': 0, 'cases': 14},
                          'actual_partial_preflight': {'terminal': '5eb816', 'exit_code': 0},
                          'preparation': {'terminal': 'd65acc', 'exit_code': 0},
                          'launch': {'terminal': '72e7ae', 'session': 35024},
                          'last_exact_backup_observation': {'terminal': 'a334d8', 'exit_code': 0},
                          'archive': {'terminal': '9fe722', 'exit_code': 0},
                          'documentation_diff_check': {'terminal': 'dd8d40', 'exit_code': 0}},
         'exact_waiter': {'pid': 875351, 'start_ticks': 322559189, 'session': 35024,
                          'observed_at_utc': '2026-09-12T22:59:48.453841+00:00', 'state': 'S'},
         'actual_upload_completed': False, 'authoritative_paper_changed': False,
         'source_release': False, 'full_goal_complete': False}
path = DEST / 'final-handoff.json'
t.write_new(path, value)
print(json.dumps({'final_handoff': binding(path), 'additional_copies': len(rows), 'original_copies_verified': len(manifest['files']),
                  'goal_turn': 'PROGRESS', 'actual_upload_completed': False, 'full_goal_complete': False}))
