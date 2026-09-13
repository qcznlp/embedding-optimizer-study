"""Preserve current document handoff and original complete probe evidence."""
from pathlib import Path
import author as a

target = a.STORY / 'reports/engineering-archive/dense-v3-complete-document-handoff-v1'
target.mkdir(parents=True, exist_ok=False)
files = {}


def save(source, name, expected=None):
    value = a.identity(source) if expected is None else a.bound(source, expected)
    a.copy_file(source, target / name, value)
    files[name] = {'origin': str(source), **value}


for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md'):
    save(a.STORY / name, 'before/' + name)
for path in sorted(a.HERE.rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts:
        save(path, 'actual/' + path.relative_to(a.HERE).as_posix())
probe = Path('/tmp/dense-v3-factorial-probe.ryorjXJK')
source_sha = 'eb4fc27738adaecd9614ba5822255a5dc37de242afabfa53ff8cd8a6c8c73334'
auth_sha = '00d6b42f922b4428352d5b9fc5e5fa7e0436a4aa20ee0c693e57b9f645398ded'
save(probe / 'probe.py', 'complete-probes/probe.py', source_sha)
save(probe / 'authorization.json', 'complete-probes/authorization.json', auth_sha)
states, actual_zero_exits = set(), 0
for pool in ('a', 'b'):
    root = probe / 'run' / ('pool-' + pool)
    complete = a.read(root / 'completed.json')
    a.need(complete['source_sha256'] == source_sha and complete['authorization_sha256'] == auth_sha
           and len(complete['checkpoints']) == 30 and not (root / 'failed.json').exists(),
           'Incomplete original probe pool')
    for name in ('started.json', 'completed.json'):
        save(root / name, 'complete-probes/pool-' + pool + '/' + name)
    for row in complete['checkpoints']:
        run, step = row['run_id'], row['step']
        a.need(step in (79, 157, 235, 313, 391) and (run, step) not in states, 'Duplicate probe state')
        states.add((run, step))
        prefix = root / f'{run}-checkpoint-{step}'
        a.bound(prefix.with_suffix('.verified.json'), row['verified'])
        exited = a.read(prefix.with_suffix('.exited.json'))
        a.need(exited['exit_code'] == 0, 'Original actual probe worker did not exit zero')
        actual_zero_exits += 1
        for suffix in ('.request.json', '.started.json', '.exited.json', '.verified.json'):
            source = prefix.with_suffix(suffix)
            save(source, 'complete-probes/pool-' + pool + '/' + source.name)
a.need(len(states) == actual_zero_exits == 60 and len({run for run, step in states}) == 12,
       'Incomplete original probe coverage')
for name, value in files.items():
    a.bound(value['origin'], value)
    a.bound(target / name, value)
a.write(target / 'archive-manifest.json', {'archived_at_utc': a.now(), 'files': files,
    'copied_files': len(files), 'copied_bytes': sum(v['bytes'] for v in files.values()),
    'source_and_copy_hashes_verified': True, 'complete_checkpoint_probes': 60,
    'original_actual_exit_zero_probe_workers': 60,
    'synthetic_layout_controls_separately_named_not_scientific_results': True,
    'native_primary_reused_not_recomputed': True, 'document_waiter_launched': True,
    'real_factorial_complete': False, 'authoritative_manuscript_installed': False,
    'portable_full_paper_complete': False, 'committed_source_release': False, 'full_goal_complete': False})
print({'archive': str(target), 'files': len(files), 'bytes': sum(v['bytes'] for v in files.values()),
       'manifest': a.identity(target / 'archive-manifest.json')}, flush=True)
