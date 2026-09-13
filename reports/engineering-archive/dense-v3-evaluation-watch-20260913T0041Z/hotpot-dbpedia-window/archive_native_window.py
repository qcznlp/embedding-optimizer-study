"""Archive native task results and separately version-bound shared metadata.

This is an observation archive, not a scientific collector or admission gate.
Original evaluators, numerical consumers and their authorities are unchanged.
"""
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/root/embedding-optimizer-story-refactor')
MONITOR = Path('/tmp/dense-v3-evaluation-monitor.tljylP9V')
ENTRY = Path('/tmp/dense-v3-factorial-evaluation.IOV7MK93')
RESULTS = Path('/root/embedding-optimizer-v3-experiment/evaluations/dense-v3-state-operator-v1')
REPORT = ROOT / 'reports/engineering-archive/dense-v3-evaluation-watch-20260913T0041Z'
META_NAMES = {'model_meta.json', 'run_settings.jsonl'}


def now():
    return datetime.now(timezone.utc).isoformat()


def payload(path):
    assert path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), str(path)
    return path.read_bytes()


def identity(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def read_json(path):
    return json.loads(payload(path))


before_name, after_name, window, *extra_names = sys.argv[1:]
assert re.fullmatch(r'[a-z0-9-]+', window)
assert all(Path(n).name == n and n.endswith('.json') for n in (before_name, after_name, *extra_names))
OUT = REPORT / window
assert not OUT.exists(), str(OUT)
OUT.mkdir()
copies = []
copied = {}


def copy(source, relative, kind, expected=None, captured=None):
    data = payload(source) if captured is None else captured
    actual = identity(data)
    if expected is not None:
        assert actual == {k: expected[k] for k in ('bytes', 'sha256')}, str(source)
    relative = str(relative)
    target = OUT / relative
    if relative in copied:
        assert copied[relative]['source'] == str(source)
        assert identity(payload(target)) == actual
        return relative
    assert not target.exists(), str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as handle:
        handle.write(data)
    assert identity(payload(target)) == actual
    row = {'source': str(source), 'path': relative, 'kind': kind, **actual}
    copies.append(row)
    copied[relative] = row
    return relative


assert identity(payload(ENTRY / 'evaluate.py'))['sha256'] == '45577537257e1dff2b7c3a26ee12f7564fbb476daa256e7b0f09300305b68cba'
assert identity(payload(ENTRY / 'authorization.json'))['sha256'] == '3a1ae32fdde63abf386603208e413ea11bf0c169eb1923f9dd7e1da529ffa29c'
auth = read_json(ENTRY / 'authorization.json')
before = read_json(MONITOR / before_name)
after = read_json(MONITOR / after_name)


def completed(observation):
    rows = set()
    for pool, state in observation['pools'].items():
        assert state['failure_receipt'] is False
        for worker in state['task_workers']:
            if worker['observation'].get('terminal'):
                assert worker['observation']['exit_code'] == 0
                assert worker['job'] in auth['jobs'][pool]
                rows.add((pool, *worker['job']))
    assert len(rows) == sum(p['completed_tasks'] for p in observation['pools'].values())
    return rows


old = completed(before)
current = completed(after)
assert old < current and len(current) <= 168
new = sorted(current - old)


def native(pool, job):
    run, step, task = job
    assert step == 391 and job in auth['jobs'][pool]
    base = ENTRY / 'run' / ('pool-' + pool) / 'jobs' / (run + '-391-' + task)
    paths = {s: base.with_suffix(s) for s in ('.started.json', '.exited.json', '.task-verified.json')}
    records = {s: read_json(p) for s, p in paths.items()}
    start, end, verified = (records[s] for s in ('.started.json', '.exited.json', '.task-verified.json'))
    assert start['job'] == end['job'] == job and start['pid'] == end['pid'] and end['exit_code'] == 0
    assert verified['task'] == task and len(verified['files']) == 3
    for item in verified['files']:
        p = Path(item['path'])
        assert p.resolve().is_relative_to(RESULTS / run / 'checkpoint-391')
        assert p.suffix in ('.json', '.jsonl')
    return paths, records


def save_native(pool, paths):
    return {s: copy(p, Path('native') / pool / p.name, 'native_receipt') for s, p in paths.items()}


new_tasks = []
run_references = {}
for pool, run, step, task in new:
    job = [run, step, task]
    paths, records = native(pool, job)
    saved = save_native(pool, paths)
    verified = records['.task-verified.json']
    raw = [f for f in verified['files'] if Path(f['path']).name not in META_NAMES]
    assert len(raw) == 1 and Path(raw[0]['path']).suffix == '.json'
    item = raw[0]
    result_path = copy(Path(item['path']), Path('results') / run / task / Path(item['path']).name, 'raw_task_result', item)
    end = records['.exited.json']
    new_tasks.append({'pool': pool, 'job': job, 'pid': end['pid'], 'completed_at_utc': end['completed_at_utc'], 'elapsed_seconds': end['elapsed_seconds'], 'exit_code': 0, 'native_records': saved, 'raw_result': result_path})
    meta = {Path(f['path']).name: f for f in verified['files'] if Path(f['path']).name in META_NAMES}
    assert set(meta) == META_NAMES
    run_references[(pool, run)] = meta

# Shared MTEB metadata is a per-run mutable artifact, not one immutable file
# per task. Authenticate each captured pair against an original native receipt;
# retain all earlier receipts without claiming their metadata bytes are current.
metadata_snapshots = []
for (pool, run), reference in sorted(run_references.items()):
    captured_at = now()
    captured = {name: payload(Path(item['path'])) for name, item in reference.items()}
    identities = {name: identity(data) for name, data in captured.items()}
    matches = []
    for job in auth['jobs'][pool]:
        if job[0] != run:
            continue
        base = ENTRY / 'run' / ('pool-' + pool) / 'jobs' / (run + '-391-' + job[2])
        if not base.with_suffix('.task-verified.json').is_file():
            continue
        paths, records = native(pool, job)
        meta = {Path(f['path']).name: f for f in records['.task-verified.json']['files'] if Path(f['path']).name in META_NAMES}
        if set(meta) != META_NAMES:
            continue
        if all(meta[name]['path'] == reference[name]['path'] and identities[name] == {k: meta[name][k] for k in ('bytes', 'sha256')} for name in META_NAMES):
            matches.append((records['.exited.json']['completed_at_utc'], job, paths, meta))
    assert matches, 'No original native receipt authenticates captured metadata: ' + run
    owner_time, owner_job, owner_paths, owner_meta = max(matches, key=lambda item: item[0])
    saved_owner = save_native(pool, owner_paths)
    saved_meta = {name: copy(Path(reference[name]['path']), Path('metadata') / run / name, 'shared_metadata_snapshot', owner_meta[name], captured[name]) for name in sorted(META_NAMES)}
    metadata_snapshots.append({
        'pool': pool, 'run': run, 'captured_at_utc': captured_at,
        'bound_to_native_job': owner_job, 'owner_completed_at_utc': owner_time,
        'owner_in_task_snapshot': (pool, *owner_job) in current,
        'owner_native_records': saved_owner, 'files': saved_meta,
        'not_a_historical_metadata_copy_for_every_task': True,
    })

old_active = {(pool, *w['job']) for pool, state in before['pools'].items() for w in state['task_workers'] if not w['observation'].get('terminal')}
active = []
replacement_starts = []
for pool, state in after['pools'].items():
    assert state['coordinator']['state'] in ('R', 'S') and state['coordinator']['command_matches'] is True
    for worker in state['task_workers']:
        if worker['observation'].get('terminal'):
            continue
        observation = worker['observation']
        assert observation['state'] in ('R', 'S') and observation['command_matches'] is True
        job = worker['job']
        assert job in auth['jobs'][pool]
        active.append({'pool': pool, 'job': job, **observation})
        if (pool, *job) in old_active:
            continue
        run, step, task = job
        path = ENTRY / 'run' / ('pool-' + pool) / 'jobs' / (run + '-391-' + task + '.started.json')
        start = read_json(path)
        assert start['job'] == job and start['pid'] == observation['pid'] and start['start_ticks'] == observation['start_ticks']
        saved = copy(path, Path('native') / pool / path.name, 'native_receipt')
        replacement_starts.append({'pool': pool, 'job': job, 'pid': start['pid'], 'start_ticks': start['start_ticks'], 'path': saved})

for name in dict.fromkeys((before_name, after_name, *extra_names)):
    copy(MONITOR / name, Path('observations') / name, 'observation')
for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md'):
    copy(ROOT / name, Path('before') / name, 'before_document')
copy(REPORT / 'README.md', Path('before/observation-report.md'), 'before_document')
copy(Path(__file__), Path('archive_native_window.py'), 'archive_helper')
paper_hashes = {
    'paper/main.tex': '45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e',
    'paper/references.bib': 'fca6bf5762a2cecab795a1a60c1338ab4eb999acb5d78c437b9df0e5f3553218',
    'reports/engineering-archive/dense-v3-current-publication-consumer-v1/actual/paper-preview-v2/main.tex': '36a6b3509d1ad71a0c9609832e37869bde97379daf34bf69832af0f68f7f95fe',
}
for path, expected in paper_hashes.items():
    assert identity(payload(ROOT / path))['sha256'] == expected, path
for row in copies:
    expected = {k: row[k] for k in ('bytes', 'sha256')}
    assert identity(payload(OUT / row['path'])) == expected
    if row['kind'] != 'shared_metadata_snapshot':
        assert identity(payload(Path(row['source']))) == expected
for snapshot in metadata_snapshots:
    verified = read_json(OUT / snapshot['owner_native_records']['.task-verified.json'])
    expected = {Path(f['path']).name: f for f in verified['files']}
    for name, path in snapshot['files'].items():
        assert identity(payload(OUT / path)) == {k: expected[name][k] for k in ('bytes', 'sha256')}

record = {
    'scope': 'Original task/raw-result authentication and separately native-bound shared-metadata snapshots; no score recomputation or full scientific admission.',
    'observed_at_utc': after['observed_at_utc'], 'archive_completed_at_utc': now(),
    'before_completed_tasks': len(old), 'completed_tasks': len(current),
    'new_completed_tasks': new_tasks, 'completed_task_counts': dict(sorted(Counter(row[3] for row in current).items())),
    'metadata_snapshots': metadata_snapshots, 'active_workers': active,
    'current_replacement_starts': replacement_starts,
    'completed_intermediate_worker_starts_are_in_their_native_record_triples': True,
    'copies': copies, 'paper_hashes': paper_hashes,
    'scientific_inference_generated': False, 'full_goal_complete': False,
}
with (OUT / 'readback.json').open('x') as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write('\n')
print(json.dumps({
    'completed_tasks': len(current), 'new_tasks': len(new_tasks),
    'completed_task_counts': record['completed_task_counts'],
    'new_task_timings': [{k: row[k] for k in ('job', 'completed_at_utc', 'elapsed_seconds')} for row in new_tasks],
    'current_metadata_snapshots': len(metadata_snapshots),
    'metadata_owners_beyond_task_snapshot': sum(not row['owner_in_task_snapshot'] for row in metadata_snapshots),
    'active_task_counts': dict(Counter(row['job'][2] for row in active)),
    'copied_files': len(copies), 'readback': identity(payload(OUT / 'readback.json')),
    'scientific_inference_generated': False,
}))
