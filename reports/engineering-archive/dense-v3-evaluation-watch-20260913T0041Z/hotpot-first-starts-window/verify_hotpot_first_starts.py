"""Archive original task receipts and result bytes; no scientific computation."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path('/root/embedding-optimizer-story-refactor')
MONITOR = Path('/tmp/dense-v3-evaluation-monitor.tljylP9V')
ENTRY = Path('/tmp/dense-v3-factorial-evaluation.IOV7MK93')
RESULTS = Path('/root/embedding-optimizer-v3-experiment/evaluations/dense-v3-state-operator-v1')
REPORT = ROOT / 'reports/engineering-archive/dense-v3-evaluation-watch-20260913T0041Z'
OUT = REPORT / 'hotpot-first-starts-window'


def identity(path):
    assert path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), str(path)
    payload = path.read_bytes()
    return {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


assert not OUT.exists(), str(OUT)
OUT.mkdir()
copies = []


def copy(source, relative, expected=None):
    actual = identity(source)
    if expected is not None:
        assert actual == {key: expected[key] for key in ('bytes', 'sha256')}, str(source)
    target = OUT / relative
    assert not target.exists(), str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert identity(source) == identity(target) == actual
    copies.append({'source': str(source), 'path': str(relative), **actual})


assert identity(ENTRY / 'evaluate.py')['sha256'] == '45577537257e1dff2b7c3a26ee12f7564fbb476daa256e7b0f09300305b68cba'
assert identity(ENTRY / 'authorization.json')['sha256'] == '3a1ae32fdde63abf386603208e413ea11bf0c169eb1923f9dd7e1da529ffa29c'
auth = json.loads((ENTRY / 'authorization.json').read_text())
before = json.loads((MONITOR / 'evaluation-msmarco-first-batch.json').read_text())
after = json.loads((MONITOR / 'evaluation-hotpot-first-starts.json').read_text())


def completed(data):
    rows = set()
    for pool, state in data['pools'].items():
        assert state['failure_receipt'] is False
        for worker in state['task_workers']:
            if worker['observation'].get('terminal'):
                assert worker['observation']['exit_code'] == 0
                rows.add((pool, *worker['job']))
    assert len(rows) == sum(p['completed_tasks'] for p in data['pools'].values())
    return rows


old = completed(before)
current = completed(after)
new = sorted(current - old)
assert len(old) == 30 and len(current) == 32 and len(new) == 2
assert all(row[2:] == (391, 'MSMARCO') and row[1].endswith('seed271828') and row[1].startswith('factorial-v3-adamw_state-') for row in new)
new_tasks = []
for pool, run, step, task in new:
    job = [run, step, task]
    assert job in auth['jobs'][pool]
    base = ENTRY / 'run' / ('pool-' + pool) / 'jobs' / (run + '-391-' + task)
    records = {suffix: json.loads(base.with_suffix(suffix).read_text()) for suffix in ('.started.json', '.exited.json', '.task-verified.json')}
    start, end, verified = (records[s] for s in ('.started.json', '.exited.json', '.task-verified.json'))
    assert start['job'] == end['job'] == job and start['pid'] == end['pid'] and end['exit_code'] == 0
    assert verified['task'] == task and len(verified['files']) == 3
    for suffix in records:
        source = base.with_suffix(suffix)
        copy(source, Path('native') / pool / source.name)
    for item in verified['files']:
        path = Path(item['path'])
        assert path.resolve().is_relative_to(RESULTS / run / 'checkpoint-391')
        assert path.name in ('MSMARCODecontaminated.json', 'model_meta.json', 'run_settings.jsonl')
        copy(path, Path('results') / run / path.name, item)
    new_tasks.append({'pool': pool, 'job': job, 'pid': end['pid'], 'completed_at_utc': end['completed_at_utc'], 'elapsed_seconds': end['elapsed_seconds'], 'exit_code': 0})

old_active = {(pool, *w['job']) for pool, state in before['pools'].items() for w in state['task_workers'] if not w['observation'].get('terminal')}
active = []
new_starts = []
for pool, state in after['pools'].items():
    assert state['coordinator']['state'] == 'S' and state['coordinator']['command_matches'] is True
    for worker in state['task_workers']:
        if worker['observation'].get('terminal'):
            continue
        observation = worker['observation']
        assert observation['state'] == 'R' and observation['command_matches'] is True
        active.append({'pool': pool, 'job': worker['job'], **observation})
        if (pool, *worker['job']) in old_active:
            continue
        run, step, task = worker['job']
        assert step == 391 and task == 'HotpotQA'
        path = ENTRY / 'run' / ('pool-' + pool) / 'jobs' / (run + '-391-HotpotQA.started.json')
        start = json.loads(path.read_text())
        assert start['job'] == worker['job'] and start['pid'] == observation['pid'] and start['start_ticks'] == observation['start_ticks']
        copy(path, Path('native') / pool / path.name)
        new_starts.append({'pool': pool, 'job': worker['job'], 'pid': start['pid'], 'start_ticks': start['start_ticks']})
assert len(active) == 8 and len(new_starts) == 2
assert sum(w['job'][2] == 'MSMARCO' for w in active) == 6
assert sum(w['job'][2] == 'HotpotQA' for w in active) == 2

observations = (
    'evaluation-msmarco-first-batch.json', 'evaluation-hotpot-first-starts.json',
    'summary-hotpot-first-starts.json', 'document-hotpot-first-starts.json', 'combined-hotpot-first-starts.json',
    'resume-hotpot-first-starts.json', 'outcomes-hotpot-first-starts.json',
    'hotpot-first-starts-session-events.json', 'hotpot-first-starts-observer-tools.json',
)
for name in observations:
    copy(MONITOR / name, Path('observations') / name)
for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md'):
    copy(ROOT / name, Path('before') / name)
copy(REPORT / 'README.md', Path('before/observation-report.md'))
paper_hashes = {
    'paper/main.tex': '45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e',
    'paper/references.bib': 'fca6bf5762a2cecab795a1a60c1338ab4eb999acb5d78c437b9df0e5f3553218',
    'reports/engineering-archive/dense-v3-current-publication-consumer-v1/actual/paper-preview-v2/main.tex': '36a6b3509d1ad71a0c9609832e37869bde97379daf34bf69832af0f68f7f95fe',
}
for path, expected in paper_hashes.items():
    assert identity(ROOT / path)['sha256'] == expected, path
for row in copies:
    assert identity(Path(row['source'])) == identity(OUT / row['path']) == {key: row[key] for key in ('bytes', 'sha256')}
record = {
    'scope': 'Original task receipts/current result-byte verification; no score recomputation or scientific inference.',
    'observed_at_utc': after['observed_at_utc'], 'before_completed_tasks': 30, 'completed_tasks': 32,
    'new_completed_tasks': new_tasks, 'new_starts': new_starts, 'active_workers': active,
    'copies': copies, 'paper_hashes': paper_hashes, 'scientific_inference_generated': False, 'full_goal_complete': False,
}
with (OUT / 'readback.json').open('x') as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write('\n')
print(json.dumps({'completed_tasks': 32, 'new_completed_tasks': new_tasks, 'new_starts': new_starts, 'copied_files': len(copies), 'readback': identity(OUT / 'readback.json'), 'scientific_inference_generated': False}))
