"""Read bounded log tails only for currently registered, freshly observed BEIR jobs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
EVAL = Path('/tmp/dense-v3-factorial-evaluation.IOV7MK93')
observation = json.loads((HERE / 'evaluation-current.json').read_text())
rows = []
for pool, value in observation['pools'].items():
    for item in value['task_workers']:
        process = item['observation']
        if process.get('terminal'):
            continue
        if process.get('command_matches') is not True:
            raise ValueError('No fresh registered live worker')
        run, step, task = item['job']
        prefix = EVAL / 'run' / ('pool-' + pool) / 'jobs' / f'{run}-{step}-{task}'
        record = json.loads(prefix.with_suffix('.started.json').read_text())
        if record['pid'] != process['pid'] or record['start_ticks'] != process['start_ticks']:
            raise ValueError('Registered log owner differs')
        path = prefix.with_suffix('.log')
        if path.is_symlink() or any(p.is_symlink() for p in path.parents):
            raise ValueError('Nonordinary registered log')
        size = path.stat().st_size
        with path.open('rb') as stream:
            stream.seek(max(0, size - 65536))
            tail = stream.read(65536).decode('utf-8', errors='replace')
        progress = re.findall(r'Batches:\s*(\d+)%[^\r\n]*?(\d+)/(\d+)\s*\[([^\r\n\]]+)\]', tail)
        last = progress[-1] if progress else None
        rows.append({'pool': pool, 'run_id': run, 'task': task, 'pid': process['pid'],
            'log_bytes_at_read': size, 'last_reported_batch_progress': None if last is None else
            {'percent': int(last[0]), 'completed': int(last[1]), 'total': int(last[2]), 'timing': last[3]}})
result = {'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'jobs': rows,
    'source_of_counts': 'bounded tails of exact registered worker logs; not whole-task completion',
    'gpu_or_process_enumeration': False, 'scientific_completion': False}
with (HERE / 'evaluation-batch-progress.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps(result))
