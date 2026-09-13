"""Nonblocking stack samples of only freshly authenticated registered study workers."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
EVAL = Path('/tmp/dense-v3-factorial-evaluation.IOV7MK93')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def identity(record):
    root = Path('/proc') / str(record['pid'])
    first = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    need(int(first[19]) == record['start_ticks'], 'Original registered PID is gone')
    command = [x.decode() for x in (root / 'cmdline').read_bytes().split(b'\0') if x]
    need(command == record['command'], 'Exact registered command differs')
    after = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    need(first[19] == after[19] and first[1] == after[1], 'Identity raced')
    return {'state': after[0], 'user_ticks': int(after[11]), 'system_ticks': int(after[12])}


for name, digest in (
    ('evaluate.py', '45577537257e1dff2b7c3a26ee12f7564fbb476daa256e7b0f09300305b68cba'),
    ('authorization.json', '3a1ae32fdde63abf386603208e413ea11bf0c169eb1923f9dd7e1da529ffa29c'),
):
    need(hashlib.sha256((EVAL / name).read_bytes()).hexdigest() == digest, 'Live binding changed')
observed = json.loads((HERE / 'evaluation-final.json').read_text())
rows = []
for pool, value in observed['pools'].items():
    for item in value['task_workers']:
        if item['observation'].get('terminal'):
            continue
        run, step, task = item['job']
        prefix = EVAL / 'run' / ('pool-' + pool) / 'jobs' / f'{run}-{step}-{task}'
        record = json.loads(prefix.with_suffix('.started.json').read_text())
        need(record['pid'] == item['observation']['pid']
             and record['start_ticks'] == item['observation']['start_ticks'], 'Wrong registered owner')
        need(not prefix.with_suffix('.exited.json').exists(), 'Worker already completed')
        before = identity(record)
        result = subprocess.run(['/usr/local/bin/py-spy', 'dump', '--nonblocking',
                                 '--pid', str(record['pid'])], capture_output=True,
                                text=True, timeout=20, check=False)
        after = identity(record)
        log_path = prefix.with_suffix('.log')
        need(not any(x.is_symlink() for x in (log_path, *log_path.parents)), 'Unexpected log path')
        with log_path.open('rb') as stream:
            stream.seek(max(0, log_path.stat().st_size - 65536))
            log = stream.read(65536).decode('utf-8', errors='replace')
        rows.append({'pool': pool, 'run_id': run, 'task': task, 'pid': record['pid'],
                     'start_ticks': record['start_ticks'], 'before': before, 'after': after,
                     'sample_exit_code': result.returncode, 'stdout': result.stdout,
                     'stderr': result.stderr,
                     'bounded_log_oom_retry_lines': [line for line in log.splitlines()
                         if line.startswith('RETRYING ClimateFEVER (OOM) -> encode_char_budget=')]})
output = {'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'jobs': rows,
          'nonblocking': True, 'unrelated_process_or_gpu_enumeration': False,
          'source_or_worker_mutation': False, 'scientific_completion': False,
          'scope': 'one stack sample per exact registered worker; not throughput or GPU utilization'}
with (HERE / 'evaluation-stacks.json').open('x') as stream:
    json.dump(output, stream, indent=2, sort_keys=True)
    stream.write('\n')
for row in rows:
    print(json.dumps({'pid': row['pid'], 'run_id': row['run_id'],
                      'sample_exit_code': row['sample_exit_code'],
                      'stdout': row['stdout'], 'stderr': row['stderr'],
                      'bounded_log_oom_retry_lines': row['bounded_log_oom_retry_lines']}))
