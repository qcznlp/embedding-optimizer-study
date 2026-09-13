"""Read only the source-authorized functional coordinator and its exact children."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / 'run'
SOURCE = '3b02f1d8486f3c2b41d0403cbf291580d0ca32868b8b29cb05514a22770257f5'
AUTH = 'd72527595e2c963eae7fd46b0a1de2bd83d1ea15a863ae467a1065cfc5b2f336'
STARTED = 'c1fcac34c21d248c3c7217fa2cc87d63132ff19b71eeb2d8f705bb1aba9f6f41'


def identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Not an ordinary bound file: {path}')
    before = path.stat()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('Bound file changed during read')
    return {'bytes': after.st_size, 'sha256': sha}


def load(path, sha=None):
    bound = identity(path)
    if sha is not None and bound['sha256'] != sha:
        raise ValueError(f'Bound identity differs: {path}')
    value = json.loads(Path(path).read_text())
    if identity(path) != bound:
        raise ValueError('JSON changed during read')
    return value, bound


def process(record):
    pid, start = record['pid'], record['start_ticks']
    path = Path('/proc') / str(pid)
    try:
        raw = (path / 'stat').read_text()
    except FileNotFoundError:
        return {'pid': pid, 'start_ticks': start, 'state': 'missing_reconcile_with_terminal_records'}
    fields = raw[raw.rfind(')') + 2:].split()
    if int(fields[19]) != start:
        raise ValueError('Owned PID was reused; do not read its command')
    command = [v.decode() for v in (path / 'cmdline').read_bytes().split(b'\0') if v]
    if command != record['command']:
        raise ValueError('Exact owned command differs')
    raw = (path / 'stat').read_text()
    again = raw[raw.rfind(')') + 2:].split()
    if int(again[19]) != start:
        raise ValueError('Owned handle raced')
    return {'pid': pid, 'start_ticks': start, 'ppid': int(again[1]),
            'state': again[0], 'command_matches': True}


def main():
    if identity(HERE / 'dispatch.py')['sha256'] != SOURCE:
        raise ValueError('Frozen dispatcher differs')
    authority, authority_bound = load(HERE / 'authorization.json', AUTH)
    inputs, input_bound = load(HERE / 'inputs.json', authority['inputs_sha256'])
    for path, expected in authority['sources'].items():
        if identity(path) != expected:
            raise ValueError('Frozen functional dependency differs')
    for record in [authority['tests'], authority['test_source']]:
        if identity(record['path']) != {k: record[k] for k in ('bytes', 'sha256')}:
            raise ValueError('Frozen test evidence differs')
    started, started_bound = load(RUN / 'coordinator.started.json', STARTED)
    prefix = ['/usr/bin/python', '-B', str(HERE / 'dispatch.py'), '--source-sha', SOURCE,
              '--authorization-sha', AUTH]
    if (started['pid'], started['start_ticks'], started['command']) != (42916, 299159282, prefix + ['--coordinate']):
        raise ValueError('Coordinator is not the actual launched handle')
    plans = {job['plan']['state']['cell']: job for job in inputs['jobs']}
    if len(plans) != 61 or list(plans) != authority['state_order']:
        raise ValueError('Declared state order differs')
    workers, encoded, verified, feature_states = [], [], [], []
    active_tokens = []
    for path in sorted((RUN / 'jobs').glob('*.started.json')):
        record, bound = load(path)
        cell, token = record['cell'], record['gpu_token']
        if cell not in plans or token not in [str(i) for i in range(8)]:
            raise ValueError('Undeclared worker cell or token')
        argv = record['command']
        expected = prefix + ['--worker', cell, '--plan-sha', plans[cell]['plan_sha256'], '--gpu-token', token]
        if (record['ppid'] != 42916 or record['source_sha256'] != SOURCE
                or record['authorization_sha256'] != AUTH or argv[:-4] != expected
                or argv[-4] != '--lease-fd' or argv[-2] != '--lease-fd'
                or len({int(argv[-3]), int(argv[-1])}) != 2):
            raise ValueError('Worker does not match this exact source-authorized dispatch')
        ended = RUN / 'jobs' / f'{cell}.exited.json'
        status = {'cell': cell, 'gpu_token': token, 'started': {'path': str(path), **bound}}
        if ended.exists():
            exit_record, exit_bound = load(ended)
            if (exit_record['pid'], exit_record['start_ticks'], exit_record['command'], exit_record['authorization_sha256']) != (
                    record['pid'], record['start_ticks'], argv, AUTH):
                raise ValueError('Actual child terminal receipt differs')
            status.update(terminal=True, exit_code=exit_record['exit_code'],
                          exited={'path': str(ended), **exit_bound})
        else:
            status.update(terminal=False, observed=process(record))
            if status['observed']['state'] not in ('missing_reconcile_with_terminal_records', 'Z'):
                active_tokens.append(token)
        workers.append(status)
    if len(active_tokens) > 1 or len(set(active_tokens)) != len(active_tokens):
        raise ValueError('More than one functional GPU worker is active')
    for suffix, target in [('encoded', encoded), ('verified', verified), ('features-verified', feature_states)]:
        for path in sorted((RUN / 'jobs').glob(f'*.{suffix}.json')):
            record, bound = load(path)
            if record['cell'] not in plans:
                raise ValueError('Undeclared state receipt')
            target.append({'cell': record['cell'], 'receipt': {'path': str(path), **bound}})
    features = None
    if (RUN / 'features.started.json').exists():
        record, _ = load(RUN / 'features.started.json')
        if record['command'] != prefix + ['--features'] or record['ppid'] != 42916:
            raise ValueError('Undeclared CPU feature worker')
        if (RUN / 'features.exited.json').exists():
            features = {'terminal': True, 'receipt': load(RUN / 'features.exited.json')[0]}
        else:
            features = {'terminal': False, 'observed': process(record)}
    result = {
        'scope': 'exact_read_only_primary_functional_campaign_observation',
        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'sources_authenticated': len(authority['sources']), 'authorization': authority_bound,
        'inputs': input_bound, 'coordinator_started': started_bound,
        'coordinator': process(started), 'workers': workers, 'active_gpu_tokens': active_tokens,
        'encoded_states': encoded, 'verified_vector_states': verified,
        'verified_feature_states': feature_states, 'feature_worker': features,
        'waiting_for_validation_record_present': (RUN / 'waiting-for-validation.json').exists(),
        'vectors_completed': (RUN / 'vectors.completed.json').exists(),
        'features_completed': (RUN / 'features.completed.json').exists(),
        'completed': (RUN / 'completed.json').exists(), 'failed': (RUN / 'failed.json').exists(),
        'other_process_or_gpu_process_enumeration': False, 'writes_or_signals': False,
        'fresh_model_or_numeric_replay_by_observer': False, 'scientific_completion': False}
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
