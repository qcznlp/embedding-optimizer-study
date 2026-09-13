"""Read only the exact registered complete-paper replay coordinator/CPU child."""
import argparse
import json
from pathlib import Path
import combined as c

SOURCE = '1a9c50dba2f58cc68b110749ccbc40d427445d53f0dc6c994b8f78c02096d566'
AUTH = 'c66a9e434789d9d428e7fbe273df4db2fc2f383aabfb2145e97d8473dfcb7d41'


def process(record, expected):
    path = Path('/proc') / str(record['pid'])
    try:
        before = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if int(before[19]) != record['start_ticks']:
            return {'original_handle_absent': True, 'pid_reused': True}
        argv = (path / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        after = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return {'original_handle_absent': True}
    c.need(before[19] == after[19] and all(value in argv for value in expected), 'Owned process identity differs')
    return {'pid': record['pid'], 'start_ticks': record['start_ticks'], 'state': after[0],
            'ppid': int(after[1]), 'command_matches': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    c.need(args.output.is_absolute() and not args.output.exists(), 'Use a new absolute observation')
    c.bound(c.HERE / 'combined.py', SOURCE)
    c.bound(c.HERE / 'authorization.json', AUTH)
    run = c.HERE / 'run'
    started = c.read(run / 'started.json')
    c.need(started['source_sha256'] == SOURCE and started['authorization_sha256'] == AUTH, 'Wrong coordinator')
    child = {'started': False}
    if (run / 'replay.started.json').exists():
        record = c.read(run / 'replay.started.json')
        child = c.read(run / 'replay.exited.json') if (run / 'replay.exited.json').exists() else process(record, record['command'])
    completion = c.read(run / 'completed.json') if (run / 'completed.json').exists() else None
    result = {'observed_at_utc': c.now(),
              'coordinator': process(started, [str(c.HERE / 'combined.py'), 'coordinate', SOURCE, AUTH]),
              'native_document_complete': (c.DOC / 'run/completed.json').exists(),
              'closed_combined_bundle_created': (run / 'bundle.json').exists(),
              'replay_child': child, 'failed': (run / 'failed.json').exists(), 'completion': completion,
              'read_only_exact_registered_handles': True, 'gpu_access': False, 'full_goal_complete': False}
    c.write(args.output, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
