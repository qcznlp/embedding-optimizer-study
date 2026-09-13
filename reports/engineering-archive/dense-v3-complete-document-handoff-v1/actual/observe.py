"""Read only the single registered complete-document waiter; no GPU enumeration."""
import argparse
from pathlib import Path
import json

import author as a

SOURCE_SHA = 'cac9486b6e92acedb190bad67e34a5fd647d1c10a691071483fbf001406b1f9d'
AUTH_SHA = 'ca7a0503b9324d51ce62a9c19cd1ec380e06311a0ff498b84ea30e683670e0e4'


def observe_process(record):
    root = Path('/proc') / str(record['pid'])
    try:
        before = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        if int(before[19]) != record['start_ticks']:
            return {'pid': record['pid'], 'original_handle_absent': True, 'pid_reused': True}
        argv = [x.decode() for x in (root / 'cmdline').read_bytes().split(b'\0') if x]
        after = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return {'pid': record['pid'], 'original_handle_absent': True}
    a.need(after[19] == before[19] and after[1] == before[1], 'Registered waiter identity raced')
    a.need(all(value in argv for value in (str(a.HERE / 'author.py'), 'coordinate', SOURCE_SHA, AUTH_SHA)),
           'Registered paper waiter command differs')
    return {'pid': record['pid'], 'start_ticks': record['start_ticks'], 'ppid': int(after[1]),
            'state': after[0], 'command_matches': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    a.need(args.output.is_absolute() and not args.output.exists(), 'Use a new absolute observation path')
    a.bound(a.HERE / 'author.py', SOURCE_SHA)
    a.bound(a.HERE / 'authorization.json', AUTH_SHA)
    record = a.read(a.HERE / 'run/started.json')
    a.need(record['source_sha256'] == SOURCE_SHA and record['authorization_sha256'] == AUTH_SHA,
           'Wrong registered document waiter')
    value = {'observed_at_utc': a.now(), 'coordinator': observe_process(record),
        'complete': (a.HERE / 'run/completed.json').exists(),
        'failed': (a.HERE / 'run/failed.json').exists(),
        'factorial_inference_complete': (a.SUMMARY / 'run/completed.json').exists(),
        'document_output_started': a.OUTPUT.exists(), 'gpu_access': False,
        'read_only_exact_registered_handle': True, 'full_goal_complete': False}
    a.write(args.output, value)
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
