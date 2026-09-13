"""Read only the new registered resume coordinators and their direct children."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Nonordinary owned record')
    return json.loads(path.read_bytes())


def process(record):
    root = Path('/proc') / str(record['pid'])
    try:
        raw = (root / 'stat').read_text()
    except FileNotFoundError:
        return dict(pid=record['pid'], original_handle_absent=True, exit_code=None)
    before = raw[raw.rfind(')') + 2:].split()
    if int(before[19]) != record['start_ticks']:
        return dict(pid=record['pid'], original_handle_absent=True, pid_reused=True, exit_code=None)
    if before[0] == 'Z':
        return dict(pid=record['pid'], start_ticks=record['start_ticks'], state='Z', exit_code=None)
    argv = [x.decode() for x in (root / 'cmdline').read_bytes().split(b'\0') if x]
    need(argv == record['command'], 'Exact owned recovery command changed')
    raw = (root / 'stat').read_text()
    after = raw[raw.rfind(')') + 2:].split()
    need(int(after[19]) == record['start_ticks'], 'Registered recovery handle raced')
    return dict(pid=record['pid'], start_ticks=record['start_ticks'], ppid=int(after[1]),
                state=after[0], command_matches=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-sha', required=True)
    p.add_argument('--authorization-sha', required=True)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    need(args.output.is_absolute() and not args.output.exists(), 'Use a new absolute observation path')
    for name, sha in [('resume.py', args.source_sha), ('authorization.json', args.authorization_sha)]:
        path = HERE / name
        read(path) if name.endswith('.json') else need(path.is_file() and not path.is_symlink(), 'Missing live source')
        need(hashlib.sha256(path.read_bytes()).hexdigest() == sha, 'Registered recovery source/authority changed')
    pools = {}
    for pool in ('a', 'b'):
        root = HERE / 'run' / ('pool-' + pool)
        start = root / 'started.json'
        if not start.exists():
            pools[pool] = dict(started=False)
            continue
        record = read(start)
        need(record['source_sha256'] == args.source_sha and record['authorization_sha256'] == args.authorization_sha,
             'Wrong original recovery registration')
        ranks = []
        exits = read(root / 'ranks.exited.json') if (root / 'ranks.exited.json').exists() else None
        for rank in range(4):
            path = root / f'rank-{rank}.started.json'
            if path.exists():
                status = dict(exit_code=exits['actual_rank_exits'][str(rank)], terminal=True) if exits else process(read(path))
                ranks.append(dict(rank=rank, observation=status))
        reader = None
        if (root / 'reader.started.json').exists():
            reader = read(root / 'reader.exited.json') if (root / 'reader.exited.json').exists() else process(read(root / 'reader.started.json'))
        pools[pool] = dict(started=True, coordinator=process(record), ranks=ranks, reader=reader,
            gpu_admission=(root / 'admission.json').exists(), failure=(root / 'failed.json').exists(),
            completion=read(root / 'completed.json') if (root / 'completed.json').exists() else None)
    result = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(), pools=pools,
        read_only_exact_registered_handles=True, scientific_completion=False)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
