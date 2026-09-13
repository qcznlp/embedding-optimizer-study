"""Read only the two registered evaluator handles and their own job records."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
SOURCE = '45577537257e1dff2b7c3a26ee12f7564fbb476daa256e7b0f09300305b68cba'
AUTH = '3a1ae32fdde63abf386603208e413ea11bf0c169eb1923f9dd7e1da529ffa29c'
HANDLES = {'a': (802691, 320351837), 'b': (802714, 320351956)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    require(path.is_file() and not path.is_symlink(), 'Missing or symlinked owned record')
    return json.loads(path.read_text())


def process(pid, start, required_argument):
    path = Path('/proc') / str(pid)
    try:
        first = (path / 'stat').read_text()
    except FileNotFoundError:
        return dict(pid=pid, original_handle_absent=True, exit_code=None)
    fields = first[first.rfind(')') + 2:].split()
    if int(fields[19]) != start:
        return dict(pid=pid, original_handle_absent=True, pid_reused=True, exit_code=None)
    if fields[0] == 'Z':
        return dict(pid=pid, start_ticks=start, state='Z', exit_code=None)
    argv = [v.decode() for v in (path / 'cmdline').read_bytes().split(b'\0') if v]
    require(required_argument in argv, 'Exact owned command changed')
    second = (path / 'stat').read_text()
    after = second[second.rfind(')') + 2:].split()
    require(int(after[19]) == start, 'Owned handle raced')
    return dict(pid=pid, start_ticks=start, ppid=int(after[1]), state=after[0], command_matches=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    require(a.output.is_absolute() and not a.output.exists(), 'New absolute observation output required')
    for name, digest in (('evaluate.py', SOURCE), ('authorization.json', AUTH)):
        require(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest, 'Bound live entry changed')
    results = {}
    for pool, (pid, start) in HANDLES.items():
        root = HERE / 'run' / ('pool-' + pool)
        record = read(root / 'started.json')
        require(record['pid'] == pid and record['start_ticks'] == start and record['source_sha256'] == SOURCE
                and record['authorization_sha256'] == AUTH, 'Original coordinator identity differs')
        children = []
        for path in sorted((root / 'jobs').glob('*.started.json')):
            item = read(path)
            exit_path = path.with_name(path.name.replace('.started.json', '.exited.json'))
            if exit_path.exists():
                observed = dict(terminal=True, exit_code=read(exit_path)['exit_code'])
            else:
                observed = process(item['pid'], item['start_ticks'], item['command'][2])
            children.append(dict(job=item['job'], observation=observed))
        results[pool] = dict(coordinator=process(pid, start, str(HERE / 'evaluate.py')),
            training_pool_admitted=(root / 'training-pool-admitted.json').exists(),
            completed_tasks=len(list((root / 'jobs').glob('*.task-verified.json'))), task_workers=children,
            failure_receipt=(root / 'failed.json').exists(),
            complete=read(root / 'completed.json') if (root / 'completed.json').exists() else None)
    result = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(), pools=results,
                  read_only_exact_registered_handles=True, scientific_completion=False)
    with a.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
