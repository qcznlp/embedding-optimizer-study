"""Read only the registered inference waiter and its two exact CPU collectors."""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).parent.resolve()
SOURCE_SHA = '4be265ce12cc649a595afaec9e5ef8b68a50eaa347ccb583d4d32ba3e100c5a1'
AUTH_SHA = '47485b8a16af1b0be419e67c4cf45ede08cdacbd01059e42357e16c97b9eb92c'
spec = importlib.util.spec_from_file_location('_registered_factorial_inference_observer', HERE / 'summarize.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
entry.e.bound(HERE / 'summarize.py', SOURCE_SHA)
entry.e.read(HERE / 'authorization.json', AUTH_SHA)


def process(record, expected):
    root = Path('/proc') / str(record['pid'])
    try:
        first = (root / 'stat').read_text()
        fields = first[first.rfind(')') + 2:].split()
        if int(fields[19]) != record['start_ticks']:
            return dict(pid=record['pid'], original_handle_absent=True, pid_reused=True)
        argv = [x.decode() for x in (root / 'cmdline').read_bytes().split(b'\0') if x]
        last = (root / 'stat').read_text()
        after = last[last.rfind(')') + 2:].split()
    except FileNotFoundError:
        return dict(pid=record['pid'], original_handle_absent=True)
    entry.e.need(int(after[19]) == record['start_ticks'] and after[1] == fields[1], 'Registered process identity raced')
    entry.e.need(all(value in argv for value in expected), 'Registered inference command changed')
    return dict(pid=record['pid'], start_ticks=record['start_ticks'], ppid=int(after[1]), state=after[0], command_matches=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    entry.e.need(args.output.is_absolute() and not args.output.exists(), 'Use a new absolute observation')
    root = HERE / 'run'
    started = entry.e.read(root / 'started.json')
    entry.e.need(started['source_sha256'] == SOURCE_SHA and started['authorization_sha256'] == AUTH_SHA, 'Wrong original waiter')
    children = {}
    for kind in ('beir', 'probe'):
        path = root / (kind + '.started.json')
        if not path.exists():
            children[kind] = dict(started=False)
        elif (root / (kind + '.exited.json')).exists():
            children[kind] = dict(started=True, terminal=entry.e.read(root / (kind + '.exited.json')))
        else:
            children[kind] = dict(started=True, process=process(entry.e.read(path), [str(HERE / 'collect.py'), kind, entry.COLLECT_SHA]))
    value = dict(observed_at_utc=entry.e.now(), coordinator=process(started, [str(HERE / 'summarize.py'), 'coordinate', SOURCE_SHA, AUTH_SHA]),
        collectors=children, failed=(root / 'failed.json').exists(), complete=(root / 'completed.json').exists(),
        expected=dict(beir_tasks=168, checkpoint_probes=60, probe_task_rows=840, estimands=3),
        gpu_access=False, read_only_exact_registered_handles=True, scientific_completion=False)
    entry.e.write(args.output, value)
    print(json.dumps(value))


if __name__ == '__main__':
    main()
