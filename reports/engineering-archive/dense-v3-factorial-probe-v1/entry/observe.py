"""Read only the two registered probe coordinators and their exact direct jobs."""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).parent.resolve()
SOURCE_SHA = 'eb4fc27738adaecd9614ba5822255a5dc37de242afabfa53ff8cd8a6c8c73334'
AUTH_SHA = '00d6b42f922b4428352d5b9fc5e5fa7e0436a4aa20ee0c693e57b9f645398ded'
spec = importlib.util.spec_from_file_location('_registered_probe_observation', HERE / 'probe.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
entry.bound(HERE / 'probe.py', SOURCE_SHA)
auth = entry.read(HERE / 'authorization.json', AUTH_SHA)


def process(record, required):
    root = Path('/proc') / str(record['pid'])
    try:
        first = (root / 'stat').read_text()
        fields = first[first.rfind(')') + 2:].split()
        if int(fields[19]) != record['start_ticks']:
            return dict(pid=record['pid'], original_handle_absent=True, pid_reused=True)
        command = [x.decode() for x in (root / 'cmdline').read_bytes().split(b'\0') if x]
        last = (root / 'stat').read_text()
        final = last[last.rfind(')') + 2:].split()
    except FileNotFoundError:
        return dict(pid=record['pid'], original_handle_absent=True)
    entry.need(int(final[19]) == record['start_ticks'] and fields[1] == final[1], 'Registered process identity raced')
    matched = all(x in command for x in required)
    entry.need(matched, 'Registered probe command changed')
    return dict(pid=record['pid'], start_ticks=record['start_ticks'], ppid=int(final[1]), state=final[0], command_matches=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    entry.need(args.output.is_absolute() and not args.output.exists(), 'Use a new absolute observation path')
    pools = {}
    for pool, runs in auth['queues'].items():
        root = HERE / 'run' / ('pool-' + pool)
        started = entry.read(root / 'started.json')
        entry.need(started['source_sha256'] == SOURCE_SHA and started['authorization_sha256'] == AUTH_SHA,
                   'Wrong registered probe coordinator')
        children, completed = [], 0
        for run in runs:
            for step in auth['steps']:
                prefix = root / f'{run}-checkpoint-{step}'
                if prefix.with_suffix('.verified.json').exists():
                    completed += 1
                if prefix.with_suffix('.started.json').exists() and not prefix.with_suffix('.exited.json').exists():
                    record = entry.read(prefix.with_suffix('.started.json'))
                    children.append(dict(run_id=run, step=step, process=process(record,
                        [str(HERE / 'probe.py'), 'worker', str(prefix.with_suffix('.request.json')), SOURCE_SHA, AUTH_SHA])))
        pools[pool] = dict(coordinator=process(started, [str(HERE / 'probe.py'), 'coordinate', pool, SOURCE_SHA, AUTH_SHA]),
            whole_training_pool_admitted=(root / 'training-pool-admitted.json').exists(),
            completed_checkpoint_probes=completed, active_workers=children,
            failure_receipt=(root / 'failed.json').exists(), complete=(root / 'completed.json').exists())
    result = dict(observed_at_utc=entry.now(), pools=pools, pretrained_reference_complete=True,
        reference_newly_encoded=False, expected_checkpoint_probes=60, read_only_exact_registered_handles=True,
        scientific_completion=False)
    entry.write(args.output, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
