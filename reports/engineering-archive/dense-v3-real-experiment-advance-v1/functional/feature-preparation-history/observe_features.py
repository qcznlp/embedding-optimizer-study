"""Read the exact new CPU-only coordinator/child and native-verified feature receipts."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path('/root/embedding-optimizer-v3-experiment/launch/functional-features-recovery-v3')
SOURCE = '694069c2d0e2866bba1126ea891cc9360ff03923691afa26ae04724165d35d9f'
APPROVAL = '4c151939a90937f27488620291f64d58105dc1f1a8e962eaa7e3949379113514'
TESTS = 'e3083be2dad6970c24b728efe626dd18be7182d90d498a12fb5a0ac89332ffd1'
STARTED = 'a558e8c7d6f3e9bb14bb291d03edd136043d73b64e32844f42d148db07c3bbc4'
sys.path.insert(0, str(ROOT))
import features as entry


def snapshot():
    entry.need(entry.identity(ROOT / 'features.py')['sha256'] == SOURCE, 'CPU feature source changed')
    entry.read(ROOT / 'owner-approval.json', APPROVAL)
    tests = entry.read(ROOT / 'tests.json', TESTS)
    entry.need(entry.identity(ROOT / 'test_features.py') == tests['test_source'], 'Feature tests changed')
    entry.require_inputs()
    original = entry.read(entry.PRIOR / 'authorization.json', entry.PINS[entry.PRIOR / 'authorization.json'])
    for path, binding in original['sources'].items():
        entry.need(entry.identity(path) == binding, 'Original functional dependency changed')
    started = entry.read(ROOT / 'run/coordinator.started.json', STARTED)
    prefix = ['/usr/bin/python', '-B', str(ROOT / 'features.py'), '--source-sha', SOURCE,
              '--approval-sha', APPROVAL, '--tests-sha', TESTS]
    entry.need((started['pid'], started['start_ticks'], started['command']) ==
               (779352, 319719253, prefix + ['--coordinate']), 'Wrong exact feature coordinator')
    old = entry.EXPERIMENT / 'launch/functional-dimensions/observe.py'
    entry.need(entry.identity(old)['sha256'] == 'c314e4d540305bb67e0bcdcb6ad921d22926e4cbd40f59e00e84415be18cd078',
               'Exact process reader changed')
    spec = importlib.util.spec_from_file_location('feature_read_only_exact_process', old)
    reader = importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)
    terminal = [name for name in ('coordinator.completed', 'failed') if (ROOT / 'run' / f'{name}.json').exists()]
    entry.need(len(terminal) <= 1, 'Conflicting feature terminals')
    child = entry.read(ROOT / 'run/features.started.json')
    entry.need(child['ppid'] == started['pid'] and child['source_sha256'] == SOURCE
               and child['approval_sha256'] == APPROVAL and child['tests_sha256'] == TESTS
               and child['command'] == prefix + ['--compute'], 'Wrong exact CPU feature child')
    end = ROOT / 'run/features.exited.json'
    if end.exists():
        exited = entry.read(end)
        entry.need(exited['pid'] == child['pid'] and exited['source_sha256'] == SOURCE
                   and exited['approval_sha256'] == APPROVAL, 'Wrong actual feature exit')
        worker = {'terminal': True, 'exit_code': exited['exit_code']}
    else:
        worker = {'terminal': False, 'observed': reader.process(child)}
    raw = entry.read(entry.VECTOR_ROOT / 'manifest.json')
    accepted = {}
    for path in sorted((ROOT / 'run/jobs').rglob('*.features-verified.json')):
        value = entry.read(path); cell = value['cell']
        entry.need(cell in raw['states'] and cell not in accepted
                   and path == ROOT / 'run/jobs' / f'{cell}.features-verified.json'
                   and value['fresh_raw_vectors_fully_recomputed'] is True,
                   'Unknown, duplicate or incompletely recomputed feature state')
        bound = value['manifest']
        entry.need(bound['path'] == f'states/{cell}/manifest.json'
                   and entry.identity(entry.OUTPUT / bound['path']) == {k: bound[k] for k in ('bytes', 'sha256')},
                   'Verified feature bundle changed')
        accepted[cell] = value
    if terminal == ['coordinator.completed']:
        result = entry.read(ROOT / 'run/features.completed.json')
        complete = entry.read(ROOT / 'run/coordinator.completed.json')
        entry.need(len(accepted) == result['states'] == 61 and worker == {'terminal': True, 'exit_code': 0}
                   and complete['actual_feature_worker_exit'] == 0
                   and complete['feature_completion'] == entry.identity(ROOT / 'run/features.completed.json'),
                   'Incomplete real feature campaign')
    return {'observed_at_utc': datetime.now(timezone.utc).isoformat(),
            'coordinator': {'terminal_record': terminal[0]} if terminal else reader.process(started),
            'worker': worker, 'accepted_vector_states': 61, 'native_verified_feature_states': len(accepted),
            'accepted_features': accepted, 'source_sha256': SOURCE, 'approval_sha256': APPROVAL,
            'read_only_owned_handles': True, 'scientific_completion': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = snapshot()
    with args.output.open('x') as stream: stream.write(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'accepted_features'}, sort_keys=True), flush=True)
