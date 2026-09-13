"""Preserve the tested launch and first real native receipts; no model/remote operations."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import sys

WORK = Path('/tmp/dense-v3-functional-production-candidate.krPeuQIL')
LIVE = Path('/root/embedding-optimizer-v3-experiment/launch/functional-dimensions-recovery-v1')
ARCHIVE = Path('/root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-v3-functional-recovery-launch-v1')
sys.path.insert(0, str(LIVE))
import recovery_support as support


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


ARCHIVE.mkdir(exist_ok=False)
copies = {}
def copy(src, name):
    expected = support.identity(src)
    target = ARCHIVE / name; target.parent.mkdir(parents=True, exist_ok=True)
    with Path(src).open('rb') as source, target.open('xb') as destination:
        shutil.copyfileobj(source, destination)
    support.need(support.identity(target) == support.identity(src) == expected, 'Archive copy changed')
    copies[name] = {'source': str(src), **expected}


for name in support.FILES:
    copy(LIVE / name, 'source/' + name)
for name in ('proposal.json', 'owner-approval.json', 'authorization.json', 'tests.json', 'README.md'):
    copy(LIVE / name, 'launch/' + name)
for name in ('coordinator.started.json', 'pretrained.reused.json'):
    copy(LIVE / 'run' / name, 'launch/run/' + name)
for path in sorted((WORK / 'source-first').iterdir()):
    copy(path, 'source-first/' + path.name)
for name in ('tests-first.json', 'tests-second.json', 'observation-first.json',
             'observation-first.process.json', 'observation-first.stderr',
             'observation-second.json', 'observation-second.process.json', 'observation-second.stderr'):
    copy(WORK / 'actual' / name, 'actual/' + name)
for path in sorted((WORK / 'before').iterdir()):
    copy(path, 'before/' + path.name)
for name in ('plan.md', 'prepare_metadata.py', 'observe_once.py', 'archive_launch.py'):
    copy(WORK / name, name)
observation = support.read(WORK / 'actual/observation-second.json')
support.need(len(observation['states']['new_verified']) == 3, 'Use exact first-three snapshot')
timings = []
for cell in observation['states']['new_verified']:
    for suffix in ('admission', 'started', 'exited', 'encoded', 'verified'):
        copy(LIVE / 'run/jobs' / f'{cell}.{suffix}.json', f'launch/run/jobs/{cell}.{suffix}.json')
    start = support.read(LIVE / 'run/jobs' / f'{cell}.started.json')
    end = support.read(LIVE / 'run/jobs' / f'{cell}.exited.json')
    encoded = support.read(LIVE / 'run/jobs' / f'{cell}.encoded.json')
    support.need(end['exit_code'] == 0 and encoded['authorization_sha256'] ==
                 '50414c3541ea3031c4388c9b21274b33faf0676ce74abd44d3ccd1bfea398122',
                 'Real terminal evidence differs')
    timings.append({'cell': cell, 'encoding_seconds': encoded['encoding_seconds'],
                    'worker_wall_seconds': (datetime.fromisoformat(end['exited_at_utc']) -
                                            datetime.fromisoformat(start['started_at_utc'])).total_seconds(),
                    'worker_start_ticks': start['start_ticks']})
cycles = [(b['worker_start_ticks'] - a['worker_start_ticks']) / 100
          for a, b in zip(timings, timings[1:])]
timing = {'scope': 'first_three_real_checkpoint_encodings_only', 'states': timings,
          'mean_encoding_seconds': statistics.mean(t['encoding_seconds'] for t in timings),
          'mean_worker_wall_seconds': statistics.mean(t['worker_wall_seconds'] for t in timings),
          'note': 'Cycle seconds use the host clock-tick rate recorded below; estimate excludes complete vector readback and all CPU features.'}
import os
tick_rate = os.sysconf('SC_CLK_TCK')
cycles = [(b['worker_start_ticks'] - a['worker_start_ticks']) / tick_rate
          for a, b in zip(timings, timings[1:])]
timing.update(clock_ticks_per_second=tick_rate, observed_start_to_start_seconds=cycles,
              estimated_remaining_encoding_minutes=57 * statistics.mean(cycles) / 60)
write(ARCHIVE / 'actual/timing.json', timing)
protected = support.read(support.STORY / 'reports/engineering-archive/dense-v3-pretrained-native-copy-v1/actual/readout.json')['original_bound_files_unchanged']
for path, expected in protected.items():
    support.need(support.identity(path) == expected, 'Protected original file changed')
write(ARCHIVE / 'verification.json', {
    'verified_at_utc': datetime.now(timezone.utc).isoformat(),
    'copied_original_identities': copies,
    'files': {str(p.relative_to(ARCHIVE)): support.identity(p) for p in sorted(ARCHIVE.rglob('*')) if p.is_file()},
    'protected_original_files_unchanged': protected,
    'native_preparation_tool': {'session_id': 86421, 'terminal_chunk': '84f65b', 'exit_code': 0},
    'actual_coordinator_session': 27585, 'original_attempt_restarted': False,
    'scientific_completion': False})
print(json.dumps({'archive': str(ARCHIVE), 'verification': support.identity(ARCHIVE / 'verification.json'),
                  'timing': timing, 'protected_original_files': len(protected)}, sort_keys=True), flush=True)
