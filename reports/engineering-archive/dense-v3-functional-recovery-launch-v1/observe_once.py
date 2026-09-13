"""Capture one real observation of the admitted new recovery only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path('/root/embedding-optimizer-v3-experiment/launch/functional-dimensions-recovery-v1')
observer = root / 'observe_recovery.py'
if hashlib.sha256(observer.read_bytes()).hexdigest() != '9b0220be167465d8d5d4992115be6b9c099bf3ab3af30ba4ec7a463b255e0f84':
    raise ValueError('Observer source changed')
command = ['/usr/bin/python', '-B', str(observer), '--source-sha',
           'f0d732610f715b6463e92a9acc1c6c3e9fc414d981c5384ae7ab0aca3e491040',
           '--authorization-sha', '50414c3541ea3031c4388c9b21274b33faf0676ce74abd44d3ccd1bfea398122',
           '--started-sha', '6d6c33eccc9b0987e165c6e1d3519e266907e4197c51d62472657ab7aac18a1c']
child = subprocess.run(command, cwd=root, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                       stdin=subprocess.DEVNULL, capture_output=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open('xb') as stream:
    stream.write(child.stdout)
with args.output.with_suffix('.stderr').open('xb') as stream:
    stream.write(child.stderr)
with args.output.with_suffix('.process.json').open('x') as stream:
    stream.write(json.dumps({'command': command, 'exit_code': child.returncode,
        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'stdout_sha256': hashlib.sha256(child.stdout).hexdigest()}, sort_keys=True, indent=2) + '\n')
if child.returncode:
    print(child.stderr.decode(), flush=True)
    raise SystemExit(child.returncode)
record = json.loads(child.stdout)
states = record['states']
print(json.dumps({'observed_at_utc': record['observed_at_utc'], 'coordinator': record['coordinator'],
    'accepted_vectors': len(states['new_verified']) + int(states['reused_pretrained']),
    'features': len(states['features_verified']), 'workers': states['workers'],
    'terminal_record_names': list(record['terminal_records']),
    'snapshot': str(args.output)}, sort_keys=True), flush=True)
