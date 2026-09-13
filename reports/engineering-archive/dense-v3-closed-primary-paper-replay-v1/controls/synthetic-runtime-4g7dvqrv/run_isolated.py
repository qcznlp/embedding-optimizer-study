"""Fresh-process numerical replay with old producer file access and network refused.

This is an explicit Python I/O boundary, not a physical-host or OS-sandbox claim.
The original component code is unmodified. Failures and partial outputs remain.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = root / 'manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != args.manifest_sha256:
        raise ValueError('Portable input anchor differs')
    if not args.output.is_absolute() or args.output.exists() or any(
            p.is_symlink() for p in (args.output, *args.output.parents)):
        raise ValueError('Use a new ordinary absolute attempt directory')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('PYTHONPATH', '') != '':
        raise ValueError('Hide CUDA and remove the producer Python path')
    args.output.mkdir(exist_ok=False)
    runtime = args.output / 'runtime'
    runtime.mkdir()
    for key, relative in (('MPLCONFIGDIR', 'matplotlib'), ('XDG_CACHE_HOME', 'cache'), ('TMPDIR', 'tmp')):
        target = runtime / relative
        target.mkdir()
        os.environ[key] = str(target)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    sys.dont_write_bytecode = True
    sys.path[:] = [p for p in sys.path if not p.startswith(('/root/embedding-optimizer', '/tmp/dense-'))]
    observations = {'closed_bundle_files_opened': set(), 'producer_reads_refused': [], 'network_refused': 0}

    def audit(event, values):
        if event == 'socket.connect':
            observations['network_refused'] += 1
            raise PermissionError('Network use is forbidden in numerical reconstruction')
        if event != 'open' or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).absolute().resolve(strict=False)
        if path.is_relative_to(root):
            observations['closed_bundle_files_opened'].add(path.relative_to(root).as_posix())
            return
        if path.is_relative_to(args.output):
            return
        if str(path).startswith(('/root/embedding-optimizer', '/tmp/dense-')):
            observations['producer_reads_refused'].append(str(path))
            raise PermissionError('Original producer directory fallback is forbidden')

    sys.addaudithook(audit)
    sys.argv = [str(root / 'replay.py'), '--bundle', str(root), '--manifest-sha256',
                args.manifest_sha256, '--output', str(args.output / 'reconstructed')]
    failed = None
    try:
        runpy.run_path(str(root / 'replay.py'), run_name='__main__')
    except BaseException as error:
        failed = {'exception_type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        record = {'observed_at_utc': datetime.now(timezone.utc).isoformat(),
            'manifest_sha256': args.manifest_sha256,
            'closed_bundle_files_opened': sorted(observations['closed_bundle_files_opened']),
            'producer_reads_refused': observations['producer_reads_refused'],
            'network_refused': observations['network_refused'], 'failure': failed,
            'physical_second_host': False, 'os_sandbox_claimed': False,
            'full_goal_complete': False}
        with (args.output / 'io-boundary.json').open('x') as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write('\n')


if __name__ == '__main__':
    main()
