"""Isolated source assembly and exact CPU diagnostic launch; no scientific admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


STORY = Path('/root/embedding-optimizer-story-refactor')
PRIMARY = Path('/root/embedding-optimizer-primary-v3')
PARENT = STORY / 'reports/engineering-archive/dense-v3-primary-launch-v1/source-assembly.json'
PARENT_SHA = 'e603ef175c0f691a1dd83c4d8b474d90da2fc67c8c65f5a77f3c3180e78332b8'
COMPONENTS = tuple(f'src/embed_optim/factorial_v3_{name}.py' for name in
                   ('batches', 'optimizer', 'checkpoint', 'trainer'))
WORKER = 'reports/engineering-archive/dense-v3-factorial-trainer-v1/worker.py'


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected ordinary source file: {path}')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def assemble(root):
    if not root.is_dir() or any(root.iterdir()):
        raise ValueError('Assembly requires an existing fresh empty mktemp directory')
    if identity(PARENT)['sha256'] != PARENT_SHA:
        raise ValueError('Original primary source manifest changed')
    parent = json.loads(PARENT.read_text())
    if len(parent['files']) != 56:
        raise ValueError('Original primary source coverage changed')
    sources = {}
    for relative, row in parent['files'].items():
        source = PRIMARY / relative
        if identity(source) != row['identity']:
            raise ValueError(f'Original primary source changed: {relative}')
        sources[relative] = {'origin': str(source), **row['identity']}
    for relative in (*COMPONENTS, WORKER):
        if relative in sources:
            raise ValueError('A component may not overwrite original primary source')
        source = STORY / relative
        sources[relative] = {'origin': str(source), **identity(source)}
    assembled = root / 'source-first'
    for relative, row in sources.items():
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Invalid relative source path')
        destination = assembled / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row['origin'], destination)
        if identity(destination) != {k: row[k] for k in ('bytes', 'sha256')}:
            raise ValueError(f'Copy verification failed: {relative}')
    manifest = {
        'scope': 'cpu-only-factorial-trainer-source-assembly',
        'scientific_admission': False,
        'created_at_utc': now(),
        'root': str(assembled),
        'parent_manifest': {'path': str(PARENT), **identity(PARENT)},
        'primary_files': 56,
        'factorial_components': 4,
        'files': sources,
        'driver': {'path': str(Path(__file__).resolve()), **identity(__file__)},
    }
    save(root / 'source-first.json', manifest)
    print(json.dumps({'assembled': str(assembled), 'files': len(sources),
                      'manifest': identity(root / 'source-first.json')}), flush=True)


def run(args):
    manifest_path = args.manifest.resolve()
    manifest_id = identity(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest_id['sha256'] != args.manifest_sha:
        raise ValueError('Require the externally anchored source manifest')
    if manifest['driver'] != {'path': str(Path(__file__).resolve()), **identity(__file__)}:
        raise ValueError('Launch driver changed after assembly')
    source = Path(manifest['root'])
    for relative, row in manifest['files'].items():
        if identity(source / relative) != {k: row[k] for k in ('bytes', 'sha256')}:
            raise ValueError(f'Assembled source changed: {relative}')
    attempt = args.attempt.resolve()
    attempt.mkdir(parents=False, exist_ok=False)
    output = attempt / 'output'
    command = ['/usr/bin/python', '-B', '-m', 'torch.distributed.run', '--standalone',
               '--nnodes=1', '--nproc-per-node=4', str(source / WORKER),
               '--output', str(output), '--mode', args.mode,
               '--optimizer', args.optimizer, '--seed', str(args.seed), '--stop', str(args.stop)]
    if args.resume_binding:
        command.extend(['--resume-binding', str(args.resume_binding.resolve())])
    environment = dict(os.environ)
    declared_environment = {
        'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONPATH': f'{source / "src"}:{source}',
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1', 'WANDB_MODE': 'disabled',
    }
    environment.update(declared_environment)
    original_nice = os.nice(10)
    started_at = now()
    with (attempt / 'worker.log').open('x') as log:
        process = subprocess.Popen(command, cwd=source, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
        # Only inspect the exact child created above. Never enumerate processes.
        stat = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()
        started = {
            'scope': 'cpu-only-factorial-trainer-diagnostic', 'scientific_admission': False,
            'started_at_utc': started_at, 'pid': process.pid,
            'start_ticks': int(stat[19]), 'ppid': int(stat[1]),
            'command': command, 'cwd': str(source), 'declared_environment': declared_environment,
            'nice': original_nice, 'source_manifest': {'path': str(manifest_path), **manifest_id},
            'resume_binding': None if not args.resume_binding else {
                'path': str(args.resume_binding.resolve()), **identity(args.resume_binding)},
        }
        save(attempt / 'started.json', started)
        print(json.dumps({'started': str(attempt / 'started.json'),
                          'pid': process.pid, 'start_ticks': started['start_ticks']}), flush=True)
        code = process.wait()
    exited = {'finished_at_utc': now(), 'exit_code': code,
              'started': identity(attempt / 'started.json'), 'log': identity(attempt / 'worker.log')}
    save(attempt / 'exited.json', exited)
    print(json.dumps({'attempt': str(attempt), **exited}), flush=True)
    if code:
        print((attempt / 'worker.log').read_text()[-24000:], flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    preparation = commands.add_parser('assemble')
    preparation.add_argument('--root', type=Path, required=True)
    execution = commands.add_parser('run')
    execution.add_argument('--manifest', type=Path, required=True)
    execution.add_argument('--manifest-sha', required=True)
    execution.add_argument('--attempt', type=Path, required=True)
    execution.add_argument('--mode', choices=('loader', 'train'), required=True)
    execution.add_argument('--optimizer', choices=('hybrid_adamw', 'muon'), default='hybrid_adamw')
    execution.add_argument('--seed', type=int, choices=(314159, 271828, 161803), default=314159)
    execution.add_argument('--stop', type=int, choices=(0, 79, 157, 235, 313), default=0)
    execution.add_argument('--resume-binding', type=Path)
    args = parser.parse_args()
    if args.action == 'assemble':
        assemble(args.root.resolve())
        return 0
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
