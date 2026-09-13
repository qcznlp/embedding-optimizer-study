"""Preserve completed transport/replay evidence and the new live evaluator entry.

This copies existing control records only. It does not copy the 7.5 GB binary
payload, inspect processes, launch evaluation, change source, or publish a claim.
"""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent.resolve()
STORY = HERE.parents[2]
BACKUP = Path('/tmp/dense-v3-functional-durability.EB3m5dZB')
EVAL = Path('/tmp/dense-v3-factorial-evaluation.IOV7MK93')
SENSITIVITY = Path('/tmp/dense-v3-functional-sensitivity.4Jn3BfPW')
WATCHER = Path('/tmp/dense-v3-factorial-backup-launch.75Chtt86')
TRAIN = Path('/root/embedding-optimizer-v3-experiment/launch/factorial-training-v1')


def identity(path):
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Not an ordinary input: ' + str(path))
    before = path.stat()
    with path.open('rb') as stream:
        value = dict(bytes=before.st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())
    after = path.stat()
    if any(getattr(before, k) != getattr(after, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
        raise ValueError('Input changed during copy')
    return value


def main():
    if (HERE / 'verification.json').exists():
        raise FileExistsError('Preserve the existing archive')
    pairs = []
    for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md'):
        pairs.append((STORY / name, 'before/' + name))
    for path in sorted(BACKUP.iterdir()):
        if path.is_file():
            pairs.append((path, 'functional-durability/' + path.name))
    for name in ('docs/functional-analysis-restoration.md', 'scripts/restore_functional_analysis.py'):
        pairs.append((STORY / name, 'functional-durability/restoration/' + name))
    for path in sorted(EVAL.iterdir()):
        if path.is_file():
            pairs.append((path, 'factorial-evaluation/' + path.name))
    for pool in ('a', 'b'):
        pairs.append((EVAL / 'run' / ('pool-' + pool) / 'started.json',
                      'factorial-evaluation/run/pool-' + pool + '/started.json'))
    for path in sorted((SENSITIVITY / 'relocated-replay').iterdir()):
        if path.is_file():
            pairs.append((path, 'functional-sensitivity-relocated-replay/' + path.name))
    for state, operator, pool in (('adamw_state', 'adamw', 'a'), ('adamw_state', 'muon', 'b'),
                                  ('muon_state', 'muon', 'a'), ('muon_state', 'adamw', 'b')):
        run = f'factorial-v3-{state}-{operator}-seed314159'
        for name in ('verified.json', 'upload.returned.json'):
            pairs.append((WATCHER / 'run' / run / name, 'checkpoint-backups/' + run + '/' + name))
        for name in ('completed.json', 'fresh-native-readback.json', 'ranks.exited.json', 'reader.exited.json'):
            pairs.append((TRAIN / 'run' / ('pool-' + pool) / run / name,
                          'training-complete/' + run + '/' + name))
    result = []
    for source, relative in pairs:
        target = HERE / relative
        first = identity(source)
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if identity(target) != first or identity(source) != first:
            raise ValueError('Archive copy differs')
        result.append(dict(original=str(source), copied=relative, **first))
    original_result = STORY / 'reports/dense-v3-functional-sensitivity-v1/actual/result.json'
    replay_result = HERE / 'functional-sensitivity-relocated-replay/result.json'
    if identity(original_result) != identity(replay_result):
        raise ValueError('Actual copied-source sensitivity replay differs')
    record = dict(scope='completed-control-evidence-and-live-evaluation-handoff',
        verified_at_utc=datetime.now(timezone.utc).isoformat(), source=identity(Path(__file__)),
        copied_files=result, copied_file_count=len(result),
        sensitivity_result_byte_exact=True,
        retained_binary_root=str(BACKUP / 'download'),
        live_evaluation_sessions={'a': 41515, 'b': 37455},
        evaluation_tasks_complete=0, continuation_training_complete=4,
        source_publication=False, scientific_completion=False)
    with (HERE / 'verification.json').open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(copied_files=len(result), verification=identity(HERE / 'verification.json'),
                          sensitivity_result_byte_exact=True, scientific_completion=False)))


if __name__ == '__main__':
    main()
