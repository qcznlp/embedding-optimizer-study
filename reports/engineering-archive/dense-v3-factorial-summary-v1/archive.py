"""Archive the actual registered inference waiter and latest training/backup proof."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent.resolve()
STORY = HERE.parents[2]
WORK = Path('/tmp/dense-v3-factorial-summary.BQ08HjeP')
TRAIN = Path('/root/embedding-optimizer-v3-experiment/launch/factorial-training-v1')
BACKUP = Path('/tmp/dense-v3-factorial-backup-launch.75Chtt86')


def identity(path):
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Nonordinary archive input')
    before = path.stat()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    if any(getattr(before, k) != getattr(after, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
        raise ValueError('Archive input changed')
    return dict(bytes=after.st_size, sha256=sha)


def main():
    if (HERE / 'verification.json').exists():
        raise FileExistsError('Keep the existing archive')
    pairs = [(p, 'entry/' + p.relative_to(WORK).as_posix()) for p in sorted(WORK.rglob('*')) if p.is_file()]
    pairs.append((STORY / 'src/embed_optim/state_operator_factorial_summary.py', 'source-original/state_operator_factorial_summary.py'))
    for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md'):
        pairs.append((STORY / name, 'before/' + name))
    for pool, operator in (('a', 'muon'), ('b', 'adamw')):
        run = f'factorial-v3-adamw_state-{operator}-seed271828'
        for name in ('completed.json', 'fresh-native-readback.json', 'ranks.exited.json', 'reader.exited.json'):
            pairs.append((TRAIN / 'run' / ('pool-' + pool) / run / name, 'third-pair/' + run + '/' + name))
        for name in ('verified.json', 'upload.returned.json'):
            pairs.append((BACKUP / 'run' / run / name, 'third-pair/' + run + '/backup/' + name))
    copied = []
    for source, relative in pairs:
        first = identity(source)
        target = HERE / relative
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if identity(source) != first or identity(target) != first:
            raise ValueError('Archive copy differs')
        copied.append(dict(original=str(source), copied=relative, **first))
    value = dict(scope='actual-factorial-inference-waiter-and-third-pair-completion',
        verified_at_utc=datetime.now(timezone.utc).isoformat(), source=identity(Path(__file__)), copied_files=copied,
        registered_session=4529, actual_collectors_started=0, actual_factorial_statistical_results=False,
        actual_training_branches_complete=6, new_checkpoint_backups_verified=30,
        numerical_functions_changed=False, source_publication=False, scientific_completion=False)
    with (HERE / 'verification.json').open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(copied_files=len(copied), verification=identity(HERE / 'verification.json'), scientific_completion=False)))


if __name__ == '__main__':
    main()
