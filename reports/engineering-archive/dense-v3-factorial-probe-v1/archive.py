"""Preserve the real five-stage probe preparation and registered waiting jobs."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent.resolve()
STORY = HERE.parents[2]
WORK = Path('/tmp/dense-v3-factorial-probe.ryorjXJK')
OUTPUT = Path('/root/embedding-optimizer-v3-experiment/analyses/dense-v3-factorial-probe-v1')


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
    pairs = [(p, 'entry/' + p.name) for p in sorted(WORK.iterdir()) if p.is_file()]
    pairs += [(p, 'source/' + p.relative_to(WORK / 'source').as_posix())
              for p in sorted((WORK / 'source').rglob('*.py'))]
    for pool in ('a', 'b'):
        pairs.append((WORK / 'run' / ('pool-' + pool) / 'started.json', 'run/pool-' + pool + '/started.json'))
    for name in ('metrics.json', 'origin.json'):
        pairs.append((OUTPUT / 'pretrained' / name, 'pretrained/' + name))
    for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md'):
        pairs.append((STORY / name, 'before/' + name))
    copied = []
    for source, relative in pairs:
        before = identity(source)
        target = HERE / relative
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if identity(source) != before or identity(target) != before:
            raise ValueError('Archive copy differs')
        copied.append(dict(original=str(source), copied=relative, **before))
    record = dict(scope='real-factorial-probe-preparation-and-waiting-entries',
        verified_at_utc=datetime.now(timezone.utc).isoformat(), source=identity(Path(__file__)),
        copied_files=copied, retained_baseline_arrays={str(OUTPUT / 'pretrained' / name): identity(OUTPUT / 'pretrained' / name)
            for name in ('vectors-fp16.npz', 'scores.npz')},
        owned_sessions={'pool_a': 54795, 'pool_b': 34578}, new_checkpoint_encodings_complete=0,
        reference_reused_not_encoded=True, numerical_sources_changed=False,
        historical_guards_modified=False, source_publication=False, scientific_completion=False)
    with (HERE / 'verification.json').open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(copied_files=len(copied), verification=identity(HERE / 'verification.json'), scientific_completion=False)))


if __name__ == '__main__':
    main()
