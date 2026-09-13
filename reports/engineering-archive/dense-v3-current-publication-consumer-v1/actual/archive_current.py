"""Copy this terminal current-primary consumer and paper previews into a new local evidence archive."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent.resolve()
STORY = Path('/root/embedding-optimizer-story-refactor')
DEST = STORY / 'reports/engineering-archive/dense-v3-current-publication-consumer-v1'


def identity(path):
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Require ordinary source artifacts')
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    if DEST.exists():
        raise ValueError('Archive destination already exists; never overwrite evidence')
    if identity(STORY / 'paper/main.tex')['sha256'] != '45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e':
        raise ValueError('Authoritative manuscript changed; reconcile first')
    sources = {}
    for path in sorted(HERE.rglob('*')):
        if path.is_symlink():
            raise ValueError('Unexpected symlink in scoped consumer artifacts')
        if path.is_file() and '__pycache__' not in path.parts:
            sources['actual/' + str(path.relative_to(HERE))] = path
    for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md'):
        sources['before/' + name] = STORY / name
    training = json.loads((HERE / 'training-1959.json').read_bytes())
    if training['completed_branches'] != 12:
        raise ValueError('This closeout archive requires twelve actual completed continuations')
    train_root = Path('/root/embedding-optimizer-v3-experiment/launch/factorial-training-v1/run')
    for pool, value in training['pools'].items():
        base = train_root / ('pool-' + pool)
        sources['complete-training/pool-' + pool + '/coordinator.completed.json'] = base / 'coordinator.completed.json'
        for run in value['runs']:
            if run['status'] != 'full_training_and_fresh_readback_complete':
                raise ValueError('Incomplete continuation in terminal archive')
            for path in sorted((base / run['run_id']).glob('*.json')):
                sources['complete-training/pool-' + pool + '/' + run['run_id'] + '/' + path.name] = path
    backup = Path('/tmp/dense-v3-factorial-backup-launch.75Chtt86/run')
    for path in sorted(backup.glob('factorial-v3-*/verified.json')):
        sources['completed-backup-receipts/' + path.parent.name + '/verified.json'] = path
        sources['completed-backup-receipts/' + path.parent.name + '/artifact_manifest.json'] = path.parent / 'artifact_manifest.json'
    records = {name: {'origin': str(path), **identity(path)} for name, path in sources.items()}
    DEST.mkdir()
    for name, source in sources.items():
        path = DEST / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
        expected = {k: records[name][k] for k in ('bytes', 'sha256')}
        if identity(source) != expected or identity(path) != expected:
            raise ValueError('Source changed during archive copy')
    value = {'scope': 'current_primary_native_consumer_and_actual_development_manuscript_archive_v1',
        'archived_at_utc': datetime.now(timezone.utc).isoformat(), 'files': records, 'copied_files': len(records),
        'copied_bytes': sum(r['bytes'] for r in records.values()), 'copy_hashes_verified': True,
        'producer_paths_still_required_for_upstream_native_replay': True, 'whole_source_portability_claimed': False,
        'authoritative_manuscript_installed': False, 'committed_source_release': False, 'scientific_completion': False}
    with (DEST / 'archive-manifest.json').open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in value.items() if k != 'files'}))


if __name__ == '__main__':
    main()
