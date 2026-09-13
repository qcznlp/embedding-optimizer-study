"""Preserve this new recovery handoff without changing any live entry or output."""
import argparse
from pathlib import Path
import shutil
import resume as r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', required=True, type=Path)
    args = p.parse_args()
    r.no_existing(args.destination)
    args.destination.mkdir(parents=True)
    names = ['resume.py', 'test_resume.py', 'run_tests.py', 'tests.json', 'PLAN.md', 'observe.py',
             'authorization.json', 'downloaded.json', 'archive.py', 'commands.json',
             'training-initial.json', 'training-1845.json', 'evaluation-1845.json',
             'probe-1845.json', 'summary-1845.json', 'resume-started.json',
             'run/pool-a/started.json', 'run/pool-b/started.json']
    rows = []

    def copy(source, target):
        original = r.identity(source)
        destination = args.destination / target
        r.no_existing(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        r.need(r.identity(source) == r.identity(destination) == original, 'Archive copy changed')
        rows.append(dict(source=str(source), destination=target, **original))

    for name in names:
        copy(r.HERE / name, name)
    repository = Path('/root/embedding-optimizer-story-refactor')
    for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md'):
        copy(repository / name, 'before/' + name)
    remote = []
    for path in sorted((r.BACKUP / 'run').glob('*/verified.json')):
        value = r.read(path)
        copy(path, 'backup-receipts/' + path.parent.name + '.json')
        remote.append(dict(run_id=path.parent.name, revision=value['commit_oid'],
                           verified_at_utc=value['verified_at_utc'], receipt=r.identity(path)))
    r.write(args.destination / 'verification.json', dict(archived_at_utc=r.now(), copies=rows,
        immutable_verified_backup_runs=remote, current_resume_outputs_copied=False,
        downloaded_large_payloads=r.identity(r.HERE / 'downloaded.json'),
        original_numerical_source_already_archived=True, live_sources_changed=False,
        gpu_resume_success_claimed=False, scientific_completion=False, source_publication=False))
    print(r.identity(args.destination / 'verification.json'))


if __name__ == '__main__':
    main()
