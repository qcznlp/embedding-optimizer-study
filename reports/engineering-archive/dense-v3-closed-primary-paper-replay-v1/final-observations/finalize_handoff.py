"""Add final observations/document copies without rewriting the accepted archive."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
STORY = Path('/root/embedding-optimizer-story-refactor')
REPORT = STORY / 'reports/engineering-archive/dense-v3-closed-primary-paper-replay-v1'


def identity(path):
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Ordinary file required')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}


def main():
    old = REPORT / 'archive-manifest.json'
    assert identity(old)['sha256'] == '172db76419a8efb9af2174c4fd6dc02eedec816171a318ede81ed1ce274d7986'
    manifest = json.loads(old.read_bytes())
    for name, expected in manifest['files'].items():
        assert identity(REPORT / name) == {k: expected[k] for k in ('bytes', 'sha256')}
    final = REPORT / 'final-observations'
    final.mkdir(exist_ok=False)
    copies = {}

    def copy(original, name):
        target = final / name
        target.parent.mkdir(parents=True, exist_ok=True)
        before = identity(original)
        if target.exists():
            raise ValueError('Preserve existing handoff')
        shutil.copyfile(original, target)
        assert identity(target) == identity(original) == before
        copies['final-observations/' + name] = {**before, 'origin': str(original)}

    for original in sorted((HERE / 'final-observations').iterdir()):
        copy(original, original.name)
    copy(Path(__file__), 'finalize_handoff.py')
    for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md',
                 'docs/paper-results-reproduction.md'):
        copy(STORY / name, 'after/' + name)
    copies['README.md'] = identity(REPORT / 'README.md')
    patterns = [rb'wandb_v1_[A-Za-z0-9_-]{40,}', rb'\bhf_[A-Za-z0-9]{30,}',
                rb'\bgh[pousr]_[A-Za-z0-9]{30,}', rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
    for name in copies:
        assert not any(re.search(p, (REPORT / name).read_bytes()) for p in patterns), 'Credential-shaped text'
    protected = manifest['original_source_files_unchanged']
    for name, expected in protected.items():
        assert identity(Path(name))['sha256'] == expected
    result = {'scope': 'closed-primary-numerical-replay-final-documentation-and-observations',
              'observed_at_utc': datetime.now(timezone.utc).isoformat(),
              'original_manifest': identity(old), 'additional_files': copies,
              'additional_files_count': len(copies),
              'additional_bytes': sum(v['bytes'] for v in copies.values()),
              'original_archive_inputs_unchanged': True,
              'protected_scientific_sources_and_authoritative_manuscript_unchanged': True,
              'source_release': False, 'full_goal_complete': False,
              'goal_turn_classification': 'PROGRESS'}
    target = REPORT / 'final-handoff.json'
    with target.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'handoff': identity(target), 'additional_files': len(copies),
                      'original_archive_unchanged': True, 'full_goal_complete': False}))


if __name__ == '__main__':
    main()
