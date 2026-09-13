"""Copy only this completed audit and its exact metadata inputs, with a digest inventory."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

WORK = Path(__file__).resolve().parent
STORY = Path('/root/embedding-optimizer-story-refactor')
DEST = STORY / 'reports/engineering-archive/dense-v3-final-tracking-audit-v1'
PATTERNS = (rb'wandb_v1_[A-Za-z0-9_-]{40,}', rb'\bhf_[A-Za-z0-9]{30,}',
            rb'\bgh[pousr]_[A-Za-z0-9]{30,}', rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')


def identity(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    if DEST.exists():
        raise ValueError('Refuse existing archive')
    inputs = json.loads((WORK / 'actual/input-bindings.json').read_bytes())['files']
    plan = [(p, 'work/' + p.relative_to(WORK).as_posix(), None)
            for p in sorted(WORK.rglob('*')) if p.is_file()]
    plan.extend((Path(p), 'native-inputs/' + f'{i:03d}-' + Path(p).name, bound)
                for i, (p, bound) in enumerate(sorted(inputs.items())))
    plan.extend((STORY / p, 'before/' + p, None)
                for p in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md', 'paper/main.tex'))
    plan.append((WORK / 'README.md', 'README.md', None))
    rows = []
    # Validate all explicit source files before any copy; never read credential stores.
    for source, relative, expected in plan:
        if source.name == 'gpu.py' or any(p.is_symlink() for p in (source, *source.parents)):
            raise ValueError('Excluded or nonordinary source')
        data = source.read_bytes()
        if any(re.search(pattern, data) for pattern in PATTERNS):
            raise ValueError('Credential-shaped content refused')
        got = identity(data)
        if expected is not None and got != expected:
            raise ValueError('Native source binding differs')
        rows.append({'source': str(source), 'path': relative, **got})
    DEST.mkdir(parents=True, exist_ok=False)
    for row in rows:
        data = Path(row['source']).read_bytes()
        if identity(data) != {k: row[k] for k in ('bytes', 'sha256')}:
            raise ValueError('Copy input raced')
        target = DEST / row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
        if identity(target.read_bytes()) != {k: row[k] for k in ('bytes', 'sha256')}:
            raise ValueError('Copy readback differs')
    result = {'scope': 'actual-read-only-tracking-archive', 'observed_at_utc': datetime.now(timezone.utc).isoformat(),
              'files': rows, 'file_count': len(rows), 'bytes': sum(r['bytes'] for r in rows),
              'credential_shaped_findings': 0, 'source_release': False, 'scientific_completion': False}
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with (DEST / 'copy-inventory.json').open('xb') as stream:
        stream.write(data)
    print(json.dumps({'archive': str(DEST), 'files': len(rows), 'bytes': result['bytes'],
                      'inventory': identity(data), 'credential_shaped_findings': 0}))


if __name__ == '__main__':
    main()
