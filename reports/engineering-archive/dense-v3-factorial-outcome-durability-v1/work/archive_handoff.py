"""Preserve the live new entry, bounded checks, actual observations and ops preimages."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import outcomes as o

DEST = o.STORY / 'reports/engineering-archive/dense-v3-factorial-outcome-durability-v1'


def binding(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    o.need(not DEST.exists(), 'Preserve any earlier archive')
    names = ('PLAN.md', 'ARTIFACT_README.md', 'related-artifacts.json', 'outcomes.py',
             'test_outcomes.py', 'tests.json', 'actual_preflight.py', 'actual-preflight.json',
             'authorization.json', 'observe.py', 'observation-initial.json', 'evaluation-initial.json',
             'run/started.json', 'README.md', 'archive_handoff.py')
    plan = [(o.HERE / n, 'work/' + n) for n in names]
    plan.append((o.HERE / 'README.md', 'README.md'))
    plan.extend((path, 'bound-parents/' + str(i) + '-' + path.name) for i, path in enumerate(o.PINS))
    plan.extend((o.STORY / n, 'before/' + n) for n in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md', 'paper/main.tex'))
    rows = []
    for path, name in plan:
        o.sha(path)
        raw = path.read_bytes()
        o.need(not re.search(rb'wandb_v1_[A-Za-z0-9_-]{40,}|\bhf_[A-Za-z0-9]{30,}|\bgh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', raw),
               'Credential-shaped archive input refused')
        rows.append({'source': str(path), 'path': name, **binding(path)})
    DEST.mkdir(parents=True, exist_ok=False)
    for row in rows:
        source = Path(row['source'])
        expected = {k: row[k] for k in ('bytes', 'sha256')}
        o.need(binding(source) == expected, 'Copy input changed')
        target = DEST / row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
        o.need(binding(target) == expected, 'Archive copy differs')
    result = {'scope': 'new-complete-outcome-durability-waiter-handoff',
        'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'files': rows,
        'file_count': len(rows), 'bytes': sum(r['bytes'] for r in rows),
        'credential_shaped_findings': 0, 'actual_upload_completed': False,
        'source_release': False, 'scientific_completion': False}
    path = DEST / 'copy-inventory.json'
    with path.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps({'archive': str(DEST), 'manifest': binding(path), 'files': len(rows), 'bytes': result['bytes']}))


if __name__ == '__main__':
    main()
