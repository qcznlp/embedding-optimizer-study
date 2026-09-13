"""Read-only high-confidence credential scan; never print matching secret bytes."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import zipfile

from embed_optim.distribution_audit import _credential_patterns

root = Path('/root/embedding-optimizer-story-refactor')
work = Path(__file__).resolve().parent
inventory = json.loads((work / 'full-source-tests-v1/inputs.json').read_text())['files']
patterns = {**_credential_patterns(), 'Hugging Face access token': re.compile(b'h' + b'f_' + rb'[A-Za-z0-9]{25,}')}
findings = []
counts = {'candidate_files': 0, 'candidate_bytes': 0, 'history_objects': 0, 'history_bytes': 0, 'archive_members': 0, 'archive_bytes': 0}

def inspect(scope, name, payload, depth=0):
    for label, pattern in patterns.items():
        if pattern.search(payload):
            findings.append({'scope': scope, 'path': name, 'pattern': label})
    if depth >= 2:
        return
    if payload.startswith(b'PK\x03\x04'):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for item in archive.infolist():
                if item.is_dir() or Path(item.filename).name == 'gpu.py':
                    continue
                if item.file_size > 150_000_000:
                    raise ValueError('Oversized archive member needs separate audit: ' + name)
                raw = archive.read(item)
                counts['archive_members'] += 1
                counts['archive_bytes'] += len(raw)
                inspect(scope, name + '::' + item.filename, raw, depth + 1)
    elif name.endswith(('.tar.gz', '.tgz', '.tar')):
        with tarfile.open(fileobj=io.BytesIO(payload), mode='r:*') as archive:
            for item in archive:
                if not item.isfile() or Path(item.name).name == 'gpu.py':
                    continue
                if item.size > 150_000_000:
                    raise ValueError('Oversized archive member needs separate audit: ' + name)
                raw = archive.extractfile(item).read()
                counts['archive_members'] += 1
                counts['archive_bytes'] += len(raw)
                inspect(scope, name + '::' + item.name, raw, depth + 1)

for name, binding in inventory.items():
    assert Path(name).name != 'gpu.py'
    raw = (root / name).read_bytes()
    # Record any concurrent candidate change; do not silently claim the old identity.
    if hashlib.sha256(raw).hexdigest() != binding['sha256']:
        raise ValueError('Candidate changed since full-suite inventory: ' + name)
    inspect('candidate', name, raw)
    counts['candidate_files'] += 1
    counts['candidate_bytes'] += len(raw)

objects = subprocess.check_output(['git', 'rev-list', '--objects', '--all'], cwd=root).decode().splitlines()
protected = {line.split(' ', 1)[0] for line in objects if ' ' in line and Path(line.split(' ', 1)[1]).name == 'gpu.py'}
child = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
try:
    for line in objects:
        oid, _, name = line.partition(' ')
        if oid in protected:
            continue
        child.stdin.write((oid + '\n').encode())
        child.stdin.flush()
        header = child.stdout.readline().decode().strip().split()
        if len(header) != 3 or header[0] != oid:
            raise ValueError('Unexpected Git object response')
        size = int(header[2])
        raw = child.stdout.read(size)
        assert len(raw) == size and child.stdout.read(1) == b'\n'
        # Commit/tag messages can contain credentials too. Tree binary names do not
        # confer authorization to inspect protected helper blobs, excluded above.
        inspect('git-history-' + header[1], name or oid, raw)
        counts['history_objects'] += 1
        counts['history_bytes'] += size
finally:
    child.stdin.close()
    code = child.wait()
    if code:
        raise ValueError('Git object reader failed')

result = {'complete': not findings, 'patterns': sorted(patterns), 'counts': counts, 'findings': findings,
          'protected_helper_blob_ids_excluded': len(protected), 'scope': 'candidate inventory and reachable Git objects, including readable ZIP/tar members; high-confidence patterns, not a guarantee against every possible secret representation',
          'candidate_inventory_sha256': hashlib.sha256((work / 'full-source-tests-v1/inputs.json').read_bytes()).hexdigest()}
with (work / 'release-credential-audit.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
print(json.dumps(result), flush=True)
raise SystemExit(0 if result['complete'] else 1)
