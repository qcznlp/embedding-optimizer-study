"""Add only exact existing archival bytes, including the linked reviewed PDF."""
import base64
import csv
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
base = '9c594ff89c86f189ce47289f8a199c6c6aca6bb7'
data = json.loads((work / 'ignored-report-members.json').read_text())
files = data['omitted_files']
assert len(files) == 84 and data['omitted_bytes'] == 51747992
remote_manifest = 'reports/engineering-archive/dense-v3-second-stage-pool-b-artifact-backup-v1/artifact_manifest.json'
assert {item['manifest'] for item in data['unresolved_bindings']} == {remote_manifest}
assert len(data['unresolved_bindings']) == 280
remote_guide = (root / Path(remote_manifest).parent / 'README.md').read_text()
assert 'the exact externally anchored\n  remote manifest' in remote_guide
assert '35ce505d3cf3389f4f1a2f35b46259749bc1d7ed' in remote_guide
patterns = {**_credential_patterns(), 'HF': re.compile(b'h' + b'f_' + rb'[A-Za-z0-9]{25,}')}
verified_hash_matches = []
scanned_members = 0

def scan(name, payload, zip_owner=None, zip_name=None):
    global scanned_members
    scanned_members += 1
    for label, pattern in patterns.items():
        if not pattern.search(payload):
            continue
        if label != 'HF' or zip_owner is None or not zip_name.endswith('.dist-info/RECORD'):
            raise ValueError('Unresolved credential pattern: ' + name + ' (' + label + ')')
        count = 0
        for member, digest, size in csv.reader(io.StringIO(payload.decode())):
            assert not pattern.search(member.encode()) and not pattern.search(size.encode())
            matches = list(pattern.finditer(digest.encode()))
            if not matches:
                continue
            assert Path(member).name != 'gpu.py'
            raw = zip_owner.read(member)
            expected = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')
            assert digest == expected and int(size) == len(raw)
            count += len(matches)
        assert count == len(list(pattern.finditer(payload)))
        verified_hash_matches.append({'record': name, 'matches': count})
    if payload.startswith(b'PK\x03\x04'):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for item in archive.infolist():
                if not item.is_dir():
                    assert Path(item.filename).name != 'gpu.py' and item.file_size < 50_000_000
                    scan(name + '::' + item.filename, archive.read(item), archive, item.filename)
    elif name.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
            for item in archive:
                if item.isfile():
                    assert Path(item.name).name != 'gpu.py' and item.size < 50_000_000
                    scan(name + '::' + item.name, archive.extractfile(item).read())

assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip() == base
assert subprocess.check_output(['git', 'status', '--porcelain'], cwd=root) == b''
for name, binding in files.items():
    path = root / name
    assert not path.is_symlink() and Path(name).name != 'gpu.py'
    raw = path.read_bytes()
    assert len(raw) == binding['bytes'] and hashlib.sha256(raw).hexdigest() == binding['sha256']
    scan(name, raw)
report = root / 'reports/engineering-archive/dense-v3-archival-ignore-fix-v1'
report.mkdir(exist_ok=False)
readme = '''# Archived PDF and ignored-output delivery correction

Final anonymous-link review found that the reviewed PDF existed locally but was
excluded from the first public payload by the generic build-directory ignore rule.
The scientific/source payload and complete numerical closure were present, but
the README PDF link was not yet usable. This correction does not rewrite that history.

The top-level local archive inventories identify 84 omitted files / 51,747,992
bytes: nine PDFs, their original compilation outputs and eight wheel/sdist pairs.
Every added file matches its pre-existing manifest SHA-256 and byte count. No
source, test, protocol, scientific result, manuscript text or old manifest changes.
The reviewed PDF retains its original SHA-256
7f72c08717e4538cc498259043457f424d598abb92846b261461611b24f46d09.

All added payloads and readable archive members are credential-scanned. Any HF-like
RECORD substring is accepted only after independently recomputing the referenced
member's complete SHA-256 and size. No other match is admitted or pattern relaxed.

The 280 locally unresolved entries all belong to one explicitly external HF artifact
manifest, documented in the second-stage pool-B backup guide at immutable revision
35ce505d3cf3389f4f1a2f35b46259749bc1d7ed. They are not local archive omissions and
are not copied into another namespace. No completed HF download is repeated.

This is a publication correction only. Earlier 3,800 source-version test passes,
complete numerical/paper builds and distribution checks concern unchanged code.
The original failed link check and exact omitted-member inventory are retained here.
'''
(report / 'README.md').write_text(readme)
(report / 'ignored-report-members.json').write_bytes((work / 'ignored-report-members.json').read_bytes())
(report / 'publish_ignored_report_members.py').write_bytes(Path(__file__).read_bytes())
receipt = {'verified': True, 'omitted_file_count': len(files), 'omitted_bytes': data['omitted_bytes'],
           'scanned_payload_count': scanned_members, 'independently_verified_hash_substrings': verified_hash_matches,
           'unresolved_credential_findings': 0, 'source_commit_before': base,
           'failed_pdf_link_tool': '9f4dda / exit 128', 'scientific_or_executable_sources_changed': False}
(report / 'verification.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
paths = sorted(set(files) | {path.relative_to(root).as_posix() for path in report.iterdir()})
for name in paths:
    if name not in files:
        scan(name, (root / name).read_bytes())
subprocess.run(['git', 'add', '--force', '--pathspec-from-file=-', '--pathspec-file-nul'], cwd=root,
               input=b'\0'.join(name.encode() for name in paths) + b'\0', check=True)
for name in paths:
    staged = subprocess.check_output(['git', 'show', ':' + name], cwd=root)
    assert staged == (root / name).read_bytes()
with (work / 'archival-members-commit.log').open('xb') as stream:
    subprocess.run(['git', 'commit', '-m', 'Include manifest-bound reviewed PDF and archived build artifacts'], cwd=root,
                   stdout=stream, stderr=subprocess.STDOUT, check=True)
receipt['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip()
with (work / 'archival-members-commit.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
print(json.dumps(receipt), flush=True)
