"""Independently classify the ten candidate findings without weakening scanning."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

root = Path('/root/embedding-optimizer-story-refactor')
work = Path(__file__).resolve().parent
scan = json.loads((work / 'release-credential-audit.json').read_text())
pattern = re.compile(b'h' + b'f_' + rb'[A-Za-z0-9]{25,}')
verified = []
for finding in scan['findings']:
    assert finding['scope'] == 'candidate' and finding['pattern'] == 'Hugging Face access token'
    archive_name, record_name = finding['path'].split('::')
    assert archive_name.endswith('.whl') and record_name.endswith('.dist-info/RECORD')
    with zipfile.ZipFile(root / archive_name) as archive:
        raw = archive.read(record_name)
        total = len(list(pattern.finditer(raw)))
        matched_rows = []
        for row in csv.reader(io.StringIO(raw.decode())):
            assert len(row) == 3
            member_name, declared_hash, declared_size = row
            assert not pattern.search(member_name.encode()) and not pattern.search(declared_size.encode())
            matches = list(pattern.finditer(declared_hash.encode()))
            if not matches:
                continue
            assert declared_hash.startswith('sha256=') and Path(member_name).name != 'gpu.py'
            payload = archive.read(member_name)
            actual = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip('=')
            assert declared_hash == actual and int(declared_size) == len(payload)
            matched_rows.append({'member': member_name, 'match_count': len(matches), 'actual_sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)})
        assert total > 0 and total == sum(row['match_count'] for row in matched_rows)
    verified.append({'finding': finding, 'classification': 'substring of independently recomputed wheel RECORD SHA-256, not a credential', 'verified_members': matched_rows})
assert len(verified) == 10
result = {'complete': True, 'scan_sha256': hashlib.sha256((work / 'release-credential-audit.json').read_bytes()).hexdigest(),
          'original_scan_modified': False, 'scanner_relaxed': False, 'unresolved_findings': 0, 'classifications': verified}
with (work / 'release-credential-classification.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
print(json.dumps({'complete': True, 'verified_wheel_hash_false_positives': len(verified), 'unresolved_findings': 0}), flush=True)
