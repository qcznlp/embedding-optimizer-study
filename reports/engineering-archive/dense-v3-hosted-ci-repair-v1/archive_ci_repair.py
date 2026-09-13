"""Preserve the actual failed hosted artifact and bounded local repair evidence."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile

from embed_optim.distribution_audit import _credential_patterns

root = Path('/root/embedding-optimizer-story-refactor')
work = Path(__file__).resolve().parent
out = root / 'reports/engineering-archive/dense-v3-hosted-ci-repair-v1'
out.mkdir(exist_ok=False)
base = '65c584ce11435dc84aec675a6ed733a9a5fc97d1'
def git(*args):
    return subprocess.check_output(['git', *args], cwd=root)
assert git('rev-parse', 'HEAD').decode().strip() == base
assert git('diff', base, '--', 'src', 'scripts', 'configs', 'paper', 'requirements-formal.lock', 'requirements-formal-flash.txt') == b''
changed = git('diff', '--name-only').decode().splitlines()
patterns = {**_credential_patterns(), 'HF': re.compile(b'h' + b'f_' + rb'[A-Za-z0-9]{25,}')}
def save(name, raw):
    assert not any(p.search(raw) for p in patterns.values()), name
    target = out / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(raw)
for name in changed:
    save('before/' + name, git('show', base + ':' + name))
    save('after/' + name, (root / name).read_bytes())
artifact = work / 'ci-source-803c70dc.zip'
assert hashlib.sha256(artifact.read_bytes()).hexdigest() == '8b42a41245fe672611dc5ca1f8cebe24710da9ac7da99f8003738c5e5f1b6eaf'
with zipfile.ZipFile(artifact) as archive:
    for name in archive.namelist():
        raw = archive.read(name)
        assert not any(p.search(raw) for p in patterns.values()), name
save('hosted/ci-source-803c70dc.zip', artifact.read_bytes())
for name in ['ci-repair-current-tests.log', 'ci-repair-current-tests.xml', 'ci-primary-fixture-tests.log', 'ci-primary-fixture-tests.xml', 'ci-relocated-factorial-fixture/original-factorial.log', 'ci-relocated-factorial-fixture/original-factorial.xml', 'ci-relocated-factorial-fixture/original-factorial-result.json', 'ci-relocated-factorial-fixture/inputs.json', 'distribution-ci-repair.json', 'build-ci-repair.log']:
    save('local/' + name, (work / name).read_bytes())
cases = {}
for name in ['ci-repair-current-tests.xml', 'ci-relocated-factorial-fixture/original-factorial.xml']:
    suites = ET.parse(work / name).getroot().findall('testsuite')
    assert all(s.attrib[k] == '0' for s in suites for k in ['errors', 'failures', 'skipped'])
    cases[name] = sum(int(s.attrib['tests']) for s in suites)
save('README.md', '''# Hosted CI repair — verification pending

The first public-source hosted run [34756580699](https://github.com/qcznlp/embedding-optimizer-study/actions/runs/34756580699)
failed. Its unmodified artifact ZIP is retained: 2,873 current cases had five
failures and 34 errors; 733 original-analysis cases had one failure; 194 original-
factorial cases had 77 failures. No test cases were skipped; the later paper step
did not run. Local 3,800-case acceptance did not establish hosted CI acceptance.

The hosted developer environment differed from the frozen scientific runtime
and lacked FlashAttention. CI now installs the genuine hash-locked runtime and
compiles its real CUDA extension. It neither fabricates package metadata nor
changes native guards. Subsequent commands use no-sync to preserve formal pins.
Two synthetic fixtures now explicitly relocate an authenticated runtime spec
and create an uncommitted repository, instead of depending on the producer host.
Production source, immutable runtime locks, scientific outputs, source-role
overlays, numerical assertions and tolerances are unchanged.

Local corrected original-factorial execution: all 194 cases pass, coordinator
27932 / b71966 / exit 0. The focused current XML records 85 passing cases; its
original terminal tool result was lost, so its OS exit is not asserted. Full
hosted verification of this repair is pending. No new training or GPU work.

Before/after edited files, exact failed hosted artifact, local logs/XML/source
inventories and actual distribution audit are retained. This report does not
supersede old failures or label an unexecuted CI revision successful.
'''.encode())
save('archive_ci_repair.py', Path(__file__).read_bytes())
files = {}
for path in sorted(out.rglob('*')):
    if path.is_file():
        raw = path.read_bytes()
        files[str(path.relative_to(out))] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
save('manifest.json', (json.dumps({'base_commit': base, 'hosted_run': 34756580699, 'hosted_conclusion': 'failure', 'repair_hosted_verification': 'pending', 'local_cases': cases, 'files': files}, indent=2, sort_keys=True) + '\n').encode())
print(json.dumps({'archive': str(out), 'files': len(files) + 1, 'local_cases': cases, 'manifest_sha256': hashlib.sha256((out / 'manifest.json').read_bytes()).hexdigest()}))
