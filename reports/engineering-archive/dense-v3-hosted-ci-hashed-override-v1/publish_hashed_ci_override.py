"""Preserve the real installer failure and publish its bounded installer fix."""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from embed_optim.distribution_audit import _credential_patterns

root = Path('/root/embedding-optimizer-story-refactor')
work = Path(__file__).resolve().parent
out = root / 'reports/engineering-archive/dense-v3-hosted-ci-hashed-override-v1'
out.mkdir(exist_ok=False)
def git(*args):
    return subprocess.check_output(['git', *args], cwd=root)
base = '0878698e52a20b691c5705cbe1ebc0495da58357'
assert git('rev-parse', 'HEAD').decode().strip() == base
assert git('diff', '--cached', '--name-only') == b''
assert git('diff', base, '--', 'src', 'scripts', 'configs', 'paper', 'requirements-formal.lock', 'requirements-formal-flash.txt') == b''
changes = set(git('diff', '--name-only').decode().splitlines())
assert changes == {'.github/workflows/ci.yml', 'AGENTS.md', 'CONTRIBUTING.md', 'CURRENT_EXPERIMENT.md', 'tests/test_source_roles.py'}
def save(name, raw):
    assert not any(p.search(raw) for p in _credential_patterns().values()), name
    target = out / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(raw)
for name in changes:
    save('before/' + name, git('show', base + ':' + name))
    save('after/' + name, (root / name).read_bytes())
for name in ['ci-0878698e-job-raw.log', 'ci-formal-install-overrides-dry-run.log', 'ci-formal-install-overrides-isolated.log', 'ci-formal-install-hashed-overrides-isolated.log', 'ci-hashed-override-tests.xml', 'ci-hashed-override-tests.log']:
    save('actual/' + name, (work / name).read_bytes())
suites = ET.parse(work / 'ci-hashed-override-tests.xml').getroot().findall('testsuite')
assert all(s.attrib[k] == '0' for s in suites for k in ['errors', 'failures', 'skipped'])
count = sum(int(s.attrib['tests']) for s in suites)
save('README.md', f'''# Formal hashed installer override — hosted verification pending

Actual hosted repair run [34758294285](https://github.com/qcznlp/embedding-optimizer-study/actions/runs/34758294285)
failed before tests: fast-plaid requests Torch 2.9.0, while the immutable formal
environment records Torch 2.9.1+cu129. The installer omitted the original formal
version override. CUDA compiler installation succeeded. No test verdict follows.

Using the same original requirements-formal.lock as both hashed requirements and
hashed overrides resolves successfully in an empty isolated Python 3.12 environment
(12a81c / exit 0, dry run only). The first dry run targeting the system interpreter
was refused as externally managed; no bypass or installation was performed. An
isolated unhashed-constraints attempt was then rejected by require-hashes. Both
failures are preserved. No lock, native guard, numerical source or result changed.

Current focused JUnit: {count} cases, zero failures/errors/skips. Ruff and diff-check
pass. The hosted log is retained unchanged, including its terminal failure.
This commit repairs CI and contributor commands only; full hosted execution is
still pending. No GPU or live study environment changes.
'''.encode())
save('publish_hashed_ci_override.py', Path(__file__).read_bytes())
files = {}
for p in sorted(out.rglob('*')):
    if p.is_file():
        raw = p.read_bytes()
        files[str(p.relative_to(out))] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
save('manifest.json', (json.dumps({'base_commit': base, 'hosted_run': 34758294285, 'hosted_conclusion': 'failure', 'repair_hosted_verification': 'pending', 'files': files}, indent=2, sort_keys=True) + '\n').encode())
paths = sorted(changes | {str(p.relative_to(root)) for p in out.rglob('*') if p.is_file()})
assert set(git('ls-files', '--others', '--exclude-standard').decode().splitlines()).issubset(paths)
subprocess.run(['git', 'diff', '--check'], cwd=root, check=True)
subprocess.run(['git', 'add', '--force', '--pathspec-from-file=-', '--pathspec-file-nul'], cwd=root, input=b'\0'.join(p.encode() for p in paths) + b'\0', check=True)
for name in paths:
    assert git('show', ':' + name) == (root / name).read_bytes()
with (work / 'ci-hashed-override-commit.log').open('xb') as log:
    subprocess.run(['git', 'commit', '-m', 'Retain formal hashed dependency overrides during CI installation'], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
print(json.dumps({'commit': git('rev-parse', 'HEAD').decode().strip(), 'files': len(paths), 'tests': count, 'push_pending': True}))
