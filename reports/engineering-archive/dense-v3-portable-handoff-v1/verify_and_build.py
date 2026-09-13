"""Verify lossless handoff separation and run the unchanged full distribution audit."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WORK = Path('/tmp/dense-v3-portable-handoff.u0r0Q1W0')
ACTUAL = HERE / 'actual-verified'
ACTUAL.mkdir()

def identity(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

before_hashes = {
    'AGENTS.md': '8e9ae951d0f3a71c3329de50083d5b20b8f8f39821103aaf2cd96746960929fd',
    'PROJECT_STATUS.md': 'fc5bc3c19ddc00a42cdf7dc822cc169cc3e0c5c09f37eefeccca341c697e4969',
    'CURRENT_EXPERIMENT.md': 'be9857e55d8d1e7817b5c2cac75f5dfd745fa856a54fe5d9667199090bdbff5a',
}
for name, digest in before_hashes.items():
    assert identity(HERE / 'before' / name)['sha256'] == digest
agents = (ROOT / 'AGENTS.md').read_text()
for token in ('gpu.py or its processes', 'No broad process/GPU-process inspection',
              '196647/start246327790', '870313/start257721545', '870864/start257733056',
              'both lease FDs', 'LOCK_UN', '403', '9,629 paths / 366,252,912,201',
              'Do not self-waive', 'unobserved', 'No subagents', 'source-release gates',
              'historical archive', 'Do not hand-edit generated findings'):
    assert token in agents, token
links = []
for path in (ROOT / 'AGENTS.md', ROOT / 'PROJECT_STATUS.md', ROOT / 'CURRENT_EXPERIMENT.md',
             ROOT / 'README.md', HERE / 'README.md'):
    for link in re.findall(r'\]\(([^)]+)\)', path.read_text()):
        if '://' not in link and not link.startswith('#'):
            target = path.parent / link.split('#')[0]
            assert target.exists(), (path, link)
            links.append({'source': str(path.relative_to(ROOT)), 'link': link})
old_readme = (HERE / 'before/README.md').read_text()
new_readme = (ROOT / 'README.md').read_text()
marker = r'<!-- FINAL-CONCLUSION:BEGIN -->.*?<!-- FINAL-CONCLUSION:END -->'
assert re.search(marker, old_readme, re.S).group() == re.search(marker, new_readme, re.S).group()
assert old_readme.split('-->', 1)[0] == new_readme.split('-->', 1)[0]
assert identity(HERE / 'before/pyproject.toml') == identity(ROOT / 'pyproject.toml')

from embed_optim import current_paper
from embed_optim.distribution_audit import _checkout_path_findings
current_paper.read_snapshot(ROOT / 'paper/current')
source_identity = identity(ROOT / 'src/embed_optim/distribution_audit.py')
findings = {}
for name in before_hashes:
    findings[name] = {'before': _checkout_path_findings('documentation', name, (HERE / 'before' / name).read_bytes()),
                      'after': _checkout_path_findings('documentation', name, (ROOT / name).read_bytes())}
assert not any(row['after'] for row in findings.values())
with (ACTUAL / 'documentation.json').open('x') as stream:
    json.dump({'originals_unchanged': before_hashes, 'checked_local_links': links,
               'current_files': {name: identity(ROOT / name) for name in (*before_hashes, 'README.md')},
               'original_scanner': source_identity, 'documentation_findings': findings,
               'reviewed_paper_snapshot_unchanged': True,
               'readme_release_marker_and_attribution_unchanged': True,
               'packaging_declarations_unchanged': True}, stream, indent=2, sort_keys=True)
print(json.dumps({'local_links_checked': len(links), 'original_documents_preserved': len(before_hashes)}), flush=True)

env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(ROOT / 'src')}
def run(label, command):
    with (ACTUAL / (label + '.stdout')).open('xb') as out, (ACTUAL / (label + '.stderr')).open('xb') as err:
        child = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err)
    with (ACTUAL / (label + '.exit.json')).open('x') as stream:
        json.dump({'command': command, 'exit_code': child.returncode}, stream, indent=2)
    print(json.dumps({'label': label, 'actual_exit_code': child.returncode}), flush=True)
    return child.returncode

assert run('tests', ['/usr/bin/python', '-m', 'pytest', 'tests/test_current_paper.py',
    'tests/test_paper_layout.py', 'tests/test_current_distribution_surface.py', 'tests/test_distribution.py',
    '-q', '--junitxml=' + str(ACTUAL / 'tests.xml')]) == 0
assert run('build', ['/usr/bin/python', '-m', 'build', '--no-isolation', '--outdir', str(WORK / 'dist')]) == 0
audit_code = run('audit', ['/usr/bin/python', '-m', 'embed_optim.distribution_audit',
                          '--repo-root', str(ROOT), '--dist-dir', str(WORK / 'dist')])
audit = json.loads((ACTUAL / 'audit.stdout').read_bytes())
assert audit_code == 1 and audit['complete'] is False
previous = json.loads((ROOT / 'reports/engineering-archive/dense-v3-current-paper-entry-v1/actual/audit-final.stdout').read_bytes())
expected = [p for p in previous['problems'] if not any(name in p for name in ('AGENTS.md', 'PROJECT_STATUS.md'))]
assert audit['problems'] == expected and len(expected) == 6
assert identity(ROOT / 'src/embed_optim/distribution_audit.py') == source_identity
with (ACTUAL / 'comparison.json').open('x') as stream:
    json.dump({'previous_problems': previous['problems'], 'current_problems': audit['problems'],
               'resolved_documentation_findings': 4, 'remaining_findings': 6,
               'original_audit_unchanged': True, 'full_distribution_admission': False,
               'source_release_verified': False}, stream, indent=2)
