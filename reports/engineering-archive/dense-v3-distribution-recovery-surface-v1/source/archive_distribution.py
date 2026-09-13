"""Preserve this bounded packaging attempt, including both failed full audits."""

from pathlib import Path
import argparse
import difflib
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone

HERE = Path(__file__).parent
ROOT = Path('/root/embedding-optimizer-story-refactor')
DEST = ROOT / 'reports/engineering-archive/dense-v3-distribution-recovery-surface-v1'
OPS = ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def copy(source, relative):
    if source.name == 'gpu.py' or source.is_symlink() or not source.is_file():
        raise ValueError('Not a regular in-scope input')
    target = DEST / relative
    if target.exists():
        raise ValueError('Preserve existing archive material')
    target.parent.mkdir(parents=True, exist_ok=True)
    original = sha(source)
    shutil.copy2(source, target)
    if sha(source) != original or sha(target) != original:
        raise ValueError('Copy or source changed')


def prepare():
    baseline = json.loads((HERE / 'distribution-inputs.json').read_text())
    expanded = json.loads((HERE / 'distribution-expanded-inputs.json').read_text())
    for name, binding in expanded['files'].items():
        if name.endswith('/gpu.py') or sha(ROOT / name) != binding['sha256']:
            raise ValueError('Expanded build input changed before archive')
    changed = sorted(name for name in baseline['files'] if name in expanded['files']
                     and baseline['files'][name] != expanded['files'][name])
    if changed != ['pyproject.toml']:
        raise ValueError('Unexpected existing-input edit')
    added = sorted(set(expanded['declared_data_files']) - set(baseline['declared_data_files']))
    if len(added) != 40 or set(baseline['declared_data_files']) - set(expanded['declared_data_files']):
        raise ValueError('Unexpected distribution data delta')
    for name in OPS:
        copy(ROOT / name, 'ops-before/' + name)
    for name in ('build_distribution_baseline.py', 'build_distribution_expanded.py',
                 'rebuild_from_sdist.py', 'sample_owned_evaluation.py', 'archive_distribution.py'):
        copy(HERE / name, 'source/' + name)
    records = (
        'distribution-inputs.json', 'distribution-expanded-inputs.json',
        'distribution-build.json', 'distribution-build.log',
        'distribution-expanded-build.json', 'distribution-expanded-build.log',
        'distribution-audit-baseline.json', 'distribution-audit-expanded.json',
        'distribution-tests.xml', 'distribution-tests-final.xml',
        'sdist-rebuild.log', 'sdist-roundtrip.json',
        'evaluation-initial.json', 'evaluation-final.json', 'evaluation-stacks.json',
        'evaluation-after-build.json', 'evaluation-closeout.json', 'summary-closeout.json',
        'document-closeout.json', 'combined-closeout.json', 'resume-closeout.json',
        'outcome-durability-closeout.json',
    )
    for name in records:
        copy(HERE / name, 'actual/' + name)
    for stage in ('baseline', 'expanded'):
        copy(HERE / ('distribution-' + stage) / 'pyproject.toml', stage + '/pyproject.toml')
        for name in ('embedding_optimizer_study-0.1.0-py3-none-any.whl',
                     'embedding_optimizer_study-0.1.0.tar.gz'):
            copy(HERE / ('built-' + stage) / name, stage + '/' + name)
    copy(HERE / 'rebuilt-wheel/embedding_optimizer_study-0.1.0-py3-none-any.whl',
         'roundtrip/embedding_optimizer_study-0.1.0-py3-none-any.whl')
    for name in ('tests/test_current_distribution_surface.py', 'tests/test_distribution.py',
                 'src/embed_optim/distribution_audit.py', 'MANIFEST.in'):
        copy(ROOT / name, 'source-current/' + name)
    for name in added:
        copy(ROOT / name, 'added-data/' + name)
    before = (DEST / 'baseline/pyproject.toml').read_text().splitlines(keepends=True)
    after = (DEST / 'expanded/pyproject.toml').read_text().splitlines(keepends=True)
    with (DEST / 'actual/pyproject-task-only.diff').open('x') as stream:
        stream.writelines(difflib.unified_diff(before, after,
                          fromfile='before-this-task/pyproject.toml',
                          tofile='after-this-task/pyproject.toml'))
    write(DEST / 'actual/task-delta.json', {
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'baseline_data_files': len(baseline['declared_data_files']),
        'expanded_data_files': len(expanded['declared_data_files']),
        'added_data_files': added, 'removed_data_files': [],
        'changed_preexisting_build_inputs': changed,
        'new_test': 'tests/test_current_distribution_surface.py',
        'all_expanded_build_inputs_unchanged_before_ops_update': True,
        'original_full_distribution_audit_passed': False,
        'scientific_sources_changed': False, 'scientific_completion': False,
        'source_release': False,
    })
    print(json.dumps({'prepared': True, 'archive': str(DEST), 'added_data_files': len(added)}))


def seal():
    for name in OPS:
        copy(ROOT / name, 'ops-after/' + name)
    # Use the unchanged original scanner; print no matching contents.
    sys.path.insert(0, str(ROOT / 'src'))
    from embed_optim.distribution_audit import _credential_findings
    bindings = {}
    findings = []
    for path in sorted(DEST.rglob('*')):
        if path.is_symlink() or path.name == 'gpu.py':
            raise ValueError('Out-of-scope archive member')
        if not path.is_file():
            continue
        name = path.relative_to(DEST).as_posix()
        payload = path.read_bytes()
        findings.extend(_credential_findings('archive', name, payload))
        bindings[name] = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
    if findings:
        raise ValueError('Credential-shaped archive material; do not publish')
    # Wheel/tar members were separately checked by the unchanged full audit.
    write(DEST / 'archive-manifest.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'bounded_distribution_recovery_coverage_not_release',
        'files': bindings, 'files_count': len(bindings),
        'total_bytes': sum(value['bytes'] for value in bindings.values()),
        'credential_shaped_findings': [],
        'original_full_audits': {'baseline_complete': False, 'expanded_complete': False},
        'compressed_member_scanning': 'unchanged original distribution audit, original failures retained',
        'same_physical_host': True, 'scientific_completion': False, 'source_release': False,
    })
    for name, binding in bindings.items():
        if sha(DEST / name) != binding['sha256']:
            raise ValueError('Archive readback differs')
    print(json.dumps({'archive_files': len(bindings),
                      'archive_bytes': sum(value['bytes'] for value in bindings.values()),
                      'manifest_sha256': sha(DEST / 'archive-manifest.json'),
                      'original_full_distribution_audit_passed': False,
                      'scientific_completion': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'seal'))
    args = parser.parse_args()
    {'prepare': prepare, 'seal': seal}[args.mode]()
