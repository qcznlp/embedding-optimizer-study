"""Build revised distributions, retain original audit result, run relocated wheel."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = Path('/root/embedding-optimizer-story-refactor')
REVIEW = ROOT / 'reports/paper-review/dense-v3-complete-manuscript-revision-v1/actual/paper/build/main.pdf'

def run(label, argv, cwd, env):
    with (HERE / (label + '.stdout')).open('xb') as out, (HERE / (label + '.stderr')).open('xb') as err:
        child = subprocess.run(argv, cwd=cwd, env=env, stdout=out, stderr=err)
    with (HERE / (label + '.exit.json')).open('x') as stream:
        json.dump({'command': argv, 'cwd': str(cwd), 'exit_code': child.returncode}, stream, indent=2)
    print(json.dumps({'label': label, 'actual_exit_code': child.returncode}), flush=True)
    return child.returncode

env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(ROOT / 'src')}
assert run('build-final', ['/usr/bin/python', '-m', 'build', '--no-isolation',
                          '--outdir', str(HERE / 'dist-final')], ROOT, env) == 0
audit_code = run('audit-final', ['/usr/bin/python', '-m', 'embed_optim.distribution_audit',
    '--repo-root', str(ROOT), '--dist-dir', str(HERE / 'dist-final')], ROOT, env)
# Preserve and report the full original failure. A passing relocated document
# must not be promoted to a passing distribution gate.
audit = json.loads((HERE / 'audit-final.stdout').read_bytes())
assert audit_code == 1 and audit['complete'] is False
assert len(audit['problems']) == 10 and all('producer checkout path:' in p for p in audit['problems'])
wheel = Path(audit['wheel']['path'])
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == audit['wheel']['sha256']
package = HERE / 'extracted'
package.mkdir()
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
    assert len(names) == len(set(names))
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        assert not path.is_absolute() and '..' not in path.parts and not stat.S_ISLNK(mode)
        target = package / info.filename
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        assert not stat.S_IFMT(mode) or stat.S_ISREG(mode)
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = archive.read(info.filename)
        with target.open('xb') as stream:
            stream.write(raw)
        assert target.read_bytes() == raw
text = subprocess.run(['pdftotext', '-layout', str(REVIEW), '-'], check=True, capture_output=True).stdout
with (HERE / 'reviewed-pdf-text.txt').open('xb') as stream:
    stream.write(text)
env['PYTHONPATH'] = str(package)
assert run('packaged-child', ['/usr/bin/python', '-B', str(HERE / 'packaged_child.py')], HERE, env) == 0
with (HERE / 'package-check.json').open('x') as stream:
    json.dump({'wheel': audit['wheel'], 'sdist': audit['sdist'], 'wheel_files_extracted': len(names),
               'declared_data_files': audit['declared_data_files'],
               'actual_relocated_document_passed': True, 'full_distribution_audit_passed': False,
               'unchanged_original_audit_problems': audit['problems'],
               'physical_second_host': False, 'source_release_verified': False}, stream, indent=2)
