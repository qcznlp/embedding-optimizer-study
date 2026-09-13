"""Actual source-distribution round trip without original source fallback."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime, timezone

HERE = Path(__file__).parent
ARCHIVE = HERE / 'built-expanded/embedding_optimizer_study-0.1.0.tar.gz'
WHEEL = HERE / 'built-expanded/embedding_optimizer_study-0.1.0-py3-none-any.whl'
RESTORED = HERE / 'restored-sdist'
OUTPUT = HERE / 'rebuilt-wheel'
EXPECTED = '4c246130848502e29aeb414b94e79921d1bbaaabdb276b1c34454c64f9b10a77'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('PYTHONPATH') != '':
    raise ValueError('Use hidden CUDA and empty PYTHONPATH')
if RESTORED.exists() or OUTPUT.exists() or sha(ARCHIVE) != EXPECTED:
    raise ValueError('Preserve old attempts; require exact original source archive')
RESTORED.mkdir()
with tarfile.open(ARCHIVE) as archive:
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or '..' in path.parts or path.parts[0] != 'embedding_optimizer_study-0.1.0' or not (member.isfile() or member.isdir()):
            raise ValueError('Unexpected source archive member')
    archive.extractall(RESTORED, filter='data')
project = RESTORED / 'embedding_optimizer_study-0.1.0'
command = [sys.executable, '-B', '-m', 'build', '--no-isolation', '--wheel', '--outdir', str(OUTPUT)]
result = subprocess.run(command, cwd=project, capture_output=True, text=True, check=False)
with (HERE / 'sdist-rebuild.log').open('x') as stream:
    stream.write(result.stdout)
    stream.write(result.stderr)
if result.returncode:
    raise SystemExit(result.returncode)
rebuilt = OUTPUT / WHEEL.name
with zipfile.ZipFile(WHEEL) as original, zipfile.ZipFile(rebuilt) as replay:
    names = {x for x in original.namelist() if not x.endswith('/')}
    actual_names = {x for x in replay.namelist() if not x.endswith('/')}
    if names != actual_names:
        raise ValueError('Rebuilt wheel inventory differs')
    differences = sorted(x for x in names if original.read(x) != replay.read(x))
    if differences:
        raise ValueError('Rebuilt wheel member payloads differ')
    helper_root = HERE / 'wheel-helper-roundtrip'
    helper_root.mkdir()
    helper_results = []
    for name in ('restore_primary_v3.py', 'restore_functional_analysis.py', 'restore_factorial_probes.py'):
        member = 'embedding_optimizer_study-0.1.0.data/data/share/embedding-optimizer-study/scripts/' + name
        with (helper_root / name).open('xb') as stream:
            stream.write(replay.read(member))
    for name in ('restore_primary_v3.py', 'restore_functional_analysis.py', 'restore_factorial_probes.py'):
        run = subprocess.run([sys.executable, '-B', str(helper_root / name), '--help'], cwd=helper_root,
                             capture_output=True, text=True, check=False)
        if run.returncode:
            raise ValueError('Relocated installed recovery CLI did not start')
        helper_results.append({'name': name, 'exit_code': run.returncode, 'stdout_sha256': hashlib.sha256(run.stdout.encode()).hexdigest(),
                               'source_sha256': sha(helper_root / name), 'mode': 'help_only_not_download_or_native_admission'})
record = {'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(Path(__file__)),
          'scope': 'actual_sdist_to_wheel_payload_roundtrip_not_scientific_release',
          'sdist_sha256': EXPECTED, 'build_exit_code': result.returncode,
          'original_wheel_sha256': sha(WHEEL), 'rebuilt_wheel_sha256': sha(rebuilt),
          'wheel_member_count': len(names), 'all_member_payloads_byte_exact': True,
          'relocated_helpers': helper_results, 'same_physical_host': True,
          'original_producer_pythonpath': False, 'installed_environment_modified': False,
          'original_full_distribution_audit_passed': False, 'scientific_completion': False}
with (HERE / 'sdist-roundtrip.json').open('x') as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps(record))
