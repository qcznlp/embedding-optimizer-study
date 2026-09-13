"""Bind genuine hosted/local CPU diagnostics, without accepting a failed replay."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

work = Path(__file__).resolve().parent
root = Path('/root/embedding-optimizer-story-refactor')
old = Path('/tmp/dense-v3-flash-wheel.L7A0IiP0')
control = Path('/tmp/ci-numerical-runtime.AQg9TtWv')
out = root / 'reports/engineering-archive/dense-v3-ci-cpu-emulation-v1'
zip_path = old / 'ci-ccdbcf27-receipts.zip'
assert zip_path.stat().st_size == 14432
assert hashlib.sha256(zip_path.read_bytes()).hexdigest() == '0b7befc2e3f77779b01d0e16f9711eede6985567f62f1e716f9de268d26b3cbc'
with zipfile.ZipFile(zip_path) as archive:
    hosted = json.loads(archive.read('complete-paper/diagnostics/functional.json'))
    cpu = archive.read('complete-paper/diagnostics/cpu.txt').decode()
local = json.loads((control / 'functional-haswell.json').read_text())
assert hosted['actual_decisions'] == local['actual_decisions']
assert hosted['differences'] == local['differences']
assert hosted['functional_tables_exact'] is True and hosted['decisions_exact'] is False
assert len(hosted['differences']) == 136 and 'EPYC 7763' in cpu
assert 'avx2' in cpu and 'avx512' not in cpu
maximum = max(row['absolute_difference'] for row in hosted['differences'])
assert maximum == 4.440892098500626e-15
forced = json.loads((work / 'functional-forced-skx.json').read_text())
assert forced['functional_tables_exact'] is forced['decisions_exact'] is True
assert forced['differences'] == [] and forced['acceptance_override'] is False
kit = work / 'sde-external-10.13.1-2026-07-28-lin.tar.xz'
assert hashlib.sha256(kit.read_bytes()).hexdigest() == '94e97d623fec54385686e1e7ba65ebc9941748c05ee451423948334892bf2b50'
files = [zip_path, old/'ci-ccdbcf27-job.log', work/'functional-forced-skx.json',
         work/'functional-forced-skx.log', work/'functional-skx.json',
         work/'functional-skx.log', work/'smoke.log', work/'pin-log.txt',
         work/'focused-tests.log', work/'focused-tests.xml', Path(__file__)]
for source in files:
    assert not (out / source.name).exists()
    shutil.copyfile(source, out/source.name)
receipt = {'scope': 'cpu-backend-diagnosis-and-local-instruction-emulation',
           'hosted_run': 34768399864, 'hosted_job': 103753403488,
           'hosted_full_replay_passed': False, 'hosted_functional_tables_exact': True,
           'whole_hosted_decisions_match_local_haswell': True,
           'diagnostic_differences': 136, 'max_absolute_difference': maximum,
           'forced_emulation_probe_exit': 0, 'forced_emulation_tables_and_decisions_exact': True,
           'first_nonforced_probe_exit': None, 'chip_list_exit': 127,
           'complete_emulated_replay_claimed': False, 'source_or_assertions_modified': False,
           'gpu_execution': False, 'security_settings_changed': False,
           'sde_binary_redistributed': False}
with (out/'verified.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
manifest = {'files': {p.name: {'bytes': p.stat().st_size,
                             'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in sorted(out.iterdir())}}
with (out/'manifest.json').open('x') as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
print(json.dumps({'files':len(manifest['files']), 'manifest_sha256':hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()}))
