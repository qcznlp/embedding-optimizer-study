"""Check the bounded initial launch archive and append-only later observations."""
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path('/root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-v3-functional-recovery-launch-v1')
sys.path.insert(0, str(ROOT / 'source'))
import recovery_support as support

original = support.read(ROOT / 'verification.json',
                        '45e0059a85a529a92df53112b29989e29009c0f762f80dc7863d7285d1fa9fbe')
for name, expected in original['files'].items():
    support.need(support.identity(ROOT / name) == expected, 'Initial archive payload changed')
for path, expected in original['protected_original_files_unchanged'].items():
    support.need(support.identity(path) == expected, 'Original dependency changed')
for number in ('third', 'fourth'):
    receipt = support.read(ROOT / 'actual' / f'observation-{number}.process.json')
    support.need(receipt['exit_code'] == 0 and support.identity(ROOT / 'actual' / f'observation-{number}.json')['sha256']
                 == receipt['stdout_sha256'], 'Later observer output differs')
latest = support.read(ROOT / 'actual/observation-fourth.json')
support.need(latest['coordinator']['pid'] == 766302 and latest['coordinator']['start_ticks'] == 319358599
             and len(latest['states']['new_verified']) == 10 and latest['states']['reused_pretrained']
             and not latest['states']['features_verified'], 'Latest snapshot count/identity differs')
result = {'verified_at_utc': datetime.now(timezone.utc).isoformat(),
          'initial_verification': support.record(ROOT / 'verification.json'),
          'files': {str(p.relative_to(ROOT)): support.identity(p) for p in sorted(ROOT.rglob('*')) if p.is_file()},
          'latest_observed_at_utc': latest['observed_at_utc'], 'accepted_vectors': 11, 'features': 0,
          'protected_original_files_unchanged': len(original['protected_original_files_unchanged']),
          'new_scientific_inference': False, 'scientific_completion': False}
with (ROOT / 'verification-final.json').open('x') as stream:
    stream.write(json.dumps(result, sort_keys=True, indent=2) + '\n')
print(json.dumps({'verification': support.record(ROOT / 'verification-final.json'),
                  'sealed_files': len(result['files']), 'accepted_vectors': 11,
                  'original_files_unchanged': result['protected_original_files_unchanged']}, sort_keys=True))
