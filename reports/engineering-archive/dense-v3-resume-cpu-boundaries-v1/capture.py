"""Capture the terminal CPU diagnostic; do not rerun or admit GPU recovery."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

here = Path(__file__).resolve().parent
root = here.parents[2]

def identity(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

completed = json.loads((here / 'completed.json').read_bytes())
reference = root / 'reports/engineering-archive/dense-v3-portable-factorial-checkpoint-v1/reference-component.json'
assert identity(reference)['sha256'] == '00e0dab0a4aaf52a5783ffb6ae2d5399952d55f2055b65b033fb5c4a0baa400d'
sources = json.loads(reference.read_bytes())['identity']['bound_factorial_run']['sources']
for name in ('factorial_v3_batches.py', 'factorial_v3_optimizer.py', 'optimizers.py', 'config.py'):
    relative = 'src/embed_optim/' + name
    assert identity(root / relative) == sources[relative]
for module in ('factorial_v3_batches', 'factorial_v3_optimizer', 'saved_factorial_checkpoint'):
    assert identity(root / 'src/embed_optim' / (module + '.py'))['sha256'] == completed['source_sha256']['embed_optim.' + module]
unit = ET.parse(here / 'unit.xml').getroot().find('testsuite').attrib
assert unit['tests'] == '14' and all(unit[k] == '0' for k in ('errors', 'failures', 'skipped'))
for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'tests/test_factorial_resume_order.py'):
    destination = here / 'after' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as out, (root / name).open('rb') as original:
        shutil.copyfileobj(original, out)
    assert identity(destination) == identity(root / name)
(here / 'checks.json').write_text(json.dumps({
    'actual_numeric_modules_equal_checkpoint_sources': True,
    'diagnostic_sources_unchanged': True, 'unit': unit,
    'diagnostic_tool_exit': 0, 'diagnostic_terminal': '470a2d',
    'regression_tool_exit': 0, 'regression_terminal': '15ebd9',
    'full_cause_localized': False, 'exact_gpu_resume_accepted': False,
    'new_scientific_results': False, 'full_source_release': False}, indent=2))
rows = {p.relative_to(here).as_posix(): identity(p) for p in sorted(here.rglob('*')) if p.is_file()}
with (here / 'manifest.json').open('x') as out:
    json.dump({'files': rows}, out, indent=2, sort_keys=True)
print(json.dumps({'files': len(rows), 'bytes': sum(v['bytes'] for v in rows.values()),
                  'manifest': identity(here / 'manifest.json')}))
