"""Preserve this completed, bounded post-result readout without changing inputs."""
import hashlib
import json
import shutil
from pathlib import Path

WORK = Path(__file__).parent
REPORT = Path('/root/embedding-optimizer-story-refactor/reports/paper-review/dense-v3-cosine-sensitivity-v1')
STORY = Path('/root/embedding-optimizer-story-refactor')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if REPORT.exists():
    raise ValueError('Preserve an existing report; do not overwrite it')
if sha(WORK / 'actual/completed.json') != 'b7bb5584fb3e462dc734ca30b0705c75de3f9e751270ebdb7edfbf41343ee75a':
    raise ValueError('Unexpected actual completion')
shutil.copytree(WORK, REPORT)
bindings = json.loads((WORK / 'actual/input_bindings.json').read_text())['frozen_sources']
for name, expected in bindings.items():
    src = Path(name)
    if sha(src) != expected:
        raise ValueError('Original frozen source changed')
    dst = REPORT / 'frozen-source' / src.relative_to(STORY)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if sha(dst) != expected:
        raise ValueError('Source copy differs')
primary = Path('/root/embedding-optimizer-primary-v3/src/embed_optim/optimizers.py')
if sha(primary) != '7d7d4f200c410cb1b24e1773ae72811582f43a0c794a5d1b2af269c123e69b1e':
    raise ValueError('Reviewed primary optimizer routing changed')
(REPORT / 'input-provenance').mkdir()
shutil.copy2(primary, REPORT / 'input-provenance/primary-optimizers.py')
for name in ('artifact_manifest.json', 'download-verified.json'):
    shutil.copy2(STORY / 'reports/engineering-archive/dense-v3-close-loop-handoff-v1/functional-durability' / name,
                 REPORT / 'input-provenance' / name)
record = {'actual_cpu_session': 29917, 'actual_terminal': '457651', 'actual_exit_code': 0,
          'literal_analysis_completion_sha256': sha(REPORT / 'actual/completed.json'),
          'note': 'Inspection path misses during source discovery were not scientific run failures. No original source or failed historical evidence was changed.',
          'manuscript_installed': False, 'source_published': False}
with (REPORT / 'execution.json').open('x') as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps({'report': str(REPORT), 'files_copied': sum(p.is_file() for p in REPORT.rglob('*')), 'original_source_changes': False}))
