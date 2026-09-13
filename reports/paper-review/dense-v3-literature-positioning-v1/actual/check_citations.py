"""Compile the deferred prose/citations only; never touch live paper inputs."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone

HERE = Path(__file__).parent
ROOT = Path('/root/embedding-optimizer-story-refactor')
REVIEW = ROOT / 'reports/paper-review/dense-v3-literature-positioning-v1'
DEST = HERE / 'citation-check'
EXPECTED_BIB = 'fca6bf5762a2cecab795a1a60c1338ab4eb999acb5d78c437b9df0e5f3553218'
EXPECTED_MAIN = '45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise ValueError('Require hidden CUDA')
if sha(ROOT / 'paper/references.bib') != EXPECTED_BIB or sha(ROOT / 'paper/main.tex') != EXPECTED_MAIN:
    raise ValueError('Original bound paper changed')
inputs = {
    'original-references.bib': ROOT / 'paper/references.bib',
    'references-additions.bib': REVIEW / 'references-additions.bib',
    'related-work-candidate.tex': REVIEW / 'related-work-candidate.tex',
    'acl.sty': ROOT / 'paper/vendor/acl.sty',
    'acl_natbib.bst': ROOT / 'paper/vendor/acl_natbib.bst',
}
bindings = {name: {'original_path': str(path), 'sha256': sha(path)} for name, path in inputs.items()}
for name, path in inputs.items():
    if (DEST / name).exists():
        raise ValueError('Preserve existing citation attempt')
    shutil.copy2(path, DEST / name)
    if sha(DEST / name) != bindings[name]['sha256']:
        raise ValueError('Copy differs')
old_keys = set(re.findall(r'@\w+\s*\{\s*([^,\s]+)', (DEST / 'original-references.bib').read_text()))
new_keys = set(re.findall(r'@\w+\s*\{\s*([^,\s]+)', (DEST / 'references-additions.bib').read_text()))
citations = set()
for group in re.findall(r'\\cite[pt]\{([^}]+)\}', (DEST / 'related-work-candidate.tex').read_text()):
    citations.update(key.strip() for key in group.split(','))
if len(new_keys) != 6 or old_keys & new_keys or citations - (old_keys | new_keys) or new_keys - citations:
    raise ValueError('Citation keys do not close over exactly six new references')
write(HERE / 'citation-inputs.json', {'sources': bindings, 'snippet_sha256': sha(DEST / 'snippet.tex'),
                                    'source_sha256': sha(Path(__file__)), 'cited_keys': sorted(citations)})
env = dict(os.environ, TEXINPUTS=str(DEST) + ':', BSTINPUTS=str(DEST) + ':', BIBINPUTS=str(DEST) + ':')
latex = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-no-shell-escape', '-recorder', 'snippet.tex']
commands = [latex, ['bibtex', 'snippet'], latex, latex]
records = []
for number, command in enumerate(commands, 1):
    result = subprocess.run(command, cwd=DEST, env=env, capture_output=True, text=True, check=False)
    with (DEST / f'command-{number}.log').open('x') as stream:
        stream.write(result.stdout)
        stream.write(result.stderr)
    records.append({'command': command, 'exit_code': result.returncode})
    write(DEST / f'command-{number}.json', records[-1])
    if result.returncode:
        raise SystemExit(result.returncode)
log = (DEST / 'snippet.log').read_text()
bbl = (DEST / 'snippet.bbl').read_text()
if 'There were undefined citations' in log or 'Citation ' in log and 'undefined' in log:
    raise ValueError('Unresolved final citations')
if 'Overfull \\hbox' in log or 'Overfull \\vbox' in log:
    raise ValueError('Snippet has an overfull box')
for name, path in inputs.items():
    if sha(path) != bindings[name]['sha256']:
        raise ValueError('Original input changed during compilation')
if sha(ROOT / 'paper/main.tex') != EXPECTED_MAIN:
    raise ValueError('Original manuscript changed')
record = {'completed_at_utc': datetime.now(timezone.utc).isoformat(),
          'scope': 'actual_deferred_related_work_and_bibliography_compile_only',
          'source_sha256': sha(Path(__file__)), 'commands': records,
          'new_references': len(new_keys), 'cited_references': len(citations),
          'bibliography_entries_rendered': bbl.count('\\bibitem'),
          'undefined_citations': False, 'overfull_boxes': False,
          'pdf_sha256': sha(DEST / 'snippet.pdf'),
          'original_main_sha256': EXPECTED_MAIN, 'original_bibliography_sha256': EXPECTED_BIB,
          'original_inputs_unchanged': True, 'full_document_validation': False,
          'scientific_completion': False, 'source_release': False}
write(HERE / 'citation-completed.json', record)
print(json.dumps(record))
