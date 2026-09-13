"""Promote authenticated current document inputs and the unchanged build component."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/root/embedding-optimizer-story-refactor')
HERE = Path(__file__).resolve().parent
REVIEW = ROOT / 'reports/paper-review/dense-v3-complete-manuscript-revision-v1'
manifest = REVIEW / 'archive-manifest.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == '318a06fcb5d045ef2f88bb1225938191686db3d77476e31e4308dcc6410e8a6c'
records = json.loads(manifest.read_bytes())['files']

def identity(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def copy(source, target):
    assert source.is_file() and not source.is_symlink() and not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert identity(source) == identity(target)

for name in ('pyproject.toml', 'paper/Makefile', 'paper/README.md', 'README.md',
             'AGENTS.md', 'PROJECT_STATUS.md', 'CURRENT_EXPERIMENT.md'):
    copy(ROOT / name, HERE / 'before' / name)
document = json.loads((REVIEW / 'actual/document.json').read_bytes())
inputs = document['source_inspection']['inputs']
assert len(inputs) == 12
for name, expected in inputs.items():
    source = REVIEW / 'actual/paper' / name
    assert identity(source) == expected == records['actual/paper/' + name]
    copy(source, ROOT / 'paper/current' / name)
component = REVIEW / 'provenance/document_component.py'
assert identity(component) == records['provenance/document_component.py']
assert identity(component)['sha256'] == 'dbed8e2d4d418ce89d5d56d75da4e433e9b9b0888194a9c44ca053dd03d2684b'
copy(component, ROOT / 'src/embed_optim/complete_paper_document.py')
snapshot = {'schema_version': 1, 'scope': 'reviewed_complete_dense_document_v1',
    'document_component_sha256': identity(component)['sha256'],
    'review_archive_sha256': identity(manifest)['sha256'],
    'reviewed_pdf_sha256': records['actual/paper/build/main.pdf']['sha256'],
    'scientific_numerical_replay_sha256': '6a0a8a3657d88a20529db8d45d32c31c5b74be783c0abb9d57994b0009d3c275',
    'upstream_scientific_recomputation': False, 'source_release_verified': False,
    'inputs': inputs}
with (ROOT / 'paper/current/document-snapshot.json').open('x') as stream:
    json.dump(snapshot, stream, sort_keys=True, indent=2)
    stream.write('\n')
print(json.dumps({'snapshot': identity(ROOT / 'paper/current/document-snapshot.json'),
                  'inputs': len(inputs), 'component_copied_without_modification': True}))
