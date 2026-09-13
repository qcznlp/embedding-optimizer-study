"""Strict original document checks for a prose-only revision of completed findings."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE / sys.argv[1]
source = Path('/tmp/dense-v3-document-integration.Xz1qvTME/document_component.py')
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'dbed8e2d4d418ce89d5d56d75da4e433e9b9b0888194a9c44ca053dd03d2684b'
spec = importlib.util.spec_from_file_location('unchanged_document_component', source)
doc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doc)
old = json.loads((HERE / 'original-inputs.json').read_bytes())
inputs = {}
for name in doc.BUILD_INPUTS:
    raw = (ROOT / 'paper' / name).read_bytes()
    inputs[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    if name not in ('main.tex', 'references.bib'):
        assert inputs[name] == old[name], name
with (ROOT / 'review-inputs.json').open('x') as stream:
    json.dump({'scope': 'prose_and_bibliography_only', 'inputs': inputs,
               'original_document_component_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'all_generated_results_figures_constants_vendor_bytes_unchanged': True}, stream, indent=2)
expected = {name: (ROOT / 'paper' / name).read_bytes() for name in doc.RESULT_FILES}
receipt = doc.compile_fresh(ROOT, expected, (ROOT / 'paper/results.tex').read_bytes())
print(json.dumps({'abstract_words': receipt['source_inspection']['actual_abstract'],
                  'main_end_page': receipt['layout']['main_end_page'],
                  'pdf_pages': receipt['pdf_pages'], 'font_count': receipt['font_count'],
                  'final_release_verified': False}))
