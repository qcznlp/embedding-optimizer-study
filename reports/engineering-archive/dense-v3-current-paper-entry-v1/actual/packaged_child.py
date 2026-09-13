"""Actual relocated-wheel document execution; no checkout imports permitted."""
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PACKAGE = HERE / 'extracted'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert os.environ.get('PYTHONPATH') == str(PACKAGE)
from embed_optim import current_paper
assert Path(current_paper.__file__).is_relative_to(PACKAGE)
paper = PACKAGE / 'embedding_optimizer_study-0.1.0.data/data/share/embedding-optimizer-study/paper/current'
result = current_paper.build_current_paper(paper, HERE / 'packaged-document')
loaded = {}
for name, module in list(sys.modules.items()):
    if name == 'embed_optim' or name.startswith('embed_optim.'):
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(PACKAGE), (name, path)
        loaded[name] = str(path.relative_to(PACKAGE))
pdf_text = subprocess.run(['pdftotext', '-layout', result['pdf'], '-'],
    check=True, capture_output=True).stdout
assert pdf_text == (HERE / 'reviewed-pdf-text.txt').read_bytes()
with (HERE / 'packaged-document/relocation.json').open('x') as stream:
    json.dump({'loaded_package_modules': loaded, 'all_package_imports_from_extracted_wheel': True,
               'pdf_text_identical_to_reviewed_manuscript': True,
               'physical_second_host': False, 'source_release_verified': False}, stream, indent=2)
print(json.dumps(result), flush=True)
