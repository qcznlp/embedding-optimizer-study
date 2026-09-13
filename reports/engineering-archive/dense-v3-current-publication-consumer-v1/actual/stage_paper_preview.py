"""Stage a full-paper layout preview with genuine primary results and pending factorial.

The authoritative manuscript is not edited or installed by this staging step.
Only explicit current generated files are substituted in a new directory.
"""
from pathlib import Path
import shutil
from assemble_publication import read, STORY, HERE
from native_primary import need, identity, write_new

output = HERE / 'paper-preview'
result = HERE / 'scientific-publication-v3'
completion = read(result / 'completed.json', '1941d86b42972a53dbca8ff86c8cc346282cc10f17446d62f66d67fcedb826cc')
need(completion['current_primary_scientific_result_assembly_complete'] is True
     and completion['manuscript_installed'] is False, 'Current native result assembly incomplete')
for name, bound in completion['outputs'].items():
    need(identity(result / name) == bound, 'Current scientific output changed')
need(not output.exists(), 'Use a new paper preview directory')
sources = {name: STORY / 'paper' / name for name in (
    'main.tex', 'results.tex', 'references.bib', 'Makefile',
    'vendor/acl.sty', 'vendor/acl_natbib.bst',
    'figures/optimizer-weight-dimension-map.pdf', 'generated/state-operator-factorial.tex')}
for name in ('optimizer-primary.tex', 'dimension-utilization.tex'):
    sources['generated/' + name] = result / 'exact-publication' / name
sources['generated/recipe-sensitivity.tex'] = result / 'recipe-sensitivity.tex'
sources['main.original.tex'] = STORY / 'paper/main.tex'
need(identity(STORY / 'paper/main.tex')['sha256'] == '45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e',
     'Authoritative paper changed; reconcile before staging')
output.mkdir()
inputs = {}
for name, source in sources.items():
    bound = identity(source)
    destination = output / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    need(not destination.exists(), 'Preview copy would overwrite')
    shutil.copyfile(source, destination)
    need(identity(source) == identity(destination) == bound, 'Preview copy differs')
    inputs[name] = {'origin': str(source), **bound}
write_new(output / 'staged-inputs.json', {'scope': 'actual_primary_full_paper_development_preview',
    'inputs': inputs, 'primary_assembly': identity(result / 'completed.json'),
    'factorial_results_pending': True, 'authoritative_manuscript_installed': False,
    'final_publication_gate_passed': False, 'scientific_completion': False})
print('Staged full paper with actual primary/functional results; factorial remains explicitly pending.')
