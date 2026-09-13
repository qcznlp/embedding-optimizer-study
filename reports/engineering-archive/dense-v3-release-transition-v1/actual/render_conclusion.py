"""Generate the current README conclusion only from admitted complete v3 outputs."""
import hashlib
import json
from pathlib import Path

from embed_optim import current_paper
from embed_optim import paper_reproduction as replay

work = Path(__file__).resolve().parent
root = Path('/root/embedding-optimizer-story-refactor')
completed = work / 'release-paper'
assert replay.identity(completed / 'complete.json')['sha256'] == '37756f9bc25114a59538be60fb727abecd03f60f09d326343b7b9a457f3ec3c1'
outer = replay.read(completed / 'complete.json')
assert outer['complete'] is True and outer['scope'] == replay.SCOPE
assert outer['source']['sha256'] == 'd5013dba447d8affc61e547ea09d395499b3539b95468cc52aa63240640eeb11'
assert outer['repository_publication_complete'] is False
bundle, _ = replay.verify_bundle(root / 'reports/engineering-archive/dense-v3-complete-replay-entry-v1/closed')
native = replay.validate_numerical(completed / 'numerical', bundle)
paper, _, _ = current_paper.read_snapshot(root / 'paper/current')
assert replay.shared_inputs(completed / 'numerical/document/paper', paper) == outer['shared_inputs']
assert replay.shared_inputs(completed / 'reviewed/paper', paper) == outer['shared_inputs']
assert replay.identity(completed / 'reviewed/paper/build/main.pdf') == outer['pdf']
doc = replay.read(completed / 'reviewed/document.json')
assert doc['compiled_pdf_verified'] is True and doc['layout']['main_end_page'] <= 8
assert doc['source_inspection']['actual_abstract']['words_conservative'] <= 200
rebuilt = completed / 'numerical/primary/reconstructed'
primary_receipt = replay.read(rebuilt / 'complete.json')
summary_path = rebuilt / 'exact-publication/primary-summary.json'
assert replay.identity(summary_path) == primary_receipt['all_17_original_publication_outputs_byte_exact']['exact-publication/primary-summary.json']
summary = replay.read(summary_path)
assert len(summary['primary']) == len(summary['secondary']) == 3
assert all(r['support'] == 'inconclusive' for r in summary['primary'])
assert summary['outcome_table_counts']['all_task_scores'] == 840
secondary = {(r['treatment'], r['baseline']): r for r in summary['secondary']}
muon, normuon = secondary[('muon', 'adamw')], secondary[('normuon', 'adamw')]
assert muon['support'] == 'inconclusive' and muon['mean'] > 0 and muon['lower'] < 0 < muon['upper']
assert normuon['support'] == 'positive' and 0 < normuon['lower'] < normuon['mean'] < normuon['upper']
assert native['factorial_counts'] == replay.FACTORIAL_COUNTS
block = (
    '<!-- FINAL-CONCLUSION:BEGIN -->\n'
    '**Complete DenseOn results.** All three comparisons averaged over the declared\n'
    'four-rate grids are inconclusive. Validation-selected NorMuon exceeds AdamW by\n'
    f'**{100*normuon["mean"]:+.3f} nDCG@10 points** (simultaneous 95% interval\n'
    f'**[{100*normuon["lower"]:+.3f}, {100*normuon["upper"]:+.3f}]**); selected Muon is\n'
    f'**{100*muon["mean"]:+.3f}**, with its interval crossing zero. These are one-seed,\n'
    'fixed-grid results, not a universal optimizer ranking. The complete weight,\n'
    'representation and crossed-continuation findings are in the\n'
    '[reviewed paper](paper/current/README.md) and [evidence summary](PROJECT_STATUS.md).\n'
    '<!-- FINAL-CONCLUSION:END -->\n'
)
with (work / 'conclusion.md').open('x') as stream:
    stream.write(block)
with (work / 'conclusion-binding.json').open('x') as stream:
    json.dump({'completion': replay.identity(completed / 'complete.json'), 'summary': replay.identity(summary_path),
               'document': replay.identity(completed / 'reviewed/document.json'), 'block_sha256': hashlib.sha256(block.encode()).hexdigest(),
               'source': replay.identity(Path(__file__).resolve()), 'historical_contracts_modified': False,
               'source_publication_complete': False}, stream, indent=2, sort_keys=True)
print(block)
