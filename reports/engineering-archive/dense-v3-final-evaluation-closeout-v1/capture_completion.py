"""Copy completed native evidence; no statistical recomputation or admission change."""
import hashlib
import json
import shutil
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
EXP = Path('/root/embedding-optimizer-v3-experiment')
SUMMARY = Path('/tmp/dense-v3-factorial-summary.BQ08HjeP')
DOC = Path('/tmp/dense-v3-document-integration.Xz1qvTME/actual-complete-paper')
RESUME = Path('/tmp/dense-v3-resume-verification.wcsmQnDm')
copies = []


def identity(path):
    assert path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def read(path):
    return json.loads(path.read_text())


def copy(path, relative, expected=None):
    actual = identity(path)
    if expected is not None:
        assert actual == expected, str(path)
    target = OUT / relative
    assert not target.exists(), str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    assert identity(target) == actual == identity(path)
    copies.append({'source': str(path), 'path': relative, **actual})


results = EXP / 'analyses/dense-v3-factorial-inference-v1'
assert identity(results / 'readout.json')['sha256'] == 'b6b69d9b087a6797c54330064316aa9f9fe71887797ef59313efce8c3214781d'
receipt = read(results / 'readout.json')
assert receipt['actual_collector_exits'] == {'beir': 0, 'probe': 0}
assert receipt['table_counts'] == dict(beir_seed_task_scores=168, factorial_cell_summary=4,
    estimand_seed_task_contrasts=126, estimand_summary=3, probe_checkpoint_metrics=60,
    probe_task_metrics=840)
assert receipt['independent_arithmetic_verified'] is True
copy(results / 'readout.json', 'inference/readout.json')
for name, expected in receipt['outputs'].items():
    copy(results / name, 'inference/' + name, expected)
for kind, expected in receipt['actual_native_collectors'].items():
    copy(SUMMARY / 'run' / (kind + '.json'), 'native-collectors/' + kind + '.json', expected)
    for suffix in ('.started.json', '.exited.json', '.log'):
        copy(SUMMARY / 'run' / (kind + suffix), 'native-collectors/' + kind + suffix)
copy(SUMMARY / 'run/completed.json', 'native-collectors/completed.json')
copy(SUMMARY / 'summarize.py', 'native-collectors/summarize.py', receipt['source'])
copy(SUMMARY / 'authorization.json', 'native-collectors/authorization.json')
assert identity(OUT / 'native-collectors/authorization.json')['sha256'] == receipt['authorization_sha256']

document = read(DOC / 'completed.json')
assert document['compiled_current_document_verified'] is True
assert document['main_end_page'] == 8 and document['abstract_words'] == 161
copy(DOC / 'completed.json', 'document/completed.json')
copy(DOC / 'document.json', 'document/document.json', document['document'])
for name, expected in read(DOC / 'document.json')['artifacts'].items():
    copy(DOC / name, 'document/' + name, expected)

for pool in ('a', 'b'):
    run = RESUME / 'run' / ('pool-' + pool)
    assert read(run / 'ranks.exited.json')['actual_rank_exits'] == {str(i): 1 for i in range(4)}
    assert (run / 'failed.json').is_file() and not (run / 'comparison.json').exists()
    for path in sorted(run.iterdir()):
        assert path.is_file() and path.suffix in ('.json', '.log')
        copy(path, 'failed-resume/pool-' + pool + '/' + path.name)
    for rank in range(4):
        log = (run / f'rank-{rank}.log').read_text()
        assert 'state_steps is on cpu' in log and f'cuda:{rank}' in log
        assert 'wrapper_CUDA___fused_adamw_' in log
copy(RESUME / 'resume.py', 'failed-resume/resume.py')
copy(RESUME / 'authorization.json', 'failed-resume/authorization.json')
source = ROOT / 'reports/engineering-archive/dense-v3-factorial-worker-v1/actual/source-final/src/embed_optim'
for name in ('optimizers.py', 'factorial_v3_optimizer.py', 'factorial_v3_trainer.py'):
    copy(source / name, 'failed-resume/original-source/' + name)
copy(Path('/usr/local/lib/python3.12/dist-packages/torch/optim/optimizer.py'),
     'failed-resume/original-source/pytorch_optimizer.py')
with (OUT / 'completion-copies.json').open('x') as stream:
    json.dump(copies, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps({'copied_files': len(copies), 'bytes': sum(x['bytes'] for x in copies),
    'actual_tables': receipt['table_counts'], 'compiled_document': True,
    'failed_resume_cases': 2, 'failed_resume_rank_logs': 8, 'recomputation': False,
    'full_goal_complete': False}))
