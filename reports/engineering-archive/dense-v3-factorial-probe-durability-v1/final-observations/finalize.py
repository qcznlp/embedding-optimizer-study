"""Preserve final observations and updated handoffs without changing prior evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
STORY = Path('/root/embedding-optimizer-story-refactor')
REPORT = STORY / 'reports/engineering-archive/dense-v3-factorial-probe-durability-v1'
DEST = REPORT / 'final-observations'


def digest(path):
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Ordinary exact file required')
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def read(path):
    return json.loads(path.read_text())


old = REPORT / 'archive-manifest.json'
assert digest(old)['sha256'] == '7187391eca10184199d4b1a338030520a56a9d29be84956c18101b858efc5faf'
records = read(old)['files']
for relative, record in records.items():
    assert digest(REPORT / relative) == {key: record[key] for key in ('bytes', 'sha256')}
assert len(records) == 25

unchanged = {
    STORY / 'paper/main.tex': '45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e',
    Path('/tmp/dense-v3-factorial-evaluation.IOV7MK93/evaluate.py'): '45577537257e1dff2b7c3a26ee12f7564fbb476daa256e7b0f09300305b68cba',
    Path('/tmp/dense-v3-factorial-summary.BQ08HjeP/summarize.py'): '4be265ce12cc649a595afaec9e5ef8b68a50eaa347ccb583d4d32ba3e100c5a1',
    Path('/tmp/dense-v3-document-integration.Xz1qvTME/author.py'): 'cac9486b6e92acedb190bad67e34a5fd647d1c10a691071483fbf001406b1f9d',
    Path('/tmp/dense-v3-resume-verification.wcsmQnDm/resume.py'): '1746b8a382bb45fefc2ab17beaf7706984fe91802c528bec48c88a2daa038bb2',
}
for path, sha in unchanged.items():
    assert digest(path)['sha256'] == sha

destinations = {
    name: HERE / name for name in (
        'evaluation-current.json', 'evaluation-batch-progress.json',
        'evaluation_log_progress.py', 'evaluation-final.json', 'summary-final.json',
        'document-final.json', 'resume-final.json', 'sample_owned_evaluation.py',
        'evaluation-stacks.json', 'finalize.py',
    )
}
for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md',
             'docs/checkpoint-restoration.md', 'docs/continuation-probe-restoration.md',
             'scripts/restore_factorial_probes.py'):
    destinations['after/' + name] = STORY / name
DEST.mkdir(exist_ok=False)
copied = {}
for name, original in destinations.items():
    target = DEST / name
    before = digest(original)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, target)
    assert digest(target) == digest(original) == before
    copied[name] = dict(before, origin=str(original))

evaluation = read(HERE / 'evaluation-final.json')
stacks = read(HERE / 'evaluation-stacks.json')
assert sum(row['completed_tasks'] for row in evaluation['pools'].values()) == 2
assert len(stacks['jobs']) == 8 and all(row['sample_exit_code'] == 0 for row in stacks['jobs'])
assert not any(row['failure_receipt'] for row in evaluation['pools'].values())
summary = read(HERE / 'summary-final.json')
document = read(HERE / 'document-final.json')
resume = read(HERE / 'resume-final.json')
assert not summary['failed'] and not document['failed']
assert not any(row['failure'] for row in resume['pools'].values())
result = {
    'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
    'goal_turn_classification': 'PROGRESS', 'full_goal_complete': False,
    'completed_this_turn': [
        '638-file additions-only immutable probe backup',
        'actual anonymous full download and independent file-hash verification',
        'exact copied-source numerical replay of 61 scores and 5490 metric values',
        'all sixty immutable continuation model locations indexed',
        'restoration guide and current operational handoffs updated',
    ],
    'probe_snapshot': {
        'repo': 'qcz/embedding-optimizer-study-analysis-artifacts',
        'revision': 'cff3f190e169548931fbd33eadcf1279439798e1',
        'manifest_sha256': 'bcb092fe16e11150abc678a6b1e977497afbc6bf977b9dd9085cebff37004081',
        'files': 638, 'bytes': 586119383, 'npz_files': 183,
    },
    'actual_numerical_replay': digest(REPORT / 'actual/numerical-replay/actual-replay.json'),
    'original_archive_manifest': digest(old), 'original_archive_files_unchanged': 25,
    'final_observation_files': len(copied),
    'final_observation_bytes': sum(row['bytes'] for row in copied.values()),
    'files': copied, 'unchanged_live_and_manuscript': {str(p): sha for p, sha in unchanged.items()},
    'latest_evaluation_observed_at_utc': evaluation['observed_at_utc'],
    'complete_beir_tasks': 2, 'required_beir_tasks': 168,
    'registered_worker_samples': 8, 'no_batch_counter_claimed': True,
    'continuous_throughput_or_gpu_utilization_claimed': False,
    'pending': ['remaining original full-corpus BEIR queue',
                'original complete native factorial collectors and inference',
                'actual complete paper generation, numerical reconstruction and visual review',
                'result durability after actual inference',
                'two already queued original-Trainer GPU-resume endpoint checks',
                'full current source/portable-paper/tracking/package/release gates'],
    'do_not_repeat': ['completed training and checkpoint backup',
                     'completed probes and this full download/numerical replay',
                     'prior complete functional, geometry and result-fragment replays'],
    'protected_helper_access': False, 'historical_controller_transition': False,
    'github_write_retry': False, 'hf_deletion': False,
}
with (REPORT / 'final-handoff.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps({'final_handoff': digest(REPORT / 'final-handoff.json'),
                  'copied_files': len(copied), 'copied_bytes': result['final_observation_bytes'],
                  'all_original_25_files_unchanged': True, 'full_goal_complete': False}))
