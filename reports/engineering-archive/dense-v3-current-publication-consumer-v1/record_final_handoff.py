"""Retain the now-complete backup and exact dated downstream observations, without launching work."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKUP = Path('/tmp/dense-v3-factorial-backup-launch.75Chtt86/run')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def identity(path):
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Ordinary owned evidence required')
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def read(path):
    identity(path)
    return json.loads(path.read_bytes())


def main():
    need(not (ROOT / 'final-handoff.json').exists() and not (ROOT / 'final-complete-backup').exists(), 'Do not overwrite completed handoff')
    need(identity(ROOT / 'archive-manifest.json')['sha256'] == '82563d6a98625c96cb41bbfcc7c2239a01e0f4f0a6261f130268032837c7c630', 'Original archive changed')
    training = read(ROOT / 'actual/training-1959.json')
    runs = {r['run_id'] for v in training['pools'].values() for r in v['runs'] if r['status'] == 'full_training_and_fresh_readback_complete'}
    complete = read(BACKUP / 'completed.json')
    need(complete['verified_runs'] == 12 and complete['verified_checkpoints'] == 60 and set(complete['receipts']) == runs, 'Incomplete current backup')
    sources = {'completed.json': BACKUP / 'completed.json', 'started.json': BACKUP / 'started.json'}
    revisions = {}
    for run, binding in sorted(complete['receipts'].items()):
        path = BACKUP / run / 'verified.json'
        need(identity(path) == binding, 'Original complete backup receipt changed')
        receipt = read(path)
        need(receipt['run_id'] == run and receipt['all_five_checkpoints'] is True and receipt['anonymous_remote_metadata_verified'] is True, 'Wrong native remote backup')
        manifest = path.parent / 'artifact_manifest.json'
        need(identity(manifest) == receipt['artifact_manifest'], 'Original backup manifest changed')
        sources[run + '/verified.json'] = path
        sources[run + '/artifact_manifest.json'] = manifest
        revisions[run] = {k: receipt[k] for k in ('repo_id', 'prefix', 'commit_oid', 'artifact_manifest')}
    records = {}
    for name, source in sources.items():
        bound = identity(source)
        target = ROOT / 'final-complete-backup' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        need(not target.exists(), 'Refuse overwrite')
        shutil.copyfile(source, target)
        need(identity(source) == identity(target) == bound, 'Original/copy backup evidence differs')
        records[str(target.relative_to(ROOT))] = {'origin': str(source), **bound}
    observations = {str(p.relative_to(ROOT)): identity(p) for p in sorted((ROOT / 'final-observations').glob('*.json'))}
    evidence = {'archive-manifest.json': identity(ROOT / 'archive-manifest.json'),
        'actual/scientific-publication-v3/completed.json': identity(ROOT / 'actual/scientific-publication-v3/completed.json'),
        'actual/training-1959.json': identity(ROOT / 'actual/training-1959.json'),
        'actual/paper-preview-v2/build/main.pdf': identity(ROOT / 'actual/paper-preview-v2/build/main.pdf'),
        'actual/paper-preview-v2/main.tex': identity(ROOT / 'actual/paper-preview-v2/main.tex')}
    value = {'scope': 'actual_complete_training_backup_and_running_outcome_handoff',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'source': identity(Path(__file__)),
        'evidence': evidence, 'observations': observations, 'backup_copies': records, 'backup_revisions': revisions,
        'training_runs_complete': 12, 'native_checkpoints_complete': 60, 'remote_backups_complete': 60,
        'backup_network_reads_repeated_now': False, 'beir_units_at_200651_utc': 2, 'probe_states_at_200725_utc': 25,
        'active_gpu_work': 'Six original full-corpus BEIR workers and two checkpoint-probe workers at the dated observations.',
        'training_supervisor_exits': [0, 0], 'backup_tool_exit': 0, 'backup_tool_terminal_chunk': '2f23de',
        'primary_scientific_result_assembly_complete': True, 'factorial_outcomes_complete': False,
        'authoritative_manuscript_installed': False, 'old_scientific_or_document_guards_waived': False,
        'committed_source_release': False, 'goal_turn_classification': 'progress', 'scientific_completion': False}
    with (ROOT / 'final-handoff.json').open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'complete_backup_copies': len(records), 'backup_runs': 12, 'checkpoints': 60, 'goal_turn_classification': 'progress', 'scientific_completion': False}))


if __name__ == '__main__':
    main()
