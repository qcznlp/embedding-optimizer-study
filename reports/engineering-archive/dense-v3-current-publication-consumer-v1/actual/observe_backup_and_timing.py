"""Read completed owned backup receipts and actual prior task durations; no network/process access."""
import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent.resolve()
EXP = Path('/root/embedding-optimizer-v3-experiment')
BACKUP = Path('/tmp/dense-v3-factorial-backup-launch.75Chtt86')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def identity(path):
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Ordinary owned artifact required')
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def read(path):
    identity(path)
    return json.loads(path.read_bytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    need(a.output.is_absolute() and not a.output.exists(), 'New absolute observation required')
    training = read(a.training)
    need(training['authorization_sha256'] == '00c52f0bec979fb163a734ad0f9eadcd6ea09f7393133fd35c65bc62878ce11f', 'Wrong training observation')
    need(identity(BACKUP / 'backup.py')['sha256'] == 'e71c965a9c727420f68adc14985d428084db9296163eeb6c66ec347d0855ecc6', 'Original backup source changed')
    need(identity(BACKUP / 'owner-approval.json')['sha256'] == 'a64390df28fcbe32ca12e64acba99747ee5a598e27334de9f1ddf9770b37e1a2', 'Backup approval changed')
    completed = {run['run_id'] for pool in training['pools'].values() for run in pool['runs'] if run.get('status') == 'full_training_and_fresh_readback_complete'}
    backups = []
    for run_id in sorted(completed):
        root = BACKUP / 'run' / run_id
        if not (root / 'verified.json').exists():
            continue
        receipt = read(root / 'verified.json')
        need(receipt['run_id'] == run_id and receipt['repo_id'] == 'qcz/embedding-optimizer-study-checkpoints'
            and receipt['all_five_checkpoints'] is True and receipt['anonymous_remote_metadata_verified'] is True, 'Wrong or incomplete original backup')
        need(identity(root / 'artifact_manifest.json') == receipt['artifact_manifest'], 'Original backup manifest changed')
        backups.append({k: receipt[k] for k in ('run_id', 'commit_oid', 'prefix', 'verified_at_utc', 'artifact_manifest',
            'all_five_checkpoints', 'anonymous_remote_metadata_verified', 'all_payloads_downloaded_again')} |
            {'receipt_path': str(root / 'verified.json'), 'receipt': identity(root / 'verified.json')})
    outcome = read(HERE / 'scientific-publication-v3/outcomes.json')
    expected = {(r['run_id'], r['step'], r['task']) for r in outcome['all_task_scores']}
    need(len(expected) == 840, 'Native score population differs')
    observed, timings, durations = set(), {}, []
    for pool in ('a', 'b'):
        root = EXP / 'launch/evaluation-handoff' / ('pool-' + pool) / 'jobs'
        for path in sorted(root.glob('*.exited.json')):
            receipt = read(path)
            job = tuple(receipt['job'])
            need(job in expected and job not in observed and receipt['exit_code'] == 0, 'Unexpected, repeated or failed actual primary task')
            start_path = path.with_name(path.name.replace('.exited.json', '.started.json'))
            start = read(start_path)
            need(tuple(start['job']) == job and start['pid'] == receipt['pid']
                and start['source_sha256'] == '5b8a89bed5e0eaa02a12585ee3f6c883ca3e162b550898d68322321ee41d8427'
                and start['authorization_sha256'] == '2351e225387f42fe607d7e002a0717de79a78cba9f3c93f66652510083f0be7c', 'Original worker timing identity differs')
            seconds = receipt['elapsed_seconds']
            need(isinstance(seconds, (int, float)) and 0 < seconds < float('inf'), 'Invalid actual duration')
            observed.add(job)
            timings.setdefault(job[2], []).append(seconds)
            durations.append({'job': list(job), 'elapsed_seconds': seconds, 'started': identity(start_path), 'exited': identity(path)})
    need(observed == expected and len(timings) == 14 and all(len(v) == 60 for v in timings.values()), 'Incomplete prior timing surface')
    tasks = {k: {'samples': len(v), 'median_seconds': statistics.median(v), 'mean_seconds': statistics.mean(v),
        'min_seconds': min(v), 'max_seconds': max(v)} for k, v in sorted(timings.items())}
    median_work = 12 * sum(v['median_seconds'] for v in tasks.values())
    result = {'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'training_observation': identity(a.training),
        'training_complete_at_earlier_observation': len(completed), 'verified_backup_runs': len(backups),
        'verified_backup_checkpoints': 5 * len(backups), 'original_backup_receipts': backups,
        'backup_network_reverification_performed_now': False, 'backup_process_liveness_observed_now': False,
        'prior_primary_task_timings': tasks, 'all_840_timing_records': durations,
        'rough_factorial_beir_planning_hours': {'serial_median_task_work': median_work / 3600,
            'ideal_eight_gpu': median_work / (3600 * 8), 'ideal_six_gpu': median_work / (3600 * 6)},
        'eta_boundary': 'Planning extrapolation from completed primary same-worker tasks, not a deadline. Excludes admission, tail imbalance, contention, probe and final reading.',
        'read_only_owned_artifacts': True, 'scientific_completion': False}
    with a.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('original_backup_receipts', 'all_840_timing_records')}))


if __name__ == '__main__':
    main()
