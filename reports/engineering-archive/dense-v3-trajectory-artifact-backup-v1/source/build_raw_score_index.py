"""Join all five previously recovered immutable raw-score snapshots; no network/model."""
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path, PurePosixPath

ARCHIVES = Path('/root/embedding-optimizer-story-refactor/reports/engineering-archive')
REPORT = Path('/root/embedding-optimizer-story-refactor/reports/dense-v3-complete-trajectories-v1')
PROTOCOL = '4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b'
REPO = 'qcz/embedding-optimizer-study-analysis-artifacts'
SOURCES = (
    (782, 'dense-v3-first-stage-artifact-backup-v1', 'first-stage', '17b90cd5d8b2d9b47ba2849b5a9e3d0ffd17998e5ffcb9fc8b1f176e55ea0e7b', 'ca72b1c6c13795d4403866e6bba3c2e29e22a74cc50fecf922ae91f638d58ce0', '8555e5849b56862948b0fc0688085708ce3e67cf', 557),
    (1563, 'dense-v3-complete-second-stage-artifact-backup-v1', 'second-stage', '8ead2e55fa3c800199fc1da8051f1b1d8c0fe15246f9d0f0e5f688e34942c277', 'af20f4df3306b4749feb678bdff543a8a146ec7197908c5b12ee011786768040', 'aa1515cbf98cb1b2af0be58cf05a4985bf1254cb', 557),
    (2345, 'dense-v3-complete-third-stage-artifact-backup-v1', 'third-stage', '4014de3241535c339d29e7d08d3bc2901597db546440218f78b44ac4fc6dce34', 'cc8253c2cb75d19fa88cb01fea4026670f81706cd83e52b7de740055c18cf409', 'fdad53c239d9ed59141fcafbd92a0438bb1a657f', 557),
    (3126, 'dense-v3-complete-fourth-stage-artifact-backup-v1', 'fourth-stage', '21f7c01540162767e86180ca3119e81df41e5b95e55b3044614fd9aeec0f604c', 'd680eda01aac97173c325e80f7879828ee26ed3f7d706b01787176ba4d094c0f', '5eb4cceed5c3dbee05d46850ec0d76aedff7d4cc', 557),
    (3907, 'dense-v3-evaluation-artifact-backup-v1', 'final', 'bf443d5623c1312f8661a918ab93f9aea5f119058307733d477c8eac0c6da74f', '3e55dc65506405277d9dc9aeb08c3a7b8381055454e293b6d2376d877bd2a10d', '3883b677f87b1982f06016e9fadb8bb95e0cfc96', 660),
)
RUNS = tuple('verified-v3-' + opt + '-' + rate for opt, rates in (
    ('adamw', ('1e-6', '3e-6', '1e-5', '3e-5')),
    ('muon', ('1e-4', '3e-4', '1e-3', '3e-3')),
    ('normuon', ('1e-4', '3e-4', '1e-3', '3e-3')),
) for rate in rates)
TASKS = ('ArguAna', 'ClimateFEVER', 'DBPedia', 'FEVER', 'FiQA2018', 'HotpotQA',
         'MSMARCO', 'NFCorpus', 'NQ', 'QuoraRetrieval', 'SCIDOCS', 'SciFact', 'TRECCOVID', 'Touche2020')


def safe(name):
    path = PurePosixPath(name)
    assert path.parts and not path.is_absolute() and '..' not in path.parts
    assert path.as_posix() == name and '\\' not in name
    return name


def read(path, expected=None):
    assert path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), str(path)
    raw = path.read_bytes()
    identity = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                'git_blob_sha1': hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()}
    if expected:
        assert all(identity[k] == v for k, v in expected.items()), str(path)
    return raw, identity


def build():
    table_raw, table_identity = read(REPORT / 'tables/all_task_scores.csv',
        {'sha256': 'ac87e909c2dd07df6feb8d76a02cc2016ea5e2a202cf241196eae4220198d6dc'})
    table = list(csv.DictReader(table_raw.decode().splitlines()))
    reference = {(r['run_id'], int(r['step']), r['task']): float(r['ndcg_at_10']) for r in table}
    assert len(table) == len(reference) == 840
    snapshots, scores, total_files, total_bytes = [], {}, 0, 0
    for step, archive_name, branch, digest, download_sha, revision, file_count in SOURCES:
        archive = ARCHIVES / archive_name
        raw, download_identity = read(archive / 'actual/download-verified.json', {'sha256': download_sha})
        download = json.loads(raw)
        assert download['revision'] == revision and download['manifest']['sha256'] == digest
        assert download['all_payload_hashes_match'] is True and download['files'] == file_count
        prefix = download['prefix']
        safe(prefix)
        root = (Path(download['download_root']) / prefix if step == 3907
                else Path(download['downloaded_root']))
        assert str(root).startswith('/tmp/dense-v3-') and root.as_posix().endswith('/download/' + prefix)
        manifest_raw, manifest_identity = read(root / 'artifact_manifest.json', {'sha256': digest})
        archived_raw, _ = read(archive / 'actual/artifact_manifest.json', {'sha256': digest})
        assert manifest_raw == archived_raw
        manifest = json.loads(manifest_raw)
        files = manifest['files']
        assert len(files) == file_count - 1
        assert {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} == set(files) | {'artifact_manifest.json'}
        assert not any(p.is_symlink() for p in root.rglob('*'))
        for name, identity in files.items():
            read(root / safe(name), identity)
        native_prefix = 'native/beir-' + branch + '/'
        complete_names = sorted(n for n in files if n.startswith(native_prefix)
                                and n.endswith('/all-fourteen-tasks-verified.json'))
        assert len(complete_names) == 12
        checkpoints, seen = [], set()
        for name in complete_names:
            complete = json.loads(read(root / name, files[name])[0])
            plan = complete['plan']
            checkpoint = plan['checkpoint']
            run = checkpoint['run_id']
            assert run in RUNS and run not in seen and checkpoint['step'] == step
            assert checkpoint['protocol_sha256'] == plan['protocol_sha256'] == PROTOCOL
            assert complete['complete_task_files_verified'] is True
            assert complete['scientific_completion'] is False and plan['tasks'] == list(TASKS)
            assert [r['task'] for r in complete['tasks']] == list(TASKS)
            seen.add(run)
            values, task_files = [], []
            for task in complete['tasks']:
                task_name = task['task']
                item = next(f for f in task['files']
                            if PurePosixPath(f['path']).name == task_name + 'Decontaminated.json')
                relative = PurePosixPath(item['path']).relative_to('/root/embedding-optimizer-v3-experiment/evaluations/dense-primary-v3').as_posix()
                recovered_name = native_prefix + safe(relative)
                assert recovered_name in files
                raw_score = json.loads(read(root / recovered_name,
                    {k: item[k] for k in ('bytes', 'sha256')})[0])
                split = 'dev' if task_name == 'MSMARCO' else 'test'
                assert raw_score['task_name'] == task_name + 'Decontaminated'
                assert list(raw_score['scores']) == [split] and len(raw_score['scores'][split]) == 1
                row = raw_score['scores'][split][0]
                score = row['ndcg_at_10']
                assert type(score) is float and math.isfinite(score) and 0 <= score <= 1
                assert score == row['main_score'] == task['ndcg_at_10']
                key = (run, step, task_name)
                assert key not in scores and score == reference[key]
                scores[key] = score
                values.append(Fraction.from_float(score))
                task_files.append({'task': task_name, 'path': recovered_name,
                                   'identity': files[recovered_name], 'ndcg_at_10': score})
            mean = sum(values, Fraction()) / 14
            checkpoints.append({'run_id': run, 'step': step,
                                'run_identity_sha256': checkpoint['run_identity_sha256'],
                                'checkpoint_seal': checkpoint['checkpoint_seal'],
                                'native_complete': {'path': name, 'identity': files[name]},
                                'task_files': task_files, 'macro_ndcg_at_10': float(mean),
                                'macro_score_0_to_100': float(mean * 100)})
        assert seen == set(RUNS)
        total = sum(f['bytes'] for f in files.values()) + manifest_identity['bytes']
        assert total == download['bytes']
        total_files += file_count
        total_bytes += total
        snapshots.append({'step': step, 'repo_id': REPO, 'repo_type': 'dataset', 'revision': revision,
                          'prefix': prefix, 'manifest': manifest_identity, 'files': file_count, 'bytes': total,
                          'download_receipt_identity': download_identity,
                          'checkpoints': sorted(checkpoints, key=lambda c: RUNS.index(c['run_id']))})
    assert scores == reference and total_files == 2888
    return {'scope': 'complete_primary_v3_immutable_raw_score_recovery_index',
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'primary_protocol_sha256': PROTOCOL,
            'runs': list(RUNS), 'steps': [s[0] for s in SOURCES], 'tasks': list(TASKS),
            'snapshots': snapshots, 'checkpoint_count': 60, 'raw_task_score_count': 840,
            'accepted_all_task_table_identity': table_identity,
            'previously_recovered_files_rechecked': total_files, 'recovered_logical_bytes_rechecked': total_bytes,
            'every_raw_score_matches_complete_trajectory_table': True,
            'source_code_included': False, 'new_network_transfer': False, 'new_model_or_statistical_execution': False,
            'same_physical_host_reconstruction': True, 'scientific_completion': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.output.is_absolute() and not args.output.exists()
    result = build()
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'identity': read(args.output)[1],
                      'snapshots': 5, 'checkpoints': 60, 'raw_scores_reconstructed': 840,
                      'recovered_files_rechecked': result['previously_recovered_files_rechecked'],
                      'recovered_bytes_rechecked': result['recovered_logical_bytes_rechecked'],
                      'new_model_or_network_execution': False, 'scientific_completion': False}, sort_keys=True))


if __name__ == '__main__':
    main()
