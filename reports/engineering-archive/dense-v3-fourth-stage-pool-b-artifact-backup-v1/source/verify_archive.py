"""Seal this actual backup's local evidence; no network, GPU or model execution."""
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/root/embedding-optimizer-story-refactor')
PRIMARY = Path('/root/embedding-optimizer-primary-v3')
EXPERIMENT = Path('/root/embedding-optimizer-v3-experiment')
WORK = Path('/tmp/dense-v3-fourth-stage-pool-b-artifact-backup.DK40TDg0')
ARCHIVE = Path('/root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-v3-fourth-stage-pool-b-artifact-backup-v1')
ORIGINAL = ROOT / 'reports/engineering-archive/dense-v3-fourth-stage-pool-b-evaluations-v1'
MANIFEST_SHA = 'ba3e103b503d52533ead4f2f0823a5937cf09111245563835369f2493fcb41d7'
REVISION = 'dd16fe4cfe3e21ba8c8b4583038ed93fe4ce6e92'
PARENT = '9b2925acd9339b5ce25fd332ab8840ac50319407'
PROTOCOL = '4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b'
PREFIX = 'corrected-dense-correctness-v3/partial-fourth-stage-pool-b-evaluations/' + PROTOCOL + '/' + MANIFEST_SHA
SOURCES = {
    'backup.py': 'ce1a87f1fef337fd200a291296977b79ad6b1dfdda2ad264fc3b52aa4ed25d52',
    'verify_recovered.py': '0017a2462b5c099f1b82b4f1ad4c5a96166ce8cdead5a7e087d88478058e2773',
    'test_backup.py': '33c8f769fba743f93141f1a39862b521bee3fa0409e173f7553046e302ec5528',
    'test_verify_recovered.py': 'c028f9394b868b845b18fbad901858d7f3082b9464e0356e7d0192fff8fc2bb3',
    'ARTIFACT_README.md': '1a1916046cc48ab2df0969e3fa8748846cc3faec881bb663ca40ac2cf9bbd9ee',
}
COPIED = [
    'adapter-tests.xml', 'adapter-final-tests.xml', 'reader-tests.xml', 'preflight.json',
    'upload-mode-preflight.json', 'upload-started.json', 'upload.json',
    'remote-audit.json', 'download-verified.json',
    'staged-independent-readback.json', 'recovered-independent-readback.json',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_bytes(path):
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            'Not an ordinary evidence file: ' + str(path))
    return path.read_bytes()


def identity(path):
    raw = read_bytes(path)
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'git_blob_sha1': hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()}


def read(path):
    return json.loads(read_bytes(path))


def main():
    output = ARCHIVE / 'verification.json'
    require(not output.exists(), 'Preserve earlier verification')
    original_verification = ORIGINAL / 'verification.json'
    require(identity(original_verification)['sha256'] ==
            '87645cf835349856fc26741fe71fe79c36106b7c9a2fcd0a5f3fdc29002d036a',
            'Original accepted archive verification changed')
    original = read(original_verification)
    for record in [*original['payloads'].values(), *original['protected_originals']]:
        got = identity(Path(record['path']))
        require(all(got[key] == record[key] for key in ('bytes', 'sha256')),
                'Original archived/protected input changed')
    assembly = read(PRIMARY / 'source-assembly.json')
    source_checks = 0
    for root in (PRIMARY, EXPERIMENT / 'launch/source-snapshot'):
        for name, record in assembly['files'].items():
            got = identity(root / name)
            require(all(got[k] == record['identity'][k] for k in ('bytes', 'sha256')),
                    'Frozen assembled source changed')
            source_checks += 1
    require(source_checks == 112, 'Wrong original source coverage')

    for name, sha in SOURCES.items():
        require(identity(ARCHIVE / 'source' / name)['sha256'] == sha
                and read_bytes(ARCHIVE / 'source' / name) == read_bytes(WORK / 'source' / name),
                'Executed source copy changed')
    for name in COPIED:
        require(read_bytes(ARCHIVE / 'actual' / name) == read_bytes(WORK / name),
                'Actual receipt/test copy differs')
    require(read_bytes(ARCHIVE / 'actual/artifact_manifest.json') ==
            read_bytes(WORK / 'staging/artifact_manifest.json'), 'Manifest copy differs')
    before = identity(ARCHIVE / 'before/CURRENT_EXPERIMENT.md')
    require(before['sha256'] == '8193f6a7c31999527781063077753fb4c6db5834f9f76a8466a9b79e02ee67c4',
            'Previous handoff changed')

    test_counts = {}
    for kind, count, filename in [('adapter_initial', 34, 'adapter-tests.xml'),
                                  ('adapter', 46, 'adapter-final-tests.xml'),
                                  ('reader', 18, 'reader-tests.xml')]:
        tree = ET.fromstring(read_bytes(ARCHIVE / 'actual' / filename))
        cases = list(tree.iter('testcase'))
        suites = [tree] if tree.tag == 'testsuite' else list(tree.findall('testsuite'))
        require(len(cases) == count and sum(int(s.get('tests', '0')) for s in suites) == count
                and all(int(s.get(k, '0')) == 0 for s in suites
                        for k in ('failures', 'errors', 'skipped'))
                and not any(c.find(k) is not None for c in cases
                            for k in ('failure', 'error', 'skipped')), 'Incomplete test evidence')
        test_counts[kind] = count

    preflight = read(ARCHIVE / 'actual/preflight.json')
    manifest_identity = identity(ARCHIVE / 'actual/artifact_manifest.json')
    require(manifest_identity['sha256'] == MANIFEST_SHA and manifest_identity == preflight['manifest']
            and preflight['parent_revision'] == PARENT and preflight['prefix'] == PREFIX
            and preflight['upload_files'] == 281 and preflight['upload_bytes'] == 983157,
            'Prepared target differs')
    manifest = read(ARCHIVE / 'actual/artifact_manifest.json')
    expected = {**manifest['files'], 'artifact_manifest.json': manifest_identity}
    remote = read(ARCHIVE / 'actual/remote-audit.json')
    upload = read(ARCHIVE / 'actual/upload.json')
    download = read(ARCHIVE / 'actual/download-verified.json')
    require(remote['revision'] == upload['revision'] == download['revision'] == REVISION
            and remote['prefix'] == download['prefix'] == PREFIX
            and remote['durability_verified'] is True
            and remote['old_root_entries_unchanged'] == 20
            and remote['old_corrected_subtrees_unchanged'] == 11
            and remote['root_attributes_unchanged'] is True
            and remote['remote_paths_deleted_or_overwritten'] is False
            and remote['source_code_uploaded'] is False
            and upload['durability_verified'] is False
            and download['all_payload_hashes_match'] is True
            and download['files'] == 281 and download['bytes'] == 983157,
            'Actual transport evidence differs')
    require(set(remote['remote_inventory']) == set(expected) and len(expected) == 281,
            'Remote payload population differs')
    for name, record in remote['remote_inventory'].items():
        require(record['kind'] == 'git_blob_sha1'
                and record['digest'] == expected[name]['git_blob_sha1']
                and record['bytes'] == expected[name]['bytes'], 'Remote regular blob differs')
    modes = read(ARCHIVE / 'actual/upload-mode-preflight.json')['modes']
    require(set(modes) == {PREFIX + '/' + n for n in expected}
            and all(value == {'mode': 'regular', 'ignored': False, 'remote_oid': None}
                    for value in modes.values()), 'Upload modes changed')
    recovered_root = WORK / 'download' / PREFIX
    require(download['downloaded_root'] == str(recovered_root), 'Wrong actual download root')
    require({p.relative_to(recovered_root).as_posix() for p in recovered_root.rglob('*') if p.is_file()}
            == set(expected), 'Recovered inventory changed')
    for name, want in expected.items():
        require(identity(recovered_root / name) == want, 'Recovered data changed')

    staged = read(ARCHIVE / 'actual/staged-independent-readback.json')
    recovered = read(ARCHIVE / 'actual/recovered-independent-readback.json')
    for proof in (staged, recovered):
        require(proof['manifest'] == manifest_identity and proof['files_verified'] == 281
                and proof['raw_task_scores_reconstructed'] == 84
                and proof['native_exit_zero_workers_checked'] == 84
                and proof['complete_checkpoint_means_reconstructed'] == 6
                and proof['csv_rows_reconstructed'] == 90
                and proof['reads_only_supplied_snapshot'] is True
                and proof['network_transport_verified_by_this_program'] is False
                and proof['new_model_or_statistical_execution'] is False
                and proof['scientific_completion'] is False, 'Wrong independent replay scope')
    require(staged['fourth_stage_pool_b_means_0_to_100'] == recovered['fourth_stage_pool_b_means_0_to_100'],
            'Recovered exact means differ')
    commands = read(ARCHIVE / 'commands.json')
    for key in ('adapter_tests', 'reader_tests', 'prepare', 'check', 'staged_reader',
                'upload', 'download', 'recovered_reader'):
        require(commands['executions'][key]['terminal_result']['exit_code'] == 0,
                'Missing actual terminal tool receipt')
    observations = read(ARCHIVE / 'observations.json')
    latest = observations['final_primary']
    require(latest['observer']['live'] is True and latest['observer']['command_matches'] is True
            and not latest['failed_receipt_present']
            and len(latest['worker_observations_from_that_snapshot']) == 4
            and all(w['live'] and w['command_matches']
                    for w in latest['worker_observations_from_that_snapshot']),
            'Final separately timestamped original worker observation differs')

    guide = ROOT / 'docs/fourth-stage-pool-b-evaluation-restoration.md'
    current = ROOT / 'CURRENT_EXPERIMENT.md'
    require(REVISION in guide.read_text() and MANIFEST_SHA in guide.read_text()
            and SOURCES['verify_recovered.py'] in guide.read_text()
            and '54 / 60' in current.read_text()
            and str(latest['primary_tasks'][0]) + ' / 840' in current.read_text(),
            'Recovery guide or current status lacks actual bindings')
    expected_paths = {'README.md', 'commands.json', 'observations.json', 'actual/tests.json',
                      'before/CURRENT_EXPERIMENT.md', 'source/verify_archive.py',
                      'actual/artifact_manifest.json'}
    expected_paths |= {'source/' + name for name in SOURCES}
    expected_paths |= {'actual/' + name for name in COPIED}
    entries = list(ARCHIVE.rglob('*'))
    require(not any(p.is_symlink() for p in entries), 'Archive symlink refused')
    files = {p.relative_to(ARCHIVE).as_posix(): p for p in entries if p.is_file()}
    require(set(files) == expected_paths, 'Unexpected local archive file population')
    secret = re.compile(rb'(wandb_v1_[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})')
    for path in [*files.values(), guide, current]:
        require(secret.search(read_bytes(path)) is None, 'Credential-shaped content refused')
    links = 0
    for document in (guide, current, ARCHIVE / 'README.md'):
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', document.read_text()):
            target = target.split('#', 1)[0]
            if not target or target.startswith(('https://', 'http://')):
                continue
            resolved = (document.parent / target).resolve()
            require(resolved == output or resolved.exists(), 'Missing local document link')
            links += 1
    proof = {
        'scope': 'fourth_stage_pool_b_data_backup_local_evidence_verification',
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'revision': REVISION, 'manifest': manifest_identity, 'remote_files': 281,
        'anonymous_recovery_bytes': 983157, 'raw_task_scores_reconstructed': 84,
        'complete_checkpoint_means_reconstructed': 6, 'csv_rows_reconstructed': 90,
        'adapter_tests': test_counts['adapter'], 'reader_tests': test_counts['reader'],
        'initial_adapter_tests': test_counts['adapter_initial'],
        'original_assembled_source_files_unchanged': source_checks,
        'original_local_readback_payloads_unchanged': len(original['payloads']),
        'old_remote_root_entries_unchanged': 20, 'old_remote_corrected_subtrees_unchanged': 11,
        'newly_backed_up_checkpoint_outcomes': 6, 'overlapping_checkpoint_outcomes': 0,
        'complete_checkpoint_outcomes_now_backed_up': 54,
        'payloads': {name: identity(path) for name, path in sorted(files.items())},
        'guide': identity(guide), 'current_handoff': identity(current),
        'local_links_checked': links, 'credential_pattern_matches': 0,
        'same_physical_host_recovery': True, 'source_code_published': False,
        'functional_recovery_authorized': False, 'functional_recovery_launched': False,
        'new_model_or_statistical_execution': False, 'scientific_completion': False,
    }
    with output.open('x') as stream:
        stream.write(json.dumps(proof, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in proof.items() if k != 'payloads'}, sort_keys=True))


if __name__ == '__main__':
    main()
