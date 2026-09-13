"""Read-only closure of this transfer, its actual receipts and unchanged parents."""
import hashlib
import importlib.util
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

STORY = Path('/root/embedding-optimizer-story-refactor')
EXPERIMENT = Path('/root/embedding-optimizer-v3-experiment')
WORK = Path('/tmp/dense-v3-evaluation-artifact-backup.NyzIC4XW')
ARCHIVE = STORY / 'reports/engineering-archive/dense-v3-evaluation-artifact-backup-v1'
MANIFEST = 'bf443d5623c1312f8661a918ab93f9aea5f119058307733d477c8eac0c6da74f'
COMMIT = '3883b677f87b1982f06016e9fadb8bb95e0cfc96'


def identity(path):
    assert path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), path
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'git_blob_sha1': hashlib.sha1(b'blob ' + str(len(raw)).encode() + bytes([0]) + raw).hexdigest()}


def check(path, expected):
    got = identity(path)
    assert all(got[k] == v for k, v in expected.items()), path
    return got


def read(path):
    return json.loads(path.read_text())


def main():
    assert not (ARCHIVE / 'verification.json').exists(), 'Preserve previous verification'
    check(WORK / 'backup.py', {'sha256': 'ab565206b22d0138c8227fef3f26eb28e2abeb7653c6f7137b7b532a289ec046'})
    spec = importlib.util.spec_from_file_location('exact_evaluation_backup', WORK / 'backup.py')
    backup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backup)
    preflight, expected = backup.prepared(WORK)  # CPU file reads only; no network/write.
    native = backup.select_sources()
    for _, (path, binding) in native.items():
        check(path, binding)
    assert len(native) == 659 and len(expected) == 660
    assert preflight['manifest']['sha256'] == MANIFEST

    pairs = [(WORK / name, ARCHIVE / 'source' / name)
             for name in ('backup.py', 'replay_scores.py', 'test_evaluation_backup.py')]
    names = ('preflight.json', 'upload-mode-preflight.json', 'upload-started.json',
             'upload.json', 'remote-audit.json', 'download-verified.json',
             'staged-score-replay.json', 'recovered-score-replay.json', 'transport-tests.xml')
    pairs += [(WORK / name, ARCHIVE / 'actual' / name) for name in names]
    pairs += [(WORK / 'staging/artifact_manifest.json', ARCHIVE / 'actual/artifact_manifest.json'),
              (WORK / 'ARTIFACT_README.md', ARCHIVE / 'actual/HF_README.md'),
              (WORK / 'CURRENT_EXPERIMENT.before.md', ARCHIVE / 'before/CURRENT_EXPERIMENT.md'),
              (WORK / 'commands.json', ARCHIVE / 'commands.json'),
              (WORK / 'observations.json', ARCHIVE / 'observations.json'),
              (WORK / 'RESTORATION_GUIDE.md', STORY / 'docs/evaluation-analysis-restoration.md')]
    for original, copied in pairs:
        check(copied, identity(original))

    remote = read(ARCHIVE / 'actual/remote-audit.json')
    download = read(ARCHIVE / 'actual/download-verified.json')
    modes = read(ARCHIVE / 'actual/upload-mode-preflight.json')
    assert remote['revision'] == download['revision'] == COMMIT
    assert remote['durability_verified'] is True and remote['files'] == 660
    assert remote['old_root_entries_unchanged'] == 20
    assert remote['old_corrected_subtrees_unchanged'] == 2
    assert remote['corrected_root_tree_changed_only_by_new_evaluation_subtree'] is True
    assert remote['remote_paths_deleted_or_overwritten'] is False
    assert download['all_payload_hashes_match'] is True and download['bytes'] == 29526987
    assert modes['commit_created'] is False and modes['lfs_payloads_uploaded'] is False
    assert len(modes['files']) == 660 and all(
        r == {'ignored': False, 'mode': 'regular', 'remote_oid': None} for r in modes['files'].values())
    recovered = read(ARCHIVE / 'actual/recovered-score-replay.json')
    assert recovered['numeric_replay_passed'] is True and recovered['verified_files'] == 660
    assert recovered['validation_rows'] == 49152 and recovered['scalar_validation_checks'] == 294912
    assert recovered['beir_task_scores'] == 168 and recovered['beir_endpoint_means'] == 12
    assert recovered['validation_group_metric_means'] == 576 and recovered['tables_verified'] == 4
    commands = read(ARCHIVE / 'commands.json')['records']
    terminal = ('evaluationDurabilityPrepareTerminal', 'evaluationDurabilityTestsLaunch',
                'evaluationDurabilityUploadPoll1', 'evaluationDurabilityDownloadPoll4',
                'evaluationDurabilityStagedReplayLaunch', 'evaluationDurabilityRecoveredReplayLaunch',
                'evaluationDurabilityGuideOffline')
    for name in terminal:
        entry = commands[name]
        result = entry.get('result', entry)
        assert result['exit_code'] == 0 and result['chunk_id']
    guide = json.loads(commands['evaluationDurabilityGuideOffline']['result']['output'])
    check(STORY / 'docs/evaluation-analysis-restoration.md', {'sha256': guide['guide_sha256']})
    assert guide['offline_block_executed'] and guide['staged_and_recovered_numeric_receipts_match_excluding_path_time']
    xml = ET.parse(ARCHIVE / 'actual/transport-tests.xml').getroot()
    suites = list(xml.iter('testsuite'))
    assert sum(int(s.attrib['tests']) for s in suites) == len(list(xml.iter('testcase'))) == 40
    assert all(int(s.attrib[k]) == 0 for s in suites for k in ('errors', 'failures', 'skipped'))

    previous = STORY / 'reports/engineering-archive/dense-v3-validation-selection-readback-v1'
    check(previous / 'verification.json', {'sha256': '863e2ba7e69ef66d19c4fb9036ac8bc2ced42dc96d3e4375cff88baf271943c6'})
    previous_verification = read(previous / 'verification.json')
    for name, binding in previous_verification['archive_files'].items():
        check(previous / name, binding)
    check(ARCHIVE / 'before/CURRENT_EXPERIMENT.md', previous_verification['live_files']['CURRENT_EXPERIMENT.md'])
    endpoint = STORY / 'reports/engineering-archive/dense-v3-all-final-evaluations-v1'
    check(endpoint / 'verification.json', {'sha256': '131cd6df358a270e85471db76f0b1971e7849934c245e9f76cd1f0f7b10036ff'})
    endpoint_verification = read(endpoint / 'verification.json')
    for name, binding in endpoint_verification['archive_files'].items():
        check(endpoint / name, binding)
    check(STORY / 'reports/engineering-archive/dense-v3-training-artifact-backup-v1/verification.json',
          {'sha256': 'a9c55c4b1d8f2878fb2c8418316d6bb09135934fa12bcd3c9b3bae39bc9dafcf'})
    check(STORY / 'reports/engineering-archive/dense-v3-training-artifact-backup-v1/backup.py',
          {'sha256': '4dbfac65e1aaee5f004bd2af1d4cc4200d9d193329c1d7aaf0858d45bfbdad41'})
    check(STORY / 'AGENTS.md', {'sha256': '2ea0ac433747a4406823013f36701f49fb8cd806a5c01bdbad73c88ad93b6946'})
    for root in (Path('/root/embedding-optimizer-primary-v3'), EXPERIMENT / 'launch/source-snapshot'):
        check(root / 'source-assembly.json', {'sha256': 'e603ef175c0f691a1dd83c4d8b474d90da2fc67c8c65f5a77f3c3180e78332b8'})
        assembly = read(root / 'source-assembly.json')
        assert len(assembly['files']) == 56
        for name, binding in assembly['files'].items():
            check(root / name, binding['identity'])
    dispatch = {
        'evaluation-handoff/dispatch.py': '5b8a89bed5e0eaa02a12585ee3f6c883ca3e162b550898d68322321ee41d8427',
        'evaluation-handoff/authorization.json': '2351e225387f42fe607d7e002a0717de79a78cba9f3c93f66652510083f0be7c',
        'validation-handoff/validation.py': 'd2fc55c67670ea5b42cacb055f48379ef278a465ce86d95808231ab86049d913',
        'validation-handoff/authorization.json': '6a8dcdd579bd6c2f6ea82e3f4cd808f88e92e9a28b9e97f41ffa6447eb0e1f6f',
        'functional-dimensions/dispatch.py': '3b02f1d8486f3c2b41d0403cbf291580d0ca32868b8b29cb05514a22770257f5',
        'functional-dimensions/authorization.json': 'd72527595e2c963eae7fd46b0a1de2bd83d1ea15a863ae467a1065cfc5b2f336'}
    for name, digest in dispatch.items():
        check(EXPERIMENT / 'launch' / name, {'sha256': digest})
    framing = STORY / 'reports/paper-review/dense-v3-retrieval-usefulness-v1'
    frame = read(framing / 'verification.json')
    for name, binding in read(framing / 'before-bindings.json').items():
        if name not in frame['live_files']:
            check(STORY / name, binding)
    for name, binding in frame['live_files'].items():
        if name != 'CURRENT_EXPERIMENT.md':
            check(STORY / name, binding)
    check(STORY / 'paper/build/main.pdf', frame['archive_files']['draft/main.pdf'])

    linked = 0
    documents = (ARCHIVE / 'README.md', STORY / 'CURRENT_EXPERIMENT.md', STORY / 'docs/evaluation-analysis-restoration.md')
    for document in documents:
        for link in re.findall(r'\]\(([^)]+)\)', document.read_text()):
            if link.startswith(('https://', 'http://', '#')):
                continue
            target = (document.parent / link.split('#', 1)[0]).resolve()
            if target == ARCHIVE / 'verification.json':
                continue  # Its actual result is written after this call exits zero.
            assert target.exists(), (document, link)
            linked += 1
    archive_files = {p.relative_to(ARCHIVE).as_posix(): identity(p)
                     for p in sorted(ARCHIVE.rglob('*')) if p.is_file()}
    live_files = {str(p.relative_to(STORY)): identity(p) for p in documents[1:]}
    for path in [*(ARCHIVE / name for name in archive_files), *(STORY / name for name in live_files)]:
        matches = backup.SECRETS.findall(path.read_bytes())
        if path == ARCHIVE / 'actual/transport-tests.xml':
            # Exact, local-only negative-test parameter; never a public payload exception.
            check(path, {'sha256': '5d3ddbb0b136935c19805192b584b0c8433a8024bb88aeb8b7fb47c07090be12'})
            dummy = b'hf_' + b'a' * 24
            assert matches == [dummy]
            assert '"hf_" + "a" * 24' in (ARCHIVE / 'source/test_evaluation_backup.py').read_text()
            expected_name = 'test_code_or_credential_shaped_text_is_refused[' + dummy.decode() + ']'
            assert sum(e.attrib['name'] == expected_name for e in xml.iter('testcase')) == 1
        else:
            assert not matches, path
    result = {'scope': 'complete_endpoint_validation_artifact_durability_and_offline_score_recovery',
              'observed_at_utc': datetime.now(timezone.utc).isoformat(),
              'archive_files': archive_files, 'live_files': live_files, 'local_links_checked': linked,
              'native_original_files_freshly_rehashed': len(native), 'staged_files_reverified': len(expected),
              'anonymous_recovered_files': 660, 'anonymous_recovered_bytes': 29526987,
              'original_terminal_successful_calls_checked': len(terminal), 'transport_tests': 40,
              'recovered_numerical_replay': recovered, 'old_remote_root_entries_unchanged': 20,
              'old_remote_corrected_subtrees_unchanged': 2, 'remote_revision': COMMIT,
              'prior_validation_archive_files_preserved': len(previous_verification['archive_files']),
              'prior_endpoint_archive_files_preserved': len(endpoint_verification['archive_files']),
              'prior_handoff_preserved_in_before_copy': True, 'prior_training_backup_caveat_unchanged': True,
              'primary_source_assemblies_verified': 2, 'files_per_primary_assembly': 56,
              'dispatch_authorization_files_unchanged': len(dispatch),
              'exact_local_synthetic_negative_test_id_occurrences': 1,
              'public_payload_credential_scan_unchanged': True,
              'original_local_archive_scan_failure_preserved': True,
              'manuscript_pdf_generated_results_and_protocols_unchanged': True,
              'verification_network_or_model_execution': False, 'source_code_published': False,
              'physical_second_host_experiment': False, 'resource_handoff': False,
              'scientific_completion': False, 'whole_goal_complete': False, 'passed_in_declared_scope': True}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
