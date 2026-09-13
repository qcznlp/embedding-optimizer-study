"""Verify the actual immutable fourth-stage transport and isolated reconstruction."""
import importlib.util
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ARCHIVE = Path(__file__).parents[1]
STORY = Path('/root/embedding-optimizer-story-refactor')
WORK = Path('/tmp/dense-v3-complete-fourth-stage-artifact-backup.Gh7ngvO3')


def main():
    output = ARCHIVE / 'verification.json'
    assert not output.exists(), 'Preserve verification'
    spec = importlib.util.spec_from_file_location('fourth_stage_archive_backup', ARCHIVE / 'source/backup.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    transport = module.load_transport()
    preflight, expected = module.prepared(transport, WORK)
    audit = transport.read(ARCHIVE / 'actual/remote-audit.json')
    download = transport.read(ARCHIVE / 'actual/download-verified.json')
    reconstructed = transport.read(ARCHIVE / 'actual/isolated-reconstruction.json')
    started = transport.read(ARCHIVE / 'actual/upload-started.json')
    assert preflight['upload_files'] == audit['files'] == download['files'] == 557
    assert audit['revision'] == download['revision'] == '5eb4cceed5c3dbee05d46850ec0d76aedff7d4cc'
    assert audit['durability_verified'] is True and download['all_payload_hashes_match'] is True
    assert audit['source_code_uploaded'] is False and audit['root_attributes_unchanged'] is True
    assert audit['remote_paths_deleted_or_overwritten'] is False
    assert audit['old_root_entries_unchanged'] == 20 and audit['old_corrected_subtrees_unchanged'] == 12
    assert preflight['manifest'] == audit['manifest'] == download['manifest']
    assert download['bytes'] == preflight['upload_bytes'] == 1959719
    assert started['preflight'] == preflight
    modes = transport.read(ARCHIVE / 'actual/upload-mode-preflight.json')
    module.check_modes(modes['modes'], preflight['prefix'], expected)
    assert modes['commit_created'] is False and modes['new_root_attributes_allowed'] is False
    assert reconstructed['files_verified'] == 557 and reconstructed['bytes_verified'] == 1959719
    assert reconstructed['raw_task_scores_reconstructed'] == 168
    assert reconstructed['native_exit_zero_workers_checked'] == 168
    assert reconstructed['complete_checkpoint_means_reconstructed'] == 12
    assert reconstructed['csv_rows_reconstructed'] == 180
    assert reconstructed['reads_only_supplied_snapshot'] is True
    assert reconstructed['network_transport_verified_by_this_program'] is False
    assert reconstructed['new_model_or_statistical_execution'] is False
    assert reconstructed['scientific_completion'] is False
    recovered = Path(download['downloaded_root'])
    assert recovered == WORK / 'download' / preflight['prefix']
    for name, item in expected.items():
        transport.compare_file(recovered / transport.safe_name(name), item)
    transport.compare_remote(expected, audit['remote_inventory'])
    bundle, rows, _ = module.source_inputs(transport)
    assert reconstructed['fourth_stage_means_0_to_100'] == {
        row['run_id']: row['macro_score_0_to_100'] for row in rows}
    for name in ('backup.py', 'verify_recovered.py', 'test_backup.py',
                 'test_verify_recovered.py', 'ARTIFACT_README.md'):
        assert transport.file_identity(ARCHIVE / 'source' / name) == transport.file_identity(WORK / 'source' / name)
    assert transport.file_identity(ARCHIVE / 'source/verify_recovered.py')['sha256'] == 'a0050dd292e68df684812c2789f3588a27574228b25e2a6a3421727b52179cda'
    tests = {}
    for name, count in (('adapter-tests.xml', 50), ('reader-tests.xml', 18)):
        suites = ET.parse(ARCHIVE / 'actual' / name).getroot().findall('testsuite')
        result = {key: sum(int(s.attrib[key]) for s in suites)
                  for key in ('tests', 'failures', 'errors', 'skipped')}
        assert result == {'tests': count, 'failures': 0, 'errors': 0, 'skipped': 0}
        tests[name] = result
    verification = transport.read(module.ORIGINAL / 'verification.json', module.VERIFICATION_SHA)
    for name, item in verification['payloads'].items():
        transport.compare_file(module.ORIGINAL / transport.safe_name(name), item)
    for name, item in verification['protected_inputs'].items():
        transport.compare_file(Path(name), item)
    assert transport.file_identity(ARCHIVE / 'before/CURRENT_EXPERIMENT.md')['sha256'] == verification['current_handoff']['sha256']
    payloads, links = {}, 0
    secret = re.compile(rb'(?:wandb_v1_[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})')
    for path in sorted(ARCHIVE.rglob('*')):
        assert not path.is_symlink(), str(path)
        if not path.is_file():
            continue
        raw = path.read_bytes()
        assert not secret.search(raw), 'Credential-like content'
        payloads[path.relative_to(ARCHIVE).as_posix()] = transport.file_identity(path)
        if path.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)', raw.decode()):
                if target.startswith(('http:', 'https:', '#')):
                    continue
                origin = STORY if path.parent == ARCHIVE / 'before' else path.parent
                resolved = (origin / target.split('#', 1)[0]).resolve()
                assert resolved.exists() or resolved == output, (str(path), target)
                links += 1
    report = {'scope': 'complete_fourth_stage_data_backup_host_local_verification',
              'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'payloads': payloads,
              'revision': audit['revision'], 'prefix': audit['prefix'], 'manifest': preflight['manifest'],
              'files_anonymously_recovered': 557, 'bytes_anonymously_recovered': 1959719,
              'raw_scores_independently_reconstructed': 168, 'exact_means_reconstructed': 12,
              'all_sixty_checkpoint_outcomes_now_have_raw_score_backups': True,
              'new_checkpoint_outcomes_backed_up': 6, 'prior_pool_b_overlap': 6,
              'old_root_entries_unchanged': 20, 'old_corrected_subtrees_unchanged': 12,
              'root_attributes_unchanged': True, 'test_xml_counts': tests,
              'complete_trajectory_archive_unchanged': transport.file_identity(module.ORIGINAL / 'verification.json'),
              'protected_inputs_unchanged': len(verification['protected_inputs']),
              'markdown_local_links_checked': links, 'credential_findings': 0,
              'current_handoff': transport.file_identity(STORY / 'CURRENT_EXPERIMENT.md'),
              'guide': transport.file_identity(STORY / 'docs/fourth-stage-evaluation-restoration.md'),
              'same_physical_host_recovery': True, 'new_model_or_statistical_execution': False,
              'source_code_uploaded': False, 'source_or_manuscript_release': False,
              'functional_recovery': False, 'scientific_completion': False}
    transport.write_new(output, report)
    print(json.dumps({'verification': str(output), 'identity': transport.file_identity(output),
                      'payloads': len(payloads), 'files_recovered': 557, 'raw_scores_reconstructed': 168,
                      'checkpoint_outcomes_with_backups': 60, 'scientific_completion': False}, sort_keys=True))


if __name__ == '__main__':
    main()
