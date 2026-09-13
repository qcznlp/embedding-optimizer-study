"""Verify the actual complete trajectory backup, raw index and unchanged parents."""
import importlib.util
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ARCHIVE = Path(__file__).parents[1]
STORY = Path('/root/embedding-optimizer-story-refactor')
WORK = Path('/tmp/dense-v3-trajectory-artifact-backup.hAJsqqOm')


def main():
    output = ARCHIVE / 'verification.json'
    assert not output.exists(), 'Preserve verification'
    spec = importlib.util.spec_from_file_location('trajectory_archive_transport', ARCHIVE / 'source/backup.py')
    backup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backup)
    t = backup.load_transport()
    preflight, expected = backup.prepared(t, WORK)
    audit = t.read(ARCHIVE / 'actual/remote-audit.json')
    download = t.read(ARCHIVE / 'actual/download-verified.json')
    replay = t.read(ARCHIVE / 'actual/isolated-reconstruction.json')
    staged = t.read(ARCHIVE / 'actual/staged-reconstruction.json')
    modes = t.read(ARCHIVE / 'actual/upload-mode-preflight.json')
    backup.check_modes(modes['modes'], preflight['prefix'])
    assert modes['commit_created'] is False and modes['new_root_attributes_allowed'] is False
    assert preflight['upload_files'] == audit['files'] == download['files'] == replay['files_verified'] == 26
    assert preflight['upload_bytes'] == download['bytes'] == replay['bytes_verified'] == 1093911
    assert audit['revision'] == download['revision'] == 'dbcd12376f483347cc570584abc588503c0a5fa0'
    assert audit['durability_verified'] is True and download['all_payload_hashes_match'] is True
    assert audit['source_code_uploaded'] is False and audit['remote_paths_deleted_or_overwritten'] is False
    assert audit['root_attributes_unchanged'] is True
    assert audit['old_root_entries_unchanged'] == 20 and audit['old_corrected_subtrees_unchanged'] == 13
    assert audit['manifest'] == download['manifest'] == preflight['manifest'] == replay['manifest']
    recovered = Path(download['downloaded_root'])
    assert recovered == WORK / 'download' / preflight['prefix']
    for name, item in expected.items():
        t.compare_file(recovered / t.safe_name(name), item)
    t.compare_remote(expected, audit['remote_inventory'])
    assert {k: v for k, v in staged.items() if k != 'observed_at_utc'} == {
        k: v for k, v in replay.items() if k != 'observed_at_utc'}
    assert replay['task_scores_crosschecked_with_index_and_observer'] == 840
    assert replay['run_stage_means_and_medians_reconstructed'] == 120
    assert replay['optimizer_stage_means_and_medians_reconstructed'] == 30
    assert replay['areas_and_normalized_means_reconstructed'] == 24
    assert replay['endpoint_contrasts_matched'] == 6 and replay['plotted_checkpoints_reconstructed'] == 60
    assert replay['new_bootstrap_or_model_execution'] is False and replay['scientific_completion'] is False
    first = t.read(ARCHIVE / 'raw-score-index.json', backup.INDEX_SHA)
    second = t.read(ARCHIVE / 'actual/source-relocated-raw-score-index.json')
    assert {k: v for k, v in first.items() if k != 'created_at_utc'} == {
        k: v for k, v in second.items() if k != 'created_at_utc'}
    assert first['previously_recovered_files_rechecked'] == 2888
    assert first['recovered_logical_bytes_rechecked'] == 37364806
    assert first['checkpoint_count'] == 60 and first['raw_task_score_count'] == 840
    assert first['every_raw_score_matches_complete_trajectory_table'] is True
    for name in ('backup.py', 'verify_recovered.py', 'test_backup.py', 'test_verify_recovered.py',
                 'build_raw_score_index.py', 'ARTIFACT_README.md'):
        assert t.file_identity(ARCHIVE / 'source' / name) == t.file_identity(WORK / 'source' / name)
    assert t.file_identity(ARCHIVE / 'source/verify_recovered.py')['sha256'] == '30c71d09767b29cd9e6164bc9e0e8ccd609b9773d116f79185337e64a61078ea'
    tests = {}
    for name, count in (('adapter-tests.xml', 18), ('reader-tests.xml', 24)):
        suites = ET.parse(ARCHIVE / 'actual' / name).getroot().findall('testsuite')
        counts = {k: sum(int(s.attrib[k]) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
        assert counts == {'tests': count, 'failures': 0, 'errors': 0, 'skipped': 0}
        tests[name] = counts
    parent = t.read(backup.ORIGINAL / 'verification.json', backup.VERIFICATION_SHA)
    for name, item in parent['payloads'].items():
        t.compare_file(backup.ORIGINAL / t.safe_name(name), item)
    for name, item in parent['protected_inputs'].items():
        t.compare_file(Path(name), item)
    primary = Path('/root/embedding-optimizer-primary-v3')
    experiment = Path('/root/embedding-optimizer-v3-experiment')
    assembly = t.read(primary / 'source-assembly.json', 'e603ef175c0f691a1dd83c4d8b474d90da2fc67c8c65f5a77f3c3180e78332b8')
    count = 0
    for base in (primary, experiment / 'launch/source-snapshot'):
        for name, item in assembly['files'].items():
            t.compare_file(base / t.safe_name(name), item['identity'])
            count += 1
    assert count == 112
    assert t.file_identity(ARCHIVE / 'before/CURRENT_EXPERIMENT.md')['sha256'] == '134566ea4dd55dbb95af420306659e9d30f3a3c0c01ef845635aceb28c0d32eb'
    payloads, links = {}, 0
    secret = re.compile(rb'(?:wandb_v1_[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})')
    for path in sorted(ARCHIVE.rglob('*')):
        assert not path.is_symlink(), str(path)
        if not path.is_file():
            continue
        raw = path.read_bytes()
        assert not secret.search(raw), 'Credential-like string'
        payloads[path.relative_to(ARCHIVE).as_posix()] = t.file_identity(path)
        if path.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)', raw.decode()):
                if target.startswith(('http:', 'https:', '#')):
                    continue
                origin = STORY if path.parent == ARCHIVE / 'before' else path.parent
                resolved = (origin / target.split('#', 1)[0]).resolve()
                assert resolved.exists() or resolved == output, (str(path), target)
                links += 1
    guide = STORY / 'docs/retrieval-trajectories-restoration.md'
    for target in re.findall(r'\]\(([^)]+)\)', guide.read_text()):
        if not target.startswith(('http:', 'https:', '#')):
            assert (guide.parent / target.split('#', 1)[0]).exists(), target
            links += 1
    report = {'scope': 'complete_trajectory_data_backup_host_local_verification',
              'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'payloads': payloads,
              'revision': audit['revision'], 'prefix': audit['prefix'], 'manifest': preflight['manifest'],
              'files_anonymously_recovered': 26, 'bytes_anonymously_recovered': 1093911,
              'complete_task_score_grid_checked': 840, 'plotted_checkpoints_reconstructed': 60,
              'original_endpoint_contrasts_unchanged': 6, 'staged_and_recovered_readback_agree': True,
              'raw_index_copied_source_replay_equal_except_time': True,
              'previously_recovered_raw_snapshot_files_rechecked': 2888,
              'previously_recovered_raw_snapshot_bytes_rechecked': 37364806,
              'old_root_entries_unchanged': 20, 'old_corrected_subtrees_unchanged': 13,
              'root_attributes_unchanged': True, 'test_xml_counts': tests,
              'protected_inputs_unchanged': len(parent['protected_inputs']),
              'original_source_assembly_files_unchanged': count,
              'complete_trajectory_parent_unchanged': t.file_identity(backup.ORIGINAL / 'verification.json'),
              'markdown_local_links_checked': links, 'credential_findings': 0,
              'current_handoff': t.file_identity(STORY / 'CURRENT_EXPERIMENT.md'),
              'current_readme': t.file_identity(STORY / 'README.md'),
              'current_project_status': t.file_identity(STORY / 'PROJECT_STATUS.md'),
              'guide': t.file_identity(guide), 'same_physical_host_recovery': True,
              'new_model_or_bootstrap_execution': False, 'functional_recovery': False,
              'source_or_manuscript_release': False, 'scientific_completion': False}
    t.write_new(output, report)
    print(json.dumps({'verification': str(output), 'identity': t.file_identity(output),
                      'payloads': len(payloads), 'files_recovered': 26, 'scores_reconstructed': 840,
                      'raw_index_source_snapshots': 5, 'original_source_files_unchanged': count,
                      'scientific_completion': False}, sort_keys=True))


if __name__ == '__main__':
    main()
