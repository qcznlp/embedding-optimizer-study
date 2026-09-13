"""One-shot, additions-only transport of existing complete retrieval trajectories, never code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import re
import shutil
from pathlib import Path

ROOT = Path('/root/embedding-optimizer-story-refactor')
ORIGINAL = ROOT / 'reports/dense-v3-complete-trajectories-v1'
TRANSPORT = ROOT / 'reports/engineering-archive/dense-v3-evaluation-artifact-backup-v1/source/backup.py'
TRANSPORT_SHA = 'ab565206b22d0138c8227fef3f26eb28e2abeb7653c6f7137b7b532a289ec046'
VERIFICATION_SHA = 'c21ac61b21fc00907011758ca64d92f1e27478942644462d834477e88054b4bc'
HELPER_SHA = 'bebdfd3d773e550665660ec12fe7451f70e7e5fd419b1ef21814f774762ef33e'
PARENT = '5eb4cceed5c3dbee05d46850ec0d76aedff7d4cc'
REPO = 'qcz/embedding-optimizer-study-analysis-artifacts'
PROTOCOL = '4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b'
NAMESPACE = 'corrected-dense-correctness-v3'
ADDITION = f'{NAMESPACE}/complete-retrieval-trajectories'
ATTR_SHA = 'd7518c3119b61f65892c8c1b1608536862f89fcc91180c0511f1e75ce0edf7df'
TABLES = ('all_task_scores.csv', 'optimizer_stage_scores.csv', 'primary_summary.csv',
          'primary_task_effects.csv', 'run_observed_auc.csv', 'run_stage_scores.csv',
          'secondary_summary.csv', 'secondary_task_effects.csv', 'readout.json')
FIGURES = ('retrieval_trajectories.pdf', 'retrieval_trajectories.png', 'retrieval_trajectories.svg',
           'figure_data.csv', 'figure.json')
DESCRIPTIVE = ('description.json', 'summary.md', 'selected_stage_contrasts.csv')
OBSERVATIONS = ('observer-completed.json', 'pool-a-completed.json', 'pool-b-completed.json')
DATA_DIRECTORIES = (('tables', TABLES), ('figures', FIGURES),
                    ('descriptive', DESCRIPTIVE), ('observations', OBSERVATIONS))
INDEX_SHA = '0418c68b9a78cc80173c9cf6af5d5bb57d223d7e763bc6da03d2fe4f682bbf96'


def payload_names():
    return {directory + '/' + name for directory, names in DATA_DIRECTORIES for name in names} | {
        'provenance/source-relocated-readout.json', 'provenance/original-verification.json',
        'provenance/accepted-endpoint-readout.json', 'provenance/raw-score-index.json',
        'README.md', 'artifact_manifest.json'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_transport():
    require(hashlib.sha256(TRANSPORT.read_bytes()).hexdigest() == TRANSPORT_SHA,
            'Original transport source changed')
    require(hashlib.sha256((ROOT / 'scripts/restore_primary_v3.py').read_bytes()).hexdigest()
            == HELPER_SHA, 'Original hashing helper changed')
    spec = importlib.util.spec_from_file_location('trajectory_transport_parent', TRANSPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_sources(t):
    verification = t.read(ORIGINAL / 'verification.json', VERIFICATION_SHA)
    require(verification['scope'] == 'complete_actual_retrieval_readout_and_host_local_preservation'
            and verification['checkpoint_count'] == 60 and verification['task_score_count'] == 840
            and verification['copied_source_csvs_byte_identical'] == 8
            and verification['native_original_worker_exit_zero_count'] == 840
            and verification['all_six_endpoint_contrasts_unchanged'] is True
            and verification['scientific_completion'] is False,
            'Require the complete, bounded actual trajectory readout')
    selected = {}
    for directory, names in DATA_DIRECTORIES:
        for name in names:
            rel = directory + '/' + name
            selected[rel] = (ORIGINAL / rel, verification['payloads'][rel])
    for name, original in (
        ('source-relocated-readout.json', 'actual/source-relocated/readout.json'),
        ('accepted-endpoint-readout.json', 'inputs/accepted-endpoint-readout.json'),
    ):
        selected['provenance/' + name] = (ORIGINAL / original, verification['payloads'][original])
    selected['provenance/original-verification.json'] = (
        ORIGINAL / 'verification.json', {'sha256': VERIFICATION_SHA})
    index_path = Path(__file__).parents[1] / 'raw-score-index.json'
    index = t.read(index_path, INDEX_SHA)
    require(index['scope'] == 'complete_primary_v3_immutable_raw_score_recovery_index'
            and index['primary_protocol_sha256'] == PROTOCOL
            and index['checkpoint_count'] == 60 and index['raw_task_score_count'] == 840
            and len(index['snapshots']) == 5 and index['previously_recovered_files_rechecked'] == 2888
            and index['every_raw_score_matches_complete_trajectory_table'] is True
            and index['source_code_included'] is False and index['scientific_completion'] is False,
            'Require the complete recovered raw-score index')
    selected['provenance/raw-score-index.json'] = (index_path, {'sha256': INDEX_SHA})
    document = Path(__file__).with_name('ARTIFACT_README.md')
    selected['README.md'] = (document, t.file_identity(document))
    require(len(selected) == 25 and set(selected) == payload_names() - {'artifact_manifest.json'},
            'Wrong payload population')
    return selected


def inspect_payload(t, name, path, identity):
    t.safe_name(name)
    allowed = payload_names()
    require(name in allowed, 'Unselected payload refused')
    result = t.compare_file(path, identity)
    t.scan_text(path)
    raw = path.read_bytes()
    if name.endswith('.svg'):
        require(re.search(rb'<(?:script|foreignObject)\b|\bon\w+\s*=|(?:href|src)\s*=\s*[\"\'](?:https?:|data:|javascript:)',
                          raw, flags=re.I) is None, 'Active or external SVG content refused')
    if name.endswith('.pdf'):
        require(raw.startswith(b'%PDF-') and not any(x in raw for x in
                (b'/JavaScript', b'/JS ', b'/Launch', b'/EmbeddedFiles', b'/OpenAction')),
                'Active PDF content refused')
    if name.endswith('.png'):
        require(raw.startswith(b'\x89PNG\r\n\x1a\n'), 'Wrong PNG format')
    return result


def prepare(t, work):
    staging = work / 'staging'
    require(not staging.exists(), 'Preserve existing staging')
    sources = select_sources(t)
    staging.mkdir()
    files = {}
    for name, (source, identity) in sorted(sources.items()):
        inspect_payload(t, name, source, identity)
        destination = staging / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        files[name] = inspect_payload(t, name, destination, identity)
    manifest = {
        'schema_version': 1, 'scope': 'complete-v3-retrieval-trajectories-backup',
        'created_at_utc': t.stamp(), 'primary_protocol_sha256': PROTOCOL,
        'source_verification_sha256': VERIFICATION_SHA, 'parent_revision': PARENT,
        'raw_score_index_sha256': INDEX_SHA, 'raw_score_snapshots': 5,
        'checkpoint_count': 60, 'task_score_count': 840,
        'scientific_completion': False, 'source_code_included': False,
        'new_inference': False, 'contrasts': 6, 'families': 2, 'files': files,
        'payload_files': len(files), 'payload_bytes': sum(x['bytes'] for x in files.values())}
    t.write_new(staging / 'artifact_manifest.json', manifest)
    identity = t.file_identity(staging / 'artifact_manifest.json')
    result = {'scope': manifest['scope'], 'repo_id': REPO, 'parent_revision': PARENT,
              'prefix': f'{ADDITION}/{PROTOCOL}/{identity["sha256"]}', 'manifest': identity,
              'source': t.file_identity(Path(__file__)), 'parent_transport_sha256': TRANSPORT_SHA,
              'upload_files': len(files) + 1,
              'upload_bytes': manifest['payload_bytes'] + identity['bytes'],
              'remote_mutations': False, 'scientific_completion': False}
    t.write_new(work / 'preflight.json', result)
    return result


def prepared(t, work):
    preflight = t.read(work / 'preflight.json')
    require(preflight['source'] == t.file_identity(Path(__file__))
            and preflight['parent_transport_sha256'] == TRANSPORT_SHA
            and preflight['parent_revision'] == PARENT and preflight['repo_id'] == REPO,
            'Prepared source or target changed')
    require(preflight['prefix'] == f'{ADDITION}/{PROTOCOL}/{preflight["manifest"]["sha256"]}',
            'Prepared prefix changed')
    manifest = t.read(work / 'staging/artifact_manifest.json', preflight['manifest']['sha256'])
    sources = select_sources(t)
    require(set(manifest['files']) == set(sources) and manifest['source_code_included'] is False
            and manifest['scientific_completion'] is False and manifest['new_inference'] is False,
            'Prepared scope changed')
    for name, (source, identity) in sources.items():
        require(inspect_payload(t, name, source, identity) == manifest['files'][name],
                'Original payload identity changed')
    expected = {**manifest['files'], 'artifact_manifest.json': preflight['manifest']}
    actual = {p.relative_to(work / 'staging').as_posix()
              for p in (work / 'staging').rglob('*') if p.is_file()}
    require(len(expected) == 26 and actual == set(expected), 'Prepared inventory differs')
    for name, identity in expected.items():
        inspect_payload(t, name, work / 'staging' / name, identity)
    return preflight, expected


def check_preservation(before, after, before_sub, after_sub):
    require(set(before) == set(after), 'Root population changed')
    require(NAMESPACE in before and before[NAMESPACE]['kind'] == 'RepoFolder'
            and after[NAMESPACE]['kind'] == 'RepoFolder', 'Missing corrected namespace')
    require({k: v for k, v in before.items() if k != NAMESPACE}
            == {k: v for k, v in after.items() if k != NAMESPACE}, 'Old root entry changed')
    require(ADDITION not in before_sub and set(after_sub) == set(before_sub) | {ADDITION},
            'Unexpected corrected subtree added')
    require(all(after_sub[k] == v for k, v in before_sub.items()), 'Old corrected subtree changed')


def check_modes(modes, prefix):
    names = payload_names()
    require(set(modes) == {f'{prefix}/{n}' for n in names}, 'Wrong upload population')
    for path, row in modes.items():
        expected = 'lfs' if path == f'{prefix}/figures/retrieval_trajectories.png' else 'regular'
        require(row == {'mode': expected, 'ignored': False, 'remote_oid': None},
                'Unexpected upload mode, ignore or overwrite')


def upload(t, work):
    require(not (work / 'upload-started.json').exists(), 'Previous upload attempt: inspect, do not retry')
    preflight, expected = prepared(t, work)
    from huggingface_hub import CommitOperationAdd, HfApi, _commit_api, hf_api, hf_hub_download
    api = HfApi(endpoint='https://huggingface.co', token=True)
    require(api.whoami(token=True).get('name') == 'qcz', 'Configured owner differs')
    info = api.repo_info(REPO, repo_type='dataset', token=False)
    require(info.sha == PARENT and info.private is False, 'Remote parent/privacy changed')
    attrs = Path(hf_hub_download(REPO, repo_type='dataset', revision=PARENT,
                               filename='.gitattributes', token=False,
                               local_dir=work / 'parent-metadata'))
    require(t.file_identity(attrs)['sha256'] == ATTR_SHA
            and b'*.png filter=lfs diff=lfs merge=lfs -text' in attrs.read_bytes().splitlines(),
            'Require the existing PNG LFS rule; no new root rule allowed')
    before = t.root_inventory(api, PARENT)
    before_sub = t.root_inventory(api, PARENT, NAMESPACE)
    require(ADDITION not in before_sub, 'Existing complete trajectories: inspect, do not overwrite')
    for module, digest in ((_commit_api, '250ed0e5a5a39383974cab5baae08a964e60f3f40c394c0d13a52e119e1f5f39'),
                           (hf_api, '659636025aa3a7efefa69ca7f16741d8d6cc12f9301beac8c69f9b3d51f81cd4')):
        require(t.file_identity(Path(inspect.getsourcefile(module)))['sha256'] == digest,
                'HF upload implementation changed')
    operations = [CommitOperationAdd(path_in_repo=f'{preflight["prefix"]}/{name}',
                  path_or_fileobj=str(work / 'staging' / name)) for name in sorted(expected)]
    _commit_api._fetch_upload_modes(additions=operations, repo_type='dataset', repo_id=REPO,
        headers=api._build_hf_headers(token=True), revision=PARENT,
        endpoint='https://huggingface.co', create_pr=False)
    modes = {op.path_in_repo: {'mode': op._upload_mode, 'ignored': op._should_ignore,
                             'remote_oid': op._remote_oid} for op in operations}
    t.write_new(work / 'upload-mode-preflight.json', {'observed_at_utc': t.stamp(),
                'modes': modes, 'new_root_attributes_allowed': False, 'commit_created': False})
    check_modes(modes, preflight['prefix'])
    t.write_new(work / 'upload-started.json', {'observed_at_utc': t.stamp(), 'preflight': preflight,
                'original_root': before, 'original_corrected_subtree': before_sub})
    commit = api.create_commit(REPO, repo_type='dataset', parent_commit=PARENT,
        operations=operations, token=True, num_threads=2,
        commit_message='Preserve complete corrected DenseOn retrieval trajectories and recovery index (data only)')
    require(re.fullmatch('[0-9a-f]{40}', commit.oid) is not None, 'Missing immutable revision')
    result = {'uploaded_at_utc': t.stamp(), 'repo_id': REPO, 'revision': commit.oid,
              'prefix': preflight['prefix'], 'manifest': preflight['manifest'],
              'files': len(expected), 'durability_verified': False, 'scientific_completion': False}
    t.write_new(work / 'upload.json', result)
    actual = {}
    for item in api.list_repo_tree(REPO, repo_type='dataset', revision=commit.oid,
                                  path_in_repo=preflight['prefix'], recursive=True, token=False):
        if type(item).__name__ == 'RepoFile':
            actual[item.path.removeprefix(preflight['prefix'] + '/')] = {
                'bytes': item.size, 'kind': 'sha256' if item.lfs else 'git_blob_sha1',
                'digest': item.lfs.sha256 if item.lfs else item.blob_id}
    t.compare_remote(expected, actual)
    check_preservation(before, t.root_inventory(api, commit.oid), before_sub,
                       t.root_inventory(api, commit.oid, NAMESPACE))
    prepared(t, work)
    audit = {**result, 'verified_at_utc': t.stamp(), 'durability_verified': True,
             'remote_inventory': actual, 'old_root_entries_unchanged': len(before) - 1,
             'old_corrected_subtrees_unchanged': len(before_sub), 'source_code_uploaded': False,
             'remote_paths_deleted_or_overwritten': False, 'root_attributes_unchanged': True}
    t.write_new(work / 'remote-audit.json', audit)
    return {k: v for k, v in audit.items() if k != 'remote_inventory'}


def download_verify(t, work):
    from huggingface_hub import hf_hub_download
    preflight, expected = prepared(t, work)
    audit = t.read(work / 'remote-audit.json')
    require(audit['durability_verified'] is True and audit['prefix'] == preflight['prefix']
            and re.fullmatch('[0-9a-f]{40}', audit['revision']) is not None, 'No accepted remote audit')
    destination = work / 'download'
    require(not destination.exists(), 'Preserve previous download')
    destination.mkdir()
    for name, identity in sorted(expected.items()):
        filename = f'{audit["prefix"]}/{name}'
        path = Path(hf_hub_download(REPO, repo_type='dataset', revision=audit['revision'],
                    filename=filename, local_dir=destination, token=False,
                    endpoint='https://huggingface.co'))
        require(path == destination / filename, 'Different recovered location')
        inspect_payload(t, name, path, identity)
    root = destination / audit['prefix']
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
            == set(expected), 'Recovered population differs')
    result = {'scope': 'actual-complete-anonymous-retrieval-trajectories-recovery',
              'observed_at_utc': t.stamp(), 'revision': audit['revision'], 'prefix': audit['prefix'],
              'downloaded_root': str(root), 'files': len(expected),
              'bytes': sum(x['bytes'] for x in expected.values()), 'manifest': preflight['manifest'],
              'all_payload_hashes_match': True, 'same_physical_host': True,
              'new_model_or_statistical_execution': False, 'scientific_completion': False}
    t.write_new(work / 'download-verified.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'upload', 'download-verify'))
    parser.add_argument('--workdir', type=Path, required=True)
    args = parser.parse_args()
    require(args.workdir.parent == Path('/tmp')
            and args.workdir.name.startswith('dense-v3-trajectory-artifact-backup.')
            and args.workdir.is_dir() and not args.workdir.is_symlink(), 'Wrong task workdir')
    transport = load_transport()
    result = {'prepare': prepare, 'upload': upload, 'download-verify': download_verify}[
        args.action](transport, args.workdir)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
