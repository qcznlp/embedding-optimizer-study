"""One-shot, additions-only preservation of original v3 training completion evidence."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import shutil

import verify_recovered as reader

ROOT = Path('/root/embedding-optimizer-story-refactor')
EXP = Path('/root/embedding-optimizer-v3-experiment')
ORIGINAL = ROOT / 'reports/engineering-archive/dense-v3-campaign-evidence-candidate-v1/additional-records'
TRANSPORT = ROOT / 'reports/engineering-archive/dense-v3-evaluation-artifact-backup-v1/source/backup.py'
TRANSPORT_SHA = 'ab565206b22d0138c8227fef3f26eb28e2abeb7653c6f7137b7b532a289ec046'
HELPER_SHA = 'bebdfd3d773e550665660ec12fe7451f70e7e5fd419b1ef21814f774762ef33e'
REPO = 'qcz/embedding-optimizer-study-analysis-artifacts'
PARENT = '37702ccb55820dbf5956a57257c5582d56733da9'
NAMESPACE = 'corrected-dense-correctness-v3'
ADDITION = NAMESPACE + '/training-completion-evidence'
ATTR_SHA = 'd7518c3119b61f65892c8c1b1608536862f89fcc91180c0511f1e75ce0edf7df'
require = reader.require


def transport():
    reader.bound(TRANSPORT, {'sha256': TRANSPORT_SHA})
    reader.bound(ROOT / 'scripts/restore_primary_v3.py', {'sha256': HELPER_SHA})
    spec = importlib.util.spec_from_file_location('campaign_evidence_backup_transport', TRANSPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sources(t):
    reader.bound(ORIGINAL.parent / 'verification.json', {
        'sha256': '9aa010869f410c20b607c48539b78eb8d067cb02d5bdef6ef7132d848760a155'})
    reader.check_originals(ORIGINAL)
    out = {}
    for name, (size, sha) in reader.ORIGINALS.items():
        p = ORIGINAL / name
        value = t.compare_file(p, {'bytes': size, 'sha256': sha})
        t.scan_text(p)
        out[name] = (p, value)
    doc = Path(__file__).with_name('ARTIFACT_README.md')
    t.scan_text(doc)
    out['README.md'] = (doc, t.file_identity(doc))
    return out


def source_identities(t):
    return {name: t.file_identity(Path(__file__).with_name(name))
            for name in ('backup.py', 'verify_recovered.py', 'ARTIFACT_README.md')}


def prepare(t, work):
    selected = sources(t)
    dest = work / 'staging'
    require(not dest.exists() and not (work / 'preflight.json').exists(), 'Preserve previous preparation')
    require(shutil.disk_usage(work).free > 50_000_000, 'Insufficient staging/recovery space')
    dest.mkdir()
    for name, (source, ident) in selected.items():
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        t.compare_file(target, ident)
    manifest = {'scope': 'original_primary_training_completion_evidence_data_only',
                'primary_protocol_sha256': reader.PROTOCOL, 'training_runs': 12,
                'checkpoint_states': 60, 'original_exit_zero_runs': 10,
                'original_exit_unobserved_runs': 2,
                'source_code_included': False, 'scientific_completion': False,
                'files': {name: ident for name, (_, ident) in selected.items()}}
    t.write_new(dest / 'artifact_manifest.json', manifest)
    ident = t.file_identity(dest / 'artifact_manifest.json')
    result = {'prepared_at_utc': t.stamp(), 'repo_id': REPO, 'parent_revision': PARENT,
              'prefix': f'{ADDITION}/{reader.PROTOCOL}/{ident["sha256"]}',
              'manifest': ident, 'sources': source_identities(t),
              'parent_transport_sha256': TRANSPORT_SHA, 'helper_sha256': HELPER_SHA,
              'files': 41, 'bytes': sum(v['bytes'] for _, v in selected.values()) + ident['bytes'],
              'remote_mutations': False, 'scientific_completion': False}
    reader.verify(dest, ident['sha256'])
    t.write_new(work / 'preflight.json', result)
    return result


def prepared(t, work):
    pre = t.read(work / 'preflight.json')
    require(pre['sources'] == source_identities(t) and pre['parent_revision'] == PARENT
            and pre['repo_id'] == REPO and pre['parent_transport_sha256'] == TRANSPORT_SHA
            and pre['helper_sha256'] == HELPER_SHA, 'Prepared source or target changed')
    require(pre['prefix'] == f'{ADDITION}/{reader.PROTOCOL}/{pre["manifest"]["sha256"]}', 'Prefix changed')
    reader.verify(work / 'staging', pre['manifest']['sha256'])
    manifest = t.read(work / 'staging/artifact_manifest.json')
    selected = sources(t)
    require(manifest['files'] == {k: v for k, (_, v) in selected.items()}, 'Original sources changed')
    expected = {**manifest['files'], 'artifact_manifest.json': pre['manifest']}
    for name in expected:
        t.scan_text(work / 'staging' / name)
    return pre, expected


def check_modes(modes, prefix, names):
    require(set(names) == set(reader.ORIGINALS) | {'README.md', 'artifact_manifest.json'}, 'Wrong population')
    require(set(modes) == {prefix + '/' + name for name in names}, 'Upload population differs')
    for name in names:
        expected = {'mode': 'regular',
                    'ignored': False, 'remote_oid': None}
        require(modes[prefix + '/' + name] == expected, 'Unexpected mode, ignored file or overwrite')


def check_preservation(before, after, old_sub, new_sub):
    require(set(before) == set(after) and NAMESPACE in before
            and before[NAMESPACE]['kind'] == after[NAMESPACE]['kind'] == 'RepoFolder', 'Root population changed')
    require({k: v for k, v in before.items() if k != NAMESPACE}
            == {k: v for k, v in after.items() if k != NAMESPACE}, 'Old root entry changed')
    require(len(old_sub) == 10 and ADDITION not in old_sub
            and set(new_sub) == set(old_sub) | {ADDITION}
            and all(new_sub[k] == v for k, v in old_sub.items()), 'Old subtree changed')


def upload(t, work):
    require(not (work / 'upload-started.json').exists(), 'Prior attempt: inspect outcome, never retry')
    pre, expected = prepared(t, work)
    from huggingface_hub import CommitOperationAdd, HfApi, _commit_api, hf_api, hf_hub_download
    api = HfApi(endpoint='https://huggingface.co', token=True)
    require(api.whoami(token=True).get('name') == 'qcz', 'Owner differs')
    info = api.repo_info(REPO, repo_type='dataset', token=False)
    require(info.sha == PARENT and info.private is False, 'Parent or privacy changed; no commit')
    attrs = Path(hf_hub_download(REPO, repo_type='dataset', revision=PARENT, filename='.gitattributes',
                                local_dir=work / 'parent-metadata', token=False))
    reader.bound(attrs, {'sha256': ATTR_SHA})
    before, old_sub = t.root_inventory(api, PARENT), t.root_inventory(api, PARENT, NAMESPACE)
    require(len(old_sub) == 10 and ADDITION not in old_sub, 'Unexpected existing namespace')
    for module, sha in ((_commit_api, '250ed0e5a5a39383974cab5baae08a964e60f3f40c394c0d13a52e119e1f5f39'),
                        (hf_api, '659636025aa3a7efefa69ca7f16741d8d6cc12f9301beac8c69f9b3d51f81cd4')):
        reader.bound(Path(inspect.getsourcefile(module)), {'sha256': sha})
    ops = [CommitOperationAdd(path_in_repo=pre['prefix'] + '/' + n,
                              path_or_fileobj=str(work / 'staging' / n)) for n in sorted(expected)]
    _commit_api._fetch_upload_modes(additions=ops, repo_type='dataset', repo_id=REPO,
        headers=api._build_hf_headers(token=True), revision=PARENT,
        endpoint='https://huggingface.co', create_pr=False)
    modes = {op.path_in_repo: {'mode': op._upload_mode, 'ignored': op._should_ignore,
                              'remote_oid': op._remote_oid} for op in ops}
    t.write_new(work / 'upload-mode-preflight.json', {'observed_at_utc': t.stamp(), 'modes': modes,
        'new_root_attributes_allowed': False, 'commit_created': False})
    check_modes(modes, pre['prefix'], expected)
    t.write_new(work / 'upload-started.json', {'observed_at_utc': t.stamp(), 'preflight': pre,
        'original_root': before, 'original_corrected_subtree': old_sub})
    commit = api.create_commit(REPO, repo_type='dataset', parent_commit=PARENT, operations=ops,
        token=True, num_threads=2, commit_message='Preserve original corrected DenseOn training completion evidence (data only)')
    require(re.fullmatch('[0-9a-f]{40}', commit.oid) is not None, 'Missing immutable commit')
    result = {'uploaded_at_utc': t.stamp(), 'repo_id': REPO, 'revision': commit.oid,
              'prefix': pre['prefix'], 'manifest': pre['manifest'], 'files': 41,
              'scientific_completion': False, 'durability_verified': False}
    t.write_new(work / 'upload.json', result)
    actual = {}
    for item in api.list_repo_tree(REPO, repo_type='dataset', revision=commit.oid,
                                   path_in_repo=pre['prefix'], recursive=True, token=False):
        if type(item).__name__ == 'RepoFile':
            actual[item.path.removeprefix(pre['prefix'] + '/')] = {
                'bytes': item.size, 'kind': 'sha256' if item.lfs else 'git_blob_sha1',
                'digest': item.lfs.sha256 if item.lfs else item.blob_id}
    t.compare_remote(expected, actual)
    check_preservation(before, t.root_inventory(api, commit.oid), old_sub,
                       t.root_inventory(api, commit.oid, NAMESPACE))
    prepared(t, work)
    audit = {**result, 'verified_at_utc': t.stamp(), 'durability_verified': True,
             'remote_inventory': actual, 'old_root_entries_unchanged': len(before) - 1,
             'old_corrected_subtrees_unchanged': 10, 'root_attributes_unchanged': True,
             'remote_paths_deleted_or_overwritten': False, 'source_code_uploaded': False}
    t.write_new(work / 'remote-audit.json', audit)
    return audit


def recover(t, work):
    from huggingface_hub import hf_hub_download
    pre, expected = prepared(t, work)
    audit = t.read(work / 'remote-audit.json')
    require(audit['durability_verified'] is True and audit['prefix'] == pre['prefix'], 'No accepted remote audit')
    dest = work / 'download'
    require(not dest.exists(), 'Preserve previous recovery attempt')
    dest.mkdir()
    for name, ident in expected.items():
        filename = pre['prefix'] + '/' + name
        path = Path(hf_hub_download(REPO, repo_type='dataset', revision=audit['revision'],
                                   filename=filename, local_dir=dest, token=False))
        require(path == dest / filename, 'Unexpected download path')
        reader.bound(path, ident)
    root = dest / pre['prefix']
    checked = reader.verify(root, pre['manifest']['sha256'])
    result = {'completed_at_utc': t.stamp(), 'revision': audit['revision'], 'prefix': pre['prefix'],
              'downloaded_root': str(root), 'files': 41, 'bytes': pre['bytes'],
              'manifest': pre['manifest'], 'all_payload_hashes_match': True,
              'anonymous_download': True, 'same_physical_host': True, 'readback': checked}
    t.write_new(work / 'download-verified.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check', 'upload', 'recover'))
    parser.add_argument('--workdir', type=Path, required=True)
    args = parser.parse_args()
    require(args.workdir == Path(__file__).parent and args.workdir.parent == Path('/tmp')
            and args.workdir.name.startswith('dense-v3-campaign-evidence-backup.')
            and not args.workdir.is_symlink(), 'Wrong exact task workdir')
    t = transport()
    result = prepared(t, args.workdir)[0] if args.action == 'check' else {
        'prepare': prepare, 'upload': upload, 'recover': recover}[args.action](t, args.workdir)
    print(json.dumps(result, sort_keys=True), flush=True)
