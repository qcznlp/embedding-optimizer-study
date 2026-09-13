"""Additions-only durability of genuine vectors/features/calibration and results.

No GPU/process access, numerical recomputation, source-code upload or old-path
change. Reuses unchanged transport hashing, text screening and remote comparison.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
STORY = Path('/root/embedding-optimizer-story-refactor')
EXP = Path('/root/embedding-optimizer-v3-experiment')
REPO = 'qcz/embedding-optimizer-study-analysis-artifacts'
PARENT = '42689be00e1644aaf24721ecc239ec1c4d425a2a'
NAMESPACE = 'corrected-dense-correctness-v3'
ADDITION = NAMESPACE + '/functional-and-calibration-v1'
TRANSPORT = STORY / 'reports/engineering-archive/dense-v3-evaluation-artifact-backup-v1/source/backup.py'
TRANSPORT_SHA = 'ab565206b22d0138c8227fef3f26eb28e2abeb7653c6f7137b7b532a289ec046'
HELPER_SHA = 'bebdfd3d773e550665660ec12fe7451f70e7e5fd419b1ef21814f774762ef33e'
VECTOR_SHA = '9bf90d652d54ee6b9b571bf9dc88cd919da8c78bb70db0d476e307733e8c23e7'
FEATURE_SHA = 'a5ef6021dc9ab535c5888e753f0a1397406a614f229894db28bd6c802f7353f5'
INFERENCE_SHA = '608965f5297fc196adcce6ca7417bd40e7f35d05e26133f20c0811dd1fd7027e'
SENSITIVITY_SHA = '8cb9b93da12a245e214429ffcf61f6922803c72c9c9ea496e572d24ecdc3e231'
CALIBRATION_SHA = '2e8233e256f3b8b5d9e20302ad62520c93780a3a831fe796e744d81fdd57c262'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Nonordinary source')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def transport():
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU/network-only operation')
    need(sha(TRANSPORT) == TRANSPORT_SHA and sha(STORY / 'scripts/restore_primary_v3.py') == HELPER_SHA,
         'Original transport source changed')
    sys.path.insert(0, str(STORY))
    spec = importlib.util.spec_from_file_location('_unchanged_analysis_transport', TRANSPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select(t):
    sources = {}

    def add(name, path, expected=None):
        t.safe_name(name)
        need(name not in sources and path.suffix in {'.json', '.csv', '.md', '.npz', '.safetensors', '.pdf', '.png'},
             'Duplicate or out-of-scope artifact')
        binding = t.compare_file(path, expected or {})
        if path.suffix not in {'.npz', '.safetensors', '.pdf', '.png'}:
            t.scan_text(path)
        sources[name] = dict(source=str(path), **binding)
        return json.loads(path.read_text()) if path.suffix == '.json' else None

    vr = EXP / 'analyses/dense-primary-v3-functional-dimensions-recovery-v1/vectors'
    fr = EXP / 'analyses/dense-primary-v3-functional-features-recovery-v3'
    vm = add('vectors/manifest.json', vr / 'manifest.json', {'sha256': VECTOR_SHA})
    fm = add('features/manifest.json', fr / 'manifest.json', {'sha256': FEATURE_SHA})
    need(vm['status'] == fm['status'] == 'complete' and len(vm['states']) == 61
         and set(vm['states']) == set(fm['states']), 'Incomplete or unmatched actual states')
    add('vectors/admission.json', vr / 'admission.json', {k: vm['admission'][k] for k in ('bytes', 'sha256')})
    add('features/admission.json', fr / 'admission.json',
        {'sha256': 'b7ce7d678bb88b1a124bfb0461e160bb09644a62b616720cf7b08772a7d2c83e'})
    for cell in sorted(vm['states']):
        for name, root, manifest in (('vectors', vr, vm), ('features', fr, fm)):
            item = manifest['states'][cell]
            path = root / t.safe_name(item['path'])
            value = add(name + '/' + item['path'], path, {k: item[k] for k in ('bytes', 'sha256')})
            need(value['status'] == 'complete', 'Incomplete state payload')
            files = [value['output']] if name == 'vectors' else list(value['outputs'].values())
            for output in files:
                payload = path.parent / t.safe_name(output['path'])
                add(name + '/' + payload.relative_to(root).as_posix(), payload,
                    {k: output[k] for k in ('bytes', 'sha256')})
    for item in fm['tables'].values():
        add('features/' + item['path'], fr / t.safe_name(item['path']), {k: item[k] for k in ('bytes', 'sha256')})
    for family, digest in (('functional-inference', INFERENCE_SHA), ('functional-sensitivity', SENSITIVITY_SHA)):
        root = STORY / ('reports/dense-v3-' + family + '-v1')
        receipt = add('results/' + family + '/readout.json', root / 'actual/readout.json', {'sha256': digest})
        for name, item in receipt.get('payloads', receipt.get('outputs', {})).items():
            if Path(name).suffix == '.tex':
                continue  # Generated TeX remains local; no authoring source publication.
            add('results/' + family + '/' + name, root / 'actual' / t.safe_name(name), item)
        independent = ('03f9670ab2a8a2334db439c95f1ef6264485cb816addd48457df305f64b45e23'
                       if family == 'functional-inference' else
                       '0b7f2a1b18405de7ab798d46edc6175f08d0e7ecb2b27e71a7c0458992cc246e')
        add('results/' + family + '/independent_verification.json', root / 'actual/independent_verification.json',
            {'sha256': independent})
    add('provenance/features.completed.json', EXP / 'launch/functional-features-recovery-v3/run/features.completed.json',
        {'sha256': '29ad21b3cb6b7b90283b6da2f162472d9ac402298debcd6a17fef4e6a6750304'})
    cr = EXP / 'launch/factorial-calibration-v2/run'
    completion = add('provenance/calibration-completed.json', cr / 'completed.json', {'sha256': CALIBRATION_SHA})
    need(completion['actual_gpu_worker_exits'] == {'adamw_state': 0, 'muon_state': 0}
         and completion['fresh_process_native_readbacks'] == 2, 'Wrong actual calibration completion')
    for state, item in completion['calibrations'].items():
        root = EXP / 'analyses/dense-primary-v3-factorial-calibration-v2' / state
        need(str(root) == item['path'], 'Wrong original calibration root')
        prefix = 'calibration/' + state + '/'
        calibration = add(prefix + 'directions/calibration.json', root / 'directions/calibration.json', item['calibration_binding'])
        need(calibration['status'] == 'complete', 'Incomplete calibration')
        request = add(prefix + 'request.json', root / 'request.json')
        need(request == calibration['request'], 'Original calibration request differs')
        add(prefix + 'directions/rows.json', root / 'directions/rows.json', calibration['rows'])
        gradient = add(prefix + 'gradient-receipt.json', root / 'gradient-receipt.json', calibration['gradient_receipt'])
        manifest = add(prefix + 'gradients/manifest.json', root / 'gradients/manifest.json', gradient['manifest'])
        need(manifest['status'] == 'complete' and len(manifest['gradient_shards']) == 8
             and {r['step_index'] for r in manifest['gradient_shards']} == set(range(8)), 'Incomplete gradient history')
        for shard in manifest['gradient_shards']:
            add(prefix + 'gradients/' + shard['path'], root / 'gradients' / t.safe_name(shard['path']),
                {k: shard[k] for k in ('bytes', 'sha256')})
        for suffix in ('exited.json', 'verified.json', 'verify.exited.json'):
            proof = add('provenance/calibration/' + state + '.' + suffix, cr / (state + '.' + suffix))
            if suffix == 'verified.json':
                need(proof['calibration_binding'] == item['calibration_binding'] and proof['fresh_process_native_readback'] is True,
                     'Fresh calibration readback differs')
    add('README.md', HERE / 'ARTIFACT_README.md')
    need(sum(n.endswith('.npz') for n in sources) == 122 and sum(n.endswith('.safetensors') for n in sources) == 16,
         'Missing binary analysis payload')
    return sources


def prepare(t):
    need(not (HERE / 'preflight.json').exists(), 'Preserve previous preparation')
    sources = select(t)
    manifest = dict(scope='actual-dense-v3-functional-and-calibration-durability-v1',
        created_at_utc=t.stamp(), files={n: {k: r[k] for k in ('bytes', 'sha256', 'git_blob_sha1')}
                                     for n, r in sources.items()},
        vector_states=61, attribution_states=61, calibration_states=2, gradient_shards=16,
        functional_primary_contrasts=9, functional_predictions=240, exploratory_predictions=1440,
        source_code_included=False, raw_text_examples_included=False, scientific_completion=False)
    t.write_new(HERE / 'artifact_manifest.json', manifest)
    mid = t.file_identity(HERE / 'artifact_manifest.json')
    sources['artifact_manifest.json'] = dict(source=str(HERE / 'artifact_manifest.json'), **mid)
    result = dict(scope=manifest['scope'], repo_id=REPO, repo_type='dataset', parent_commit=PARENT,
        prefix=ADDITION + '/' + mid['sha256'], artifact_manifest=mid, sources=sources,
        files=len(sources), bytes=sum(r['bytes'] for r in sources.values()),
        source=t.file_identity(Path(__file__)), transport_sha256=TRANSPORT_SHA,
        source_code_included=False, remote_mutations=False, scientific_completion=False)
    t.write_new(HERE / 'preflight.json', result)
    return result


def prepared(t, digest):
    preflight = t.read(HERE / 'preflight.json', digest)
    need(preflight['source'] == t.file_identity(Path(__file__)) and preflight['repo_id'] == REPO
         and preflight['parent_commit'] == PARENT and preflight['prefix'] == ADDITION + '/' + preflight['artifact_manifest']['sha256'],
         'Prepared source or remote target differs')
    for name, record in preflight['sources'].items():
        t.safe_name(name)
        t.compare_file(Path(record['source']), {k: record[k] for k in ('bytes', 'sha256', 'git_blob_sha1')})
    return preflight


def upload(t, digest):
    need(not (HERE / 'upload-started.json').exists(), 'Preserve upload attempt; inspect before any retry')
    preflight = prepared(t, digest)
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    need(api.whoami()['name'] == 'qcz', 'Unexpected HF account')
    info = api.repo_info(REPO, repo_type='dataset', token=False)
    need(info.sha == PARENT and not info.private, 'Remote parent or visibility changed')
    before, before_sub = t.root_inventory(api, PARENT), t.root_inventory(api, PARENT, NAMESPACE)
    need(ADDITION not in before_sub, 'Existing addition is protected')
    operations = [CommitOperationAdd(path_in_repo=preflight['prefix'] + '/' + name, path_or_fileobj=r['source'])
                  for name, r in sorted(preflight['sources'].items())]
    t.write_new(HERE / 'upload-started.json', dict(started_at_utc=t.stamp(), preflight_sha256=digest,
        parent_commit=PARENT, prefix=preflight['prefix'], files=len(operations), before=before, before_subtree=before_sub))
    print(json.dumps(dict(event='upload_started', files=len(operations), bytes=preflight['bytes'])), flush=True)
    commit = api.create_commit(REPO, repo_type='dataset', parent_commit=PARENT, operations=operations,
        num_threads=2, commit_message='Back up genuine DenseOn functional measurements and crossed calibration')
    need(re.fullmatch('[0-9a-f]{40}', commit.oid or '') is not None, 'No immutable remote revision')
    uploaded = dict(repo_id=REPO, repo_type='dataset', revision=commit.oid, prefix=preflight['prefix'],
                    uploaded_at_utc=t.stamp(), files=len(operations), bytes=preflight['bytes'])
    t.write_new(HERE / 'upload-returned.json', uploaded)
    after, after_sub = t.root_inventory(api, commit.oid), t.root_inventory(api, commit.oid, NAMESPACE)
    need(set(after) == set(before) and {k: v for k, v in after.items() if k != NAMESPACE} ==
         {k: v for k, v in before.items() if k != NAMESPACE}, 'Old root entries changed')
    need(set(after_sub) == set(before_sub) | {ADDITION} and
         {k: after_sub[k] for k in before_sub} == before_sub, 'Old corrected subtrees changed')
    actual = {}
    for item in api.list_repo_tree(REPO, repo_type='dataset', revision=commit.oid,
                                  path_in_repo=preflight['prefix'], recursive=True, token=False):
        if type(item).__name__ == 'RepoFile':
            actual[item.path.removeprefix(preflight['prefix'] + '/')] = dict(bytes=item.size,
                kind='sha256' if item.lfs else 'git_blob_sha1', digest=item.lfs.sha256 if item.lfs else item.blob_id)
    t.compare_remote(preflight['sources'], actual)
    prepared(t, digest)
    result = dict(**uploaded, verified_at_utc=t.stamp(), remote_inventory=actual,
        all_remote_files_verified=True, old_root_and_subtrees_preserved=True, source_code_uploaded=False,
        paths_deleted_or_overwritten=False, all_payloads_downloaded_again=False, scientific_completion=False)
    t.write_new(HERE / 'remote-verified.json', result)
    return {k: v for k, v in result.items() if k != 'remote_inventory'}


def download(t, digest):
    preflight = prepared(t, digest)
    audit = t.read(HERE / 'remote-verified.json')
    need(audit['all_remote_files_verified'] and audit['prefix'] == preflight['prefix'], 'No verified immutable upload')
    destination = HERE / 'download'
    destination.mkdir(exist_ok=False)
    from huggingface_hub import hf_hub_download

    def transfer(item):
        name, record = item
        filename = preflight['prefix'] + '/' + name
        path = Path(hf_hub_download(REPO, filename, repo_type='dataset', revision=audit['revision'],
            token=False, local_dir=destination))
        need(path == destination / filename, 'Unexpected downloaded location')
        t.compare_file(path, {k: record[k] for k in ('bytes', 'sha256', 'git_blob_sha1')})

    with ThreadPoolExecutor(max_workers=2) as workers:
        list(workers.map(transfer, sorted(preflight['sources'].items())))
    result = dict(scope='all-actual-functional-and-calibration-payloads-anonymously-recovered',
        completed_at_utc=t.stamp(), revision=audit['revision'], prefix=audit['prefix'],
        files=preflight['files'], bytes=preflight['bytes'], download_root=str(destination / audit['prefix']),
        all_payload_hashes_match=True, source_code_included=False, same_physical_host=True,
        cross_host_native_admission=False, gpu_resume_equivalence=False, scientific_completion=False)
    t.write_new(HERE / 'download-verified.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'upload', 'download'))
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--preflight-sha256')
    a = p.parse_args()
    need(sha(Path(__file__)) == a.source_sha256, 'Entry source changed')
    t = transport()
    result = prepare(t) if a.action == 'prepare' else (
        upload(t, a.preflight_sha256) if a.action == 'upload' else download(t, a.preflight_sha256))
    print(json.dumps({k: v for k, v in result.items() if k != 'sources'}), flush=True)


if __name__ == '__main__':
    main()
