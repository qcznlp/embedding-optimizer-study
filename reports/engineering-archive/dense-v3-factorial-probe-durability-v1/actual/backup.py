"""Additions-only backup of all actual crossed-continuation probe payloads.

No GPU/process access, encoder/numerical rerun, executable source upload, old
remote path modification, or scientific-paper completion claim.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
STORY = Path('/root/embedding-optimizer-story-refactor')
EXP = Path('/root/embedding-optimizer-v3-experiment')
PROBE = Path('/tmp/dense-v3-factorial-probe.ryorjXJK')
OUTPUT = EXP / 'analyses/dense-v3-factorial-probe-v1'
REPO = 'qcz/embedding-optimizer-study-analysis-artifacts'
PARENT = '9b182d77b31c93277457a578c16389935bc3fdb9'
NAMESPACE = 'corrected-dense-correctness-v3'
ADDITION = NAMESPACE + '/continuation-probes-v1'
SCOPE = 'actual-dense-v3-complete-continuation-probe-durability-v1'
PROBE_SHA = 'eb4fc27738adaecd9614ba5822255a5dc37de242afabfa53ff8cd8a6c8c73334'
PROBE_AUTH = '00d6b42f922b4428352d5b9fc5e5fa7e0436a4aa20ee0c693e57b9f645398ded'
TRANSPORT = STORY / 'reports/engineering-archive/dense-v3-evaluation-artifact-backup-v1/source/backup.py'
TRANSPORT_SHA = 'ab565206b22d0138c8227fef3f26eb28e2abeb7653c6f7137b7b532a289ec046'
HELPER_SHA = 'bebdfd3d773e550665660ec12fe7451f70e7e5fd419b1ef21814f774762ef33e'
ARCHIVE = STORY / 'reports/engineering-archive/dense-v3-complete-document-handoff-v1'
ARCHIVE_SHA = '60dd7bb39021af1dc7790851eb18405766f759f9cd9a6e0113918ad3004e84c4'
CHECKPOINTS = STORY / 'reports/engineering-archive/dense-v3-current-publication-consumer-v1'
CHECKPOINT_HANDOFF_SHA = '7da373c74e69986fbc8269b64d3e4279b9ad9b3a803b2accbe0b6cab9da0e202'
STEPS = (79, 157, 235, 313, 391)


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Nonordinary input')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def transport():
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU/network-only backup required')
    need(sha(TRANSPORT) == TRANSPORT_SHA and sha(STORY / 'scripts/restore_primary_v3.py') == HELPER_SHA,
         'Original transport implementations changed')
    sys.path.insert(0, str(STORY))
    spec = importlib.util.spec_from_file_location('_unchanged_probe_artifact_transport', TRANSPORT)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def select(t):
    need(sha(PROBE / 'probe.py') == PROBE_SHA, 'Original probe source changed')
    archive = t.read(ARCHIVE / 'archive-manifest.json', ARCHIVE_SHA)
    checkpoint_handoff = t.read(CHECKPOINTS / 'final-handoff.json', CHECKPOINT_HANDOFF_SHA)
    sources = {}

    def add(name, path, expected=None):
        t.safe_name(name)
        need(name not in sources and path.suffix in {'.json', '.npz', '.md'},
             'Duplicate or out-of-scope data-only artifact')
        binding = t.compare_file(path, expected or {})
        if path.suffix != '.npz':
            t.scan_text(path)
        sources[name] = {'source': str(path), **binding}
        return t.read(path) if path.suffix == '.json' else None

    auth = add('provenance/probe-authorization.json', PROBE / 'authorization.json', {'sha256': PROBE_AUTH})
    need(auth['source']['sha256'] == PROBE_SHA and auth['steps'] == list(STEPS)
         and auth['output_root'] == str(OUTPUT) and auth['new_checkpoint_encodings'] == 60
         and auth['reference_reused'] is True, 'Wrong completed probe scope')
    add('provenance/actual-inputs.json', PROBE / 'actual-inputs.json', auth['actual_inputs'])
    all_states, all_runs = set(), set()
    for pool in ('a', 'b'):
        run_root = PROBE / 'run' / ('pool-' + pool)
        for name in ('started.json', 'completed.json'):
            key = 'complete-probes/pool-' + pool + '/' + name
            expected = archive['files'][key]
            value = add('provenance/pool-' + pool + '/' + name, run_root / name,
                        {k: expected[k] for k in ('bytes', 'sha256')})
            if name == 'completed.json':
                complete = value
        need(not (run_root / 'failed.json').exists() and complete['source_sha256'] == PROBE_SHA
             and complete['authorization_sha256'] == PROBE_AUTH, 'Incomplete genuine probe pool')
        expected_states = [(run, step) for run in auth['queues'][pool] for step in STEPS]
        need([(row['run_id'], row['step']) for row in complete['checkpoints']] == expected_states,
             'Missing or reordered probe state')
        for row in complete['checkpoints']:
            run, step = row['run_id'], row['step']
            need((run, step) not in all_states, 'Duplicate probe checkpoint')
            all_states.add((run, step))
            all_runs.add(run)
            prefix = run_root / f'{run}-checkpoint-{step}'
            evidence = {}
            for suffix in ('.request.json', '.started.json', '.exited.json', '.verified.json'):
                path = prefix.with_suffix(suffix)
                expected = archive['files']['complete-probes/pool-' + pool + '/' + path.name]
                evidence[suffix] = add('provenance/pool-' + pool + '/' + path.name, path,
                                      {k: expected[k] for k in ('bytes', 'sha256')})
            request, verified = evidence['.request.json'], evidence['.verified.json']
            need(evidence['.exited.json']['exit_code'] == 0 and request['run_id'] == verified['run_id'] == run
                 and request['step'] == verified['step'] == step
                 and request['source_sha256'] == PROBE_SHA and request['authorization_sha256'] == PROBE_AUTH
                 and verified['all_four_arrays_and_six_metrics_reconstructed'] is True
                 and verified['samples'] == 224 and verified['tasks'] == 14,
                 'Missing actual original zero exit or full native probe readback')
            t.compare_file(prefix.with_suffix('.verified.json'), row['verified'])
            root = OUTPUT / run / f'checkpoint-{step}'
            need(str(root) == request['output'], 'Original output path differs')
            name = run + f'/checkpoint-{step}/'
            worker = add(name + 'worker.completed.json', root / 'worker.completed.json', verified['worker_completion'])
            need(worker['source']['sha256'] == PROBE_SHA and worker['authorization_sha256'] == PROBE_AUTH
                 and worker['request'] == {k: sources['provenance/pool-' + pool + '/' + prefix.with_suffix('.request.json').name][k]
                                          for k in ('bytes', 'sha256')}
                 and worker['run_id'] == run and worker['step'] == step
                 and worker['original_encoder_and_scorer_unchanged'] is True,
                 'Worker payload does not match the admitted actual request')
            native = verified['native_raw_readback']
            raw = add(name + 'raw/manifest.json', root / 'raw/manifest.json',
                      {k: native['manifest'][k] for k in ('bytes', 'sha256')})
            need(raw['status'] == 'complete' and raw['plan'] == worker['raw_plan']
                 and raw['output'] == native['output'] and raw['output']['path'] == 'vectors.npz'
                 and raw['array_metadata'] == native['array_metadata'], 'Raw vector native binding differs')
            add(name + 'raw/vectors.npz', root / 'raw/vectors.npz',
                {k: raw['output'][k] for k in ('bytes', 'sha256')})
            need(set(worker['outputs']) == {'vectors-fp16.npz', 'scores.npz', 'metrics.json'},
                 'Incomplete saved metric-input population')
            for filename, binding in worker['outputs'].items():
                add(name + filename, root / filename, binding)
    need(len(all_states) == 60 and len(all_runs) == 12, 'Incomplete actual continuation probe grid')
    for name, binding in auth['reference_files'].items():
        add('pretrained/' + name, OUTPUT / 'pretrained' / name, binding)
    origin = t.read(OUTPUT / 'pretrained/origin.json')
    reference_root = EXP / 'analyses/dense-primary-v3-functional-dimensions-recovery-v1/vectors/states/pretrained'
    add('pretrained/raw/manifest.json', reference_root / 'manifest.json', origin['manifest'])
    add('pretrained/raw/vectors.npz', reference_root / 'vectors.npz', origin['original_vectors'])
    # Exact immutable model revisions and complete inventories, not model reuploads.
    for run in sorted(all_runs):
        for name in ('artifact_manifest.json', 'verified.json'):
            key = 'final-complete-backup/' + run + '/' + name
            value = checkpoint_handoff['backup_copies'][key]
            add('checkpoints/' + run + '/' + name, CHECKPOINTS / key,
                {k: value[k] for k in ('bytes', 'sha256')})
    add('README.md', HERE / 'ARTIFACT_README.md')
    need(sum(name.endswith('.npz') for name in sources) == 183, 'Missing raw/metric/scoring arrays')
    return sources


def prepare(t):
    need(not (HERE / 'preflight.json').exists(), 'Preserve the previous preparation')
    sources = select(t)
    manifest = {'scope': SCOPE, 'created_at_utc': t.stamp(),
        'files': {name: {k: item[k] for k in ('bytes', 'sha256', 'git_blob_sha1')}
                  for name, item in sources.items()},
        'continuation_runs': 12, 'checkpoint_probes': 60, 'reference_states': 1,
        'stage_steps': list(STEPS), 'queries_per_probe': 224, 'tasks': 14, 'candidates': 8, 'dimensions': 768,
        'raw_embeddings': 'normalized float32', 'scoring_embeddings': 'float16', 'saved_scores': 'float32',
        'all_original_native_probe_reads_reused': True, 'new_model_encoding_or_metric_computation': False,
        'source_code_included': False, 'raw_example_text_included': False, 'scientific_completion': False}
    t.write_new(HERE / 'artifact_manifest.json', manifest)
    mid = t.file_identity(HERE / 'artifact_manifest.json')
    sources['artifact_manifest.json'] = {'source': str(HERE / 'artifact_manifest.json'), **mid}
    result = {'scope': SCOPE, 'repo_id': REPO, 'repo_type': 'dataset', 'parent_commit': PARENT,
        'prefix': ADDITION + '/' + mid['sha256'], 'artifact_manifest': mid, 'sources': sources,
        'files': len(sources), 'bytes': sum(item['bytes'] for item in sources.values()),
        'source': t.file_identity(Path(__file__)), 'transport_sha256': TRANSPORT_SHA,
        'remote_mutations': False, 'scientific_completion': False}
    t.write_new(HERE / 'preflight.json', result)
    return {key: value for key, value in result.items() if key != 'sources'}


def prepared(t, digest):
    value = t.read(HERE / 'preflight.json', digest)
    need(value['scope'] == SCOPE and value['source'] == t.file_identity(Path(__file__))
         and value['repo_id'] == REPO and value['repo_type'] == 'dataset' and value['parent_commit'] == PARENT
         and value['prefix'] == ADDITION + '/' + value['artifact_manifest']['sha256'],
         'Prepared source or remote target differs')
    for name, record in value['sources'].items():
        t.safe_name(name)
        t.compare_file(Path(record['source']), {k: record[k] for k in ('bytes', 'sha256', 'git_blob_sha1')})
    return value


def upload(t, digest):
    need(not (HERE / 'upload-started.json').exists(), 'Preserve upload attempt; no automatic retry')
    value = prepared(t, digest)
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    need(api.whoami()['name'] == 'qcz', 'Unexpected HF account')
    info = api.repo_info(REPO, repo_type='dataset', token=False)
    need(info.sha == PARENT and not info.private, 'Public remote parent changed')
    before, before_sub = t.root_inventory(api, PARENT), t.root_inventory(api, PARENT, NAMESPACE)
    need(ADDITION not in before_sub, 'Existing subtree is protected')
    operations = [CommitOperationAdd(path_in_repo=value['prefix'] + '/' + name, path_or_fileobj=r['source'])
                  for name, r in sorted(value['sources'].items())]
    t.write_new(HERE / 'upload-started.json', {'started_at_utc': t.stamp(), 'preflight_sha256': digest,
        'parent_commit': PARENT, 'prefix': value['prefix'], 'files': len(operations),
        'before': before, 'before_subtree': before_sub})
    print(json.dumps({'event': 'upload_started', 'files': len(operations), 'bytes': value['bytes']}), flush=True)
    commit = api.create_commit(REPO, repo_type='dataset', parent_commit=PARENT, operations=operations,
        num_threads=2, commit_message='Preserve all sixty genuine DenseOn continuation probes and checkpoint links')
    need(re.fullmatch('[0-9a-f]{40}', commit.oid or '') is not None, 'Missing immutable revision')
    returned = {'repo_id': REPO, 'repo_type': 'dataset', 'revision': commit.oid, 'prefix': value['prefix'],
        'uploaded_at_utc': t.stamp(), 'files': len(operations), 'bytes': value['bytes']}
    t.write_new(HERE / 'upload-returned.json', returned)
    after, after_sub = t.root_inventory(api, commit.oid), t.root_inventory(api, commit.oid, NAMESPACE)
    need(set(after) == set(before) and {k: v for k, v in after.items() if k != NAMESPACE} ==
         {k: v for k, v in before.items() if k != NAMESPACE}, 'Old root entries changed')
    need(set(after_sub) == set(before_sub) | {ADDITION}
         and {k: after_sub[k] for k in before_sub} == before_sub, 'Old corrected subtrees changed')
    inventory = {}
    for item in api.list_repo_tree(REPO, repo_type='dataset', revision=commit.oid,
                                  path_in_repo=value['prefix'], recursive=True, token=False):
        if type(item).__name__ == 'RepoFile':
            inventory[item.path.removeprefix(value['prefix'] + '/')] = {'bytes': item.size,
                'kind': 'sha256' if item.lfs else 'git_blob_sha1',
                'digest': item.lfs.sha256 if item.lfs else item.blob_id}
    t.compare_remote(value['sources'], inventory)
    prepared(t, digest)
    result = {**returned, 'verified_at_utc': t.stamp(), 'remote_inventory': inventory,
        'all_remote_files_verified': True, 'old_root_and_subtrees_preserved': True,
        'source_code_uploaded': False, 'paths_deleted_or_overwritten': False,
        'all_payloads_downloaded_again': False, 'scientific_completion': False}
    t.write_new(HERE / 'remote-verified.json', result)
    return {k: v for k, v in result.items() if k != 'remote_inventory'}


def download(t, digest):
    value = prepared(t, digest)
    remote = t.read(HERE / 'remote-verified.json')
    need(remote['all_remote_files_verified'] is True and remote['prefix'] == value['prefix'],
         'Missing verified immutable upload')
    destination = HERE / 'download'
    destination.mkdir(exist_ok=False)
    from huggingface_hub import hf_hub_download
    def transfer(item):
        name, record = item
        filename = value['prefix'] + '/' + name
        path = Path(hf_hub_download(REPO, filename, repo_type='dataset', revision=remote['revision'],
                                   token=False, local_dir=destination))
        need(path == destination / filename, 'Unexpected downloaded location')
        t.compare_file(path, {k: record[k] for k in ('bytes', 'sha256', 'git_blob_sha1')})
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(transfer, sorted(value['sources'].items())))
    result = {'scope': 'all-actual-continuation-probe-payloads-anonymously-recovered',
        'completed_at_utc': t.stamp(), 'revision': remote['revision'], 'prefix': remote['prefix'],
        'files': value['files'], 'bytes': value['bytes'], 'download_root': str(destination / remote['prefix']),
        'all_payload_hashes_match': True, 'source_code_included': False, 'same_physical_host': True,
        'cross_host_native_admission': False, 'gpu_resume_equivalence': False, 'scientific_completion': False}
    t.write_new(HERE / 'download-verified.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'upload', 'download'))
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--preflight-sha256')
    args = parser.parse_args()
    need(sha(Path(__file__)) == args.source_sha256, 'Backup entry source changed')
    t = transport()
    result = prepare(t) if args.action == 'prepare' else (
        upload(t, args.preflight_sha256) if args.action == 'upload' else download(t, args.preflight_sha256))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
