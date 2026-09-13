"""Wait for original complete native results, then add/verify a data-only HF snapshot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
STORY = Path('/root/embedding-optimizer-story-refactor')
EXP = Path('/root/embedding-optimizer-v3-experiment')
SUMMARY = Path('/tmp/dense-v3-factorial-summary.BQ08HjeP')
EVAL = Path('/tmp/dense-v3-factorial-evaluation.IOV7MK93')
PROBE = Path('/tmp/dense-v3-factorial-probe.ryorjXJK')
TRAIN = EXP / 'launch/factorial-training-v1'
RESULTS = EXP / 'evaluations/dense-v3-state-operator-v1'
TABLES = EXP / 'analyses/dense-v3-factorial-inference-v1'
REPO = 'qcz/embedding-optimizer-study-analysis-artifacts'
PARENT = 'cff3f190e169548931fbd33eadcf1279439798e1'
NAMESPACE = 'corrected-dense-correctness-v3'
ADDITION = NAMESPACE + '/complete-continuation-outcomes-v1'
SCOPE = 'complete-native-factorial-outcome-durability-v1'
SUMMARY_SHA = '4be265ce12cc649a595afaec9e5ef8b68a50eaa347ccb583d4d32ba3e100c5a1'
SUMMARY_AUTH = '47485b8a16af1b0be419e67c4cf45ede08cdacbd01059e42357e16c97b9eb92c'
COLLECT_SHA = '988c71fd9ad24576a017864ce9b99eeeb0ed6e5d48327483607b185fe4175366'
TRAIN_AUTH = '00c52f0bec979fb163a734ad0f9eadcd6ea09f7393133fd35c65bc62878ce11f'
TRANSPORT = STORY / 'reports/engineering-archive/dense-v3-evaluation-artifact-backup-v1/source/backup.py'
COUNTS = {'beir_seed_task_scores': 168, 'factorial_cell_summary': 4,
          'estimand_seed_task_contrasts': 126, 'estimand_summary': 3,
          'probe_checkpoint_metrics': 60, 'probe_task_metrics': 840}
PINS = {
    SUMMARY / 'summarize.py': SUMMARY_SHA, SUMMARY / 'authorization.json': SUMMARY_AUTH,
    SUMMARY / 'collect.py': COLLECT_SHA, TRAIN / 'authorization.json': TRAIN_AUTH,
    EVAL / 'evaluate.py': '45577537257e1dff2b7c3a26ee12f7564fbb476daa256e7b0f09300305b68cba',
    EVAL / 'authorization.json': '3a1ae32fdde63abf386603208e413ea11bf0c169eb1923f9dd7e1da529ffa29c',
    PROBE / 'probe.py': 'eb4fc27738adaecd9614ba5822255a5dc37de242afabfa53ff8cd8a6c8c73334',
    PROBE / 'authorization.json': '00d6b42f922b4428352d5b9fc5e5fa7e0436a4aa20ee0c693e57b9f645398ded',
    STORY / 'src/embed_optim/state_operator_factorial_summary.py': 'd6d4d8481404e38b0e7ab9291680c39a99909fc3fb87959bc42be49f4361cc27',
    STORY / 'configs/dense_no_packing_state_operator_factorial_protocol.json': '5773943a3ae9b581021a0f7b85b162c74d5c497eeca1386578ff9d2c3bcafe76',
    STORY / 'configs/dense_no_packing_state_operator_claim_wording_amendment.json': '929fe6445598b84b546670eb5eefd408c5221cacad6cc8e93a984651dbbe5de7',
    TRANSPORT: 'ab565206b22d0138c8227fef3f26eb28e2abeb7653c6f7137b7b532a289ec046',
    STORY / 'scripts/restore_primary_v3.py': 'bebdfd3d773e550665660ec12fe7451f70e7e5fd419b1ef21814f774762ef33e',
}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Ordinary file required')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transport():
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA')
    for path, expected in PINS.items():
        need(sha(path) == expected, 'Original bound source changed')
    sys.path.insert(0, str(STORY))
    spec = importlib.util.spec_from_file_location('_unchanged_outcome_durability_transport', TRANSPORT)
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    return t


def process(pid, expected_start, required):
    root = Path('/proc') / str(pid)
    try:
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        argv = (root / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return {'terminal_or_missing': True}
    need(int(fields[19]) == expected_start and all(value in argv for value in required), 'Wrong exact owned handle')
    return {'pid': pid, 'start_ticks': expected_start, 'state': fields[0], 'command_matches': True}


def own_start():
    return int((Path('/proc') / str(os.getpid()) / 'stat').read_text().rsplit(')', 1)[1].split()[19])


def ready():
    need(not (SUMMARY / 'run/failed.json').exists(), 'Original summary failed; preserve outputs')
    if (SUMMARY / 'run/completed.json').exists():
        return True
    observation = process(813139, 320676124, [str(SUMMARY / 'summarize.py'), SUMMARY_SHA, SUMMARY_AUTH])
    need(not observation.get('terminal_or_missing'), 'Original summary vanished without completion')
    return False


def population(tables, training, tasks):
    need(set(tables) == set(COUNTS) and {k: len(v) for k, v in tables.items()} == COUNTS, 'Incomplete six-table population')
    expected = {(run, task) for runs in training['queues'].values() for run in runs for task in tasks}
    rows = tables['beir_seed_task_scores']
    keys = [(row['run_id'], row['task']) for row in rows]
    need(len(keys) == len(set(keys)) == 168 and set(keys) == expected, 'Wrong complete BEIR cells')
    for row in rows:
        req = training['requests'][row['run_id']]
        need(all(row[k] == req[k] for k in ('state', 'operator', 'seed')), 'Wrong source/operator/order identity')
        value = row['ndcg_at_10']
        need(type(value) in (int, float) and 0 <= value <= 1, 'Invalid normalized score')


def select(t):
    need(ready(), 'All actual outcomes are still pending')
    sources = {}
    def add(name, path, expected=None):
        t.safe_name(name)
        path = Path(path)
        need(path.suffix in {'.json', '.jsonl', '.csv', '.md'}, 'Only data/metadata text is in scope')
        bound = t.compare_file(path, expected or {})
        t.scan_text(path)
        value = {'source': str(path), **bound}
        need(name not in sources or sources[name] == value, 'Conflicting shared metadata')
        sources[name] = value
        return t.read(path) if path.suffix == '.json' else None
    completed = add('provenance/summary/completed.json', SUMMARY / 'run/completed.json')
    need(completed['source_sha256'] == SUMMARY_SHA and completed['authorization_sha256'] == SUMMARY_AUTH,
         'Wrong original complete summary')
    readout = add('tables/readout.json', TABLES / 'readout.json', completed['readout'])
    need(readout['source']['sha256'] == SUMMARY_SHA and readout['authorization_sha256'] == SUMMARY_AUTH
         and readout['actual_collector_exits'] == {'beir': 0, 'probe': 0}
         and readout['table_counts'] == COUNTS and readout['independent_arithmetic_verified'] is True,
         'Incomplete original native summary')
    collectors = {}
    for kind in ('beir', 'probe'):
        collectors[kind] = add('provenance/summary/' + kind + '.json', SUMMARY / 'run' / (kind + '.json'),
                               readout['actual_native_collectors'][kind])
        need(collectors[kind]['collector_source']['sha256'] == COLLECT_SHA, 'Wrong original native collector')
        for suffix in ('.started.json', '.exited.json'):
            value = add('provenance/summary/' + kind + suffix, SUMMARY / 'run' / (kind + suffix))
            if suffix == '.exited.json':
                need(value['exit_code'] == 0, 'Original collector exit is not zero')
    expected_outputs = {'tables.json', 'independent_verification.json'} | {k + '.csv' for k in COUNTS}
    need(set(readout['outputs']) == expected_outputs, 'Unexpected original numerical output inventory')
    for name, binding in readout['outputs'].items():
        add('tables/' + name, TABLES / name, binding)
    training = add('provenance/training-authorization.json', TRAIN / 'authorization.json', {'sha256': TRAIN_AUTH})
    ev = add('provenance/evaluation-authorization.json', EVAL / 'authorization.json', {'sha256': PINS[EVAL / 'authorization.json']})
    tables = t.read(TABLES / 'tables.json')
    population(tables, training, ev['tasks'])
    beir, probe = collectors['beir'], collectors['probe']
    need(beir['scope'] == 'actual-complete-genuine-v3-factorial-beir-readback'
         and beir['actual_exit_zero_workers'] == 168 and beir['fresh_original_task_reader'] is True
         and beir['rows'] == tables['beir_seed_task_scores'] and len(beir['sources']) == 168,
         'Missing genuine full BEIR collector population')
    need(probe['scope'] == 'actual-complete-genuine-v3-factorial-probe-readback'
         and probe['actual_exit_zero_workers'] == 60 and probe['raw_arrays_and_metrics_reconstructed'] is True
         and probe['overall_rows'] == tables['probe_checkpoint_metrics']
         and probe['task_rows'] == tables['probe_task_metrics'], 'Missing genuine complete probe collection')
    for pool in ('a', 'b'):
        for kind, root in (('beir', EVAL), ('probe', PROBE)):
            record = collectors[kind]['pools'][pool]
            value = add('provenance/' + kind + '/pool-' + pool + '/completed.json',
                        root / 'run' / ('pool-' + pool) / 'completed.json', record['binding'])
            need(value == record['value'], 'Original complete pool record differs')
        root = EVAL / 'run' / ('pool-' + pool)
        for run in training['queues'][pool]:
            whole = add('native/beir/' + run + '/all-fourteen-tasks-verified.json',
                        RESULTS / run / 'checkpoint-391/all-fourteen-tasks-verified.json',
                        beir['pools'][pool]['value']['runs'][run])
            need(whole['run_id'] == run and whole['task_count'] == 14 and whole['step'] == 391, 'Wrong complete checkpoint')
            add('provenance/beir/native/' + run + '.json', root / 'native' / (run + '.json'), whole['native_input'])
            add('provenance/beir/native/' + run + '.exited.json', root / 'native' / (run + '.exited.json'))
    for row in beir['sources']:
        run, task = row['run_id'], row['task']
        pool = next(p for p, runs in training['queues'].items() if run in runs)
        prefix = EVAL / 'run' / ('pool-' + pool) / 'jobs' / f'{run}-391-{task}'
        for suffix, field in (('.started.json', 'worker_started'), ('.exited.json', 'worker_exited')):
            value = add('provenance/beir/jobs/' + prefix.name + suffix, prefix.with_suffix(suffix), row[field])
            need(value['job'] == [run, 391, task], 'Wrong actual evaluation worker')
            if suffix == '.exited.json':
                need(value['exit_code'] == 0, 'Nonzero evaluation exit')
        for bound in row['native_result']['files']:
            path = Path(bound['path'])
            relative = path.relative_to(RESULTS / run / 'checkpoint-391').as_posix()
            add('native/beir/' + run + '/' + relative, path, {k: bound[k] for k in ('bytes', 'sha256')})
    need(sum(name.endswith('Decontaminated.json') for name in sources) == 168
         and sum(name.endswith('/run_settings.jsonl') for name in sources) == 12
         and sum(name.endswith('/model_meta.json') for name in sources) == 12, 'Incomplete raw/final-shared-result inventory')
    for label, path in (('summary-authorization', SUMMARY / 'authorization.json'),
                        ('probe-authorization', PROBE / 'authorization.json'),
                        ('scientific-protocol', STORY / 'configs/dense_no_packing_state_operator_factorial_protocol.json'),
                        ('claim-wording-amendment', STORY / 'configs/dense_no_packing_state_operator_claim_wording_amendment.json')):
        add('provenance/' + label + '.json', path, {'sha256': PINS[path]})
    add('README.md', HERE / 'ARTIFACT_README.md')
    add('related-artifacts.json', HERE / 'related-artifacts.json')
    return sources


def preserved(before, after, before_sub, after_sub):
    need(set(before) == set(after) and {k: v for k, v in before.items() if k != NAMESPACE} ==
         {k: v for k, v in after.items() if k != NAMESPACE}, 'Old remote root changed')
    need(ADDITION not in before_sub and set(after_sub) == set(before_sub) | {ADDITION}
         and {k: after_sub[k] for k in before_sub} == before_sub, 'Old remote corrected subtree changed')


def authenticate(t, source_sha, auth_sha):
    need(sha(__file__) == source_sha, 'New frozen entry changed')
    auth = t.read(HERE / 'authorization.json', auth_sha)
    need(auth['source'] == t.file_identity(Path(__file__)) and auth['scope'] == SCOPE
         and auth['parent_revision'] == PARENT and auth['repo_id'] == REPO
         and auth['addition'] == ADDITION and auth['gpu_access'] is False, 'Wrong transport authority')
    for filename, binding in auth['local_inputs'].items():
        t.compare_file(HERE / filename, binding)
    return auth


def transfer(t, auth):
    sources = select(t)
    manifest = {'schema_version': 1, 'scope': SCOPE, 'created_at_utc': t.stamp(),
                'files': {k: {n: v[n] for n in ('bytes', 'sha256', 'git_blob_sha1')} for k, v in sorted(sources.items())},
                'table_counts': COUNTS, 'source_code_included': False, 'raw_example_text_included': False,
                'model_payloads_included': False, 'manuscript_included': False, 'scientific_completion': False}
    t.write_new(HERE / 'artifact_manifest.json', manifest)
    mid = t.file_identity(HERE / 'artifact_manifest.json')
    sources['artifact_manifest.json'] = {'source': str(HERE / 'artifact_manifest.json'), **mid}
    prefix = ADDITION + '/' + mid['sha256']
    expected = {k: {n: v[n] for n in ('bytes', 'sha256', 'git_blob_sha1')} for k, v in sources.items()}
    t.write_new(HERE / 'preflight.json', {'source': auth['source'], 'prefix': prefix, 'parent_revision': PARENT,
        'sources': sources, 'files': len(sources), 'bytes': sum(v['bytes'] for v in sources.values()),
        'manifest': mid, 'remote_mutations': False, 'scientific_completion': False})
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download, _commit_api, hf_api
    import inspect
    api = HfApi(endpoint='https://huggingface.co', token=True)
    need(api.whoami()['name'] == 'qcz', 'Wrong configured HF owner')
    info = api.repo_info(REPO, repo_type='dataset', token=False)
    need(info.sha == PARENT and info.private is False, 'Original public parent changed; no automatic retry')
    before, before_sub = t.root_inventory(api, PARENT), t.root_inventory(api, PARENT, NAMESPACE)
    need(ADDITION not in before_sub, 'Existing remote subtree is protected')
    operations = [CommitOperationAdd(path_in_repo=prefix + '/' + name, path_or_fileobj=value['source'])
                  for name, value in sorted(sources.items())]
    need(sha(inspect.getsourcefile(_commit_api)) == '250ed0e5a5a39383974cab5baae08a964e60f3f40c394c0d13a52e119e1f5f39'
         and sha(inspect.getsourcefile(hf_api)) == '659636025aa3a7efefa69ca7f16741d8d6cc12f9301beac8c69f9b3d51f81cd4',
         'Upload-mode API client changed')
    _commit_api._fetch_upload_modes(additions=operations, repo_type='dataset', repo_id=REPO,
        headers=api._build_hf_headers(token=True), revision=PARENT, endpoint='https://huggingface.co', create_pr=False)
    modes = {o.path_in_repo: {'mode': o._upload_mode, 'ignored': o._should_ignore, 'remote_oid': o._remote_oid} for o in operations}
    t.write_new(HERE / 'upload-mode-preflight.json', {'files': modes, 'commit_created': False, 'lfs_payload_uploaded': False})
    need(all(v == {'mode': 'regular', 'ignored': False, 'remote_oid': None} for v in modes.values()), 'Unexpected upload mode')
    for name, record in sources.items():
        t.compare_file(Path(record['source']), expected[name])
    t.write_new(HERE / 'upload-started.json', {'started_at_utc': t.stamp(), 'prefix': prefix,
        'parent_revision': PARENT, 'files': len(sources), 'before': before, 'before_subtree': before_sub})
    commit = api.create_commit(REPO, repo_type='dataset', parent_commit=PARENT, operations=operations,
        num_threads=2, commit_message='Preserve complete native DenseOn continuation outcomes and inference (data only)')
    need(re.fullmatch('[0-9a-f]{40}', commit.oid or '') is not None, 'Missing immutable revision')
    t.write_new(HERE / 'upload-returned.json', {'revision': commit.oid, 'prefix': prefix, 'manifest': mid,
                                              'uploaded_at_utc': t.stamp(), 'files': len(sources)})
    actual = {}
    for item in api.list_repo_tree(REPO, repo_type='dataset', revision=commit.oid, path_in_repo=prefix, recursive=True, token=False):
        if type(item).__name__ == 'RepoFile':
            actual[item.path.removeprefix(prefix + '/')] = {'bytes': item.size,
                'kind': 'sha256' if item.lfs else 'git_blob_sha1', 'digest': item.lfs.sha256 if item.lfs else item.blob_id}
    t.compare_remote(expected, actual)
    preserved(before, t.root_inventory(api, commit.oid), before_sub, t.root_inventory(api, commit.oid, NAMESPACE))
    t.write_new(HERE / 'remote-verified.json', {'revision': commit.oid, 'prefix': prefix, 'inventory': actual,
        'verified_at_utc': t.stamp(), 'old_root_and_subtrees_unchanged': True, 'paths_deleted_or_overwritten': False})
    destination = HERE / 'download'
    destination.mkdir(exist_ok=False)
    def download(item):
        name, record = item
        filename = prefix + '/' + name
        path = Path(hf_hub_download(REPO, filename, repo_type='dataset', revision=commit.oid, token=False, local_dir=destination))
        need(path == destination / filename, 'Wrong downloaded path')
        t.compare_file(path, record)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, sorted(expected.items())))
    root = destination / prefix
    need({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} == set(expected), 'Recovered population differs')
    for name, record in sources.items():
        t.compare_file(Path(record['source']), expected[name])
    result = {'scope': SCOPE, 'completed_at_utc': t.stamp(), 'repo_id': REPO, 'repo_type': 'dataset',
        'revision': commit.oid, 'prefix': prefix, 'manifest': mid, 'files': len(sources),
        'bytes': sum(v['bytes'] for v in sources.values()), 'download_root': str(root),
        'all_payloads_anonymously_downloaded_and_verified': True, 'old_remote_subtrees_preserved': True,
        'source_code_uploaded': False, 'physical_second_host_proof': False, 'scientific_completion': False}
    t.write_new(HERE / 'download-verified.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'coordinate'))
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--authorization-sha256')
    p.add_argument('--tests-sha256')
    args = p.parse_args()
    t = transport()
    need(sha(__file__) == args.source_sha256, 'Wrong transport source')
    if args.action == 'prepare':
        tests = t.read(HERE / 'tests.json', args.tests_sha256)
        need(tests['source'] == t.file_identity(Path(__file__)) and tests['failures'] == tests['errors'] == 0
             and tests['tests_run'] >= 8, 'Missing bounded transport controls')
        local = {name: t.file_identity(HERE / name) for name in ('ARTIFACT_README.md', 'related-artifacts.json', 'test_outcomes.py', 'tests.json')}
        value = {'scope': SCOPE, 'owner_message': '你有权做一切事情，目标是尽快完成任务',
            'source': t.file_identity(Path(__file__)), 'local_inputs': local, 'repo_id': REPO,
            'repo_type': 'dataset', 'parent_revision': PARENT, 'addition': ADDITION,
            'pins': {str(k): v for k, v in PINS.items()}, 'gpu_access': False, 'source_release': False,
            'automatic_upload_retry': False, 'scientific_completion': False, 'created_at_utc': t.stamp()}
        t.write_new(HERE / 'authorization.json', value)
        print(json.dumps({'authorization': t.file_identity(HERE / 'authorization.json'), 'upstream_complete': ready()}), flush=True)
        return
    auth = authenticate(t, args.source_sha256, args.authorization_sha256)
    run = HERE / 'run'
    run.mkdir(exist_ok=False)
    t.write_new(run / 'started.json', {'pid': os.getpid(), 'start_ticks': own_start(), 'started_at_utc': t.stamp(),
        'source_sha256': args.source_sha256, 'authorization_sha256': args.authorization_sha256, 'gpu_access': False})
    try:
        while not ready():
            time.sleep(30)
        transport()
        authenticate(t, args.source_sha256, args.authorization_sha256)
        result = transfer(t, auth)
        t.write_new(run / 'completed.json', {'source_sha256': args.source_sha256,
            'authorization_sha256': args.authorization_sha256, 'download_verified': t.file_identity(HERE / 'download-verified.json'),
            'completed_at_utc': t.stamp(), 'scientific_completion': False})
        print(json.dumps(result), flush=True)
    except BaseException as error:
        t.write_new(run / 'failed.json', {'failed_at_utc': t.stamp(), 'exception_type': type(error).__name__,
                                         'scientific_completion': False, 'automatic_retry': False})
        # Do not emit HTTP exception strings that could contain credential-bearing requests.
        print(json.dumps({'failed': True, 'exception_type': type(error).__name__}), flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
