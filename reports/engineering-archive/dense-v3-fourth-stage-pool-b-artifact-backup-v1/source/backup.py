"""One-shot data-only backup of the six accepted original pool-B step-3126 evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import math
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

ROOT = Path('/root/embedding-optimizer-story-refactor')
EXPERIMENT = Path('/root/embedding-optimizer-v3-experiment')
ORIGINAL = ROOT / 'reports/engineering-archive/dense-v3-fourth-stage-pool-b-evaluations-v1'
TRANSPORT = ROOT / 'reports/engineering-archive/dense-v3-evaluation-artifact-backup-v1/source/backup.py'
TRANSPORT_SHA = 'ab565206b22d0138c8227fef3f26eb28e2abeb7653c6f7137b7b532a289ec046'
HELPER_SHA = 'bebdfd3d773e550665660ec12fe7451f70e7e5fd419b1ef21814f774762ef33e'
BUNDLE_SHA = '7d12ba8f36caa05d890a75c3598176324086433408f3a322ed45c75eb66521f5'
VERIFICATION_SHA = '87645cf835349856fc26741fe71fe79c36106b7c9a2fcd0a5f3fdc29002d036a'
INDEPENDENT_SHA = 'dcc88b00d09b2df34ca22f775898fc49a5dab0933f0ae088688ca0141d26dafd'
AUTH_SHA = '2351e225387f42fe607d7e002a0717de79a78cba9f3c93f66652510083f0be7c'
PROTOCOL = '4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b'
REPO = 'qcz/embedding-optimizer-study-analysis-artifacts'
PARENT = '9b2925acd9339b5ce25fd332ab8840ac50319407'
NAMESPACE = 'corrected-dense-correctness-v3'
ADDITION = NAMESPACE + '/partial-fourth-stage-pool-b-evaluations'
ATTR_SHA = 'd7518c3119b61f65892c8c1b1608536862f89fcc91180c0511f1e75ce0edf7df'
ALL_RUNS = tuple('verified-v3-' + opt + '-' + rate for opt, rates in (
    ('adamw', ('1e-6', '3e-6', '1e-5', '3e-5')),
    ('muon', ('1e-4', '3e-4', '1e-3', '3e-3')),
    ('normuon', ('1e-4', '3e-4', '1e-3', '3e-3')),
) for rate in rates)
RUNS = (
    'verified-v3-adamw-3e-6', 'verified-v3-adamw-1e-5',
    'verified-v3-muon-3e-4', 'verified-v3-muon-3e-3',
    'verified-v3-normuon-1e-3', 'verified-v3-normuon-3e-3',
)
TASKS = ('ArguAna', 'ClimateFEVER', 'DBPedia', 'FEVER', 'FiQA2018', 'HotpotQA',
         'MSMARCO', 'NFCorpus', 'NQ', 'QuoraRetrieval', 'SCIDOCS', 'SciFact', 'TRECCOVID', 'Touche2020')
GENERATED = {'tables/checkpoint_means.csv', 'tables/task_scores.csv', 'provenance/acceptance.json'}
FILE_COUNT = 281


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_transport():
    require(hashlib.sha256(TRANSPORT.read_bytes()).hexdigest() == TRANSPORT_SHA,
            'Original transport source changed')
    require(hashlib.sha256((ROOT / 'scripts/restore_primary_v3.py').read_bytes()).hexdigest()
            == HELPER_SHA, 'Original hashing helper changed')
    spec = importlib.util.spec_from_file_location('fourth_stage_pool_b_transport_parent', TRANSPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_rows(bundle):
    require(bundle['protocol_sha256'] == PROTOCOL and bundle['scientific_completion'] is False
            and bundle['model_or_retrieval_recomputed'] is False, 'Wrong accepted bundle scope')
    records = bundle['checkpoints']
    expected = {(run, step) for run in ALL_RUNS for step in (782, 1563, 2345, 3907)}
    expected |= {(run, 3126) for run in RUNS}
    require(len(records) == 54 and {(r['run_id'], r['step']) for r in records} == expected,
            'Require the exact prior forty-eight states and six original pool-B fourth-stage states')
    rows = [next(r for r in records if r['run_id'] == run and r['step'] == 3126) for run in RUNS]
    for row in rows:
        require(row['task_count'] == 14 and len(row['original_workers']) == 14
                and [r['task'] for r in row['native_complete_reread']['tasks']] == list(TASKS),
                'Incomplete fourth-stage tasks or worker records')
    return rows

def source_inputs(t):
    verification = t.read(ORIGINAL / 'verification.json', VERIFICATION_SHA)
    require(len(verification['payloads']) == 15
            and verification['complete_checkpoint_outcomes'] == 54
            and verification['new_checkpoint_outcomes'] == 6
            and verification['raw_task_values'] == 756 and verification['new_raw_task_values'] == 84
            and verification['previous_forty_eight_records_exactly_unchanged'] is True
            and verification['scientific_completion'] is False, 'Wrong accepted archive verification')
    require(verification['payloads']['actual/complete-checkpoint-readback.json']['sha256'] == BUNDLE_SHA
            and verification['payloads']['actual/independent-raw-reconstruction.json']['sha256'] == INDEPENDENT_SHA,
            'Accepted numerical readback bindings differ')
    independent = t.read(ORIGINAL / 'actual/independent-raw-reconstruction.json', INDEPENDENT_SHA)
    require(independent['complete_checkpoint_records'] == 54
            and independent['new_checkpoint_records'] == 6
            and independent['new_raw_task_values'] == 84
            and independent['raw_task_scores_checked'] == 756
            and independent['native_exit_zero_workers_checked'] == 756
            and independent['previous_forty_eight_records_exactly_unchanged'] is True
            and independent['all_twelve_fourth_stage_configurations_complete'] is False
            and independent['scientific_completion'] is False, 'Wrong independent accepted cohort')
    authority = t.read(EXPERIMENT / 'launch/evaluation-handoff/authorization.json', AUTH_SHA)
    require(len(authority['queues']['b']) == 6 and set(authority['queues']['b']) == set(RUNS),
            'Cohort differs from the original unselected pool-B queue')
    bundle = t.read(ORIGINAL / 'actual/complete-checkpoint-readback.json', BUNDLE_SHA)
    rows = select_rows(bundle)
    selected = {}

    def add(prefix, root, item):
        source = Path(item['path'])
        name = prefix + '/' + t.safe_name(source.relative_to(root).as_posix())
        value = (source, {k: item[k] for k in ('bytes', 'sha256')})
        require(name not in selected or selected[name] == value, 'Conflicting original file')
        selected[name] = value

    for row in rows:
        for item in [row['original_complete_receipt'], row['original_operational_receipt'],
                     *row['raw_score_and_metadata_snapshots']]:
            add('native/beir-fourth-stage-pool-b', EXPERIMENT / 'evaluations/dense-primary-v3', item)
        for worker in row['original_workers']:
            require(set(worker) == {'started', 'exited'}, 'Wrong original worker proof')
            for item in worker.values():
                add('provenance/beir-workers', EXPERIMENT / 'launch/evaluation-handoff', item)
    require(len(selected) == 276, 'Require 108 native files and 168 original worker records')
    document = Path(__file__).with_name('ARTIFACT_README.md')
    selected['README.md'] = (document, t.file_identity(document))
    return bundle, rows, selected


def csv_bytes(fields, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def derived_bytes(bundle, rows):
    means, tasks = [], []
    for row in rows:
        run = row['run_id']
        optimizer, rate = run.removeprefix('verified-v3-').split('-', 1)
        common = dict(run_id=run, optimizer=optimizer, learning_rate=rate, step=3126)
        scores = [r['ndcg_at_10'] for r in row['native_complete_reread']['tasks']]
        require(all(math.isfinite(v) and 0 <= v <= 1 for v in scores), 'Undefined primary score')
        mean = sum(map(Fraction.from_float, scores), Fraction()) / 14
        require(float(mean) == row['macro_ndcg_at_10']
                and float(mean * 100) == row['macro_score_0_to_100'], 'Original mean differs')
        means.append(dict(common, task_count=14, macro_ndcg_at_10=float(mean),
                          macro_score_0_to_100=float(mean * 100)))
        tasks.extend(dict(common, task=r['task'], ndcg_at_10=r['ndcg_at_10'])
                     for r in row['native_complete_reread']['tasks'])
    acceptance = {
        'scope': 'accepted-original-pool-b-step-3126-beir-data-only', 'primary_protocol_sha256': PROTOCOL,
        'accepted_bundle_sha256': BUNDLE_SHA, 'accepted_verification_sha256': VERIFICATION_SHA,
        'native_readback_at_utc': bundle['observed_at_utc'], 'runs': list(RUNS), 'step': 3126,
        'tasks': list(TASKS), 'task_cells': 84, 'native_measurement_files': 108,
        'cohort_selection': 'original_pool_b_queue_not_scores',
        'all_twelve_fourth_stage_configurations_complete': False,
        'accepted_independent_readback_sha256': INDEPENDENT_SHA,
        'original_worker_records': 168, 'full_primary_task_cells_required': 840,
        'source_code_included': False, 'new_model_or_statistical_execution': False,
        'scientific_completion': False,
    }
    return {
        'tables/checkpoint_means.csv': csv_bytes(
            ['run_id', 'optimizer', 'learning_rate', 'step', 'task_count',
             'macro_ndcg_at_10', 'macro_score_0_to_100'], means),
        'tables/task_scores.csv': csv_bytes(
            ['run_id', 'optimizer', 'learning_rate', 'step', 'task', 'ndcg_at_10'], tasks),
        'provenance/acceptance.json': (json.dumps(acceptance, indent=2, sort_keys=True,
                                                allow_nan=False) + '\n').encode(),
    }


def prepare(t, work):
    staging = work / 'staging'
    require(not staging.exists(), 'Preserve existing staging')
    bundle, rows, selected = source_inputs(t)
    require(shutil.disk_usage(work).free > 200_000_000, 'Insufficient staging/recovery space')
    staging.mkdir()
    for name, (source, identity) in sorted(selected.items()):
        require(source.suffix in {'.json', '.jsonl', '.md'}, 'Unexpected source payload')
        t.compare_file(source, identity)
        t.scan_text(source)
        target = staging / t.safe_name(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        t.compare_file(target, identity)
    for name, raw in derived_bytes(bundle, rows).items():
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw)
    files = {}
    for path in sorted(staging.rglob('*')):
        if path.is_file():
            t.scan_text(path)
            files[path.relative_to(staging).as_posix()] = t.file_identity(path)
    require(len(files) == FILE_COUNT - 1, 'Unexpected prepared population')
    manifest = {
        'schema_version': 1, 'scope': 'partial-primary-v3-fourth-stage-pool-b-beir-artifact-backup',
        'created_at_utc': t.stamp(), 'primary_protocol_sha256': PROTOCOL,
        'accepted_bundle_sha256': BUNDLE_SHA, 'accepted_verification_sha256': VERIFICATION_SHA,
        'runs': list(RUNS), 'checkpoint_stages': [3126], 'beir_task_cells': 84,
        'cohort_selection': 'original_pool_b_queue_not_scores',
        'all_twelve_fourth_stage_configurations_complete': False,
        'accepted_independent_readback_sha256': INDEPENDENT_SHA,
        'native_measurement_files': 108, 'original_worker_records': 168,
        'source_code_included': False, 'new_model_or_statistical_execution': False,
        'scientific_completion': False, 'files': files, 'payload_files': len(files),
        'payload_bytes': sum(v['bytes'] for v in files.values()),
    }
    t.write_new(staging / 'artifact_manifest.json', manifest)
    identity = t.file_identity(staging / 'artifact_manifest.json')
    preflight = {
        'scope': manifest['scope'], 'repo_id': REPO, 'parent_revision': PARENT,
        'prefix': f'{ADDITION}/{PROTOCOL}/{identity["sha256"]}', 'manifest': identity,
        'source': t.file_identity(Path(__file__)), 'parent_transport_sha256': TRANSPORT_SHA,
        'upload_files': FILE_COUNT, 'upload_bytes': manifest['payload_bytes'] + identity['bytes'],
        'remote_mutations': False, 'scientific_completion': False,
    }
    t.write_new(work / 'preflight.json', preflight)
    return preflight


def prepared(t, work):
    preflight = t.read(work / 'preflight.json')
    require(preflight['source'] == t.file_identity(Path(__file__))
            and preflight['parent_transport_sha256'] == TRANSPORT_SHA
            and preflight['parent_revision'] == PARENT and preflight['repo_id'] == REPO,
            'Prepared source or remote target changed')
    require(preflight['prefix'] == f'{ADDITION}/{PROTOCOL}/{preflight["manifest"]["sha256"]}',
            'Prepared prefix differs')
    manifest = t.read(work / 'staging/artifact_manifest.json', preflight['manifest']['sha256'])
    bundle, rows, selected = source_inputs(t)
    generated = derived_bytes(bundle, rows)
    require(set(manifest['files']) == set(selected) | GENERATED
            and manifest['source_code_included'] is False and manifest['scientific_completion'] is False
            and manifest['checkpoint_stages'] == [3126] and manifest['beir_task_cells'] == 84
            and manifest['cohort_selection'] == 'original_pool_b_queue_not_scores'
            and manifest['all_twelve_fourth_stage_configurations_complete'] is False
            and manifest['accepted_independent_readback_sha256'] == INDEPENDENT_SHA,
            'Prepared population or scope changed')
    for name, (source, identity) in selected.items():
        require(t.compare_file(source, identity) == manifest['files'][name], 'Original input differs')
    for name, raw in generated.items():
        require((work / 'staging' / name).read_bytes() == raw, 'Derived data changed')
    expected = {**manifest['files'], 'artifact_manifest.json': preflight['manifest']}
    actual = {p.relative_to(work / 'staging').as_posix()
              for p in (work / 'staging').rglob('*') if p.is_file()}
    require(len(expected) == FILE_COUNT and actual == set(expected), 'Prepared inventory differs')
    for name, identity in expected.items():
        t.compare_file(work / 'staging' / t.safe_name(name), identity)
        t.scan_text(work / 'staging' / name)
    return preflight, expected


def check_preservation(before, after, old_sub, new_sub):
    require(set(before) == set(after) and NAMESPACE in before
            and before[NAMESPACE]['kind'] == after[NAMESPACE]['kind'] == 'RepoFolder',
            'Remote root population differs')
    require({k: v for k, v in before.items() if k != NAMESPACE}
            == {k: v for k, v in after.items() if k != NAMESPACE}, 'Old root entry changed')
    require(ADDITION not in old_sub and set(new_sub) == set(old_sub) | {ADDITION}
            and all(new_sub[k] == v for k, v in old_sub.items()), 'Old subtree or addition differs')


def check_modes(modes, prefix, names):
    require(len(names) == FILE_COUNT and set(modes) == {prefix + '/' + n for n in names},
            'Upload population differs')
    require(all(v == {'mode': 'regular', 'ignored': False, 'remote_oid': None}
                for v in modes.values()), 'LFS, ignored or overwrite mode refused before commit')


def upload(t, work):
    require(not (work / 'upload-started.json').exists(), 'Prior upload attempt: inspect, never retry')
    preflight, expected = prepared(t, work)
    from huggingface_hub import CommitOperationAdd, HfApi, _commit_api, hf_api, hf_hub_download
    api = HfApi(endpoint='https://huggingface.co', token=True)
    require(api.whoami(token=True).get('name') == 'qcz', 'Configured owner differs')
    info = api.repo_info(REPO, repo_type='dataset', token=False)
    require(info.sha == PARENT and info.private is False, 'Remote parent or privacy changed')
    attrs = Path(hf_hub_download(REPO, repo_type='dataset', revision=PARENT,
        filename='.gitattributes', local_dir=work / 'parent-metadata', token=False))
    require(t.file_identity(attrs)['sha256'] == ATTR_SHA, 'Existing root attributes changed')
    before, old_sub = t.root_inventory(api, PARENT), t.root_inventory(api, PARENT, NAMESPACE)
    require(ADDITION not in old_sub, 'Existing fourth-stage-pool-b subtree: inspect, do not overwrite')
    for module, digest in ((_commit_api, '250ed0e5a5a39383974cab5baae08a964e60f3f40c394c0d13a52e119e1f5f39'),
                           (hf_api, '659636025aa3a7efefa69ca7f16741d8d6cc12f9301beac8c69f9b3d51f81cd4')):
        require(t.file_identity(Path(inspect.getsourcefile(module)))['sha256'] == digest,
                'HF upload client source changed')
    operations = [CommitOperationAdd(path_in_repo=preflight['prefix'] + '/' + name,
                  path_or_fileobj=str(work / 'staging' / name)) for name in sorted(expected)]
    _commit_api._fetch_upload_modes(additions=operations, repo_type='dataset', repo_id=REPO,
        headers=api._build_hf_headers(token=True), revision=PARENT,
        endpoint='https://huggingface.co', create_pr=False)
    modes = {op.path_in_repo: {'mode': op._upload_mode, 'ignored': op._should_ignore,
                             'remote_oid': op._remote_oid} for op in operations}
    t.write_new(work / 'upload-mode-preflight.json', {'observed_at_utc': t.stamp(),
        'modes': modes, 'commit_created': False, 'new_root_attributes_allowed': False})
    check_modes(modes, preflight['prefix'], expected)
    t.write_new(work / 'upload-started.json', {'observed_at_utc': t.stamp(), 'preflight': preflight,
        'original_root': before, 'original_corrected_subtree': old_sub})
    commit = api.create_commit(REPO, repo_type='dataset', parent_commit=PARENT, operations=operations,
        token=True, num_threads=2, commit_message='Preserve six corrected DenseOn pool-B step-3126 BEIR evaluations (data only)')
    require(re.fullmatch('[0-9a-f]{40}', commit.oid) is not None, 'No immutable commit identity')
    result = {'uploaded_at_utc': t.stamp(), 'repo_id': REPO, 'revision': commit.oid,
        'prefix': preflight['prefix'], 'manifest': preflight['manifest'], 'files': len(expected),
        'durability_verified': False, 'scientific_completion': False}
    t.write_new(work / 'upload.json', result)
    actual = {}
    for item in api.list_repo_tree(REPO, repo_type='dataset', revision=commit.oid,
                                  path_in_repo=preflight['prefix'], recursive=True, token=False):
        if type(item).__name__ == 'RepoFile':
            actual[item.path.removeprefix(preflight['prefix'] + '/')] = {
                'bytes': item.size, 'kind': 'sha256' if item.lfs else 'git_blob_sha1',
                'digest': item.lfs.sha256 if item.lfs else item.blob_id}
    t.compare_remote(expected, actual)
    check_preservation(before, t.root_inventory(api, commit.oid), old_sub,
                       t.root_inventory(api, commit.oid, NAMESPACE))
    prepared(t, work)
    audit = {**result, 'verified_at_utc': t.stamp(), 'durability_verified': True,
        'remote_inventory': actual, 'old_root_entries_unchanged': len(before) - 1,
        'old_corrected_subtrees_unchanged': len(old_sub), 'source_code_uploaded': False,
        'root_attributes_unchanged': True, 'remote_paths_deleted_or_overwritten': False}
    t.write_new(work / 'remote-audit.json', audit)
    return {k: v for k, v in audit.items() if k != 'remote_inventory'}


def download_verify(t, work):
    from huggingface_hub import hf_hub_download
    preflight, expected = prepared(t, work)
    audit = t.read(work / 'remote-audit.json')
    require(audit['durability_verified'] is True and audit['prefix'] == preflight['prefix']
            and re.fullmatch('[0-9a-f]{40}', audit['revision']) is not None, 'No accepted remote audit')
    destination = work / 'download'
    require(not destination.exists(), 'Preserve existing recovery attempt')
    destination.mkdir()

    def transfer(name):
        filename = audit['prefix'] + '/' + t.safe_name(name)
        path = Path(hf_hub_download(REPO, repo_type='dataset', revision=audit['revision'],
            filename=filename, local_dir=destination, token=False, endpoint='https://huggingface.co'))
        require(path == destination / filename, 'Unexpected recovered location')
        t.compare_file(path, expected[name])
        return name

    with ThreadPoolExecutor(max_workers=2) as executor:
        for done, _ in enumerate(executor.map(transfer, sorted(expected)), 1):
            if done % 50 == 0:
                print(json.dumps({'recovered_files': done, 'required_files': FILE_COUNT}), flush=True)
    root = destination / audit['prefix']
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} == set(expected),
            'Recovered inventory differs')
    result = {'scope': 'actual-complete-anonymous-fourth-stage-pool-b-beir-recovery',
        'observed_at_utc': t.stamp(), 'revision': audit['revision'], 'prefix': audit['prefix'],
        'downloaded_root': str(root), 'files': len(expected), 'manifest': preflight['manifest'],
        'bytes': sum(v['bytes'] for v in expected.values()), 'all_payload_hashes_match': True,
        'same_physical_host': True, 'new_model_or_statistical_execution': False,
        'scientific_completion': False}
    t.write_new(work / 'download-verified.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check', 'upload', 'download-verify'))
    parser.add_argument('--workdir', type=Path, required=True)
    args = parser.parse_args()
    require(args.workdir.parent == Path('/tmp')
            and args.workdir.name.startswith('dense-v3-fourth-stage-pool-b-artifact-backup.')
            and args.workdir.is_dir() and not args.workdir.is_symlink(), 'Wrong task workdir')
    transport = load_transport()
    if args.action == 'check':
        preflight, expected = prepared(transport, args.workdir)
        result = {'prepared_files_verified': len(expected), 'preflight': preflight,
                  'remote_mutations': False, 'scientific_completion': False}
    else:
        result = {'prepare': prepare, 'upload': upload, 'download-verify': download_verify}[
            args.action](transport, args.workdir)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
