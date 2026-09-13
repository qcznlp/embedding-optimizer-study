"""Narrow, owner-authorized recovery wiring; no numerical or lease replacements."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

EXPERIMENT = Path('/root/embedding-optimizer-v3-experiment')
STORY = Path('/root/embedding-optimizer-story-refactor')
ORIGINAL = EXPERIMENT / 'launch/functional-dimensions'
DEPLOY = EXPERIMENT / 'launch/functional-dimensions-recovery-v1'
OUTPUT = EXPERIMENT / 'analyses/dense-primary-v3-functional-dimensions-recovery-v1'
OLD_OUTPUT = EXPERIMENT / 'analyses/dense-primary-v3-functional-dimensions'
SCOPE = 'owner_authorized_functional_recovery_v1_20260912'
OWNER_MESSAGE = '你有权做一切事情，目标是尽快完成任务'
FILES = ('recovery.py', 'recovery_support.py', 'record_layout.py', 'flow_candidate.py',
         'observe_recovery.py', 'test_recovery.py')
PINS = {
    ORIGINAL / 'dispatch.py': '3b02f1d8486f3c2b41d0403cbf291580d0ca32868b8b29cb05514a22770257f5',
    ORIGINAL / 'functional.py': 'c55d3fa101fa681070509a95d987023b7ed442d89ce385a8885f940ea3cc05b8',
    ORIGINAL / 'inputs.json': 'be6d92fda2f7f0ba3e86f2c2866fa9dca530f066d29ce92d0887fdad91684067',
    ORIGINAL / 'authorization.json': 'd72527595e2c963eae7fd46b0a1de2bd83d1ea15a863ae467a1065cfc5b2f336',
    ORIGINAL / 'run/failed.json': 'c84f93bc431612be2b885d7a467828d08a18d5f259829c4b925a4f23ea7def2e',
    ORIGINAL / 'observe.py': 'c314e4d540305bb67e0bcdcb6ad921d22926e4cbd40f59e00e84415be18cd078',
    STORY / 'reports/engineering-archive/dense-v3-pretrained-native-copy-v1/actual/readout.json':
        '773c184b474c0508ff22a549db5303abefe922c95272bd539c32ed5c0b503195',
}
HELPERS = {
    'record_layout.py': 'f16dcc94d5eadc1f2a8da1fe50840b639c49919138b409fa8715fc81dbaaaa89',
    'flow_candidate.py': 'c06246c7e71416164390ee74bb5838cfb706b2fbcd28b0ba99115cce55877a2e',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
         'Require ordinary, non-symlink evidence')
    before = path.stat()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need(all(getattr(before, k) == getattr(after, k) for k in
             ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')), 'Evidence raced')
    return {'bytes': after.st_size, 'sha256': sha}


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def read(path, sha=None):
    expected = identity(path)
    need(sha is None or expected['sha256'] == sha, 'External evidence binding differs')
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=unique_pairs,
                       parse_constant=lambda _: need(False, 'Nonfinite JSON constant'))
    need(identity(path) == expected, 'JSON evidence raced')
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def record(path):
    return {'path': str(path), **identity(path)}


def verify_record(value):
    need(set(value) == {'path', 'bytes', 'sha256'} and
         identity(value['path']) == {k: value[k] for k in ('bytes', 'sha256')},
         'Bound artifact differs')


def namespace(here):
    here = Path(here)
    need(here == DEPLOY and here != ORIGINAL and OUTPUT != OLD_OUTPUT,
         'Recovery may execute only in its new approved namespace')
    need(here.is_dir() and not any(p.is_symlink() for p in (here, *here.parents)),
         'Recovery launch ancestry differs')
    need(OUTPUT.parent.is_dir() and not any(p.is_symlink() for p in (OUTPUT, *OUTPUT.parents)),
         'Recovery output ancestry differs')


def source_files(root):
    return {name: identity(Path(root) / name) for name in FILES}


def modules(here):
    for name, sha in HELPERS.items():
        need(identity(Path(here) / name)['sha256'] == sha, 'Validated recovery helper changed')
    import record_layout
    import flow_candidate
    need(Path(record_layout.__file__).resolve() == Path(here) / 'record_layout.py'
         and Path(flow_candidate.__file__).resolve() == Path(here) / 'flow_candidate.py',
         'Wrong helper import location')
    return record_layout, flow_candidate


def proposal_data(source_root, settings):
    """Metadata only. Produces no execution authority and starts no process."""
    for path, sha in PINS.items():
        need(identity(path)['sha256'] == sha, 'Original evidence changed')
    inputs = read(ORIGINAL / 'inputs.json', PINS[ORIGINAL / 'inputs.json'])
    for path, value in inputs['sources'].items():
        need(identity(path) == value, 'Original dependency changed')
    cells = [j['plan']['state']['cell'] for j in inputs['jobs']]
    # The helper import location is checked separately in the installed entry.
    import record_layout
    record_layout.validate_cells(cells)
    need(all(digest(j['plan']) == j['plan_sha256'] for j in inputs['jobs']), 'Changed state plan')
    origin = {'records': str(ORIGINAL / 'run/jobs'),
              'vectors': str(OLD_OUTPUT / 'vectors/states/pretrained'),
              'source_sha256': PINS[ORIGINAL / 'dispatch.py'],
              'authorization_sha256': PINS[ORIGINAL / 'authorization.json']}
    origin['files'] = {name: identity(Path(origin['records']) / ('pretrained.' + name))
                       for name in ('started.json', 'exited.json', 'encoded.json', 'verified.json')}
    origin['files'].update({name: identity(Path(origin['vectors']) / name)
                            for name in ('manifest.json', 'vectors.npz')})
    return {'scope': SCOPE, 'launch_root': str(DEPLOY), 'output_root': str(OUTPUT),
            'source_files': source_files(source_root), 'settings': settings,
            'original_bindings': {str(p): identity(p) for p in PINS},
            'sources': inputs['sources'], 'state_order': cells,
            'plans': {j['plan']['state']['cell']: j['plan_sha256'] for j in inputs['jobs']},
            'pretrained_origin': origin, 'remaining_new_encodings': 60,
            'feature_states': 61, 'native_guards_modified': False,
            'execution_authorized': False, 'scientific_completion': False}


def check_proposal(here, proposal, settings):
    namespace(here)
    need(proposal == proposal_data(here, settings), 'Proposal/source/inputs no longer match')
    modules(here)


def check_approval(approval, proposal_sha):
    need(approval.get('scope') == SCOPE and approval.get('approved') is True
         and approval.get('source') == 'direct_user_message'
         and approval.get('owner_message') == OWNER_MESSAGE
         and approval.get('automatic_continuation') is False
         and approval.get('proposal_sha256') == proposal_sha
         and approval.get('preserve_original_attempt') is True
         and approval.get('protected_helper_access') is False,
         'Recovery needs the explicit owner approval bound to this proposal')


def check_tests(tests, proposal):
    need(tests.get('source_files') == proposal['source_files']
         and tests.get('tests_run', 0) >= 12
         and tests.get('failures') == tests.get('errors') == tests.get('skipped') == 0
         and tests.get('production_module_composition_passed') is True
         and tests.get('original_operational_bodies_unchanged') is True
         and tests.get('synthetic_process_model_lease_fixtures') is True,
         'Missing source-bound recovery boundary/composition checks')


def prepare(args, entry, geometry, here, settings, functional_sha, inputs_sha):
    namespace(here)
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Preparation must hide CUDA')
    need(not any(p.exists() or p.is_symlink() for p in
                 (here / 'authorization.json', here / 'run', OUTPUT)), 'Use fresh recovery outputs')
    proposal = read(here / 'proposal.json', args.proposal_sha)
    check_proposal(here, proposal, settings)
    check_approval(read(here / 'owner-approval.json', args.approval_sha), args.proposal_sha)
    check_tests(read(here / 'tests.json', args.tests_sha), proposal)
    context, inputs = entry.authenticate(functional_sha, inputs_sha)
    need(inputs['sources'] == proposal['sources']
         and [j['plan']['state']['cell'] for j in context.jobs] == proposal['state_order'],
         'Native input admission differs from proposal')
    authority = {
        'scope': entry.SCOPE, 'recovery_scope': SCOPE,
        'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': args.source_sha, 'functional_source_sha256': functional_sha,
        'inputs_sha256': inputs_sha, 'settings': settings, 'sources': inputs['sources'],
        'primary_protocol_sha256': context.primary.sha256,
        'dimension_protocol_sha256': context.contract.sha256,
        'state_order': proposal['state_order'], 'pretrained_origin': proposal['pretrained_origin'],
        'proposal': record(here / 'proposal.json'), 'owner_approval': record(here / 'owner-approval.json'),
        'tests': record(here / 'tests.json'), 'test_source': record(here / 'test_recovery.py'),
        'execution_authorized': True, 'native_guards_modified': False,
        'committed_source_release': False, 'scientific_completion': False,
        'original_attempt_restarted_or_modified': False,
        'boundary': 'New owner-approved namespace, same 61 states/numerics, one reused accepted state; original release gates unchanged.'}
    check_proposal(here, proposal, settings)
    for name, sha in (('proposal.json', args.proposal_sha), ('owner-approval.json', args.approval_sha),
                      ('tests.json', args.tests_sha)):
        need(identity(here / name)['sha256'] == sha, 'Preparation evidence changed')
    geometry.write_new(here / 'authorization.json', authority)
    return {'authorization': record(here / 'authorization.json'), 'states': 61,
            'new_encodings': 60, 'native_input_admission_passed': True,
            'gpu_jobs_launched': 0, 'scientific_completion': False}


def authority(here, source_sha, authorization_sha, settings):
    namespace(here)
    need(identity(here / 'recovery.py')['sha256'] == source_sha, 'Recovery entry changed')
    value = read(here / 'authorization.json', authorization_sha)
    need(value['recovery_scope'] == SCOPE and value['source_sha256'] == source_sha
         and value['execution_authorized'] is True
         and value['original_attempt_restarted_or_modified'] is False,
         'Wrong recovery authority')
    for field, name in (('proposal', 'proposal.json'), ('owner_approval', 'owner-approval.json'),
                        ('tests', 'tests.json'), ('test_source', 'test_recovery.py')):
        need(value[field]['path'] == str(here / name), 'Recovery evidence path differs')
        verify_record(value[field])
    proposal = read(here / 'proposal.json', value['proposal']['sha256'])
    check_proposal(here, proposal, settings)
    check_approval(read(here / 'owner-approval.json'), value['proposal']['sha256'])
    check_tests(read(here / 'tests.json'), proposal)
    need(value['sources'] == proposal['sources'] and value['state_order'] == proposal['state_order']
         and value['pretrained_origin'] == proposal['pretrained_origin'], 'Recovery population differs')
    return value, proposal


def output_facade(entry):
    """Relocate only dispatch output; original native functions retain their globals."""
    need(entry.OUTPUT == OLD_OUTPUT, 'Original entry output binding changed')
    return SimpleNamespace(**{**vars(entry), 'OUTPUT': OUTPUT})


def original_process_reader():
    path = ORIGINAL / 'observe.py'
    need(identity(path)['sha256'] == PINS[path], 'Original exact-process reader changed')
    spec = importlib.util.spec_from_file_location('unchanged_functional_process_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.process
