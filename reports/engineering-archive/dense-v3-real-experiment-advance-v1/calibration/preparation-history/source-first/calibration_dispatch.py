"""Owner-scoped genuine two-state GPU calibration, separate from old controllers.

Only the already tested native calibration component computes gradients, update
directions and rates. This entry supplies fresh namespaces, original dual leases,
source/main-input admission and actual child exits. No branch is trained here.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EXPERIMENT = Path('/root/embedding-optimizer-v3-experiment')
STORY = Path('/root/embedding-optimizer-story-refactor')
PRIMARY = Path('/root/embedding-optimizer-primary-v3')
DEPLOY = EXPERIMENT / 'launch/factorial-calibration-v1'
SOURCE_ROOT = STORY / 'reports/engineering-archive/dense-v3-factorial-worker-v1/actual/source-final'
OUTPUT = EXPERIMENT / 'analyses/dense-primary-v3-factorial-calibration-v1'
RUN = HERE / 'run'
SCOPE = 'owner_authorized_v3_factorial_calibration_20260912'
STATES = ('adamw_state', 'muon_state')
GPU = {'adamw_state': '1', 'muon_state': '2'}
OWNER_MESSAGE = '你有权做一切事情，目标是尽快完成任务'
ACCEPTED = STORY / 'reports/engineering-archive/dense-v3-factorial-calibration-v1/actual/actual-inputs-third.json'
ORIGINAL_INPUTS = EXPERIMENT / 'launch/functional-dimensions/inputs.json'
GEOMETRY = EXPERIMENT / 'launch/weight-geometry/run_geometry.py'
PARENT = EXPERIMENT / 'launch/train_matrix.py'
VALIDATION = EXPERIMENT / 'launch/validation-handoff/validation.py'
DISPATCH = EXPERIMENT / 'launch/functional-dimensions/dispatch.py'
PINS = {
    ACCEPTED: '3640dc3a0deaa142d7e09eabfcc8450ab786db1b7302e04f470518557a63176f',
    ORIGINAL_INPUTS: 'be6d92fda2f7f0ba3e86f2c2866fa9dca530f066d29ce92d0887fdad91684067',
    GEOMETRY: '5ab87568a1a9050e423c2e778a4fda8b53f906eec2ee2f5a824e2522c09c0ec6',
    PARENT: 'dd823408de44d0696d0cac3a9d865ffc41936abdf2542ae815c3f5eda8dffcf3',
    VALIDATION: 'd2fc55c67670ea5b42cacb055f48379ef278a465ce86d95808231ab86049d913',
    DISPATCH: '3b02f1d8486f3c2b41d0403cbf291580d0ca32868b8b29cb05514a22770257f5',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Nonordinary bound file')
    before = path.stat()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need(all(getattr(before, k) == getattr(after, k) for k in
             ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')), 'Bound file raced')
    return {'bytes': after.st_size, 'sha256': sha}


def pairs(items):
    value = {}
    for key, item in items:
        need(key not in value, 'Duplicate JSON key')
        value[key] = item
    return value


def read(path, sha=None):
    bound = identity(path)
    need(sha is None or bound['sha256'] == sha, 'External binding differs')
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                       parse_constant=lambda _: need(False, 'Nonfinite JSON'))
    need(identity(path) == bound, 'JSON raced')
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def imported(name, path):
    need(identity(path)['sha256'] == PINS[path], 'Original execution dependency changed')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def namespace():
    need(HERE == DEPLOY and not any(p.is_symlink() for p in (HERE, *HERE.parents, OUTPUT, *OUTPUT.parents)),
         'Use only the explicit new calibration namespace')


def sources():
    accepted = read(ACCEPTED, PINS[ACCEPTED])
    files = accepted['source_files']
    need(len(files) == 66 and digest(files) ==
         '247848d22aed020f3b50a2788ffd6a72a78d96730b2e52bcdb033450c7436380',
         'The accepted 66-file numerical assembly differs')
    for path, sha in PINS.items():
        need(identity(path)['sha256'] == sha, 'Pinned original admission changed')
    for name, value in files.items():
        need(identity(SOURCE_ROOT / name) == value, 'Audited numerical file changed')
    return files, accepted['inputs']


def context(expected_sources):
    from embed_optim import factorial_v3_calibration as calibration
    from embed_optim.factorial_v3_inputs import Locations, load_inputs
    locations = Locations(STORY, PRIMARY, EXPERIMENT,
        Path('/root/embedding-optimizer-study/data'),
        STORY / 'reports/engineering-archive/dense-v3-factorial-inputs-v1')
    calibration.require_sources()
    inputs = load_inputs(locations)
    check_imports(expected_sources)
    return calibration, locations, inputs


def check_imports(expected_sources):
    for name, module in tuple(sys.modules.items()):
        if name == 'embed_optim' or name.startswith('embed_optim.'):
            path = Path(module.__file__).resolve()
            need(path.is_relative_to(SOURCE_ROOT), 'Foreign numerical package import')
            relative = str(path.relative_to(SOURCE_ROOT))
            need(relative in expected_sources and identity(path) == expected_sources[relative],
                 'Imported source is outside the accepted numerical closure')


def approval(value, source_sha):
    need(value.get('scope') == SCOPE and value.get('approved') is True
         and value.get('source') == 'direct_user_message' and value.get('owner_message') == OWNER_MESSAGE
         and value.get('automatic_continuation') is False and value.get('source_sha256') == source_sha
         and value.get('protected_helper_access') is False and value.get('old_controller_transition') is False,
         'Explicit owner-scoped calibration approval required')


def prepare(args):
    namespace()
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Preparation must hide CUDA')
    need(identity(__file__)['sha256'] == args.source_sha, 'New entry source differs')
    need(not any(p.exists() or p.is_symlink() for p in (RUN, OUTPUT, HERE / 'authorization.json')),
         'Preserve existing calibration attempts')
    approval(read(HERE / 'owner-approval.json', args.approval_sha), args.source_sha)
    tests = read(HERE / 'tests.json', args.tests_sha)
    need(tests['source_sha256'] == args.source_sha and tests['tests_run'] >= 10
         and tests['failures'] == tests['errors'] == tests['skipped'] == 0
         and tests['test_source'] == identity(HERE / 'test_dispatch.py'), 'Unverified launch boundary')
    files, accepted = sources()
    calibration, locations, inputs = context(files)
    need(inputs == accepted, 'Genuine v3 input admission changed')
    geometry = imported('calibration_original_main_reader', GEOMETRY)
    main, child = geometry.read_primary_admission(PINS[GEOMETRY])
    original = read(ORIGINAL_INPUTS, PINS[ORIGINAL_INPUTS])
    need(main['rows'] == original['actual_completion']['rows'] and len(main['rows']) == 12,
         'Complete actual primary proofs changed')
    authority = {'scope': SCOPE, 'prepared_at_utc': geometry.now(), 'source_sha256': args.source_sha,
        'test_source': tests['test_source'], 'tests': identity(HERE / 'tests.json'),
        'owner_approval': identity(HERE / 'owner-approval.json'),
        'source_root': str(SOURCE_ROOT), 'source_files': files, 'inputs_sha256': digest(inputs),
        'requests': {state: calibration.calibration_request(inputs, state) for state in STATES},
        'gpu_tokens': GPU, 'output_root': str(OUTPUT),
        'main_completion': {'rows_sha256': digest(main['rows']), 'runs': 12,
                            'ten_observed_zero_two_unobserved_exits_preserved': True, 'reader': child},
        'both_original_lease_namespaces': True, 'execution_authorized': True,
        'old_controller_transition': False, 'native_guards_modified': False,
        'committed_source_release': False, 'scientific_completion': False,
        'boundary': 'Owner-approved new v3 calibration. Uses spare tokens while original functional analysis continues; no old publication/controller gate is claimed passed. Fixed sources, 32 rows, eight gradients, 88 hidden matrices and global 5e-4 scale unchanged.'}
    need(sources()[0] == files and identity(__file__)['sha256'] == args.source_sha, 'Source changed during admission')
    geometry.write_new(HERE / 'authorization.json', authority)
    return {'authorization': identity(HERE / 'authorization.json'), 'native_inputs_and_all_twelve_proofs': True,
            'states': 2, 'gpu_jobs_launched': 0, 'scientific_completion': False}


def authenticate(args, gpu=False):
    namespace()
    need(identity(__file__)['sha256'] == args.source_sha, 'New entry source changed')
    auth = read(HERE / 'authorization.json', args.authorization_sha)
    need(auth['scope'] == SCOPE and auth['source_sha256'] == args.source_sha
         and auth['gpu_tokens'] == GPU and auth['source_root'] == str(SOURCE_ROOT)
         and auth['output_root'] == str(OUTPUT) and auth['execution_authorized'] is True
         and auth['native_guards_modified'] is False and auth['old_controller_transition'] is False
         and auth['committed_source_release'] is False and auth['scientific_completion'] is False,
         'Wrong calibration authority')
    for name, field in (('owner-approval.json', 'owner_approval'), ('tests.json', 'tests'), ('test_dispatch.py', 'test_source')):
        need(identity(HERE / name) == auth[field], 'Approval/test source changed')
    approval(read(HERE / 'owner-approval.json'), args.source_sha)
    files, accepted = sources()
    need(files == auth['source_files'] and digest(accepted) == auth['inputs_sha256'], 'Admitted inputs changed')
    parent = imported('calibration_original_lease_parent', PARENT)
    if gpu:
        need(args.worker in STATES and args.gpu_token == GPU[args.worker]
             and os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu_token, 'Worker token differs')
        validation = imported('calibration_original_inherited_checker', VALIDATION)
        validation.require_inherited_leases(args.lease_fd,
            [root / f'gpu-{args.gpu_token}.lock' for root in parent.LEASE_ROOTS])
    else:
        need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU coordinator/readback must hide CUDA')
    return auth, parent


def worker(args):
    auth, _ = authenticate(args, gpu=True)
    calibration, locations, inputs = context(auth['source_files'])
    need(digest(inputs) == auth['inputs_sha256'], 'Live calibration inputs changed')
    need(calibration.calibration_request(inputs, args.worker) == auth['requests'][args.worker], 'Worker plan changed')
    admission = read(RUN / f'{args.worker}.admission.json')
    need(admission['authorization_sha256'] == args.authorization_sha
         and admission['gpu_token'] == args.gpu_token, 'Worker admission changed')
    root = OUTPUT / args.worker
    started = time.monotonic()
    calibration.export_gradients(locations, args.worker, root)
    gradient = identity(root / 'gradient-receipt.json')
    calibration.replay_calibration(locations, args.worker, root, gradient)
    binding = identity(root / 'directions/calibration.json')
    result = calibration.read_calibration(inputs, args.worker, root, binding)
    check_imports(auth['source_files'])
    need(sources()[0] == auth['source_files'], 'Numerical source changed while calibrating')
    geometry = imported('calibration_original_writer', GEOMETRY)
    receipt = {'state': args.worker, 'completed_at_utc': geometry.now(),
        'source_sha256': args.source_sha, 'authorization_sha256': args.authorization_sha,
        'gpu_token': args.gpu_token, 'gradient_receipt': gradient, 'calibration_binding': binding,
        'elapsed_seconds': time.monotonic() - started, 'calibration': result['calibration'],
        'fresh_gpu_gradients_and_directions': True, 'scientific_completion': False}
    geometry.write_new(RUN / f'{args.worker}.produced.json', receipt)
    return receipt


def verify(args):
    auth, _ = authenticate(args)
    calibration, locations, inputs = context(auth['source_files'])
    need(digest(inputs) == auth['inputs_sha256'], 'Fresh-reader inputs changed')
    root = OUTPUT / args.verify
    binding = identity(root / 'directions/calibration.json')
    need(binding['sha256'] == args.calibration_sha, 'Fresh-reader calibration anchor differs')
    result = calibration.read_calibration(inputs, args.verify, root, binding)
    geometry = imported('calibration_fresh_reader_writer', GEOMETRY)
    receipt = {'state': args.verify, 'verified_at_utc': geometry.now(),
        'source_sha256': args.source_sha, 'authorization_sha256': args.authorization_sha,
        'calibration_binding': binding, 'fresh_process_native_readback': True,
        'calibration': result['calibration'], 'scientific_completion': False}
    geometry.write_new(RUN / f'{args.verify}.verified.json', receipt)
    return receipt


def command(args, state, descriptors=None, calibration_sha=None):
    need(state in STATES, 'Unknown source state')
    argv = ['/usr/bin/python', '-B', str(Path(__file__).resolve()), '--source-sha', args.source_sha,
            '--authorization-sha', args.authorization_sha]
    if calibration_sha is not None:
        need(descriptors is None, 'CPU readback cannot inherit GPU descriptors')
        return argv + ['--verify', state, '--calibration-sha', calibration_sha]
    need(descriptors is not None and len(descriptors) == len(set(descriptors)) == 2
         and all(type(fd) is int and fd >= 0 for fd in descriptors), 'Require both inherited leases')
    argv += ['--worker', state, '--gpu-token', GPU[state]]
    for fd in descriptors:
        argv += ['--lease-fd', str(fd)]
    return argv


def environment(parent, token=None):
    env = parent.environment([token] if token is not None else None)
    env.update(PYTHONPATH=f'{SOURCE_ROOT / "src"}:{SOURCE_ROOT}', WANDB_MODE='disabled',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    return env


def coordinate(args):
    auth, parent = authenticate(args)
    need(not RUN.exists() and not OUTPUT.exists(), 'Never restart an existing calibration attempt')
    geometry = imported('calibration_coordinator_writer', GEOMETRY)
    dispatch = imported('calibration_exact_child_identity', DISPATCH)
    RUN.mkdir(); OUTPUT.mkdir()
    geometry.write_new(RUN / 'coordinator.started.json', {
        **dispatch.process_identity(os.getpid()), 'source_sha256': args.source_sha,
        'authorization_sha256': args.authorization_sha, 'started_at_utc': geometry.now(), 'gpu_tokens': GPU})
    children, outcomes = [], {}
    try:
        with ExitStack() as leases:
            for state in STATES:
                parent.handoff()
                while True:
                    try:
                        descriptors = leases.enter_context(parent.leases([GPU[state]]))
                        break
                    except BlockingIOError:
                        time.sleep(10)
                        parent.handoff()
                argv = command(args, state, descriptors)
                geometry.write_new(RUN / f'{state}.admission.json', {
                    'state': state, 'source_sha256': args.source_sha,
                    'authorization_sha256': args.authorization_sha, 'gpu_token': GPU[state],
                    'request_sha256': digest(auth['requests'][state]), 'both_inherited_leases': True})
                with (RUN / f'{state}.log').open('xb') as log:
                    child = subprocess.Popen(argv, cwd=SOURCE_ROOT, env=environment(parent, GPU[state]),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, pass_fds=descriptors)
                # Retain every created child even if subsequent exact-handle capture fails.
                children.append((state, child, argv, None))
                handle = dispatch.process_identity(child.pid, argv)
                children[-1] = (state, child, argv, handle)
                geometry.write_new(RUN / f'{state}.started.json', {**handle,
                    'state': state, 'source_sha256': args.source_sha,
                    'authorization_sha256': args.authorization_sha, 'gpu_token': GPU[state],
                    'started_at_utc': geometry.now()})
                print(json.dumps({'event': 'genuine_calibration_worker_started', 'state': state, **handle}), flush=True)
            for state, child, argv, handle in children:
                code = child.wait()
                geometry.write_new(RUN / f'{state}.exited.json', {**handle, 'state': state,
                    'exit_code': code, 'authorization_sha256': args.authorization_sha,
                    'exited_at_utc': geometry.now()})
                outcomes[state] = code
        need(outcomes == {state: 0 for state in STATES}, 'Calibration worker failed; preserve outputs')
        for state in STATES:
            produced = read(RUN / f'{state}.produced.json')
            need(produced['source_sha256'] == args.source_sha
                 and produced['authorization_sha256'] == args.authorization_sha, 'Produced calibration differs')
            argv = command(args, state, calibration_sha=produced['calibration_binding']['sha256'])
            with (RUN / f'{state}.verify.log').open('xb') as log:
                child = subprocess.Popen(argv, cwd=SOURCE_ROOT, env=environment(parent), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, close_fds=True)
                handle = dispatch.process_identity(child.pid, argv)
                geometry.write_new(RUN / f'{state}.verify.started.json', {**handle,
                    'authorization_sha256': args.authorization_sha, 'started_at_utc': geometry.now()})
                code = child.wait()
            geometry.write_new(RUN / f'{state}.verify.exited.json', {**handle, 'exit_code': code,
                'authorization_sha256': args.authorization_sha, 'exited_at_utc': geometry.now()})
            need(code == 0, 'Fresh native calibration readback failed')
            verified = read(RUN / f'{state}.verified.json')
            need(verified['calibration_binding'] == produced['calibration_binding']
                 and verified['calibration'] == produced['calibration'], 'Fresh calibrated rates differ')
        result = {'scope': SCOPE, 'completed_at_utc': geometry.now(),
            'source_sha256': args.source_sha, 'authorization_sha256': args.authorization_sha,
            'calibrations': {state: {'path': str(OUTPUT / state),
                'calibration_binding': identity(OUTPUT / state / 'directions/calibration.json')} for state in STATES},
            'actual_gpu_worker_exits': outcomes, 'fresh_process_native_readbacks': 2,
            'formal_branches_trained': 0, 'scientific_completion': False}
        geometry.write_new(RUN / 'completed.json', result)
        return result
    except BaseException as error:
        # A sibling can finish naturally; never orphan a created child or touch other jobs.
        for state, child, argv, handle in children:
            code = child.wait()
            path = RUN / f'{state}.exited.json'
            if not path.exists() and handle is not None:
                geometry.write_new(path, {**handle, 'state': state, 'exit_code': code,
                    'authorization_sha256': args.authorization_sha, 'exited_at_utc': geometry.now()})
        geometry.write_new(RUN / 'failed.json', {'source_sha256': args.source_sha,
            'authorization_sha256': args.authorization_sha, 'failed_at_utc': geometry.now(),
            'error_type': type(error).__name__, 'automatic_retry_authorized': False,
            'preserve_partial_outputs': True, 'scientific_completion': False})
        raise


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-sha', required=True)
    p.add_argument('--authorization-sha'); p.add_argument('--approval-sha'); p.add_argument('--tests-sha')
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true'); modes.add_argument('--coordinate', action='store_true')
    modes.add_argument('--worker', choices=STATES); modes.add_argument('--verify', choices=STATES)
    p.add_argument('--calibration-sha'); p.add_argument('--gpu-token', choices=tuple(GPU.values()))
    p.add_argument('--lease-fd', type=int, action='append', default=[])
    args = p.parse_args(argv)
    if args.prepare:
        need(args.approval_sha and args.tests_sha and not args.authorization_sha, 'Preparation requires explicit approval/tests')
    else:
        need(args.authorization_sha and not args.approval_sha and not args.tests_sha, 'Exact execution authority required')
    if args.worker:
        need(args.gpu_token == GPU[args.worker] and len(args.lease_fd) == len(set(args.lease_fd)) == 2
             and all(fd >= 0 for fd in args.lease_fd), 'Worker needs exact token and both leases')
    else:
        need(not args.gpu_token and not args.lease_fd, 'CPU mode cannot inherit GPU arguments')
    need(bool(args.calibration_sha) == bool(args.verify), 'Only native readback takes a calibration anchor')
    return args


if __name__ == '__main__':
    args = parse()
    function = prepare if args.prepare else coordinate if args.coordinate else worker if args.worker else verify
    print(json.dumps(function(args), sort_keys=True, allow_nan=False), flush=True)
