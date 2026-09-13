"""Owner-scoped fresh 2x2x3 DenseOn continuation, using the unchanged native worker.

This new operational entry owns only its direct four-rank children. It supplies
dual GPU leases, exact source/runtime/input admission, real NCCL initialization,
full-horizon execution and a fresh CPU whole-run read. No old controller or
publication gate is changed; no partial or failed attempt is automatically retried.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone, timedelta
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EXPERIMENT = Path('/root/embedding-optimizer-v3-experiment')
DEPLOY = EXPERIMENT / 'launch/factorial-training-v1'
OUTPUT = EXPERIMENT / 'outputs/dense-v3-state-operator-v1'
CAL_ENTRY = EXPERIMENT / 'launch/factorial-calibration-v2/calibration_dispatch.py'
CAL_SOURCE_SHA = '3436597f8ac85b1dde1e53d4fa2fef007cf19fbbc97d168b439a61cd2ef6b598'
CAL_AUTH_SHA = 'f7d29c908fc251a2affcf8c2205992f4f44d61b4b97201a60569e79e5e78081d'
SCOPE = 'owner_authorized_v3_factorial_training_20260912'
OWNER_MESSAGE = '你有权做一切事情，目标是尽快完成任务'
POOLS = {'a': ['4', '5', '6', '7'], 'b': ['0', '1', '2', '3']}
STATES = ('adamw_state', 'muon_state')
OPERATORS = ('adamw', 'muon')
SEEDS = (314159, 271828, 161803)
PROJECT, ENTITY = 'embedding-optimizer-study', 'stevezenguom'
ADDED = {
    'src/embed_optim/factorial_v3_run_contract.py': '67cf49f0a836c687f004c9085b349407111488d504c48e152cb742c0c29f787d',
    'src/embed_optim/factorial_v3_bound_trainer.py': '5b17a43e7616243d463e811e9c93e8397b869fcd5a363ff92d3a2961466da303',
    'src/embed_optim/factorial_v3_factory.py': 'bddb443318481d3d7dbddd7b5eb5f5a0f651939537dd281486ad8c305721209b',
    'src/embed_optim/factorial_v3_worker.py': '272b24b10afd67bf8c85f26e417d3740c299fe873e3b2c29a8ffc485fea934d5',
}


def helpers():
    import hashlib
    with CAL_ENTRY.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    if value != CAL_SOURCE_SHA or any(p.is_symlink() for p in (CAL_ENTRY, *CAL_ENTRY.parents)):
        raise ValueError('The accepted calibration helper changed')
    spec = importlib.util.spec_from_file_location('accepted_calibration_operations', CAL_ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c = helpers()
need, identity, read, digest = c.need, c.identity, c.read, c.digest
SOURCE_ROOT = c.SOURCE_ROOT


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def source_files():
    files, _ = c.sources()
    files = dict(files)
    for name, sha in ADDED.items():
        item = identity(SOURCE_ROOT / name)
        need(item['sha256'] == sha, 'Accepted native worker closure changed')
        files[name] = item
    need(len(files) == 70, 'Need the complete 70-file execution closure')
    return files


def context(files):
    from embed_optim import factorial_v3_calibration as calibration
    from embed_optim import factorial_v3_worker as native
    from embed_optim.factorial_v3_inputs import Locations
    locations = Locations(c.STORY, c.PRIMARY, EXPERIMENT,
        Path('/root/embedding-optimizer-study/data'),
        c.STORY / 'reports/engineering-archive/dense-v3-factorial-inputs-v1')
    need(native.require_sources(locations, SOURCE_ROOT, files['src/embed_optim/factorial_v3_worker.py']) == files,
         'Native source admission disagrees with the outer closure')
    c.check_imports(files)
    return calibration, native, locations


def cells():
    return [(state, operator, seed) for seed in SEEDS for state in STATES for operator in OPERATORS]


def queues():
    result = {'a': [], 'b': []}
    for i, (state, operator, seed) in enumerate(cells()):
        # Both pools receive three of each reset operator and both source states;
        # alternate the diagonal assignment across the three order seeds.
        diagonal = i % 4 in (0, 3)
        pool = 'a' if diagonal != ((i // 4) % 2 == 1) else 'b'
        result[pool].append(f'factorial-v3-{state}-{operator}-seed{seed}')
    return result


def namespace():
    need(HERE == DEPLOY, 'Use only the new explicit training deployment')
    for path in (HERE, OUTPUT):
        need(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlinked execution namespace')


def owner(value, source_sha):
    need(value == {
        'scope': SCOPE, 'owner_message': OWNER_MESSAGE, 'automatic_continuation': False,
        'source_sha256': source_sha, 'protected_helper_access': False,
        'old_controller_transition': False, 'native_numerical_sources_changed': False,
        'full_horizon_50k_cells': 12, 'committed_source_release': False,
        'boundary': 'Explicit owner authority for the new source-bound four-GPU execution. Both initial full branches also provide genuine production-topology verification; no completed branch or GPU save/readback is asserted before it happens. All failed attempts and pending scientific/recovery/publication gates stay distinct.'
    }, 'Owner-scoped fixed-horizon execution approval differs')


def complete_calibration(binding):
    path = CAL_ENTRY.parent / 'run/completed.json'
    need(identity(path) == binding, 'Externally bound completed calibration differs')
    completed = read(path)
    need(completed['source_sha256'] == CAL_SOURCE_SHA
         and completed['authorization_sha256'] == CAL_AUTH_SHA
         and completed['actual_gpu_worker_exits'] == dict.fromkeys(STATES, 0)
         and completed['fresh_process_native_readbacks'] == 2
         and completed['formal_branches_trained'] == 0
         and completed['scientific_completion'] is False, 'No genuine complete calibration')
    # Reauthenticate the actual GPU+fresh-reader chain, not only its summary.
    for state in STATES:
        produced = read(CAL_ENTRY.parent / 'run' / f'{state}.produced.json')
        verified = read(CAL_ENTRY.parent / 'run' / f'{state}.verified.json')
        exited = read(CAL_ENTRY.parent / 'run' / f'{state}.exited.json')
        need(produced['calibration'] == verified['calibration']
             and produced['calibration_binding'] == verified['calibration_binding']
             == completed['calibrations'][state]['calibration_binding']
             and exited['exit_code'] == 0 and exited['authorization_sha256'] == CAL_AUTH_SHA,
             'Calibration readback or actual worker exit differs')
    return completed


def make_requests(native, locations, calibrations, files):
    result = {}
    for state, operator, seed in cells():
        run_id = f'factorial-v3-{state}-{operator}-seed{seed}'
        declared = native.request(locations, calibrations, state, operator, seed,
            SOURCE_ROOT, OUTPUT, HERE / 'native-records' / run_id,
            project=PROJECT, entity=ENTITY,
            expected_worker_source=files['src/embed_optim/factorial_v3_worker.py'])
        result[run_id] = declared
    return result


def prepare(args):
    namespace()
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU preparation must hide CUDA')
    need(identity(__file__)['sha256'] == args.source_sha, 'Entry source differs')
    need(not any(p.exists() or p.is_symlink() for p in
         (HERE / 'authorization.json', HERE / 'run', HERE / 'native-records', OUTPUT)),
         'Preserve every existing training attempt')
    owner(read(HERE / 'owner-approval.json', args.approval_sha), args.source_sha)
    tests = read(HERE / 'tests.json', args.tests_sha)
    need(tests['source_sha256'] == args.source_sha and tests['tests_run'] >= 12
         and tests['errors'] == tests['failures'] == tests['skipped'] == 0
         and tests['test_source'] == identity(HERE / 'test_dispatch.py'), 'Launch checks incomplete')
    files = source_files()
    completed_binding = identity(CAL_ENTRY.parent / 'run/completed.json')
    need(completed_binding['sha256'] == args.calibration_complete_sha, 'Unbound calibration completion')
    completed = complete_calibration(completed_binding)
    calibration, native, locations = context(files)
    native_cells = calibration.load_calibrated_cells(locations, completed['calibrations'])
    need([(x['state'], x['operator'], x['seed']) for x in native_cells] == cells(),
         'Native calibrated design changed')
    geometry = c.imported('factorial_training_actual_main_admission', c.GEOMETRY)
    main, child = geometry.read_primary_admission(c.PINS[c.GEOMETRY])
    original = read(c.ORIGINAL_INPUTS, c.PINS[c.ORIGINAL_INPUTS])
    need(main['rows'] == original['actual_completion']['rows'] and len(main['rows']) == 12,
         'Actual complete primary proofs changed')
    from embed_optim.runtime import verify_runtime_spec
    runtime = verify_runtime_spec(SOURCE_ROOT / 'configs/formal_runtime.json')
    requests = make_requests(native, locations, completed['calibrations'], files)
    authority = {
        'scope': SCOPE, 'prepared_at_utc': now(), 'source_sha256': args.source_sha,
        'test_source': tests['test_source'], 'tests': identity(HERE / 'tests.json'),
        'owner_approval': identity(HERE / 'owner-approval.json'),
        'calibration_entry': identity(CAL_ENTRY), 'calibration_completion': completed_binding,
        'calibrations': completed['calibrations'], 'native_calibrated_cells': native_cells,
        'source_root': str(SOURCE_ROOT), 'source_files': files, 'output_root': str(OUTPUT),
        'gpu_pools': POOLS, 'queues': queues(), 'requests': requests, 'runtime': runtime,
        'primary_completion': {'rows_sha256': digest(main['rows']), 'runs': 12, 'reader': child,
            'ten_observed_zero_two_unobserved_exits_preserved': True},
        'execution_authorized': True, 'both_original_lease_namespaces': True,
        'old_controller_transition': False, 'native_guards_modified': False,
        'committed_source_release': False, 'scientific_completion': False,
        'real_gpu_admission': 'Each branch requires four inherited dual-namespace leases, exact four-rank NCCL all-reduce, full runtime/source identity and native model/tensor/recipe admission. First full branches are also actual topology/save/readback verification. No mock acceptance replaces them.',
        'failure_policy': 'Stop this pool on any child/reader failure; preserve every output; terminate only direct owned siblings if a rank fails; no automatic restart or resume.'}
    c.check_imports(files)
    need(source_files() == files, 'Source changed during actual preparation')
    write_new(HERE / 'authorization.json', authority)
    return {'authorization': identity(HERE / 'authorization.json'), 'actual_calibrated_cells': 12,
            'actual_primary_proofs': 12, 'formal_branches_trained': 0}


def authenticate(args, gpu=False):
    namespace()
    need(identity(__file__)['sha256'] == args.source_sha, 'Training entry changed')
    auth = read(HERE / 'authorization.json', args.authorization_sha)
    need(auth['scope'] == SCOPE and auth['source_sha256'] == args.source_sha
         and auth['gpu_pools'] == POOLS and auth['queues'] == queues()
         and auth['source_root'] == str(SOURCE_ROOT) and auth['output_root'] == str(OUTPUT)
         and auth['execution_authorized'] is True and auth['both_original_lease_namespaces'] is True
         and all(auth[k] is False for k in ('old_controller_transition', 'native_guards_modified',
                                          'committed_source_release', 'scientific_completion')),
         'Training authority changed')
    for name, field in (('owner-approval.json', 'owner_approval'), ('tests.json', 'tests'),
                        ('test_dispatch.py', 'test_source')):
        need(identity(HERE / name) == auth[field], 'Approval or tested entry changed')
    owner(read(HERE / 'owner-approval.json'), args.source_sha)
    need(source_files() == auth['source_files'] and identity(CAL_ENTRY) == auth['calibration_entry'],
         'Admitted source closure changed')
    completed = complete_calibration(auth['calibration_completion'])
    need(completed['calibrations'] == auth['calibrations'], 'Completed calibration identity changed')
    parent = c.imported('factorial_training_original_leases', c.PARENT)
    if gpu:
        need(args.worker in auth['queues'][args.pool], 'Worker is not in this fixed pool queue')
        need(os.environ.get('CUDA_VISIBLE_DEVICES') == ','.join(POOLS[args.pool]), 'Wrong GPU pool')
        verify_leases(parent, args.pool, args.lease_fd)
    else:
        need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU entry must hide CUDA')
    return auth, parent


def verify_leases(parent, pool, descriptors):
    need(len(descriptors) == len(set(descriptors)) == 8
         and all(type(fd) is int and fd >= 0 for fd in descriptors), 'Four GPUs require eight distinct descriptors')
    validation = c.imported('factorial_training_original_inherited_checker', c.VALIDATION)
    for index, token in enumerate(POOLS[pool]):
        validation.require_inherited_leases([descriptors[index], descriptors[index + 4]],
            [root / f'gpu-{token}.lock' for root in parent.LEASE_ROOTS])


def environment(parent, *, pool=None, rank=None, run_id=None, port=None):
    env = parent.environment(POOLS[pool] if pool is not None else None)
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE', 'TORCHELASTIC_RUN_ID',
                'TORCHELASTIC_RESTART_COUNT', 'TORCHELASTIC_MAX_RESTARTS'):
        env.pop(key, None)
    env.update(PYTHONPATH=f'{SOURCE_ROOT / "src"}:{SOURCE_ROOT}', PYTHONUNBUFFERED='1')
    if pool is not None:
        need(type(rank) is int and rank in range(4) and run_id in queues()[pool]
             and type(port) is int and 1024 <= port <= 65535, 'Invalid direct four-rank environment')
        env.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='4', LOCAL_WORLD_SIZE='4',
            MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), WANDB_PROJECT=PROJECT, WANDB_ENTITY=ENTITY,
            WANDB_RUN_ID=run_id, WANDB_RESUME='never', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    return env


def command(args, run_id, *, rank=None, descriptors=None, worker_sha=None):
    need(run_id in queues()[args.pool], 'Unknown pool run')
    argv = ['/usr/bin/python', '-B', str(Path(__file__).resolve()), '--source-sha', args.source_sha,
            '--authorization-sha', args.authorization_sha, '--pool', args.pool]
    if worker_sha is not None:
        need(rank is None and descriptors is None, 'CPU reader cannot inherit rank leases')
        return argv + ['--verify', run_id, '--worker-complete-sha', worker_sha]
    need(type(rank) is int and rank in range(4) and descriptors is not None
         and len(descriptors) == len(set(descriptors)) == 8
         and all(type(fd) is int and fd >= 0 for fd in descriptors), 'Incomplete four-rank launch')
    argv += ['--worker', run_id, '--rank', str(rank)]
    for fd in descriptors:
        argv += ['--lease-fd', str(fd)]
    return argv


def call_native(native, locations, declared):
    return native.run_branch(locations, declared['calibrations'], declared['state'],
        declared['operator'], declared['seed'], declared['source_root'], declared['output_root'],
        declared['record_root'], project=declared['project'], entity=declared['entity'],
        expected_worker_source=declared['worker_source'])


def worker(args):
    auth, parent = authenticate(args, gpu=True)
    declared = auth['requests'][args.worker]
    path = HERE / 'run' / f'pool-{args.pool}' / args.worker
    launch = read(path / 'admission.json')
    need(launch['request_sha256'] == digest(declared)
         and launch['authorization_sha256'] == args.authorization_sha
         and launch['source_sha256'] == args.source_sha and launch['gpu_pool'] == POOLS[args.pool],
         'Rank admission record differs')
    need(os.getppid() == launch['coordinator_pid'], 'Rank did not start under its declared direct parent')
    expected_env = {'RANK': str(args.rank), 'LOCAL_RANK': str(args.rank), 'WORLD_SIZE': '4',
                    'LOCAL_WORLD_SIZE': '4', 'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(launch['port'])}
    need(all(os.environ.get(k) == v for k, v in expected_env.items()), 'Distributed rank identity differs')
    parent.handoff()
    from embed_optim.runtime import verify_runtime_spec
    runtime = verify_runtime_spec(SOURCE_ROOT / 'configs/formal_runtime.json')
    need(runtime == auth['runtime'], 'Admitted runtime changed')
    import torch
    import torch.distributed as dist
    need(not dist.is_initialized() and not torch.cuda.is_initialized(), 'Rank inherited initialized execution state')
    torch.cuda.set_device(args.rank)
    dist.init_process_group('nccl', init_method='env://', rank=args.rank, world_size=4,
                            timeout=timedelta(minutes=3))
    try:
        actual = torch.tensor([float(args.rank)], device='cuda', dtype=torch.float32)
        dist.all_reduce(actual)
        need(actual.item() == 6.0 and torch.cuda.device_count() == 4, 'Four-rank GPU collective admission failed')
        calibration, native, locations = context(auth['source_files'])
        need(make_requests(native, locations, auth['calibrations'], auth['source_files']) == auth['requests'],
             'Native structural requests changed')
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        write_new(path / f'rank-{args.rank}.gpu-admitted.json', {
            'admitted_at_utc': now(), 'rank': args.rank, 'pid': os.getpid(), 'ppid': os.getppid(),
            'source_sha256': args.source_sha, 'authorization_sha256': args.authorization_sha,
            'request_sha256': digest(declared), 'backend': dist.get_backend(), 'world_size': 4,
            'visible_devices': 4, 'current_device': torch.cuda.current_device(),
            'gpu_name': props.name, 'gpu_memory_bytes': props.total_memory,
            'actual_all_reduce_sum': actual.item(), 'eight_original_leases_inherited': True,
            'runtime': runtime, 'model_training_completed': False, 'scientific_completion': False})
        binding = call_native(native, locations, declared)
        c.check_imports(auth['source_files'])
        need(source_files() == auth['source_files'], 'Execution source changed during training')
        write_new(path / f'rank-{args.rank}.returned.json', {
            'returned_at_utc': now(), 'rank': args.rank, 'run_id': args.worker,
            'source_sha256': args.source_sha, 'authorization_sha256': args.authorization_sha,
            'worker_completion': binding, 'process_exit_observed': False, 'scientific_completion': False})
        return {'rank': args.rank, 'run_id': args.worker, 'worker_returned': True}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def verify(args):
    auth, _ = authenticate(args)
    need(args.verify in auth['queues'][args.pool], 'CPU reader is outside the admitted pool')
    _, native, locations = context(auth['source_files'])
    declared = auth['requests'][args.verify]
    binding = identity(Path(declared['record_root']) / native.COMPLETE)
    need(binding['sha256'] == args.worker_complete_sha, 'Unbound worker completion')
    checked = native.read_execution(locations, declared, binding)
    need(checked['worker_and_native_artifacts_verified'] is True, 'Incomplete native whole-run read')
    c.check_imports(auth['source_files'])
    need(source_files() == auth['source_files'], 'Readback source changed')
    write_new(HERE / 'run' / f'pool-{args.pool}' / args.verify / 'fresh-native-readback.json', {
        'verified_at_utc': now(), 'run_id': args.verify, 'source_sha256': args.source_sha,
        'authorization_sha256': args.authorization_sha, 'worker_completion': binding,
        'fresh_process_native_readback': True, 'native': checked, 'scientific_completion': False})
    return {'run_id': args.verify, 'fresh_native_readback': True, 'worker_completion': binding}


def supervise(children):
    """Wait for direct children only; fail closed and reap all owned siblings."""
    try:
        while True:
            codes = [child.poll() for child in children]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError('A direct owned rank failed')
            if all(code is not None for code in codes):
                return codes
            time.sleep(1)
    except BaseException:
        finish_owned(children)
        raise


def finish_owned(children):
    # Popen retains each exact unreaped direct child; no PID search or process-group signal.
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if child.poll() is None:
                child.kill()
            child.wait()


def launch_one(args, auth, parent, run_id, pool_root):
    dispatch = c.imported('factorial_training_exact_owned_identity', c.DISPATCH)
    job = pool_root / run_id
    need(not job.exists() and not Path(auth['requests'][run_id]['run_root']).exists()
         and not Path(auth['requests'][run_id]['record_root']).exists(), 'Never overwrite an existing run/attempt')
    job.mkdir()
    children, codes = [], None
    try:
        with ExitStack() as stack:
            parent.handoff()
            descriptors = stack.enter_context(parent.leases(POOLS[args.pool]))
            with socket.socket() as rendezvous:
                rendezvous.bind(('127.0.0.1', 0))
                port = rendezvous.getsockname()[1]
            write_new(job / 'admission.json', {
                'admitted_at_utc': now(), 'source_sha256': args.source_sha,
                'authorization_sha256': args.authorization_sha, 'run_id': run_id,
                'request_sha256': digest(auth['requests'][run_id]), 'gpu_pool': POOLS[args.pool],
                'port': port, 'coordinator_pid': os.getpid(), 'both_original_namespaces': True,
                'preserved_old_handoff': parent.handoff(), 'scientific_completion': False})
            try:
                for rank in range(4):
                    argv = command(args, run_id, rank=rank, descriptors=descriptors)
                    log = stack.enter_context((job / f'rank-{rank}.log').open('xb'))
                    child = subprocess.Popen(argv, cwd=SOURCE_ROOT,
                        env=environment(parent, pool=args.pool, rank=rank, run_id=run_id, port=port),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, pass_fds=descriptors)
                    children.append(child)  # Retain ownership before any fallible identity write.
                    write_new(job / f'rank-{rank}.started.json', {
                        **dispatch.process_identity(child.pid), 'rank': rank, 'command': argv,
                        'started_at_utc': now(), 'source_sha256': args.source_sha,
                        'authorization_sha256': args.authorization_sha, 'coordinator_pid': os.getpid()})
                print(json.dumps({'event': 'four_rank_training_launched', 'run_id': run_id,
                    'pids': [child.pid for child in children], 'gpu_pool': POOLS[args.pool]}), flush=True)
                codes = supervise(children)
            finally:
                if any(child.poll() is None for child in children):
                    finish_owned(children)
                write_new(job / 'ranks.exited.json', {'observed_at_utc': now(),
                    'source_sha256': args.source_sha, 'authorization_sha256': args.authorization_sha,
                    'actual_rank_exits': {str(i): child.returncode for i, child in enumerate(children)}})
        need(codes == [0, 0, 0, 0], 'Not four successful rank exits')
        returned = [read(job / f'rank-{rank}.returned.json') for rank in range(4)]
        binding = returned[0]['worker_completion']
        need(all(row['worker_completion'] == binding and row['rank'] == rank
                 and row['run_id'] == run_id and row['source_sha256'] == args.source_sha
                 and row['authorization_sha256'] == args.authorization_sha
                 for rank, row in enumerate(returned)), 'Rank completion bindings disagree')
        argv = command(args, run_id, worker_sha=binding['sha256'])
        with (job / 'fresh-native-readback.log').open('xb') as log:
            reader = subprocess.Popen(argv, cwd=SOURCE_ROOT, env=environment(parent),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
            try:
                write_new(job / 'reader.started.json', {
                    **dispatch.process_identity(reader.pid), 'command': argv, 'started_at_utc': now(),
                    'source_sha256': args.source_sha, 'authorization_sha256': args.authorization_sha})
                code = reader.wait()
            finally:
                if reader.poll() is None:
                    finish_owned([reader])
                write_new(job / 'reader.exited.json', {'observed_at_utc': now(), 'exit_code': reader.returncode})
        need(code == 0, 'Actual fresh CPU native reader failed')
        verified = read(job / 'fresh-native-readback.json')
        need(verified['worker_completion'] == binding and verified['fresh_process_native_readback'] is True,
             'Fresh native reader changed the completion')
        parent.handoff()
        completed = {'completed_at_utc': now(), 'run_id': run_id, 'actual_rank_exits': codes,
            'actual_fresh_reader_exit': code, 'worker_completion': binding,
            'native_readback_binding': identity(job / 'fresh-native-readback.json'),
            'source_sha256': args.source_sha, 'authorization_sha256': args.authorization_sha,
            'full_horizon_391_steps_verified': True, 'all_five_checkpoints_verified': True,
            'retrieval_outcomes_complete': False, 'gpu_resume_equivalence_claimed': False,
            'scientific_completion': False}
        write_new(job / 'completed.json', completed)
        return completed
    except BaseException as exc:
        if any(child.poll() is None for child in children):
            finish_owned(children)
        write_new(job / 'failed.json', {'failed_at_utc': now(), 'run_id': run_id,
            'exception_type': type(exc).__name__, 'automatic_retry': False,
            'partial_outputs_preserved': True, 'scientific_completion': False})
        raise


def coordinate(args):
    auth, parent = authenticate(args)
    root = HERE / 'run' / f'pool-{args.pool}'
    need(not root.exists(), 'Never restart an existing pool')
    parent.handoff()
    root.mkdir(parents=True)
    dispatch = c.imported('factorial_training_exact_self_identity', c.DISPATCH)
    write_new(root / 'coordinator.started.json', {
        **dispatch.process_identity(os.getpid()), 'started_at_utc': now(), 'pool': args.pool,
        'source_sha256': args.source_sha, 'authorization_sha256': args.authorization_sha,
        'gpu_pool': POOLS[args.pool], 'queue': auth['queues'][args.pool]})
    completed = []
    try:
        for run_id in auth['queues'][args.pool]:
            authenticate(args)
            completed.append(launch_one(args, auth, parent, run_id, root))
        write_new(root / 'coordinator.completed.json', {
            'completed_at_utc': now(), 'pool': args.pool, 'full_branches': len(completed),
            'source_sha256': args.source_sha, 'authorization_sha256': args.authorization_sha,
            'runs': completed, 'scientific_completion': False})
        return {'pool': args.pool, 'complete_full_horizon_branches': len(completed)}
    except BaseException as exc:
        write_new(root / 'coordinator.failed.json', {'failed_at_utc': now(), 'pool': args.pool,
            'exception_type': type(exc).__name__, 'completed_prefix': [r['run_id'] for r in completed],
            'automatic_retry': False, 'scientific_completion': False})
        raise


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-sha', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--coordinate', action='store_true')
    mode.add_argument('--worker')
    mode.add_argument('--verify')
    parser.add_argument('--pool', choices=tuple(POOLS))
    parser.add_argument('--authorization-sha')
    parser.add_argument('--approval-sha')
    parser.add_argument('--tests-sha')
    parser.add_argument('--calibration-complete-sha')
    parser.add_argument('--worker-complete-sha')
    parser.add_argument('--rank', type=int)
    parser.add_argument('--lease-fd', type=int, action='append', default=[])
    args = parser.parse_args(argv)
    if args.prepare:
        need(all((args.approval_sha, args.tests_sha, args.calibration_complete_sha))
             and args.authorization_sha is None and args.pool is None and args.rank is None
             and not args.lease_fd and args.worker_complete_sha is None, 'Invalid preparation arguments')
    else:
        need(args.authorization_sha is not None and args.pool in POOLS
             and not any((args.approval_sha, args.tests_sha, args.calibration_complete_sha)),
             'Execution requires its exact preparation authority and pool')
        if args.worker:
            need(args.worker in queues()[args.pool] and type(args.rank) is int and args.rank in range(4)
                 and len(args.lease_fd) == len(set(args.lease_fd)) == 8
                 and all(fd >= 0 for fd in args.lease_fd) and args.worker_complete_sha is None,
                 'Invalid four-rank worker arguments')
        else:
            need(args.rank is None and not args.lease_fd, 'CPU entry cannot inherit GPU rank descriptors')
            need(bool(args.worker_complete_sha) == bool(args.verify), 'Reader requires exact worker completion')
            if args.verify:
                need(args.verify in queues()[args.pool], 'Reader outside pool queue')
    return args


if __name__ == '__main__':
    arguments = parse()
    action = prepare if arguments.prepare else coordinate if arguments.coordinate else worker if arguments.worker else verify
    print(json.dumps(action(arguments), sort_keys=True), flush=True)
