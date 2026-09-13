"""Wait for complete genuine continuation pools, then run unchanged full BEIR.

New operational identity/namespace only. The original historical factorial
controller and admission guards are neither called with false identities nor
modified. One direct worker per released GPU inherits both original leases.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent.resolve()
EXP = Path('/root/embedding-optimizer-v3-experiment')
PRIMARY = Path('/root/embedding-optimizer-primary-v3')
STORY = Path('/root/embedding-optimizer-story-refactor')
NATIVE_ROOT = STORY / 'reports/engineering-archive/dense-v3-factorial-worker-v1/actual/source-final'
TRAIN = EXP / 'launch/factorial-training-v1'
RESULTS = EXP / 'evaluations/dense-v3-state-operator-v1'
READINESS = EXP / 'launch/observations/evaluation-readiness-before-final-pair-completion-20260909.json'
OLD_EVAL = EXP / 'launch/evaluation-handoff/dispatch.py'
PARENT = EXP / 'launch/train_matrix.py'
TRAIN_SHA = 'd11d3281523b1573ce7fcd650e10c538cfa84022b5db92117559d47213e57e53'
AUTH_SHA = '00c52f0bec979fb163a734ad0f9eadcd6ea09f7393133fd35c65bc62878ce11f'
READER_SHA = '625f0d5c641495304d241136510f519545833616f6287cf2b417b0dfe08f5b9f'
SCOPE = 'owner-authorized-genuine-v3-factorial-final-beir-v1'
OWNER = '你有权做一切事情，目标是尽快完成任务'
POOLS = {'a': ['4', '5', '6', '7'], 'b': ['0', '1', '2', '3']}
PINS = {
    PARENT: 'dd823408de44d0696d0cac3a9d865ffc41936abdf2542ae815c3f5eda8dffcf3',
    OLD_EVAL: '5b8a89bed5e0eaa02a12585ee3f6c883ca3e162b550898d68322321ee41d8427',
    READINESS: 'ed673fb5b09277aba0d9e00ec951092d1e369e88b2e47769ba217d6e29135ebc',
    TRAIN / 'factorial_dispatch.py': TRAIN_SHA,
    TRAIN / 'authorization.json': AUTH_SHA,
    PRIMARY / 'source-assembly.json': 'e603ef175c0f691a1dd83c4d8b474d90da2fc67c8c65f5a77f3c3180e78332b8',
    PRIMARY / 'configs/dense_primary_v3_protocol.json': '4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b',
    PRIMARY / 'configs/formal_runtime.json': 'b90de16b3796bb8e9d0badcedec688da55a037d46244a0d2a7f5838795cfe77a',
    PRIMARY / 'src/embed_optim/primary_completion.py': 'a1aa61058654bd46afb41710ed606992355dab95d7c8b171a7485a8ac24d0cd5',
    PRIMARY / 'src/embed_optim/primary_contract.py': 'e49f55d5d01408de001b8e1d56244fbcaf189de45c0209caffa0b4e34f828e19',
    PRIMARY / 'src/embed_optim/evaluation_utils.py': '529436e73759051b5c3000579fe9d66ce12c4444132eca4e7a26827054683e53',
    PRIMARY / 'src/embed_optim/corrected_input_execution.py': 'ae3a7df56165a288a679760b3de34723b08de74a050775acfa1df3a51d2e48a0',
    STORY / 'configs/dense_no_packing_state_operator_factorial_protocol.json': '5773943a3ae9b581021a0f7b85b162c74d5c497eeca1386578ff9d2c3bcafe76',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Nonordinary evidence')
    first = path.stat()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    last = path.stat()
    need(all(getattr(first, k) == getattr(last, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')),
         'Evidence changed during read')
    return dict(bytes=last.st_size, sha256=digest)


def bound(path, expected):
    got = identity(path)
    need(got['sha256'] == expected if isinstance(expected, str) else got == expected, 'External identity differs: ' + str(path))
    return got


def read(path, expected=None):
    before = identity(path) if expected is None else bound(path, expected)

    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result

    result = json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
        parse_constant=lambda _: need(False, 'Nonfinite JSON'))
    need(identity(path) == before, 'JSON changed during read')
    return result


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def queues():
    result = {'a': [], 'b': []}
    for j, seed in enumerate((314159, 271828, 161803)):
        for s, state in enumerate(('adamw_state', 'muon_state')):
            for o, operator in enumerate(('adamw', 'muon')):
                pool = 'a' if (s == o) != (j % 2 == 1) else 'b'
                result[pool].append(f'factorial-v3-{state}-{operator}-seed{seed}')
    return result


def jobs(runs, tasks, sizes):
    need(len(runs) == len(set(runs)) == 6 and len(tasks) == len(set(tasks)) == 14 and set(tasks) == set(sizes),
         'Require six complete runs and all fourteen tasks')
    result = [(run, 391, task) for task in sorted(tasks, key=lambda t: (-sizes[t], t)) for run in runs]
    pilot = (runs[0], 391, 'SciFact')
    need(pilot in result, 'Missing fixed real pilot')
    result.remove(pilot)
    result.insert(0, pilot)
    need(len(result) == len(set(result)) == 84, 'Incomplete final-checkpoint pool')
    return result


def require_pool_completion(value, pool):
    need(value['pool'] == pool and value['source_sha256'] == TRAIN_SHA
         and value['authorization_sha256'] == AUTH_SHA and value['full_branches'] == 6
         and [r['run_id'] for r in value['runs']] == queues()[pool], 'Wrong complete training pool')
    for row in value['runs']:
        need(row['source_sha256'] == TRAIN_SHA and row['authorization_sha256'] == AUTH_SHA
             and row['actual_rank_exits'] == [0, 0, 0, 0] and row['actual_fresh_reader_exit'] == 0
             and row['all_five_checkpoints_verified'] is True and row['full_horizon_391_steps_verified'] is True,
             'Incomplete or failed training branch')


def source_context():
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Coordinator must hide CUDA')
    for path, digest in PINS.items():
        bound(path, digest)
    bound(HERE / 'read_native.py', READER_SHA)
    sources = read(READINESS)['source_binding']['sources']
    for path, binding in sources.items():
        bound(path, binding)
    assembly = read(PRIMARY / 'source-assembly.json')
    need(len(assembly['files']) == 56, 'Original primary evaluation assembly differs')
    for name, item in assembly['files'].items():
        bound(PRIMARY / name, item['identity'])
    auth = read(TRAIN / 'authorization.json', AUTH_SHA)
    need(auth['queues'] == queues() and auth['gpu_pools'] == POOLS, 'Actual training queues changed')
    for name, binding in auth['source_files'].items():
        bound(NATIVE_ROOT / name, binding)
    parent = load('_original_owned_gpu_leases', PARENT)
    old = load('_original_pinned_beir_worker_command', OLD_EVAL)
    from embed_optim.runtime import verify_runtime_spec
    runtime = verify_runtime_spec(PRIMARY / 'configs/formal_runtime.json')
    from embed_optim.decontamination import DECONTAMINATED_BEIR, DECONTAMINATED_CORPUS_SIZES
    protocol = read(PRIMARY / 'configs/dense_primary_v3_protocol.json')
    tasks = protocol['evaluation']['tasks']
    need(set(tasks) == set(DECONTAMINATED_BEIR) and protocol['beir_task_revisions'] ==
         {k: dict(repo=v[0], revision=v[1]) for k, v in DECONTAMINATED_BEIR.items()}, 'Pinned task definitions changed')
    for name, module in tuple(sys.modules.items()):
        if name == 'embed_optim' or name.startswith('embed_optim.'):
            need(Path(module.__file__).resolve().is_relative_to(PRIMARY), 'Foreign evaluation package import')
    import torch
    need(not torch.cuda.is_initialized(), 'Coordinator initialized CUDA')
    return parent, old, auth, protocol, runtime, dict(DECONTAMINATED_CORPUS_SIZES)


def prepare(args):
    bound(__file__, args.source_sha256)
    need(not (HERE / 'authorization.json').exists() and not (HERE / 'run').exists() and not RESULTS.exists(),
         'Preserve all prior evaluation attempts')
    parent, old, training, protocol, runtime, sizes = source_context()
    tests = read(HERE / 'tests.json', args.tests_sha256)
    need(tests['source'] == identity(__file__) and tests['reader_source'] == identity(HERE / 'read_native.py')
         and tests['tests_run'] >= 12 and tests['failures'] == tests['errors'] == tests['skipped'] == 0,
         'Missing bounded operational checks')
    probe = read(HERE / 'actual-first-native-read.json', args.native_probe_sha256)
    need(probe['reader_source'] == identity(HERE / 'read_native.py') and probe['actual_rank_exits'] == [0, 0, 0, 0]
         and probe['native_readback']['worker_and_native_artifacts_verified'] is True
         and probe['training_authorization_sha256'] == AUTH_SHA, 'Missing genuine native input test')
    tasks = protocol['evaluation']['tasks']
    record = dict(scope=SCOPE, created_at_utc=now(), owner_message=OWNER, automatic_continuation=False,
        source=identity(__file__), reader_source=identity(HERE / 'read_native.py'),
        tests=identity(HERE / 'tests.json'), test_source=identity(HERE / 'test_evaluate.py'),
        actual_native_probe=identity(HERE / 'actual-first-native-read.json'),
        training_authorization_sha256=AUTH_SHA, source_pins={str(k): v for k, v in PINS.items()},
        original_evaluation_sources=read(READINESS)['source_binding']['sources'],
        queues=queues(), gpu_pools=POOLS, tasks=tasks, task_revisions=protocol['beir_task_revisions'],
        runtime=runtime, results_root=str(RESULTS), jobs={p: jobs(r, tasks, sizes) for p, r in queues().items()},
        final_step=391, total_tasks=168, wait_for_complete_six_run_pool=True,
        real_pilot_required=True, one_task_per_gpu=True, both_original_lease_namespaces=True,
        no_lease_before_pool_completion=True, no_automatic_retry=True, scientific_estimands_changed=False,
        original_numerical_workers_changed=False, old_controller_transition=False,
        committed_source_release=False, scientific_completion=False)
    write(HERE / 'authorization.json', record)
    return dict(authorization=identity(HERE / 'authorization.json'), actual_native_input_verified=True,
                queued_tasks=168, gpu_jobs_launched=0, scientific_completion=False)


def authorized(args):
    bound(__file__, args.source_sha256)
    auth = read(HERE / 'authorization.json', args.authorization_sha256)
    need(auth['scope'] == SCOPE and auth['source'] == identity(__file__) and auth['reader_source'] == identity(HERE / 'read_native.py')
         and auth['queues'] == queues() and auth['gpu_pools'] == POOLS and auth['results_root'] == str(RESULTS)
         and auth['owner_message'] == OWNER and auth['no_lease_before_pool_completion'] is True,
         'Changed evaluation authority')
    for name, key in (('tests.json', 'tests'), ('test_evaluate.py', 'test_source'),
                       ('actual-first-native-read.json', 'actual_native_probe')):
        bound(HERE / name, auth[key])
    parent, old, training, protocol, runtime, sizes = source_context()
    need(auth['tasks'] == protocol['evaluation']['tasks'] and auth['runtime'] == runtime
         and auth['task_revisions'] == protocol['beir_task_revisions'], 'Changed evaluation recipe/runtime')
    return auth, parent, old


def rehash_model(admission):
    checkpoint = Path(admission['checkpoint'])
    for row in admission['inference_files']:
        bound(checkpoint / row['path'], {k: row[k] for k in ('bytes', 'sha256')})
    bound(checkpoint / 'factorial_trainer_component.json', admission['component_file'])


def native_read(parent, old, pool, run, output):
    env = parent.environment()
    env['PYTHONPATH'] = f'{NATIVE_ROOT / "src"}:{NATIVE_ROOT}'
    argv = ['/usr/bin/python', '-B', str(HERE / 'read_native.py'), '--source-sha256', READER_SHA,
            '--pool', pool, '--run-id', run, '--output', str(output)]
    with output.with_suffix('.log').open('x') as log:
        child = subprocess.Popen(argv, cwd=NATIVE_ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        write(output.with_suffix('.started.json'), dict(pid=child.pid, start_ticks=old.process_start_ticks(child.pid),
             command=argv, started_at_utc=now(), gpu_access=False))
        code = child.wait()
    write(output.with_suffix('.exited.json'), dict(exit_code=code, completed_at_utc=now()))
    need(code == 0, 'Actual native continuation reader failed')
    value = read(output)
    need(value['run_id'] == run and value['pool'] == pool and value['reader_source']['sha256'] == READER_SHA,
         'Wrong actual native reader output')
    rehash_model(value)
    return value


def coordinate(args):
    auth, parent, old = authorized(args)
    root = HERE / 'run' / ('pool-' + args.pool)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'jobs').mkdir()
    (root / 'native').mkdir()
    write(root / 'started.json', dict(pid=os.getpid(), start_ticks=old.process_start_ticks(os.getpid()),
        source_sha256=args.source_sha256, authorization_sha256=args.authorization_sha256,
        started_at_utc=now(), gpu_tokens=POOLS[args.pool], expected_tasks=84, scientific_completion=False))
    pool_root = TRAIN / 'run' / ('pool-' + args.pool)
    while True:
        need(not (pool_root / 'coordinator.failed.json').exists(), 'Training pool failed; no lease requested')
        path = pool_root / 'coordinator.completed.json'
        if path.exists():
            try:
                done = read(path)
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            require_pool_completion(done, args.pool)
            for row in done['runs']:
                need(read(pool_root / row['run_id'] / 'completed.json') == row, 'Pool/branch completion differs')
            break
        parent.handoff()
        time.sleep(30)
    auth, parent, old = authorized(args)
    write(root / 'training-pool-admitted.json', dict(training_completion=identity(path), admitted_at_utc=now(),
          all_six_branches_complete=True, gpu_leases_requested=False))
    admissions = {run: native_read(parent, old, args.pool, run, root / 'native' / (run + '.json'))
                  for run in queues()[args.pool]}
    cache_roots = {}
    for run, value in admissions.items():
        target = RESULTS / run / 'checkpoint-391'
        target.mkdir(parents=True, exist_ok=False)
        write(target / 'evaluation-admission.json', dict(run_id=run, checkpoint=value['checkpoint'],
              native_readback=identity(root / 'native' / (run + '.json')), tasks=auth['tasks'],
              task_revisions=auth['task_revisions'], runtime=auth['runtime'],
              source_sha256=args.source_sha256, authorization_sha256=args.authorization_sha256,
              native_historical_factorial_guard_passed=False, committed_source_release=False, scientific_completion=False))
        cache_roots[run] = target
    pending = [tuple(row) for row in auth['jobs'][args.pool]]
    pilot = pending[0]
    running, finished, failures = {}, set(), []
    pilot_passed = False
    from embed_optim.primary_completion import task_score
    versions = read(PRIMARY / 'configs/formal_runtime.json')['packages']

    def score(run, task):
        paths = list((cache_roots[run] / 'dense').rglob(task + 'Decontaminated.json'))
        need(len(paths) == 1, 'Missing or duplicate actual task result')
        return task_score(paths[0], task, run, 391, 8192, auth['task_revisions'][task]['revision'], versions)

    try:
        while pending or running:
            for token, item in list(running.items()):
                child = item['child']
                if child.poll() is None:
                    continue
                item['log'].close()
                item['lease'].__exit__(None, None, None)
                del running[token]
                write(item['prefix'].with_suffix('.exited.json'), dict(job=item['job'], pid=child.pid,
                    exit_code=child.returncode, elapsed_seconds=time.monotonic() - item['clock'], completed_at_utc=now()))
                if child.returncode:
                    failures.append(dict(job=item['job'], exit_code=child.returncode))
                    continue
                run, step, task = item['job']
                try:
                    row = score(run, task)
                    write(item['prefix'].with_suffix('.task-verified.json'), row)
                    finished.add(item['job'])
                    if item['job'] == pilot:
                        pilot_passed = True
                except Exception as exc:
                    failures.append(dict(job=item['job'], exception_type=type(exc).__name__, reason='Native task result admission failed'))
                print(json.dumps(dict(event='task_returned', job=item['job'], finished=len(finished), failures=len(failures))), flush=True)
            if failures:
                if running:
                    time.sleep(1)
                    continue
                write(root / 'failed.json', dict(failures=failures, pending=pending, finished=len(finished), scientific_completion=False))
                raise RuntimeError('Factorial evaluation failed; no automatic retry')
            for token in POOLS[args.pool]:
                if not pending or token in running or (not pilot_passed and running):
                    continue
                bound(__file__, args.source_sha256)
                bound(HERE / 'authorization.json', args.authorization_sha256)
                for path, digest in PINS.items():
                    bound(path, digest)
                for path, binding in auth['original_evaluation_sources'].items():
                    bound(path, binding)
                parent.handoff()
                lease = parent.leases([token])
                try:
                    descriptors = lease.__enter__()
                except BlockingIOError:
                    continue
                log = None
                try:
                    job = pending[0]
                    run, step, task = job
                    rehash_model(admissions[run])
                    prefix = root / 'jobs' / f'{run}-{step}-{task}'
                    argv = old.worker_command(admissions[run]['checkpoint'], dict(results_root=str(cache_roots[run])), task)
                    log = prefix.with_suffix('.log').open('x')
                    child = subprocess.Popen(argv, cwd=PRIMARY, env=old.evaluation_environment(parent, token),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, pass_fds=descriptors)
                    running[token] = dict(child=child, log=log, lease=lease, job=job, prefix=prefix, clock=time.monotonic())
                    pending.pop(0)
                    write(prefix.with_suffix('.started.json'), dict(job=job, pid=child.pid, start_ticks=old.process_start_ticks(child.pid),
                        command=argv, gpu_token=token, both_original_leases_inherited=True, started_at_utc=now()))
                    print(json.dumps(dict(event='task_started', job=job, pid=child.pid, gpu_token=token)), flush=True)
                except BaseException:
                    if token not in running:
                        if log is not None:
                            log.close()
                        lease.__exit__(None, None, None)
                    raise
            time.sleep(1 if running else 30)
        need(len(finished) == 84, 'Incomplete six-run final task grid')
        complete_runs = {}
        for run, admission in admissions.items():
            rehash_model(admission)
            rows = [score(run, task) for task in auth['tasks']]
            need(len(list((cache_roots[run] / 'dense').rglob('*Decontaminated.json'))) == 14, 'Unexpected task result inventory')
            result = dict(run_id=run, step=391, state=admission['state'], operator=admission['operator'], seed=admission['seed'],
                task_count=14, scores=rows, macro_ndcg_at_10=sum(r['ndcg_at_10'] for r in rows) / 14,
                native_input=identity(root / 'native' / (run + '.json')), completed_at_utc=now(), scientific_completion=False)
            write(cache_roots[run] / 'all-fourteen-tasks-verified.json', result)
            complete_runs[run] = identity(cache_roots[run] / 'all-fourteen-tasks-verified.json')
        authorized(args)
        write(root / 'completed.json', dict(completed_at_utc=now(), runs=complete_runs, verified_tasks=84,
             source_sha256=args.source_sha256, authorization_sha256=args.authorization_sha256, scientific_completion=False))
    finally:
        for item in running.values():
            item['log'].close()
            item['lease'].__exit__(None, None, None)  # Inherited child FDs retain their locks.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'coordinate', 'inspect'))
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--authorization-sha256')
    p.add_argument('--tests-sha256')
    p.add_argument('--native-probe-sha256')
    p.add_argument('--pool', choices=('a', 'b'))
    args = p.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(args)), flush=True)
    elif args.action == 'inspect':
        auth, _, _ = authorized(args)
        print(json.dumps(dict(tasks=auth['total_tasks'], pools=auth['gpu_pools'],
            waits_for_entire_training_pool=True, gpu_leases_requested=False, scientific_completion=False)), flush=True)
    else:
        need(args.pool in POOLS, 'Select an existing training pool')
        coordinate(args)


if __name__ == '__main__':
    main()
