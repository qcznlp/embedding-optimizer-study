"""Owner-authorized five-stage genuine continuation probe, separate from old guards.

The encoder/loading reader and the frozen six-metric scorer are unchanged.
Two one-GPU queues wait for their whole original six-run training pools. Every
worker inherits both original leases. No gap between active training cells is
used. The pretrained reference is an authenticated reuse, not another encoding.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
STORY = Path('/root/embedding-optimizer-story-refactor')
EXP = Path('/root/embedding-optimizer-v3-experiment')
REPORT = STORY / 'reports/dense-v3-functional-inference-v1'
SOURCE = HERE / 'source'
EVAL = Path('/tmp/dense-v3-factorial-evaluation.IOV7MK93/evaluate.py')
TRAIN = EXP / 'launch/factorial-training-v1'
VECTORS = EXP / 'analyses/dense-primary-v3-functional-dimensions-recovery-v1/vectors'
INPUTS = EXP / 'launch/functional-dimensions/inputs.json'
OUTPUT = EXP / 'analyses/dense-v3-factorial-probe-v1'
PROTOCOL = STORY / 'configs/dense_no_packing_state_operator_factorial_protocol.json'
SCOPE = 'owner-authorized-genuine-v3-factorial-five-stage-probe-v1'
OWNER = '你有权做一切事情，目标是尽快完成任务'
STEPS = [79, 157, 235, 313, 391]
INFERENCE_FILES = {'model.safetensors', 'config.json', 'config_sentence_transformers.json', 'modules.json',
                   'sentence_bert_config.json', 'tokenizer.json', 'tokenizer_config.json', '1_Pooling/config.json'}
PINS = {
    EVAL: '45577537257e1dff2b7c3a26ee12f7564fbb476daa256e7b0f09300305b68cba',
    REPORT / 'actual/readout.json': '608965f5297fc196adcce6ca7417bd40e7f35d05e26133f20c0811dd1fd7027e',
    SOURCE / 'src/embed_optim/__init__.py': 'ddfea072915535be4c693c1263f18e17a47ecc7c25d2f6dbe86ff6d2fad7646e',
    VECTORS / 'manifest.json': '9bf90d652d54ee6b9b571bf9dc88cd919da8c78bb70db0d476e307733e8c23e7',
    VECTORS / 'admission.json': 'c02f83853d777ece685aa70dee396f615a425cb2f39fd2da20a13b0ea597961a',
    VECTORS / 'states/pretrained/manifest.json': '90d9f044b53c05065b00d16a53a31a5c82a37d86e33552d797f6c38364e64fac',
    VECTORS / 'states/pretrained/vectors.npz': 'e57057107312363619ebb1ba19b55892fe539568d9545726f3c9b9277325112f',
    INPUTS: 'be6d92fda2f7f0ba3e86f2c2866fa9dca530f066d29ce92d0887fdad91684067',
    PROTOCOL: '5773943a3ae9b581021a0f7b85b162c74d5c497eeca1386578ff9d2c3bcafe76',
    STORY / 'src/embed_optim/state_operator_factorial_probe.py': '7a40bafe3b8d43122cf0559bb83e1d3e6dd169da3fbf4ece8f70f5ccc720784c',
    STORY / 'src/embed_optim/representation_geometry.py': '2910a084079e2c53a6b4152fcb1a32f4ab30387975a1dbbca336db9e546469a2',
}


def original():
    if EVAL.is_symlink() or hashlib.sha256(EVAL.read_bytes()).hexdigest() != PINS[EVAL]:
        raise ValueError('Original operational helpers changed')
    spec = importlib.util.spec_from_file_location('_unchanged_factorial_eval_helpers', EVAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e = original()
need, identity, bound, read, write, now = e.need, e.identity, e.bound, e.read, e.write, e.now


def check_imports(sources):
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name == 'embed_optim' or name.startswith('embed_optim.'):
            path = str(Path(module.__file__).resolve())
            need(path in sources, 'Foreign or unbound probe package import: ' + name)
            result[name] = bound(path, sources[path])
    return result


def context():
    for path, sha in PINS.items():
        bound(path, sha)
    bound(e.PARENT, e.PINS[e.PARENT])
    bound(e.OLD_EVAL, e.PINS[e.OLD_EVAL])
    bound(TRAIN / 'factorial_dispatch.py', e.TRAIN_SHA)
    training = read(TRAIN / 'authorization.json', e.AUTH_SHA)
    need(training['queues'] == e.queues() and training['gpu_pools'] == e.POOLS, 'Training population changed')
    for name, binding in training['source_files'].items():
        bound(e.NATIVE_ROOT / name, binding)
    numerical = {}
    parents = (read(REPORT / 'actual/readout.json')['source_bindings'], read(INPUTS)['sources'])
    for parent in parents:
        for path, binding in parent.items():
            if path.startswith(str(STORY / 'src') + '/'):
                copied = SOURCE / Path(path).relative_to(STORY)
                need(numerical.setdefault(str(copied), binding) == binding, 'Original source parents disagree')
                bound(copied, binding)
    need(len(numerical) >= 42, 'Original encoder/inference source union is incomplete')
    numerical[str(SOURCE / 'src/embed_optim/__init__.py')] = identity(SOURCE / 'src/embed_optim/__init__.py')
    if str(SOURCE / 'src') not in sys.path:
        sys.path.insert(0, str(SOURCE / 'src'))
    from embed_optim.runtime import verify_runtime_spec
    from embed_optim.primary_v3_dimension_contract import ENCODING
    runtime = verify_runtime_spec(e.PRIMARY / 'configs/formal_runtime.json')
    need(ENCODING['batch_size'] == 8 and ENCODING['model_dtype'] == 'bfloat16'
         and ENCODING['storage_dtype'] == 'float32', 'Unchanged real encoder settings differ')
    inputs = read(INPUTS)
    admission = read(VECTORS / 'admission.json')
    need(inputs['admitted']['probe'] == admission['probe'] and inputs['jobs'][0]['plan'] == admission['states'][0],
         'Original reference or fixed probe differs')
    probe_root = Path(inputs['probe_root'])
    for row in admission['probe']['files']:
        bound(probe_root / row['path'], {k: row[k] for k in ('bytes', 'sha256')})
    from embed_optim.probe_export import _load_probe
    dataset, manifest, manifest_sha = _load_probe(probe_root)
    ids = admission['probe']['row_identities']
    need(list(dataset['sample_id']) == [r['sample_id'] for r in ids]
         and list(dataset['source']) == [r['source'] for r in ids], 'Actual ordered probe identities differ')
    parent = e.load('_unchanged_original_probe_dual_leases', e.PARENT)
    old = e.load('_unchanged_original_probe_start_reader', e.OLD_EVAL)
    check_imports(numerical)
    return dict(training=training, inputs=inputs, admission=admission, numerical_sources=numerical,
                dataset=dataset, identities=ids, runtime=runtime, parent=parent, old=old)


def functions():
    """Extract only two unchanged pure numerical functions, not legacy guards."""
    import torch
    namespace = dict(torch=torch, Tensor=torch.Tensor, F=torch.nn.functional, math=math)
    for path, name in ((STORY / 'src/embed_optim/representation_geometry.py', 'dense_probe_scores'),
                       (STORY / 'src/embed_optim/state_operator_factorial_probe.py', '_summary')):
        bound(path, PINS[path])
        nodes = [n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == name]
        need(len(nodes) == 1, 'Original numerical function missing')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['dense_probe_scores'], namespace['_summary']


def reference(c):
    from embed_optim.primary_v3_dimension_vector_io import inspect_vectors
    original_job = c['inputs']['jobs'][0]
    checkpoint = Path(original_job['checkpoint'])
    for row in c['admission']['reference']['files']:
        bound(checkpoint / row['path'], {k: row[k] for k in ('bytes', 'sha256')})
    arrays, checked = inspect_vectors(VECTORS / 'states/pretrained', original_job['plan'], c['identities'], checkpoint,
        expected_manifest_sha256=PINS[VECTORS / 'states/pretrained/manifest.json'])
    return arrays, dict(original_checkpoint=str(checkpoint), manifest=identity(VECTORS / 'states/pretrained/manifest.json'),
        original_vectors=identity(VECTORS / 'states/pretrained/vectors.npz'), original_native_readback=checked,
        original_plan=original_job['plan'], newly_encoded=False)


def checkpoint(c, pool, run, step):
    need(run in e.queues()[pool] and type(step) is int and step in STEPS, 'Unexpected continuation probe cell')
    directory = TRAIN / 'run' / ('pool-' + pool) / run
    complete_id = identity(directory / 'completed.json')
    complete = read(directory / 'completed.json')
    need(complete['run_id'] == run and complete['source_sha256'] == e.TRAIN_SHA
         and complete['authorization_sha256'] == e.AUTH_SHA and complete['actual_rank_exits'] == [0, 0, 0, 0]
         and complete['actual_fresh_reader_exit'] == 0 and complete['all_five_checkpoints_verified'] is True
         and complete['full_horizon_391_steps_verified'] is True, 'Incomplete actual continuation')
    old = read(directory / 'fresh-native-readback.json', complete['native_readback_binding'])
    need(old['worker_completion'] == complete['worker_completion'], 'Original native completion differs')
    need(read(directory / 'ranks.exited.json')['actual_rank_exits'] == {str(i): 0 for i in range(4)}
         and read(directory / 'reader.exited.json')['exit_code'] == 0, 'Missing actual process exits')
    native = old['native']['native_readback']['artifacts']
    need(native['whole_run_artifacts_verified'] is True and native['optimizer_steps'] == 391
         and [r['step'] for r in native['checkpoints']] == STEPS, 'Incomplete original native checkpoint horizon')
    row = native['checkpoints'][STEPS.index(step)]
    request = c['training']['requests'][run]
    cp = Path(request['run_root']) / ('checkpoint-' + str(step))
    need(row['component_binding']['path'] == str(cp) and row['model_tensors'] == 134
         and row['rank_rng_states'] == 4 and row['optimizer']['step'] == step, 'Wrong native stage')
    component = read(cp / 'factorial_trainer_component.json', row['component_binding']['sha256'])
    br = component['identity']['bound_factorial_run']
    need(component['step'] == step and br['run_id'] == run and br['state'] == request['state']
         and br['operator'] == request['operator'] and br['seed'] == request['seed'], 'Wrong saved state/operator/seed')
    selected = [r for r in component['files'] if r['path'] in INFERENCE_FILES]
    need(len(selected) == 8 and {r['path'] for r in selected} == INFERENCE_FILES, 'Wrong checkpoint inference inventory')
    for item in selected:
        bound(cp / item['path'], {k: item[k] for k in ('bytes', 'sha256')})
    bound(directory / 'completed.json', complete_id)
    return dict(run_id=run, pool=pool, state=request['state'], operator=request['operator'], seed=request['seed'],
        step=step, checkpoint=str(cp), inference_files=selected,
        component=identity(cp / 'factorial_trainer_component.json'), actual_completion=complete_id,
        original_native_whole_run_read=complete['native_readback_binding'],
        fresh_optimizer_deserialization_repeated=False, original_actual_rank_exits=[0, 0, 0, 0],
        original_actual_reader_exit=0, scientific_completion=False)


def metrics(arrays, ref, identities):
    import numpy as np
    import torch
    from embed_optim.primary_v3_dimension_vector_io import validate_arrays
    validate_arrays(arrays, identities)
    validate_arrays(ref, identities)
    scorer, summary = functions()
    def stored(a):
        return {k: v.astype(np.float16) if k in ('query_embeddings', 'document_embeddings') else v.copy() for k, v in a.items()}
    saved, saved_ref = stored(arrays), stored(ref)
    scores = scorer(torch.from_numpy(saved['query_embeddings']), torch.from_numpy(saved['document_embeddings']))
    reference_scores = scorer(torch.from_numpy(saved_ref['query_embeddings']), torch.from_numpy(saved_ref['document_embeddings']))
    groups = saved['sample_groups']
    tasks = sorted(set(groups.tolist()))
    need(len(tasks) == 14 and all(int((groups == t).sum()) == 16 for t in tasks), 'Incomplete balanced probe')
    by_task = {t: summary(scores[groups == t], reference_scores[groups == t]) for t in tasks}
    return saved, scores.numpy(), dict(overall=summary(scores, reference_scores), by_task=by_task,
        scoring='unchanged-frozen-factorial-six-metric-summary', score_dtype='float32', stored_embedding_dtype='float16',
        ties='pessimistic-positive-rank; first-index-argmax-for-agreement', temperature=0.02,
        raw_fp32_vectors_retained=True, shortlist_candidates=8, scientific_completion=False)


def save_arrays(path, arrays):
    import numpy as np
    with path.open('xb') as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def authenticate(args):
    bound(__file__, args.source_sha256)
    auth = read(HERE / 'authorization.json', args.authorization_sha256)
    need(auth['source'] == identity(__file__) and auth['scope'] == SCOPE and auth['owner_message'] == OWNER
         and auth['training_authorization_sha256'] == e.AUTH_SHA and auth['queues'] == e.queues()
         and auth['steps'] == STEPS and auth['output_root'] == str(OUTPUT), 'Probe authority changed')
    bound(HERE / 'tests-second.json', auth['tests'])
    bound(HERE / 'test_probe.py', auth['test_source'])
    c = context()
    need(c['numerical_sources'] == auth['numerical_sources'] and c['runtime'] == auth['runtime'], 'Runtime/source drift')
    return auth, c


def prepare(args):
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Prepare must hide CUDA')
    bound(__file__, args.source_sha256)
    need(not OUTPUT.exists() and not (HERE / 'authorization.json').exists(), 'Preserve all earlier attempts')
    tests = read(HERE / 'tests-second.json', args.tests_sha256)
    need(tests['source'] == identity(__file__) and tests['tests_run'] >= 12
         and tests['failures'] == tests['errors'] == tests['skipped'] == 0, 'Missing actual operational checks')
    c = context()
    arrays, origin = reference(c)
    saved, scores, reference_metrics = metrics(arrays, arrays, c['identities'])
    first = checkpoint(c, 'a', e.queues()['a'][0], 79)
    need(first['component']['sha256'] == '805a71c2217cda5f7fc98ea7d014ebbe9d9fbf6f20143b78badf07dede31b9f0',
         'Actual first native stage probe changed')
    import torch
    need(not torch.cuda.is_initialized(), 'Preparation initialized CUDA')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    base = OUTPUT / 'pretrained'
    base.mkdir()
    save_arrays(base / 'vectors-fp16.npz', saved)
    save_arrays(base / 'scores.npz', dict(scores=scores))
    write(base / 'metrics.json', reference_metrics)
    write(base / 'origin.json', origin)
    baseline_files = {p.name: identity(p) for p in base.iterdir() if p.is_file()}
    write(HERE / 'actual-inputs.json', dict(original_reference_native_readback=origin,
        actual_first_checkpoint=first, ordered_probe=c['admission']['probe'],
        baseline_files=baseline_files, gpu_access=False, new_model_encoding=False, scientific_completion=False))
    auth = dict(scope=SCOPE, created_at_utc=now(), owner_message=OWNER, automatic_continuation=False,
        source=identity(__file__), tests=identity(HERE / 'tests-second.json'), test_source=identity(HERE / 'test_probe.py'),
        actual_inputs=identity(HERE / 'actual-inputs.json'), numerical_sources=c['numerical_sources'],
        source_pins={str(k): v for k, v in PINS.items()}, runtime=c['runtime'],
        training_authorization_sha256=e.AUTH_SHA, queues=e.queues(), gpu_pools=e.POOLS,
        steps=STEPS, new_checkpoint_encodings=60, reference_reused=True, reference_files=baseline_files,
        output_root=str(OUTPUT), one_gpu_per_completed_pool=True, no_lease_before_complete_training_pool=True,
        encoding=dict(batch_size=8, model_dtype='bfloat16', raw_storage_dtype='float32', metric_storage_dtype='float16',
                      max_length=8192, flash_attention=True, normalized=True, query_prompt='query: ', document_prompt='document: '),
        scientific_estimands_changed=False, numerical_functions_changed=False, old_controller_transition=False,
        original_historical_factorial_admission=False, source_release=False, scientific_completion=False)
    check_imports(c['numerical_sources'])
    write(HERE / 'authorization.json', auth)
    return dict(authorization=identity(HERE / 'authorization.json'), actual_inputs=identity(HERE / 'actual-inputs.json'),
                reference_complete=True, new_checkpoint_encodings=0, queued_checkpoint_encodings=60, scientific_completion=False)


def wait_pool(pool, parent):
    directory = TRAIN / 'run' / ('pool-' + pool)
    while True:
        need(not (directory / 'coordinator.failed.json').exists(), 'Training failed; no GPU request')
        path = directory / 'coordinator.completed.json'
        if path.exists():
            try:
                complete = read(path)
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            e.require_pool_completion(complete, pool)
            for row in complete['runs']:
                need(read(directory / row['run_id'] / 'completed.json') == row, 'Pool/branch completion differs')
            return identity(path)
        parent.handoff()
        time.sleep(30)


def check_worker_lease(args, request, c):
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == args.token and args.token in e.POOLS[request['pool']], 'Wrong worker GPU token')
    need(request['coordinator_pid'] == os.getppid()
         and c['old'].process_start_ticks(os.getppid()) == request['coordinator_start_ticks'], 'Worker parent differs')
    descriptors = [int(v) for v in args.lease_fds.split(',')]
    need(len(descriptors) == len(c['parent'].LEASE_ROOTS) == 2 and len(set(descriptors)) == 2, 'Missing dual inherited leases')
    import fcntl
    for fd, root in zip(descriptors, c['parent'].LEASE_ROOTS):
        actual, wanted = os.fstat(fd), (root / f'gpu-{args.token}.lock').stat()
        need((actual.st_dev, actual.st_ino) == (wanted.st_dev, wanted.st_ino), 'Wrong inherited lease descriptor')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def worker(args):
    auth, c = authenticate(args)
    request = read(args.request, args.request_sha256)
    need(request['source_sha256'] == args.source_sha256 and request['authorization_sha256'] == args.authorization_sha256,
         'Wrong worker request source/authority')
    check_worker_lease(args, request, c)
    directory = OUTPUT / request['run_id'] / ('checkpoint-' + str(request['step']))
    need(str(directory) == request['output'] and not directory.exists(), 'Preserve existing probe attempt')
    cp = checkpoint(c, request['pool'], request['run_id'], request['step'])
    need(cp == request['checkpoint_admission'], 'Checkpoint changed before encoding')
    from embed_optim.primary_v3_dimension_vector_io import encode_state, save_vectors
    arrays, observed = encode_state(Path(cp['checkpoint']), c['dataset'], c['identities'])
    ref, origin = reference(c)
    saved, scores, result = metrics(arrays, ref, c['identities'])
    directory.mkdir(parents=True, exist_ok=False)
    plan = dict(scope=SCOPE, request=identity(args.request), checkpoint=cp, encoding=auth['encoding'], scientific_completion=False)
    raw = save_vectors(directory / 'raw', plan, arrays, observed, c['identities'], Path(cp['checkpoint']))
    save_arrays(directory / 'vectors-fp16.npz', saved)
    save_arrays(directory / 'scores.npz', dict(scores=scores))
    write(directory / 'metrics.json', result)
    need(checkpoint(c, request['pool'], request['run_id'], request['step']) == cp, 'Checkpoint changed during encoding')
    imports = check_imports(c['numerical_sources'])
    record = dict(run_id=request['run_id'], step=request['step'], completed_at_utc=now(),
        request=identity(args.request), source=identity(__file__), authorization_sha256=args.authorization_sha256,
        raw_native_readback=raw, raw_plan=plan, reference_origin=origin,
        outputs={name: identity(directory / name) for name in ('vectors-fp16.npz', 'scores.npz', 'metrics.json')},
        loaded_project_modules=imports, original_encoder_and_scorer_unchanged=True, scientific_completion=False)
    write(directory / 'worker.completed.json', record)
    print(json.dumps(dict(run_id=request['run_id'], step=request['step'], completion=identity(directory / 'worker.completed.json'))), flush=True)


def verify_result(c, request, expected_completion):
    import numpy as np
    from embed_optim.primary_v3_dimension_vector_io import inspect_vectors
    directory = Path(request['output'])
    record = read(directory / 'worker.completed.json', expected_completion)
    need(record['source'] == identity(__file__) and record['request'] == identity(Path(request['request_file']))
         and record['run_id'] == request['run_id'] and record['step'] == request['step'], 'Wrong completed probe worker')
    cp = checkpoint(c, request['pool'], request['run_id'], request['step'])
    need(cp == request['checkpoint_admission'], 'Checkpoint changed during readback')
    arrays, checked = inspect_vectors(directory / 'raw', record['raw_plan'], c['identities'], Path(cp['checkpoint']),
        expected_manifest_sha256=record['raw_native_readback']['manifest']['sha256'])
    ref, origin = reference(c)
    saved, scores, result = metrics(arrays, ref, c['identities'])
    for name, binding in record['outputs'].items():
        bound(directory / name, binding)
    with np.load(directory / 'vectors-fp16.npz', allow_pickle=False) as stored:
        need(set(stored.files) == set(saved) and all(np.array_equal(stored[k], saved[k]) for k in saved), 'Stored probe vectors differ')
    with np.load(directory / 'scores.npz', allow_pickle=False) as stored:
        need(stored.files == ['scores'] and np.array_equal(stored['scores'], scores), 'Original full probe scores differ')
    need(read(directory / 'metrics.json') == result and record['reference_origin'] == origin, 'All six metrics/reference differ')
    check_imports(c['numerical_sources'])
    return dict(run_id=request['run_id'], step=request['step'], verified_at_utc=now(),
        worker_completion=expected_completion, all_four_arrays_and_six_metrics_reconstructed=True,
        tasks=14, samples=224, native_raw_readback=checked, scientific_completion=False)


def coordinate(args):
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Coordinator must hide CUDA')
    auth, c = authenticate(args)
    root = HERE / 'run' / ('pool-' + args.pool)
    root.mkdir(parents=True, exist_ok=False)
    parent, old = c['parent'], c['old']
    start = old.process_start_ticks(os.getpid())
    write(root / 'started.json', dict(pid=os.getpid(), start_ticks=start, started_at_utc=now(),
        source_sha256=args.source_sha256, authorization_sha256=args.authorization_sha256, expected_checkpoints=30,
        gpu_leases_requested=False, scientific_completion=False))
    try:
        complete = wait_pool(args.pool, parent)
        write(root / 'training-pool-admitted.json', dict(completion=complete, admitted_at_utc=now(), gpu_leases_requested=False))
        auth, c = authenticate(args)
        lease, descriptors, token = None, None, None
        while descriptors is None:
            parent.handoff()
            for candidate in reversed(e.POOLS[args.pool]):
                candidate_lease = parent.leases([candidate])
                try:
                    descriptors = candidate_lease.__enter__()
                except BlockingIOError:
                    continue
                lease, token = candidate_lease, candidate
                break
            if descriptors is None:
                time.sleep(1)
        completed = []
        try:
            for run in e.queues()[args.pool]:
                for step in STEPS:
                    auth, c = authenticate(args)
                    cp = checkpoint(c, args.pool, run, step)
                    prefix = root / f'{run}-checkpoint-{step}'
                    reqpath = prefix.with_suffix('.request.json')
                    request = dict(pool=args.pool, run_id=run, step=step, checkpoint_admission=cp,
                        output=str(OUTPUT / run / ('checkpoint-' + str(step))), request_file=str(reqpath),
                        coordinator_pid=os.getpid(), coordinator_start_ticks=start,
                        source_sha256=args.source_sha256, authorization_sha256=args.authorization_sha256)
                    write(reqpath, request)
                    argv = ['/usr/bin/python', '-B', str(HERE / 'probe.py'), 'worker',
                        '--source-sha256', args.source_sha256, '--authorization-sha256', args.authorization_sha256,
                        '--request', str(reqpath), '--request-sha256', identity(reqpath)['sha256'],
                        '--token', token, '--lease-fds', ','.join(str(fd) for fd in descriptors)]
                    env = parent.environment([token])
                    env.update(PYTHONPATH=str(SOURCE / 'src'), WANDB_MODE='disabled')
                    with prefix.with_suffix('.log').open('x') as log:
                        child = subprocess.Popen(argv, cwd=SOURCE, env=env, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, pass_fds=descriptors)
                        write(prefix.with_suffix('.started.json'), dict(pid=child.pid, start_ticks=old.process_start_ticks(child.pid),
                            command=argv, token=token, both_original_leases_inherited=True, started_at_utc=now()))
                        code = child.wait()
                    write(prefix.with_suffix('.exited.json'), dict(exit_code=code, completed_at_utc=now()))
                    need(code == 0, 'Actual probe worker failed; no automatic retry')
                    result = verify_result(c, request, identity(Path(request['output']) / 'worker.completed.json'))
                    write(prefix.with_suffix('.verified.json'), result)
                    completed.append(dict(run_id=run, step=step, verified=identity(prefix.with_suffix('.verified.json'))))
                    print(json.dumps(dict(run_id=run, step=step, completed=len(completed), pool=args.pool)), flush=True)
        finally:
            lease.__exit__(None, None, None)
        need(len(completed) == 30, 'Incomplete five-stage probe pool')
        write(root / 'completed.json', dict(completed_at_utc=now(), checkpoints=completed,
             source_sha256=args.source_sha256, authorization_sha256=args.authorization_sha256, scientific_completion=False))
    except BaseException as exc:
        write(root / 'failed.json', dict(failed_at_utc=now(), exception_type=type(exc).__name__, scientific_completion=False))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'coordinate', 'worker', 'inspect'))
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--authorization-sha256')
    p.add_argument('--tests-sha256')
    p.add_argument('--pool', choices=('a', 'b'))
    p.add_argument('--request', type=Path)
    p.add_argument('--request-sha256')
    p.add_argument('--token')
    p.add_argument('--lease-fds')
    args = p.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(args)), flush=True)
    elif args.action == 'inspect':
        auth, c = authenticate(args)
        print(json.dumps(dict(states=61, new_encodings=60, gpu_leases_requested=False, scientific_completion=False)))
    elif args.action == 'worker':
        worker(args)
    else:
        need(args.pool in e.POOLS, 'Select an existing pool')
        coordinate(args)


if __name__ == '__main__':
    main()
