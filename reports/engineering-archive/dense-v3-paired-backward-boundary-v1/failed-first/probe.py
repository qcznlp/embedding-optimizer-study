"""Paired four-rank backward diagnosis; no clipping or optimizer updates.

Original source/run/input admission and numerical training functions are unchanged.
Read-only hooks observe the first four resumed microbatches in the native loop.
At its clipping boundary, a second same-weight/input/RNG pass explicitly reuses
the same native training_step and no_sync schedule through the existing DDP object.
Both sets of actual pre-clip gradients are saved before intentionally stopping.
No checkpoint completion, successful long resume or scientific result is claimed.
"""
import argparse
from contextlib import ExitStack
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
OLD = Path('/tmp/dense-v3-resume-device-recovery.uQ0ynb0k')
OLD_SHA = 'b2089cd2fae33b55a018a4471a427196143012b6a98303d23a703c8e1dc435d8'
OLD_AUTH = '1e7fbcbbde7021ece3a18e5608f8cb909991860911097d0bb54e7b7a7043989a'
DOWNLOAD_SHA = '3c48d5c7d10434fe48925ec223e55c85df59e23ee6767a9d4bffbf1f5901f332'
SCOPE = 'dense-v3-four-rank-paired-backward-boundary-v1'
FIRST = Path('/tmp/dense-v3-first-gradient.FDK2kcqs')
FIRST_REPORT = Path('/root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-v3-first-gradient-boundary-v1')


def identity(path):
    path = Path(path)
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Require ordinary bound source/input')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}


def load(name, path, expected):
    if identity(path)['sha256'] != expected:
        raise ValueError('Original source differs')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r = load('_original_resume_boundary_parent', OLD / 'resume.py', OLD_SHA)


def old_context(cpu=True):
    return r.authenticate(SimpleNamespace(source_sha=OLD_SHA, authorization_sha=OLD_AUTH), cpu=cpu)


def first_completed():
    manifest = r.read(FIRST_REPORT/'manifest.json', '3fcd0e1ffebeefbd394fe330d3c5aa631fb38ef27e286896aa7df90055a62351')
    result = {}
    for pool in ('a','b'):
        path = FIRST/'run'/('pool-'+pool)/'completed.json'
        binding = manifest['files'][f'actual/pool-{pool}/completed.json']
        r.bound(path, binding)
        value = r.read(path)
        if value['actual_rank_exits'] != [0]*4 or value['optimizer_updates'] != 0:
            raise ValueError('Require completed original first-backward diagnostic')
        result[pool] = {'path':str(path),**binding}
    return result


def prepare(args):
    auth, t, original, parent = old_context()
    r.no_existing(HERE / 'authorization.json')
    r.no_existing(HERE / 'run')
    if identity(__file__)['sha256'] != args.source_sha:
        raise ValueError('Diagnostic source changed')
    unit = ET.parse(HERE / 'tests.xml').getroot().find('testsuite').attrib
    if unit['tests'] != '14' or any(unit[k] != '0' for k in ('errors', 'failures', 'skipped')):
        raise ValueError('Require completed trace regression controls')
    r.bound(r.ORIGINAL_RESUME / 'downloaded.json', DOWNLOAD_SHA)
    downloaded = r.read(r.ORIGINAL_RESUME / 'downloaded.json')
    from embed_optim import factorial_v3_checkpoint as cp
    parent.handoff()
    for pool in ('a', 'b'):
        # Do not poll or rerun these completed jobs; authenticate their terminal prerequisites.
        values = [r.read(r.TRAIN / 'run' / ('pool-' + pool) / 'coordinator.completed.json'),
                  r.read(r.EVAL / 'run' / ('pool-' + pool) / 'completed.json'),
                  r.read(r.PROBE / 'run' / ('pool-' + pool) / 'completed.json')]
        r.require_finished(pool, values)
        cp.read(downloaded['cases'][pool]['binding'], auth['cases'][pool]['component_identity'])
    r.write(HERE / 'authorization.json', dict(scope=SCOPE, owner_message=r.OWNER,
        source=identity(__file__), trace=identity(HERE / 'trace.py'), paired=identity(HERE / 'paired.py'),
        original_first_boundary=first_completed(),
        paired_test=identity(HERE / 'test_paired.py'),
        test_source=identity(HERE / 'test_trace.py'), tests=identity(HERE / 'tests.xml'),
        original_resume_source=identity(OLD / 'resume.py'), original_resume_authority=identity(OLD / 'authorization.json'),
        downloaded=identity(r.ORIGINAL_RESUME / 'downloaded.json'), source_files=original['source_files'],
        pools=t.POOLS, cases=r.CASES, output_root=str(HERE / 'run'),
        backward_microbatches_per_rank=8, backward_passes=2, optimizer_updates=0,
        second_pass_uses_same_weight_input_rng=True,
        both_original_lease_namespaces=True, no_communication_hook=True,
        stop_before_clipping=True, automatic_retry=False,
        original_numerical_sources_changed=False, exact_endpoint_resume=False,
        scientific_completion=False, protected_helper_access=False))
    print(json.dumps({'authorization': identity(HERE / 'authorization.json'), 'gpu_started': False}), flush=True)


def authenticate(args, cpu=True):
    r.bound(__file__, args.source_sha)
    auth = r.read(HERE / 'authorization.json', args.authorization_sha)
    if (auth['scope'] != SCOPE or auth['source'] != identity(__file__)
            or auth['output_root'] != str(HERE / 'run') or auth['optimizer_updates'] != 0
            or auth['backward_microbatches_per_rank'] != 8 or auth['backward_passes'] != 2
            or auth['owner_message'] != r.OWNER):
        raise ValueError('Wrong bounded diagnostic authority')
    for path, field in ((HERE/'trace.py','trace'), (HERE/'test_trace.py','test_source'),
                        (HERE/'paired.py','paired'), (HERE/'test_paired.py','paired_test'),
                        (HERE/'tests.xml','tests'), (OLD/'resume.py','original_resume_source'),
                        (OLD/'authorization.json','original_resume_authority'),
                        (r.ORIGINAL_RESUME/'downloaded.json','downloaded')):
        r.bound(path, auth[field])
    old, t, original, parent = old_context(cpu)
    if auth['original_first_boundary'] != first_completed():
        raise ValueError('Original completed boundary changed')
    if auth['source_files'] != original['source_files'] or auth['pools'] != t.POOLS or auth['cases'] != r.CASES:
        raise ValueError('Original source, cases or device pools changed')
    return auth, old, t, original, parent


def worker(args):
    auth, old, t, original, parent = authenticate(args, cpu=False)
    job = HERE / 'run' / ('pool-' + args.pool)
    admission = r.read(job / 'admission.json')
    if os.getppid() != admission['pid'] or admission['source_sha256'] != args.source_sha:
        raise ValueError('Worker is not the admitted direct child')
    expected = dict(RANK=str(args.rank), LOCAL_RANK=str(args.rank), WORLD_SIZE='4', LOCAL_WORLD_SIZE='4',
                    MASTER_ADDR='127.0.0.1', MASTER_PORT=str(admission['port']),
                    CUDA_VISIBLE_DEVICES=','.join(t.POOLS[args.pool]), WANDB_MODE='disabled')
    if any(os.environ.get(k) != v for k, v in expected.items()):
        raise ValueError('Wrong rank, reporting or device environment')
    t.verify_leases(parent, args.pool, args.lease_fd)
    parent.handoff()
    case = old['cases'][args.pool]
    binding = r.read(r.ORIGINAL_RESUME / 'downloaded.json')['cases'][args.pool]['binding']
    trace = load('_first_backward_trace', HERE / 'trace.py', auth['trace']['sha256'])
    paired = load('_paired_native_backward', HERE / 'paired.py', auth['paired']['sha256'])
    import torch
    import torch.distributed as dist
    from transformers import TrainerCallback, set_seed
    from safetensors.torch import load_file, save_file
    from accelerate.data_loader import BatchSamplerShard, SeedableRandomSampler
    from embed_optim import factorial_v3_factory as factory, factorial_v3_run_contract as contract
    from embed_optim.factorial_v3_bound_trainer import RunBoundFactorialTrainer, collective_phase
    from embed_optim.factorial_v3_trainer import FactorialTrainingArguments
    from embed_optim.factorial_v3_batches import FactorialBatchSampler
    if torch.cuda.is_initialized() or dist.is_initialized():
        raise ValueError('Inherited device state')
    torch.cuda.set_device(args.rank)
    dist.init_process_group('nccl', rank=args.rank, world_size=4, timeout=timedelta(minutes=3))
    rank_root = job / ('rank-' + str(args.rank))
    rank_root.mkdir()
    try:
        total = torch.tensor(float(args.rank), device='cuda')
        dist.all_reduce(total)
        if total.item() != 6:
            raise ValueError('Four-rank collective failed')
        _, _, locations = t.context(original['source_files'])
        d = case['declaration']
        run_identity, dataset = collective_phase(lambda: contract.prepare_run(locations, d['calibrations'],
            d['state'], d['operator'], d['seed'], t.SOURCE_ROOT))
        if run_identity != case['component_identity']['bound_factorial_run']:
            raise ValueError('Original run identity changed')
        config, values = factory.recipe(run_identity, locations.data_store / factory.BRANCH,
            job / 'trainer-output', project=d['project'], entity=d['entity'])
        values['report_to'] = []
        train_args = collective_phase(lambda: FactorialTrainingArguments(**values))
        set_seed(d['seed'])
        model, loss, _ = collective_phase(lambda: factory.load_model(run_identity, config))
        trainer = RunBoundFactorialTrainer(run_identity=run_identity, optimizer_config=config.optimizer,
            model=model, args=train_args, train_dataset=dataset, loss=loss, resume_binding=binding)
        if trainer.component_identity() != case['component_identity']:
            raise ValueError('Original numerical component changed')
        named = dict(model.named_parameters())
        if len(named) != 134:
            raise ValueError('Incomplete model parameter population')
        sampler = SeedableRandomSampler(dataset, generator=torch.Generator().manual_seed(d['seed']), data_seed=d['seed'])
        shard = BatchSamplerShard(FactorialBatchSampler(sampler), num_processes=4,
            process_index=args.rank, split_batches=False, even_batches=False)
        indices = list(shard)[1252:1256]
        raw_batches = [trainer.data_collator([dataset[i] for i in row]) for row in indices]
        expected_features = [trainer.collect_features(batch)[0] for batch in raw_batches]
        capture = trace.Capture(expected_features, list(named))
        trainer.add_callback(r.device_restore_callback(trainer, rank_root / 'device-placement.json',
            134 if args.pool == 'a' else 46, collective_phase))

        class LoadedBoundary(TrainerCallback):
            def on_train_begin(self, _args, state, control, **kwargs):
                def check():
                    path = Path(binding['path'])
                    r.recursive_equal(trace.cpu_tree(dict(model[0].model.state_dict())),
                        load_file(path / 'model.safetensors'), 'loaded-model')
                    r.recursive_equal(trace.cpu_tree(trainer._raw_optimizer().state_dict()),
                        torch.load(path / 'optimizer.pt', map_location='cpu', weights_only=True), 'loaded-optimizer')
                    r.recursive_equal(trace.cpu_tree(trainer.lr_scheduler.state_dict()),
                        torch.load(path / 'scheduler.pt', map_location='cpu', weights_only=True), 'loaded-scheduler')
                    if state.global_step != 313 or any(p.device != torch.device('cuda', args.rank) for p in named.values()):
                        raise ValueError('Wrong device-loaded step/state')
                    r.write(rank_root / 'loaded.json', dict(step=313, model_tensors=134,
                        exact_loaded_model_optimizer_scheduler=True, original_binding=binding,
                        all_parameters_on_rank_device=True, scientific_completion=False))
                collective_phase(check)
                return control
        trainer.add_callback(LoadedBoundary())
        handles = [loss.register_forward_pre_hook(capture.loss_input)]
        handles += [p.register_hook(capture.leaf_hook(name)) for name, p in named.items()]
        original_clip = trainer.accelerator.clip_grad_norm_

        def stop_before_clip(parameters, max_norm, *unused, **kwargs):
            trace.require_stop_boundary(trainer)
            if max_norm != 1.0 or [id(p) for p in parameters] != [id(p) for p in named.values()]:
                raise ValueError('Wrong original clipping boundary')
            capture.require_complete()
            gradients = {name: p.grad.detach().cpu().clone() for name, p in named.items()}
            if any(not bool(torch.isfinite(v).all()) for v in [*gradients.values(), *capture.local_sum.values()]):
                raise ValueError('Nonfinite observed gradient')
            save_file(capture.local_sum, rank_root / 'local-leaf-sum.safetensors')
            save_file(gradients, rank_root / 'ddp-preclip.safetensors')
            r.write(rank_root / 'backward.json', dict(step=313, optimizer_updates=0,
                observed_microbatches=4, expected_rank_indices=indices,
                exact_token_features=True, token_fingerprints=capture.features,
                local_gradient_event_fingerprints=capture.gradient_hashes,
                ddp_gradient_fingerprints=trace.fingerprint(gradients),
                ddp=trainer.model_wrapped._get_ddp_logging_data(),
                local_sum=identity(rank_root/'local-leaf-sum.safetensors'),
                preclip=identity(rank_root/'ddp-preclip.safetensors'),
                local_sum_is_cpu_sum_of_observed_leaf_contributions=True,
                communication_hook_installed=False, clipping_executed=False,
                exact_endpoint_resume=False, scientific_completion=False))
            # A second pass reuses this DDP object but does not change weights or optimizer state.
            # Its local contributions are checked independently of the post-reduction result.
            for handle in handles:
                handle.remove()
            handles.clear()
            warm = trace.Capture(expected_features, list(named))
            handles.append(loss.register_forward_pre_hook(warm.loss_input))
            handles.extend(p.register_hook(warm.leaf_hook(name)) for name,p in named.items())
            before_model = trace.fingerprint(dict(model.state_dict()))
            before_optimizer = trace.fingerprint(trainer._raw_optimizer().state_dict())
            before_scheduler = trace.fingerprint(trainer.lr_scheduler.state_dict())
            losses = paired.repeated_backward(trainer, raw_batches, capture.rng_at_first_input,
                                               trace.require_stop_boundary)
            warm.require_complete()
            r.recursive_equal(capture.rng_at_first_input, warm.rng_at_first_input, 'same-first-input-rng')
            if (before_model != trace.fingerprint(dict(model.state_dict()))
                    or before_optimizer != trace.fingerprint(trainer._raw_optimizer().state_dict())
                    or before_scheduler != trace.fingerprint(trainer.lr_scheduler.state_dict())):
                raise ValueError('Weights/optimizer/scheduler changed during paired backward')
            warm_gradients = {name:p.grad.detach().cpu().clone() for name,p in named.items()}
            if any(not bool(torch.isfinite(v).all()) for v in [*warm_gradients.values(),*warm.local_sum.values()]):
                raise ValueError('Nonfinite second-pass gradient')
            save_file(warm.local_sum, rank_root/'warm-local-leaf-sum.safetensors')
            save_file(warm_gradients, rank_root/'warm-ddp-preclip.safetensors')
            changed_local = [name for name in named if warm.gradient_hashes[name] != capture.gradient_hashes[name]]
            changed_ddp = [name for name in named if trace.fingerprint(warm_gradients[name]) != trace.fingerprint(gradients[name])]
            r.write(rank_root/'paired.json', dict(step=313, optimizer_updates=0, backward_passes=2,
                same_weights_optimizer_scheduler=True, same_first_input_rng=True,
                exact_token_features=warm.features==capture.features,
                local_gradient_changed_parameters=changed_local,
                post_ddp_changed_parameters=changed_ddp,
                warm_gradient_event_fingerprints=warm.gradient_hashes,
                warm_ddp_gradient_fingerprints=trace.fingerprint(warm_gradients),
                warm_ddp=trainer.model_wrapped._get_ddp_logging_data(), warm_scaled_losses=losses,
                warm_local_sum=identity(rank_root/'warm-local-leaf-sum.safetensors'),
                warm_preclip=identity(rank_root/'warm-ddp-preclip.safetensors'),
                same_input_repeated_native_training_step=True, communication_hook_installed=False,
                exact_endpoint_resume=False, scientific_completion=False))
            dist.barrier()
            raise trace.BoundaryReached()
        # Explicit diagnostic stop trap, not a replacement clipping implementation.
        trainer.accelerator.clip_grad_norm_ = stop_before_clip
        reached = False
        try:
            trainer.train(resume_from_checkpoint=binding['path'])
        except trace.BoundaryReached:
            reached = True
        finally:
            trainer.accelerator.clip_grad_norm_ = original_clip
            for handle in handles:
                handle.remove()
        if not reached:
            raise ValueError('Original loop did not reach the specified stop boundary')
        trace.require_stop_boundary(trainer)
        t.c.check_imports(original['source_files'])
        r.write(rank_root/'completed.json', dict(scope=SCOPE, rank=args.rank,
            loaded=identity(rank_root/'loaded.json'), backward=identity(rank_root/'backward.json'),
            paired=identity(rank_root/'paired.json'),
            optimizer_updates=0, no_training_checkpoint_created=True, scientific_completion=False))
    finally:
        dist.destroy_process_group()


def argv(args, mode):
    return ['/usr/bin/python', '-B', str(HERE/'probe.py'), mode,
            '--source-sha', args.source_sha, '--authorization-sha', args.authorization_sha, '--pool', args.pool]


def coordinate(args):
    auth, _, t, _, parent = authenticate(args)
    job = HERE/'run'/('pool-'+args.pool)
    r.no_existing(job)
    job.mkdir(parents=True)
    owned = t.c.imported('_first_backward_owned_identity', t.c.DISPATCH)
    r.write(job/'started.json', dict(**owned.process_identity(os.getpid()), scope=SCOPE,
                                   source_sha256=args.source_sha, authorization_sha256=args.authorization_sha))
    children = []
    try:
        parent.handoff()
        with ExitStack() as stack:
            descriptors = stack.enter_context(parent.leases(t.POOLS[args.pool]))
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            r.write(job/'admission.json', dict(pid=os.getpid(), port=port, source_sha256=args.source_sha,
                authorization_sha256=args.authorization_sha, both_original_leases_inherited=True))
            try:
                for rank in range(4):
                    command = argv(args, 'worker') + ['--rank', str(rank)]
                    for fd in descriptors:
                        command += ['--lease-fd', str(fd)]
                    env = t.environment(parent, pool=args.pool, rank=rank, run_id=r.CASES[args.pool], port=port)
                    env['WANDB_MODE'] = 'disabled'
                    for key in ('WANDB_RUN_ID', 'WANDB_RESUME', 'WANDB_API_KEY'):
                        env.pop(key, None)
                    log = stack.enter_context((job/f'rank-{rank}.log').open('xb'))
                    child = subprocess.Popen(command, cwd=t.SOURCE_ROOT, env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, pass_fds=descriptors)
                    children.append(child)
                    r.write(job/f'rank-{rank}.started.json', dict(**owned.process_identity(child.pid, command), rank=rank))
                started = time.monotonic()
                while any(c.poll() is None for c in children):
                    if any(c.poll() not in (None, 0) for c in children) or time.monotonic()-started > 1800:
                        raise RuntimeError('Owned rank failure or bounded 30-minute diagnostic timeout')
                    time.sleep(1)
                if any(c.returncode != 0 for c in children):
                    raise RuntimeError('Owned rank failed')
            finally:
                t.finish_owned(children)
                r.write(job/'ranks.exited.json', {'actual_rank_exits': [c.returncode for c in children]})
        results = [r.read(job/f'rank-{i}/backward.json') for i in range(4)]
        if any(v['ddp_gradient_fingerprints'] != results[0]['ddp_gradient_fingerprints'] for v in results[1:]):
            raise ValueError('Observed post-DDP gradients differ across ranks')
        pairs = [r.read(job/f'rank-{i}/paired.json') for i in range(4)]
        if any(v['warm_ddp_gradient_fingerprints'] != pairs[0]['warm_ddp_gradient_fingerprints'] for v in pairs[1:]):
            raise ValueError('Second-pass post-DDP gradients differ across ranks')
        r.write(job/'completed.json', dict(scope=SCOPE, original_case=r.CASES[args.pool],
            actual_rank_exits=[0]*4, loaded_state_and_tokens_exact=True,
            all_post_ddp_gradients_equal_across_four_ranks=True, optimizer_updates=0,
            backward_passes=2, same_weight_input_rng=True,
            local_changed_parameter_counts=[len(v['local_gradient_changed_parameters']) for v in pairs],
            post_ddp_changed_parameter_counts=[len(v['post_ddp_changed_parameters']) for v in pairs],
            exact_endpoint_resume=False, cause_of_prior_endpoint_difference_established=False,
            scientific_completion=False))
    except BaseException as error:
        t.finish_owned(children)
        r.write(job/'failed.json', dict(exception_type=type(error).__name__, message=str(error),
                                      automatic_retry=False, scientific_completion=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'coordinate', 'worker'])
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--authorization-sha')
    parser.add_argument('--pool', choices=['a','b'])
    parser.add_argument('--rank', type=int, choices=range(4))
    parser.add_argument('--lease-fd', type=int, action='append', default=[])
    args = parser.parse_args()
    if args.mode != 'prepare' and (args.authorization_sha is None or args.pool is None):
        raise ValueError('Explicit bound diagnostic/pool required')
    if args.mode == 'worker' and (args.rank is None or len(set(args.lease_fd)) != 8):
        raise ValueError('Explicit rank and eight inherited leases required')
    if args.mode != 'worker' and (args.rank is not None or args.lease_fd):
        raise ValueError('CPU entry cannot inherit rank leases')
    globals()[args.mode](args)
