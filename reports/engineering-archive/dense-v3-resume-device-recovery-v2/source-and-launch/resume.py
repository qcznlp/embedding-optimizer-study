"""Two real four-rank resumes with explicit restored-counter device placement.

Restore the original native component from an anonymous immutable HF download,
then use its existing bound Trainer resume hooks and a source-bound callback
that places restored Adam counters before the first update. No original source,
output, worker receipt or numerical definition is changed. These are verification jobs,
not new scientific cells, fresh-worker completions or physical second-host runs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
EXP = Path('/root/embedding-optimizer-v3-experiment')
TRAIN = EXP / 'launch/factorial-training-v1'
TRAIN_SHA = 'd11d3281523b1573ce7fcd650e10c538cfa84022b5db92117559d47213e57e53'
TRAIN_AUTH = '00c52f0bec979fb163a734ad0f9eadcd6ea09f7393133fd35c65bc62878ce11f'
EVAL = Path('/tmp/dense-v3-factorial-evaluation.IOV7MK93')
EVAL_SHA = '45577537257e1dff2b7c3a26ee12f7564fbb476daa256e7b0f09300305b68cba'
EVAL_AUTH = '3a1ae32fdde63abf386603208e413ea11bf0c169eb1923f9dd7e1da529ffa29c'
PROBE = Path('/tmp/dense-v3-factorial-probe.ryorjXJK')
PROBE_SHA = 'eb4fc27738adaecd9614ba5822255a5dc37de242afabfa53ff8cd8a6c8c73334'
PROBE_AUTH = '00d6b42f922b4428352d5b9fc5e5fa7e0436a4aa20ee0c693e57b9f645398ded'
BACKUP = Path('/tmp/dense-v3-factorial-backup-launch.75Chtt86')
ORIGINAL_RESUME = Path('/tmp/dense-v3-resume-verification.wcsmQnDm')
ORIGINAL_RESUME_SHA = '1746b8a382bb45fefc2ab17beaf7706984fe91802c528bec48c88a2daa038bb2'
ORIGINAL_RESUME_AUTH = '51e95a2adf6cf5acf151a18976c476e275307262b6f94528b313b964e2baf7d0'
OUTPUT = EXP / 'outputs/dense-v3-factorial-resume-device-recovery-v2'
SCOPE = 'owner-authorized-two-genuine-factorial-gpu-resume-device-recovery-v2'
OWNER = '你有权做一切事情，目标是尽快完成任务'
CASES = {'a': 'factorial-v3-adamw_state-adamw-seed314159',
         'b': 'factorial-v3-adamw_state-muon-seed314159'}
REVISIONS = {'a': 'a219dcab85bda040e9d46f6f7ec18ee1b0acf598',
             'b': 'e9ab737562e50a6ec0ff838f22e97301fda93d63'}
STEPS = (79, 157, 235, 313, 391)


def original():
    import hashlib
    path = EVAL / 'evaluate.py'
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != EVAL_SHA:
        raise ValueError('Original read-only operations changed')
    spec = importlib.util.spec_from_file_location('_resume_original_operations', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e = original()
need, identity, bound, read, write, now = e.need, e.identity, e.bound, e.read, e.write, e.now


def training():
    bound(TRAIN / 'factorial_dispatch.py', TRAIN_SHA)
    bound(TRAIN / 'authorization.json', TRAIN_AUTH)
    t = e.load('_resume_original_training_operations', TRAIN / 'factorial_dispatch.py')
    auth = read(TRAIN / 'authorization.json', TRAIN_AUTH)
    need(t.source_files() == auth['source_files'], 'Original 70-file source closure changed')
    parent = t.c.imported('_resume_original_dual_leases', t.c.PARENT)
    return t, auth, parent


def pins():
    for path, sha in ((EVAL / 'evaluate.py', EVAL_SHA), (EVAL / 'authorization.json', EVAL_AUTH),
                      (PROBE / 'probe.py', PROBE_SHA), (PROBE / 'authorization.json', PROBE_AUTH)):
        bound(path, sha)


def no_existing(path):
    path = Path(path)
    need(path.is_absolute() and not path.exists()
         and not any(p.is_symlink() for p in (path, *path.parents)), 'Never overwrite or follow an existing attempt')


def complete_original(t, auth, pool, run):
    root = TRAIN / 'run' / ('pool-' + pool) / run
    value = read(root / 'completed.json')
    need(value['run_id'] == run and value['source_sha256'] == TRAIN_SHA
         and value['authorization_sha256'] == TRAIN_AUTH and value['actual_rank_exits'] == [0]*4
         and value['actual_fresh_reader_exit'] == 0 and value['all_five_checkpoints_verified'] is True,
         'Original uninterrupted branch is incomplete')
    need(read(root / 'ranks.exited.json')['actual_rank_exits'] == {str(i): 0 for i in range(4)}
         and read(root / 'reader.exited.json')['exit_code'] == 0, 'Original actual process exits differ')
    reader = read(root / 'fresh-native-readback.json', value['native_readback_binding'])
    need(reader['worker_completion'] == value['worker_completion']
         and reader['fresh_process_native_readback'] is True, 'Original native reader chain differs')
    _, native, locations = t.context(auth['source_files'])
    checked = native.read_execution(locations, auth['requests'][run], value['worker_completion'])
    need(checked == reader['native'], 'Fresh original whole-run read differs')
    return checked['native_readback'], identity(root / 'completed.json')


def historical_prepare_not_exposed(args):
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Preparation is CPU-only')
    bound(__file__, args.source_sha)
    no_existing(HERE / 'authorization.json')
    no_existing(OUTPUT)
    tests = read(HERE / 'tests.json')
    need(tests['exit_code'] == 0 and tests['tests_passed'] >= 20
         and tests['source'] == identity(__file__)
         and tests['test_source'] == identity(HERE / 'test_resume.py'), 'Require actual passing operational controls')
    pins()
    t, auth, _ = training()
    admitted, _ = t.authenticate(SimpleNamespace(source_sha=TRAIN_SHA, authorization_sha=TRAIN_AUTH))
    need(admitted == auth, 'Original execution authority differs')
    from embed_optim import factorial_v3_checkpoint as cp
    cases = {}
    for pool, run in CASES.items():
        native, completion = complete_original(t, auth, pool, run)
        checkpoints = native['artifacts']['checkpoints']
        need([r['step'] for r in checkpoints] == list(STEPS), 'Original full checkpoint population differs')
        selected = {r['step']: r['component_binding'] for r in checkpoints if r['step'] in (313, 391)}
        payload = read(Path(selected[313]['path']) / cp.NAME, selected[313]['sha256'])
        need(payload['identity']['bound_factorial_run'] == native['run_identity'], 'Wrong resume run identity')
        cp.read(selected[313], payload['identity'])
        cp.read(selected[391], payload['identity'])
        remote_path = BACKUP / 'run' / run / 'verified.json'
        remote = read(remote_path)
        need(remote['commit_oid'] == REVISIONS[pool] and remote['run_id'] == run
             and remote['repo_id'] == 'qcz/embedding-optimizer-study-checkpoints'
             and remote['repo_type'] == 'model' and remote['all_five_checkpoints'] is True,
             'No immutable verified backup for this run')
        manifest_path = remote_path.parent / 'artifact_manifest.json'
        manifest = read(manifest_path, remote['artifact_manifest'])
        files = {r['path']: {'bytes': r['bytes'], 'sha256': r['sha256']} for r in payload['files']}
        files[cp.NAME] = identity(Path(selected[313]['path']) / cp.NAME)
        for name, binding in files.items():
            row = manifest['files']['run/checkpoint-313/' + name]
            need({k: row[k] for k in ('bytes', 'sha256')} == binding, 'Backup/native checkpoint disagreement')
        cases[pool] = dict(run_id=run, declaration=auth['requests'][run], original_completion=completion,
            original_checkpoints={str(k): v for k, v in selected.items()}, component_identity=payload['identity'],
            remote_receipt=identity(remote_path), remote_manifest=identity(manifest_path),
            repository=remote['repo_id'], revision=remote['commit_oid'], prefix=remote['prefix'], files=files,
            download_checkpoint=str(HERE / 'downloads' / remote['prefix'] / 'run/checkpoint-313'))
    import torch
    need(not torch.cuda.is_initialized(), 'CPU admission initialized CUDA')
    t.c.check_imports(auth['source_files'])
    value = dict(scope=SCOPE, owner_message=OWNER, created_at_utc=now(), source=identity(__file__),
        tests=identity(HERE / 'tests.json'), test_source=identity(HERE / 'test_resume.py'), cases=cases,
        original_training_source_sha256=TRAIN_SHA, original_training_authority_sha256=TRAIN_AUTH,
        source_files=auth['source_files'], runtime=auth['runtime'], output_root=str(OUTPUT),
        restore_step=313, final_step=391, additional_updates=78, additional_queries=9936,
        all_scientific_gpu_queues_in_pool_complete_first=True, both_original_lease_namespaces=True,
        reporting_disabled=True, endpoint_comparison='exact_all_tensors_and_states_no_tolerance',
        execution_authorized=True, automatic_retry=False, source_changed=False,
        physical_second_host=False, scientific_completion=False, source_publication=False)
    write(HERE / 'authorization.json', value)
    return dict(cases=2, actual_original_whole_run_reads=2, gpu_requested=False,
                authorization=identity(HERE / 'authorization.json'))


def original_download():
    bound(ORIGINAL_RESUME / 'resume.py', ORIGINAL_RESUME_SHA)
    original_auth = read(ORIGINAL_RESUME / 'authorization.json', ORIGINAL_RESUME_AUTH)
    downloaded = read(ORIGINAL_RESUME / 'downloaded.json')
    need(downloaded['source_sha256'] == ORIGINAL_RESUME_SHA
         and downloaded['authorization_sha256'] == ORIGINAL_RESUME_AUTH
         and downloaded['anonymous_actual_download'] is True
         and downloaded['native_cpu_checks'] == 2 and downloaded['files'] == 36,
         'Require the actual original anonymous download and native readback')
    for pool, case in original_auth['cases'].items():
        need(downloaded['cases'][pool]['binding']['path'] == case['download_checkpoint']
             and downloaded['cases'][pool]['files'] == case['files'], 'Original download/case differs')
    return original_auth, downloaded


def prepare(args):
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Preparation is CPU-only')
    bound(__file__, args.source_sha)
    no_existing(HERE / 'authorization.json')
    no_existing(OUTPUT)
    original_auth, _ = original_download()
    tests = read(HERE / 'tests.json')
    need(tests['exit_code'] == 0 and tests['tests_passed'] >= 20
         and tests['source'] == identity(__file__)
         and tests['test_source'] == identity(HERE / 'test_resume.py')
         and tests['extension_test_source'] == identity(HERE / 'test_resume_device.py')
         and tests['adapter_source'] == identity(HERE / 'step_devices.py'),
         'Require passing source-bound recovery and adapter controls')
    pins()
    t, training_auth, _ = training()
    admitted, _ = t.authenticate(SimpleNamespace(source_sha=TRAIN_SHA, authorization_sha=TRAIN_AUTH))
    need(admitted == training_auth and original_auth['source_files'] == training_auth['source_files'],
         'Original training authority/source differs')
    failures = {}
    for pool in CASES:
        path = ORIGINAL_RESUME / 'run' / ('pool-' + pool) / 'ranks.exited.json'
        need(read(path)['actual_rank_exits'] == {str(i): 1 for i in range(4)},
             'Preserve and bind the original failed attempts')
        failures[pool] = identity(path)
    value = dict(original_auth)
    value.update(scope=SCOPE, created_at_utc=now(), source=identity(__file__),
        tests=identity(HERE / 'tests.json'), test_source=identity(HERE / 'test_resume.py'),
        extension_test_source=identity(HERE / 'test_resume_device.py'),
        adapter_source=identity(HERE / 'step_devices.py'), output_root=str(OUTPUT),
        original_resume_authorization=identity(ORIGINAL_RESUME / 'authorization.json'),
        original_downloaded=identity(ORIGINAL_RESUME / 'downloaded.json'),
        original_failed_rank_receipts=failures, original_native_admission_reused=True,
        original_download_reused_without_relabelling=True, source_changed=True,
        original_numerical_source_changed=False, restoration_extension='post-load Adam step device placement',
        new_checkpoint_download_performed=False)
    write(HERE / 'authorization.json', value)
    return dict(cases=2, original_native_admission_reused=True, no_duplicate_download=True,
        adapter=identity(HERE / 'step_devices.py'), authorization=identity(HERE / 'authorization.json'),
        downloaded=identity(ORIGINAL_RESUME / 'downloaded.json'))


def authenticate(args, *, cpu=True):
    bound(__file__, args.source_sha)
    auth = read(HERE / 'authorization.json', args.authorization_sha)
    need(auth['scope'] == SCOPE and auth['owner_message'] == OWNER and auth['source'] == identity(__file__)
         and auth['output_root'] == str(OUTPUT) and auth['execution_authorized'] is True
         and auth['all_scientific_gpu_queues_in_pool_complete_first'] is True
         and auth['endpoint_comparison'] == 'exact_all_tensors_and_states_no_tolerance'
         and {p: c['run_id'] for p, c in auth['cases'].items()} == CASES, 'Wrong recovery authority')
    bound(HERE / 'tests.json', auth['tests'])
    bound(HERE / 'test_resume.py', auth['test_source'])
    bound(HERE / 'test_resume_device.py', auth['extension_test_source'])
    bound(HERE / 'step_devices.py', auth['adapter_source'])
    old_auth, _ = original_download()
    bound(ORIGINAL_RESUME / 'authorization.json', auth['original_resume_authorization'])
    bound(ORIGINAL_RESUME / 'downloaded.json', auth['original_downloaded'])
    need(auth['cases'] == old_auth['cases'] and auth['original_numerical_source_changed'] is False
         and auth['original_download_reused_without_relabelling'] is True
         and auth['restoration_extension'] == 'post-load Adam step device placement',
         'Restoration extension changed original cases or scientific computation')
    pins()
    t, original_auth, parent = training()
    need(auth['source_files'] == original_auth['source_files'] and auth['runtime'] == original_auth['runtime'],
         'Original numerical source/runtime identity differs')
    if cpu:
        need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU entry must hide CUDA')
    return auth, t, original_auth, parent


def historical_download_not_exposed(args):
    auth, t, original_auth, _ = authenticate(args)
    no_existing(HERE / 'downloads')
    no_existing(HERE / 'downloaded.json')
    from huggingface_hub import hf_hub_download
    from embed_optim import factorial_v3_run_contract as contract
    result = {}
    for pool, case in auth['cases'].items():
        def fetch(item):
            name, expected = item
            path = hf_hub_download(case['repository'], case['prefix'] + '/run/checkpoint-313/' + name,
                repo_type='model', revision=case['revision'], token=False,
                local_dir=HERE / 'downloads', force_download=True)
            bound(path, expected)
            return name
        with ThreadPoolExecutor(max_workers=4) as executor:
            names = list(executor.map(fetch, case['files'].items()))
        binding = {'path': case['download_checkpoint'], 'sha256': case['files']['factorial_trainer_component.json']['sha256']}
        checked = contract.inspect_checkpoint(binding, case['component_identity']['bound_factorial_run'],
                                              case['component_identity'])
        need(checked['step'] == 313 and len(names) == 18, 'Incomplete downloaded native checkpoint')
        result[pool] = dict(binding=binding, files=case['files'], native=checked,
                           revision=case['revision'], repository=case['repository'])
    import torch
    need(not torch.cuda.is_initialized(), 'Download/native CPU read initialized CUDA')
    t.c.check_imports(original_auth['source_files'])
    write(HERE / 'downloaded.json', dict(verified_at_utc=now(), cases=result,
        source_sha256=args.source_sha, authorization_sha256=args.authorization_sha,
        anonymous_actual_download=True, forced_download=True, files=36,
        bytes=sum(v['bytes'] for c in result.values() for v in c['files'].values()),
        native_cpu_checks=2, physical_second_host=False, gpu_resume_executed=False))
    return dict(files=36, native_cpu_checks=2, downloaded=identity(HERE / 'downloaded.json'))


def require_finished(pool, values):
    """Whole-pool completion, not a between-cell gap or a partial result count."""
    train, evaluation, probe = values
    e.require_pool_completion(train, pool)
    runs = set(e.queues()[pool])
    need(evaluation['source_sha256'] == EVAL_SHA and evaluation['authorization_sha256'] == EVAL_AUTH
         and evaluation['verified_tasks'] == 84 and set(evaluation['runs']) == runs, 'Incomplete evaluation pool')
    expected = {(run, step) for run in runs for step in STEPS}
    observed = [(r['run_id'], r['step']) for r in probe['checkpoints']]
    need(probe['source_sha256'] == PROBE_SHA and probe['authorization_sha256'] == PROBE_AUTH
         and len(observed) == len(set(observed)) == 30 and set(observed) == expected,
         'Incomplete five-stage probe pool')


def wait_science(pool, parent):
    roots = [TRAIN / 'run' / ('pool-' + pool), EVAL / 'run' / ('pool-' + pool),
             PROBE / 'run' / ('pool-' + pool)]
    paths = [roots[0] / 'coordinator.completed.json', *(p / 'completed.json' for p in roots[1:])]
    while True:
        for root in roots:
            need(not any((root / name).exists() for name in ('failed.json', 'coordinator.failed.json')),
                 'An upstream scientific queue failed; no recovery GPU request')
        if all(p.exists() for p in paths):
            try:
                values = [read(p) for p in paths]
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            require_finished(pool, values)
            return {str(p): identity(p) for p in paths}
        parent.handoff()
        time.sleep(30)


def recursive_equal(left, right, label='state'):
    """Every tensor/array bit pattern and every named scalar/container; no tolerance."""
    import numpy as np
    import torch
    need(type(left) is type(right), 'Type mismatch: ' + label)
    if isinstance(left, torch.Tensor):
        need(left.dtype == right.dtype and left.shape == right.shape
             and torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                             right.contiguous().reshape(-1).view(torch.uint8)), 'Tensor mismatch: ' + label)
        return {'tensors': 1, 'arrays': 0, 'scalars': 0}
    if isinstance(left, np.ndarray):
        need(left.dtype == right.dtype and left.shape == right.shape
             and left.tobytes() == right.tobytes(), 'Array mismatch: ' + label)
        return {'tensors': 0, 'arrays': 1, 'scalars': 0}
    if isinstance(left, dict):
        need(left.keys() == right.keys(), 'Key mismatch: ' + label)
        pairs = [(k, left[k], right[k]) for k in left]
    elif isinstance(left, (list, tuple)):
        need(len(left) == len(right), 'Length mismatch: ' + label)
        pairs = [(i, a, b) for i, (a, b) in enumerate(zip(left, right))]
    else:
        if isinstance(left, float):
            import math
            import struct
            need(math.isfinite(left) and math.isfinite(right)
                 and struct.pack('!d', left) == struct.pack('!d', right), 'Scalar mismatch: ' + label)
        else:
            need(left == right, 'Scalar mismatch: ' + label)
        return {'tensors': 0, 'arrays': 0, 'scalars': 1}
    count = {'tensors': 0, 'arrays': 0, 'scalars': 0}
    for name, a, b in pairs:
        for key, value in recursive_equal(a, b, label + '/' + str(name)).items():
            count[key] += value
    return count


def device_restore_callback(trainer, output, expected_counters, collective):
    """An explicit public Trainer callback; never replaces the native loader/kernel."""
    from transformers import TrainerCallback
    adapter = e.load('_bound_adam_step_device_placement', HERE / 'step_devices.py')

    class PlaceRestoredAdamCounters(TrainerCallback):
        def __init__(self):
            self.done = False

        def on_train_begin(self, args, state, control, **kwargs):
            def place():
                need(not self.done, 'Restore placement callback may run only once')
                raw = trainer._raw_optimizer()
                need(raw.completed_steps == 313, 'Original optimizer was not restored to step 313')
                original_identity = trainer.component_identity()
                report = adapter.place_adam_step_counters(raw, expected_step=313)
                need(report['named_adam_counters'] == expected_counters,
                     'Wrong actual named Adam state population')
                need(trainer.component_identity() == original_identity and raw.completed_steps == 313,
                     'Placement changed original component identity or step')
                write(output, dict(verified_at_utc=now(), adapter=identity(HERE / 'step_devices.py'),
                    placement=report, original_numerical_component_unchanged=True,
                    moments_not_reset=True, original_loader_used=True))
                self.done = True
            collective(place)
            return control

    return PlaceRestoredAdamCounters()


def worker(args):
    auth, t, original_auth, parent = authenticate(args, cpu=False)
    case = auth['cases'][args.pool]
    root = HERE / 'run' / ('pool-' + args.pool)
    admission = read(root / 'admission.json')
    need(os.getppid() == admission['coordinator_pid']
         and admission['source_sha256'] == args.source_sha
         and admission['authorization_sha256'] == args.authorization_sha, 'Wrong direct recovery parent')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == ','.join(t.POOLS[args.pool])
         and os.environ.get('WANDB_MODE') == 'disabled', 'Wrong recovery GPU/reporting environment')
    t.verify_leases(parent, args.pool, args.lease_fd)
    expected = dict(RANK=str(args.rank), LOCAL_RANK=str(args.rank), WORLD_SIZE='4', LOCAL_WORLD_SIZE='4',
                    MASTER_ADDR='127.0.0.1', MASTER_PORT=str(admission['port']))
    need(all(os.environ.get(k) == v for k, v in expected.items()), 'Wrong four-rank recovery identity')
    downloaded = read(ORIGINAL_RESUME / 'downloaded.json', admission['downloaded'])
    need(downloaded['authorization_sha256'] == ORIGINAL_RESUME_AUTH, 'Wrong original downloaded input authority')
    parent.handoff()
    from embed_optim.runtime import verify_runtime_spec
    need(verify_runtime_spec(t.SOURCE_ROOT / 'configs/formal_runtime.json') == auth['runtime'], 'Runtime changed')
    import torch
    import torch.distributed as dist
    need(not dist.is_initialized() and not torch.cuda.is_initialized(), 'Recovery inherited initialized CUDA')
    torch.cuda.set_device(args.rank)
    dist.init_process_group('nccl', init_method='env://', rank=args.rank, world_size=4, timeout=timedelta(minutes=3))
    try:
        total = torch.tensor(float(args.rank), device='cuda')
        dist.all_reduce(total)
        need(total.item() == 6 and torch.cuda.device_count() == 4, 'Actual recovery collective failed')
        from embed_optim import factorial_v3_factory as factory, factorial_v3_run_contract as contract
        from embed_optim import factorial_v3_checkpoint as cp
        from embed_optim.factorial_v3_bound_trainer import RunBoundFactorialTrainer, collective_phase
        from embed_optim.factorial_v3_trainer import FactorialTrainingArguments
        from transformers import set_seed
        _, _, locations = t.context(auth['source_files'])
        d = case['declaration']
        run_identity, dataset = collective_phase(lambda: contract.prepare_run(locations, d['calibrations'],
            d['state'], d['operator'], d['seed'], t.SOURCE_ROOT))
        need(run_identity == case['component_identity']['bound_factorial_run'], 'Resume changed run identity')
        config, values = factory.recipe(run_identity, locations.data_store / factory.BRANCH, OUTPUT,
                                        project=d['project'], entity=d['entity'])
        values['report_to'] = []  # No W&B mutation; no numerical/component-identity field changes.
        collective_phase(lambda: no_existing(config.output_dir))
        train_args = collective_phase(lambda: FactorialTrainingArguments(**values))
        set_seed(d['seed'])
        model, loss, _ = collective_phase(lambda: factory.load_model(run_identity, config))
        trainer = RunBoundFactorialTrainer(run_identity=run_identity, optimizer_config=config.optimizer,
            model=model, args=train_args, train_dataset=dataset, loss=loss,
            resume_binding=downloaded['cases'][args.pool]['binding'])
        need(trainer.component_identity() == case['component_identity'], 'Resume altered numerical component')
        trainer.add_callback(device_restore_callback(trainer,
            root / f'rank-{args.rank}.device-placement.json',
            134 if args.pool == 'a' else 46, collective_phase))
        write(root / f'rank-{args.rank}.admitted.json', dict(pid=os.getpid(), rank=args.rank,
            admitted_at_utc=now(), actual_all_reduce_sum=total.item(), same_numerical_component=True,
            resume_binding=downloaded['cases'][args.pool]['binding'], wandb_disabled=True))
        trainer.train(resume_from_checkpoint=case['download_checkpoint'])
        import dataclasses
        collective_phase(lambda: contract.require_final_state(dataclasses.asdict(trainer.state)))
        need(trainer._raw_optimizer().completed_steps == 391 and trainer._resume_step == 313,
             'Recovery did not retain the full original horizon')
        def finish():
            need(sorted(p.name for p in config.output_dir.glob('checkpoint-*')) == ['checkpoint-391'],
                 'Recovery produced an unexpected checkpoint population')
            binding = {'path': str(config.output_dir / 'checkpoint-391'),
                       'sha256': cp.file_digest(config.output_dir / 'checkpoint-391' / cp.NAME)}
            cp.read(binding, case['component_identity'])
            return binding
        result = collective_phase(finish)
        t.c.check_imports(original_auth['source_files'])
        need(t.source_files() == auth['source_files'], 'Source changed during recovery')
        write(root / f'rank-{args.rank}.returned.json', dict(returned_at_utc=now(), rank=args.rank,
            checkpoint=result, resumed_from=313, final_step=391, actual_additional_updates=78,
            process_exit_observed=False, endpoint_equivalence_unverified=True, scientific_completion=False))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def compare(args):
    auth, t, original_auth, _ = authenticate(args)
    case = auth['cases'][args.pool]
    root = HERE / 'run' / ('pool-' + args.pool)
    exits = read(root / 'ranks.exited.json')
    need(exits['actual_rank_exits'] == {str(i): 0 for i in range(4)}, 'Not four actual successful recovery exits')
    for rank in range(4):
        placed = read(root / f'rank-{rank}.device-placement.json')
        need(placed['adapter'] == auth['adapter_source']
             and placed['placement']['named_adam_counters'] == (134 if args.pool == 'a' else 46)
             and placed['placement']['counters_moved'] == (134 if args.pool == 'a' else 46)
             and placed['placement']['counter_values_bitwise_preserved'] is True,
             'Missing actual value-preserving GPU restoration placement')
    returned = [read(root / f'rank-{r}.returned.json') for r in range(4)]
    binding = returned[0]['checkpoint']
    need(all(v['checkpoint'] == binding and v['rank'] == r and v['actual_additional_updates'] == 78
             for r, v in enumerate(returned)), 'Recovery ranks disagree')
    from embed_optim import factorial_v3_run_contract as contract
    from safetensors.torch import load_file
    import torch
    original_binding = case['original_checkpoints']['391']
    native = {name: contract.inspect_checkpoint(b, case['component_identity']['bound_factorial_run'],
                                               case['component_identity'])
              for name, b in [('uninterrupted', original_binding), ('resumed', binding)]}
    left, right = Path(original_binding['path']), Path(binding['path'])
    counts = {'model': recursive_equal(load_file(left / 'model.safetensors'), load_file(right / 'model.safetensors'), 'model')}
    need(counts['model']['tensors'] == 134, 'Wrong complete model population')
    for name in ('optimizer.pt', 'scheduler.pt', *(f'rng_state_{r}.pth' for r in range(4))):
        # Pickle-bearing RNG payloads are original/native-content-authenticated above.
        weights_only = not name.startswith('rng_state_')
        counts[name] = recursive_equal(torch.load(left / name, map_location='cpu', weights_only=weights_only),
                                      torch.load(right / name, map_location='cpu', weights_only=weights_only), name)
    fields = ('global_step', 'max_steps', 'num_train_epochs', 'train_batch_size', 'epoch')
    a, b = read(left / 'trainer_state.json'), read(right / 'trainer_state.json')
    recursive_equal({k: a[k] for k in fields}, {k: b[k] for k in fields}, 'trainer-counters')
    need(not torch.cuda.is_initialized(), 'Comparison initialized CUDA')
    t.c.check_imports(original_auth['source_files'])
    write(root / 'comparison.json', dict(verified_at_utc=now(), run_id=case['run_id'], counts=counts,
        native=native, original=original_binding, resumed=binding, exact_all_model_tensors=True,
        exact_optimizer_scheduler_and_all_rank_rng=True, reporting_disabled=True,
        resume_from=313, final_step=391, additional_updates=78, additional_queries=9936,
        physical_second_host=False, all_checkpoints_resume_verified=False, scientific_completion=False))
    return dict(run_id=case['run_id'], exact_gpu_resume_endpoint=True, model_tensors=134)


def coordinate(args):
    auth, t, _, parent = authenticate(args)
    downloaded = read(ORIGINAL_RESUME / 'downloaded.json', args.downloaded_sha)
    need(downloaded['authorization_sha256'] == ORIGINAL_RESUME_AUTH, 'Wrong original download authority')
    root = HERE / 'run' / ('pool-' + args.pool)
    no_existing(root)
    root.mkdir(parents=True)
    owned = t.c.imported('_resume_exact_registered_identity', t.c.DISPATCH)
    write(root / 'started.json', dict(**owned.process_identity(os.getpid()), started_at_utc=now(),
        source_sha256=args.source_sha, authorization_sha256=args.authorization_sha,
        downloaded=identity(ORIGINAL_RESUME / 'downloaded.json'), pool=args.pool, gpu_leases_requested=False))
    children = []
    try:
        prerequisites = wait_science(args.pool, parent)
        auth, t, _, parent = authenticate(args)
        descriptors, lease = None, None
        while descriptors is None:
            parent.handoff()
            candidate = parent.leases(t.POOLS[args.pool])
            try:
                descriptors = candidate.__enter__()
                lease = candidate
            except BlockingIOError:
                time.sleep(1)
        with ExitStack() as stack:
            stack.callback(lease.__exit__, None, None, None)
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            write(root / 'admission.json', dict(coordinator_pid=os.getpid(), port=port,
                admitted_at_utc=now(), prerequisites=prerequisites, source_sha256=args.source_sha,
                authorization_sha256=args.authorization_sha, downloaded=identity(ORIGINAL_RESUME / 'downloaded.json'),
                both_original_leases_inherited=True, all_scientific_gpu_queues_completed=True))
            try:
                for rank in range(4):
                    argv = command(args, 'worker') + ['--rank', str(rank)]
                    for fd in descriptors:
                        argv += ['--lease-fd', str(fd)]
                    env = t.environment(parent, pool=args.pool, rank=rank, run_id=CASES[args.pool], port=port)
                    env['WANDB_MODE'] = 'disabled'
                    for key in ('WANDB_RUN_ID', 'WANDB_RESUME', 'WANDB_API_KEY'):
                        env.pop(key, None)
                    log = stack.enter_context((root / f'rank-{rank}.log').open('xb'))
                    child = subprocess.Popen(argv, cwd=t.SOURCE_ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, pass_fds=descriptors)
                    children.append(child)
                    write(root / f'rank-{rank}.started.json', {**owned.process_identity(child.pid, argv),
                        'rank': rank, 'command': argv, 'started_at_utc': now()})
                t.supervise(children)
            finally:
                if any(c.poll() is None for c in children):
                    t.finish_owned(children)
                write(root / 'ranks.exited.json', dict(observed_at_utc=now(),
                    actual_rank_exits={str(i): c.returncode for i, c in enumerate(children)}))
        argv = command(args, 'compare')
        with (root / 'comparison.log').open('xb') as log:
            reader = subprocess.Popen(argv, cwd=t.SOURCE_ROOT, env=t.environment(parent),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
            try:
                write(root / 'reader.started.json', {**owned.process_identity(reader.pid, argv),
                    'command': argv, 'started_at_utc': now()})
                code = reader.wait()
            finally:
                if reader.poll() is None:
                    t.finish_owned([reader])
                write(root / 'reader.exited.json', dict(exit_code=reader.returncode, observed_at_utc=now()))
        need(code == 0, 'Actual native comparison failed; no tolerance change or automatic retry')
        write(root / 'completed.json', dict(completed_at_utc=now(), source_sha256=args.source_sha,
            authorization_sha256=args.authorization_sha, run_id=CASES[args.pool], actual_rank_exits=[0]*4,
            actual_comparison_exit=0, comparison=identity(root / 'comparison.json'),
            physical_second_host=False, scientific_completion=False))
    except BaseException as exc:
        if any(c.poll() is None for c in children):
            t.finish_owned(children)
        write(root / 'failed.json', dict(failed_at_utc=now(), exception_type=type(exc).__name__,
            automatic_retry=False, all_partial_outputs_preserved=True, scientific_completion=False))
        raise


def command(args, mode):
    return ['/usr/bin/python', '-B', str(HERE / 'resume.py'), mode, '--source-sha', args.source_sha,
            '--authorization-sha', args.authorization_sha, '--pool', args.pool]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'coordinate', 'worker', 'compare'])
    p.add_argument('--source-sha', required=True)
    p.add_argument('--authorization-sha')
    p.add_argument('--downloaded-sha')
    p.add_argument('--pool', choices=['a', 'b'])
    p.add_argument('--rank', type=int, choices=range(4))
    p.add_argument('--lease-fd', type=int, action='append', default=[])
    args = p.parse_args()
    need(args.mode == 'prepare' or args.authorization_sha is not None, 'Missing explicit authority')
    need(args.mode == 'prepare' or args.pool is not None, 'Missing exact pool')
    need(args.mode != 'coordinate' or args.downloaded_sha is not None, 'Missing bound download receipt')
    need(args.mode != 'worker' or (args.rank is not None and len(set(args.lease_fd)) == len(args.lease_fd) == 8),
         'Missing exact four-rank dual-lease identity')
    need(args.mode == 'worker' or (args.rank is None and not args.lease_fd), 'CPU entry cannot inherit GPU rank leases')
    result = globals()[args.mode](args)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
