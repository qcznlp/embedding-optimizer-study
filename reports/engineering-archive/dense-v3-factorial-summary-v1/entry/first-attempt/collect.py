"""Read complete actual factorial results in separate original source namespaces.

No GPU, training, evaluation launch, historical guard override or paper write.
BEIR uses the primary native task reader; probes use the unchanged actual
encoder-bundle reader and numerical metric reconstruction.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

EVAL_ROOT = Path('/tmp/dense-v3-factorial-evaluation.IOV7MK93')
EVAL_SHA = '45577537257e1dff2b7c3a26ee12f7564fbb476daa256e7b0f09300305b68cba'
EVAL_AUTH = '3a1ae32fdde63abf386603208e413ea11bf0c169eb1923f9dd7e1da529ffa29c'
PROBE_ROOT = Path('/tmp/dense-v3-factorial-probe.ryorjXJK')
PROBE_SHA = 'eb4fc27738adaecd9614ba5822255a5dc37de242afabfa53ff8cd8a6c8c73334'
PROBE_AUTH = '00d6b42f922b4428352d5b9fc5e5fa7e0436a4aa20ee0c693e57b9f645398ded'
PRIMARY = Path('/root/embedding-optimizer-primary-v3')


def load(name, path, digest):
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('Source changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e = load('_unchanged_actual_eval_result_reader', EVAL_ROOT / 'evaluate.py', EVAL_SHA)


def require_complete_pair(kind):
    root, source, authority = (EVAL_ROOT, EVAL_SHA, EVAL_AUTH) if kind == 'beir' else (PROBE_ROOT, PROBE_SHA, PROBE_AUTH)
    result = {}
    for pool in ('a', 'b'):
        path = root / 'run' / ('pool-' + pool) / 'completed.json'
        value = e.read(path)
        e.need(value['source_sha256'] == source and value['authorization_sha256'] == authority, 'Completion source/authority differs')
        if kind == 'beir':
            e.need(value['verified_tasks'] == 84 and set(value['runs']) == set(e.queues()[pool]), 'Incomplete final BEIR pool')
        else:
            expected = [(r, s) for r in e.queues()[pool] for s in (79, 157, 235, 313, 391)]
            e.need([(r['run_id'], r['step']) for r in value['checkpoints']] == expected, 'Incomplete five-stage probe pool')
        result[pool] = dict(value=value, path=str(path), binding=e.identity(path))
    return result


def beir():
    pools = require_complete_pair('beir')
    sys.path.insert(0, str(PRIMARY / 'src'))
    auth, _, _ = e.authorized(SimpleNamespace(source_sha256=EVAL_SHA, authorization_sha256=EVAL_AUTH))
    from embed_optim.primary_completion import task_score
    versions = e.read(PRIMARY / 'configs/formal_runtime.json')['packages']
    rows, sources = [], []
    training = e.read(e.TRAIN / 'authorization.json', e.AUTH_SHA)
    for pool, runs in e.queues().items():
        root = EVAL_ROOT / 'run' / ('pool-' + pool)
        for run in runs:
            result_root = e.RESULTS / run / 'checkpoint-391'
            whole_path = result_root / 'all-fourteen-tasks-verified.json'
            whole = e.read(whole_path, pools[pool]['value']['runs'][run])
            admission_path = root / 'native' / (run + '.json')
            admission = e.read(admission_path, whole['native_input'])
            e.need(e.read(admission_path.with_suffix('.exited.json'))['exit_code'] == 0
                and admission['reader_source']['sha256'] == e.READER_SHA
                and admission['training_authorization_sha256'] == e.AUTH_SHA
                and admission['actual_rank_exits'] == [0, 0, 0, 0]
                and admission['actual_original_reader_exit'] == 0, 'Missing genuine complete native input')
            request = training['requests'][run]
            e.need(whole['run_id'] == run and whole['step'] == 391 and whole['task_count'] == 14
                and all(whole[k] == admission[k] == request[k] for k in ('state', 'operator', 'seed')),
                'Wrong continuation state/operator/seed')
            e.rehash_model(admission)
            fresh = []
            for task in auth['tasks']:
                prefix = root / 'jobs' / f'{run}-391-{task}'
                exited = e.read(prefix.with_suffix('.exited.json'))
                started = e.read(prefix.with_suffix('.started.json'))
                expected_job = [run, 391, task]
                e.need(exited['exit_code'] == 0 and exited['job'] == started['job'] == expected_job
                    and exited['pid'] == started['pid'], 'Missing actual exit-zero evaluation worker')
                paths = list((result_root / 'dense').rglob(task + 'Decontaminated.json'))
                e.need(len(paths) == 1, 'Missing/duplicate actual full-corpus task')
                score = task_score(paths[0], task, run, 391, 8192, auth['task_revisions'][task]['revision'], versions)
                previous = e.read(prefix.with_suffix('.task-verified.json'))
                e.need(previous['task'] == task and previous['ndcg_at_10'] == score['ndcg_at_10'], 'Original accepted score changed')
                fresh.append(score)
                rows.append(dict(state=admission['state'], operator=admission['operator'], seed=admission['seed'],
                    run_id=run, task=task, ndcg_at_10=score['ndcg_at_10'], result_path=str(paths[0])))
                sources.append(dict(run_id=run, task=task, native_result=score,
                    worker_started=e.identity(prefix.with_suffix('.started.json')), worker_exited=e.identity(prefix.with_suffix('.exited.json'))))
            e.need(fresh == whole['scores'] and len(list((result_root / 'dense').rglob('*Decontaminated.json'))) == 14,
                   'Complete native final-settings readback differs')
            e.rehash_model(admission)
    e.need(len(rows) == 168 and len({(r['state'], r['operator'], r['seed'], r['task']) for r in rows}) == 168,
           'Incomplete genuine factorial retrieval panel')
    for item in pools.values():
        e.bound(item['path'], item['binding'])
    e.authorized(SimpleNamespace(source_sha256=EVAL_SHA, authorization_sha256=EVAL_AUTH))
    return dict(scope='actual-complete-genuine-v3-factorial-beir-readback', verified_at_utc=e.now(),
        rows=rows, sources=sources, pools=pools, tasks=auth['tasks'],
        actual_exit_zero_workers=168, fresh_original_task_reader=True, model_encoding_repeated=False,
        historical_factorial_guard_admission=False, scientific_completion=False)


def probe():
    pools = require_complete_pair('probe')
    p = load('_unchanged_actual_probe_result_reader', PROBE_ROOT / 'probe.py', PROBE_SHA)
    auth, c = p.authenticate(SimpleNamespace(source_sha256=PROBE_SHA, authorization_sha256=PROBE_AUTH))
    overall, tasks, sources = [], [], []
    for pool, runs in e.queues().items():
        root = PROBE_ROOT / 'run' / ('pool-' + pool)
        expected = {(r['run_id'], r['step']): r['verified'] for r in pools[pool]['value']['checkpoints']}
        for run in runs:
            for stage, step in enumerate(p.STEPS, 1):
                prefix = root / f'{run}-checkpoint-{step}'
                request_path = prefix.with_suffix('.request.json')
                request = e.read(request_path)
                verified = e.read(prefix.with_suffix('.verified.json'), expected[(run, step)])
                e.need(e.read(prefix.with_suffix('.exited.json'))['exit_code'] == 0
                    and request['source_sha256'] == PROBE_SHA and request['authorization_sha256'] == PROBE_AUTH
                    and request['run_id'] == run and request['step'] == step, 'Missing actual complete probe worker')
                fresh = p.verify_result(c, request, verified['worker_completion'])
                e.need(fresh['all_four_arrays_and_six_metrics_reconstructed'] is True and fresh['tasks'] == 14
                       and fresh['samples'] == 224, 'Incomplete fresh numerical probe reconstruction')
                result = e.read(Path(request['output']) / 'metrics.json')
                cp = request['checkpoint_admission']
                metadata = dict(state=cp['state'], operator=cp['operator'], seed=cp['seed'], stage=stage,
                                fraction=stage / 5, step=step, label=run + '/checkpoint-' + str(step))
                overall.append({**metadata, **result['overall']})
                tasks.extend({**metadata, 'task': task, **value} for task, value in sorted(result['by_task'].items()))
                sources.append(dict(run_id=run, step=step, source_verification=expected[(run, step)],
                    request=e.identity(request_path), actual_worker_exit=e.identity(prefix.with_suffix('.exited.json')),
                    fresh_native_readback=fresh, metrics=e.identity(Path(request['output']) / 'metrics.json')))
    for name, binding in auth['reference_files'].items():
        e.bound(p.OUTPUT / 'pretrained' / name, binding)
    e.need(len(overall) == 60 and len(tasks) == 840, 'Incomplete all-stage probe panel')
    for item in pools.values():
        e.bound(item['path'], item['binding'])
    p.check_imports(c['numerical_sources'])
    return dict(scope='actual-complete-genuine-v3-factorial-probe-readback', verified_at_utc=e.now(),
        overall_rows=overall, task_rows=tasks, sources=sources, pools=pools,
        pretrained_reference=e.read(p.OUTPUT / 'pretrained/metrics.json'), actual_exit_zero_workers=60,
        raw_arrays_and_metrics_reconstructed=True, model_encoding_repeated=False,
        historical_factorial_guard_admission=False, scientific_completion=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind', choices=('beir', 'probe'))
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    e.bound(__file__, args.source_sha256)
    e.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and args.output.is_absolute() and not args.output.exists(),
           'Require CPU-only new absolute collector output')
    value = beir() if args.kind == 'beir' else probe()
    import torch
    e.need(not torch.cuda.is_initialized(), 'Result collection initialized CUDA')
    value['collector_source'] = e.identity(__file__)
    e.write(args.output, value)
    print(json.dumps(dict(kind=args.kind, output=e.identity(args.output), scientific_completion=False)), flush=True)


if __name__ == '__main__':
    main()
