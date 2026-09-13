"""Wait for all actual continuation outcomes, then apply the frozen estimands.

The old historical summarizer and admission guards remain unchanged. This new
outer consumer has actual v3 source/provenance readers, fixed complete coverage,
the original numerical functions and a separate independent arithmetic check.
It cannot publish the manuscript or supply missing results from fixtures.
"""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent.resolve()
ORIGINAL = Path('/root/embedding-optimizer-story-refactor/src/embed_optim/state_operator_factorial_summary.py')
ORIGINAL_SHA = 'd6d4d8481404e38b0e7ab9291680c39a99909fc3fb87959bc42be49f4361cc27'
COLLECT_SHA = '3ae0c00fbb9f743037795536ef7091a91f9aebd61125ff46eee844310840dc90'
SCOPE = 'owner-authorized-complete-genuine-v3-factorial-inference-v1'
OWNER = '你有权做一切事情，目标是尽快完成任务'
OUTPUT = Path('/root/embedding-optimizer-v3-experiment/analyses/dense-v3-factorial-inference-v1')
PROTOCOL = ORIGINAL.parents[2] / 'configs/dense_no_packing_state_operator_factorial_protocol.json'
PROTOCOL_SHA = '5773943a3ae9b581021a0f7b85b162c74d5c497eeca1386578ff9d2c3bcafe76'
ESTIMANDS = ('weight_state_effect', 'operator_effect', 'state_operator_interaction')


def collector():
    path = HERE / 'collect.py'
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != COLLECT_SHA:
        raise ValueError('Actual collector source differs')
    spec = importlib.util.spec_from_file_location('_bound_actual_factorial_collectors', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c = collector()
e = c.e


def context():
    e.bound(ORIGINAL, ORIGINAL_SHA)
    protocol = e.read(PROTOCOL, PROTOCOL_SHA)
    e.bound(c.EVAL_ROOT / 'evaluate.py', c.EVAL_SHA)
    e.bound(c.EVAL_ROOT / 'authorization.json', c.EVAL_AUTH)
    e.bound(c.PROBE_ROOT / 'probe.py', c.PROBE_SHA)
    e.bound(c.PROBE_ROOT / 'authorization.json', c.PROBE_AUTH)
    e.bound(e.PARENT, e.PINS[e.PARENT])
    e.bound(e.OLD_EVAL, e.PINS[e.OLD_EVAL])
    e.bound(e.PRIMARY / 'configs/dense_primary_v3_protocol.json', e.PINS[e.PRIMARY / 'configs/dense_primary_v3_protocol.json'])
    tasks = e.read(e.PRIMARY / 'configs/dense_primary_v3_protocol.json')['evaluation']['tasks']
    training = e.read(e.TRAIN / 'authorization.json', e.AUTH_SHA)
    e.need(training['queues'] == e.queues() and len(tasks) == 14, 'Original population differs')
    e.need(protocol['branch_data']['order_seeds'] == [314159, 271828, 161803]
        and protocol['factorial_design']['training']['expected_checkpoint_steps'] == [79, 157, 235, 313, 391],
        'Scientific design changed')
    return protocol, tasks, training


def numerical_functions(tasks):
    import numpy as np
    e.bound(ORIGINAL, ORIGINAL_SHA)
    names = {'_effect_rows', 'two_way_cluster_bootstrap', '_estimand_summaries', '_cell_summaries'}
    nodes = [node for node in ast.parse(ORIGINAL.read_bytes()).body if isinstance(node, ast.FunctionDef) and node.name in names]
    e.need({n.name for n in nodes} == names, 'Original statistical functions missing')
    namespace = dict(np=np, Any=Any, defaultdict=defaultdict, statistics=statistics,
        DECONTAMINATED_TASK_NAMES=tuple(tasks), ESTIMANDS=ESTIMANDS)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ORIGINAL), 'exec'), namespace)
    return namespace


def validate_scores(rows, tasks, training):
    expected = set()
    for runs in e.queues().values():
        for run in runs:
            request = training['requests'][run]
            expected.update((run, request['state'], request['operator'], request['seed'], task) for task in tasks)
    found = []
    for row in rows:
        value = row['ndcg_at_10']
        e.need(type(row['seed']) is int and type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
               'Invalid normalized score or seed')
        found.append(tuple(row[k] for k in ('run_id', 'state', 'operator', 'seed', 'task')))
    e.need(len(found) == len(set(found)) == 168 and set(found) == expected, 'Missing/duplicate/unregistered score cell')


def validate_probe(value, tasks, training):
    expected = set()
    for runs in e.queues().values():
        for run in runs:
            req = training['requests'][run]
            for stage, step in enumerate((79, 157, 235, 313, 391), 1):
                expected.add((req['state'], req['operator'], req['seed'], stage, step, run + '/checkpoint-' + str(step)))
    def key(row):
        return tuple(row[k] for k in ('state', 'operator', 'seed', 'stage', 'step', 'label'))
    found = [key(row) for row in value['overall_rows']]
    e.need(len(found) == len(set(found)) == 60 and set(found) == expected, 'Incomplete five-stage probe states')
    found = [(key(row), row['task']) for row in value['task_rows']]
    e.need(len(found) == len(set(found)) == 840 and set(found) == {(x, task) for x in expected for task in tasks},
           'Incomplete all-task probe states')
    for row in value['overall_rows'] + value['task_rows']:
        e.need(row['fraction'] == row['stage'] / 5 and row['samples'] == (16 if 'task' in row else 224), 'Wrong stage/sample coverage')
        for name in ('contrastive_loss_mean', 'positive_margin_mean', 'positive_margin_p05',
                     'mean_reciprocal_rank', 'top1_accuracy', 'pretrained_top1_agreement'):
            e.need(type(row[name]) in (int, float) and math.isfinite(row[name]), 'Invalid original probe metric')


def infer(beir, probe, tasks, training):
    validate_scores(beir['rows'], tasks, training)
    validate_probe(probe, tasks, training)
    original = numerical_functions(tasks)
    effects = original['_effect_rows'](beir['rows'])
    return dict(beir_seed_task_scores=beir['rows'], factorial_cell_summary=original['_cell_summaries'](beir['rows']),
        estimand_seed_task_contrasts=effects, estimand_summary=original['_estimand_summaries'](effects),
        probe_checkpoint_metrics=probe['overall_rows'], probe_task_metrics=probe['task_rows'])


def independent(tables):
    """Rational four-cell algebra and multiplicity-weighted bootstrap, independently."""
    import numpy as np
    scores = tables['beir_seed_task_scores']
    indexed = {(r['seed'], r['task'], r['state'], r['operator']): Fraction.from_float(float(r['ndcg_at_10'])) for r in scores}
    seeds, tasks = sorted({r['seed'] for r in scores}), sorted({r['task'] for r in scores})
    exact = {}
    for seed in seeds:
        for task in tasks:
            aa, am, ma, mm = [indexed[seed, task, state, operator] for state, operator in
                (('adamw_state', 'adamw'), ('adamw_state', 'muon'), ('muon_state', 'adamw'), ('muon_state', 'muon'))]
            exact['weight_state_effect', seed, task] = (ma + mm - aa - am) / 2
            exact['operator_effect', seed, task] = (am + mm - aa - ma) / 2
            exact['state_operator_interaction', seed, task] = mm + aa - ma - am
            e.need(mm - aa == exact['weight_state_effect', seed, task] + exact['operator_effect', seed, task],
                   'Exact diagonal identity failed')
    for row in tables['estimand_seed_task_contrasts']:
        value = float(exact[row['estimand'], row['seed'], row['task']])
        e.need(abs(value - row['contrast_ndcg_at_10']) <= 2e-15, 'Independent exact effect differs')
    rng = np.random.default_rng(20260904)
    seed_counts = np.eye(3, dtype=np.int64)[rng.integers(0, 3, size=(100000, 3))].sum(axis=1)
    task_counts = np.eye(14, dtype=np.int64)[rng.integers(0, 14, size=(100000, 14))].sum(axis=1)
    checks = []
    for row in tables['estimand_summary']:
        name = row['estimand']
        values = np.array([[float(exact[name, seed, task]) for task in tasks] for seed in seeds], dtype=np.float64)
        draws = ((seed_counts @ values) * task_counts).sum(axis=1) / 42
        ordered = np.sort(draws)
        def percentile(q):
            location = q * (len(ordered) - 1)
            left = math.floor(location)
            return float(ordered[left] + (location - left) * (ordered[math.ceil(location)] - ordered[left]))
        low, high = percentile(.025), percentile(.975)
        point = float(sum((exact[name, seed, task] for seed in seeds for task in tasks), Fraction(0)) / 42)
        decision = 'supported_positive' if low > 0 else 'supported_negative' if high < 0 else 'inconclusive'
        errors = [abs(point - row['point_estimate']), abs(low - row['bootstrap_ci_95_lower']), abs(high - row['bootstrap_ci_95_upper'])]
        e.need(max(errors) <= 2e-15 and decision == row['decision'], 'Independent full bootstrap differs')
        checks.append(dict(estimand=name, max_absolute_difference=max(errors), decision=decision, draws=100000))
    for row in tables['factorial_cell_summary']:
        values = [value for (seed, task, state, op), value in indexed.items() if (state, op) == (row['state'], row['operator'])]
        mean = sum(values, Fraction(0)) / 42
        variance = sum(((v - mean) ** 2 for v in values), Fraction(0)) / 42
        e.need(abs(float(mean) - row['mean_ndcg_at_10']) <= 2e-15
               and abs(math.sqrt(float(variance)) - row['population_std_ndcg_at_10']) <= 2e-15,
               'Independent exact cell moments differ')
    return dict(independent_rational_seed_task_effects=126, independent_cell_moments=4,
        independent_multiplicity_bootstrap=checks, tolerance=2e-15, original_estimates_not_replaced=True,
        intervals='three marginal 95% intervals; not simultaneous', scientific_completion=False)


def ready():
    for root in (c.EVAL_ROOT, c.PROBE_ROOT):
        for pool in ('a', 'b'):
            e.need(not (root / 'run' / ('pool-' + pool) / 'failed.json').exists(), 'Upstream queue failed')
    try:
        c.require_complete_pair('beir')
        c.require_complete_pair('probe')
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return True


def prepare(args):
    e.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Prepare must hide CUDA')
    e.bound(__file__, args.source_sha256)
    e.need(not (HERE / 'authorization.json').exists() and not OUTPUT.exists(), 'Keep every prior attempt')
    protocol, tasks, training = context()
    tests = e.read(HERE / 'tests.json', args.tests_sha256)
    e.need(tests['source'] == e.identity(__file__) and tests['collector_source']['sha256'] == COLLECT_SHA
           and tests['tests_run'] >= 12 and tests['failures'] == tests['errors'] == tests['skipped'] == 0, 'Missing bounded inference checks')
    value = dict(scope=SCOPE, created_at_utc=e.now(), owner_message=OWNER, automatic_continuation=False,
        source=e.identity(__file__), collector_source=e.identity(HERE / 'collect.py'), tests=e.identity(HERE / 'tests.json'),
        test_source=e.identity(HERE / 'test_summary.py'), original_numerical_source_sha256=ORIGINAL_SHA,
        scientific_protocol_sha256=PROTOCOL_SHA, evaluation_authorization_sha256=c.EVAL_AUTH,
        probe_authorization_sha256=c.PROBE_AUTH, tasks=tasks, estimands=list(ESTIMANDS),
        required=dict(training_runs=12, full_beir_tasks=168, checkpoint_probes=60, probe_task_rows=840, seed_task_contrasts=126),
        bootstrap=dict(samples=100000, seed=20260904, interval='linear-percentile-marginal-95%'),
        output_root=str(OUTPUT), gpu_access=False, independent_full_arithmetic_check_required=True,
        old_guards_modified=False, manuscript_installation=False, source_release=False, scientific_completion=False)
    e.write(HERE / 'authorization.json', value)
    return dict(authorization=e.identity(HERE / 'authorization.json'), upstream_ready=ready(), gpu_access=False, scientific_completion=False)


def authenticate(args):
    e.bound(__file__, args.source_sha256)
    auth = e.read(HERE / 'authorization.json', args.authorization_sha256)
    e.need(auth['source'] == e.identity(__file__) and auth['scope'] == SCOPE and auth['owner_message'] == OWNER
        and auth['output_root'] == str(OUTPUT) and auth['estimands'] == list(ESTIMANDS), 'Inference authority changed')
    for name, key in (('collect.py', 'collector_source'), ('tests.json', 'tests'), ('test_summary.py', 'test_source')):
        e.bound(HERE / name, auth[key])
    context()
    return auth


def coordinate(args):
    e.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Inference coordinator must hide CUDA')
    auth = authenticate(args)
    root = HERE / 'run'
    root.mkdir(exist_ok=False)
    old = e.load('_unchanged_own_inference_process_identity', e.OLD_EVAL)
    parent = e.load('_unchanged_cpu_reader_environment', e.PARENT)
    e.write(root / 'started.json', dict(pid=os.getpid(), start_ticks=old.process_start_ticks(os.getpid()),
        started_at_utc=e.now(), source_sha256=args.source_sha256, authorization_sha256=args.authorization_sha256,
        gpu_access=False, scientific_completion=False))
    try:
        while not ready():
            time.sleep(30)
        authenticate(args)
        values = {}
        for kind in ('beir', 'probe'):
            output = root / (kind + '.json')
            argv = ['/usr/bin/python', '-B', str(HERE / 'collect.py'), kind, '--source-sha256', COLLECT_SHA, '--output', str(output)]
            env = parent.environment()
            env.update(PYTHONPATH=str(e.PRIMARY / 'src') if kind == 'beir' else str(c.PROBE_ROOT / 'source/src'))
            with (root / (kind + '.log')).open('x') as log:
                child = subprocess.Popen(argv, cwd=HERE, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                e.write(root / (kind + '.started.json'), dict(pid=child.pid, start_ticks=old.process_start_ticks(child.pid),
                    command=argv, started_at_utc=e.now(), gpu_access=False))
                code = child.wait()
            e.write(root / (kind + '.exited.json'), dict(exit_code=code, completed_at_utc=e.now()))
            e.need(code == 0, 'Actual original outcome collector failed; preserve attempt')
            values[kind] = e.read(output)
        _, tasks, training = context()
        result = infer(values['beir'], values['probe'], tasks, training)
        verified = independent(result)
        OUTPUT.mkdir(parents=True, exist_ok=False)
        e.write(OUTPUT / 'tables.json', result)
        e.write(OUTPUT / 'independent_verification.json', verified)
        for name, rows in result.items():
            fields = list(rows[0])
            e.need(all(list(row) == fields for row in rows), 'Output table columns differ')
            with (OUTPUT / (name + '.csv')).open('x', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        authenticate(args)
        receipt = dict(scope=SCOPE, completed_at_utc=e.now(), source=e.identity(__file__),
            authorization_sha256=args.authorization_sha256, original_statistical_source_sha256=ORIGINAL_SHA,
            actual_native_collectors={kind: e.identity(root / (kind + '.json')) for kind in values},
            actual_collector_exits={'beir': 0, 'probe': 0}, table_counts={k: len(v) for k, v in result.items()},
            outputs={p.name: e.identity(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()},
            inference=dict(samples=100000, seed=20260904, intervals='three marginal linear percentile 95%'),
            independent_arithmetic_verified=True, manuscript_installed=False, scientific_completion=False)
        e.write(OUTPUT / 'readout.json', receipt)
        e.write(root / 'completed.json', dict(completed_at_utc=e.now(), readout=e.identity(OUTPUT / 'readout.json'),
            source_sha256=args.source_sha256, authorization_sha256=args.authorization_sha256, scientific_completion=False))
        print(json.dumps(dict(readout=e.identity(OUTPUT / 'readout.json'), table_counts=receipt['table_counts'], scientific_completion=False)), flush=True)
    except BaseException as exc:
        e.write(root / 'failed.json', dict(failed_at_utc=e.now(), exception_type=type(exc).__name__, scientific_completion=False))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'coordinate', 'inspect'))
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--authorization-sha256')
    p.add_argument('--tests-sha256')
    args = p.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(args)), flush=True)
    elif args.action == 'inspect':
        authenticate(args)
        print(json.dumps(dict(upstream_ready=ready(), gpu_access=False, scientific_completion=False)), flush=True)
    else:
        coordinate(args)


if __name__ == '__main__':
    main()
