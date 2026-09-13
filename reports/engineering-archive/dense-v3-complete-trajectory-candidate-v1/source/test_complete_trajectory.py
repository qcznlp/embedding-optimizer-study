"""New input-adapter checks, not new model runs or a complete trajectory result."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location('complete_trajectory_candidate', HERE / 'complete_trajectory.py')
T = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(T)
PRIOR = Path('/root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-v3-fourth-stage-pool-b-evaluations-v1/actual/complete-checkpoint-readback.json')
ENDPOINT = Path('/root/embedding-optimizer-story-refactor/reports/dense-v3-final-inference-v1/tables/readout.json')


@pytest.fixture(scope='module')
def prior():
    return T.strict_json(T.read_bound(PRIOR, T.PRIOR_54)[0])


@pytest.mark.parametrize('index', range(54))
def test_each_actual_accepted_record_reconstructs_14_scores(prior, index):
    record = prior['checkpoints'][index]
    rows = T.validate_record(record)
    assert len(rows) == 14
    assert [r['task'] for r in rows] == list(T.TASKS)
    assert [r['ndcg_at_10'] for r in rows] == [r['ndcg_at_10'] for r in record['native_complete_reread']['tasks']]
    assert all(r['run_id'] == record['run_id'] and r['step'] == record['step'] for r in rows)


def replace_embedded(item, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    item.update(bytes=len(raw), sha256=T.sha(raw), text=raw.decode())


@pytest.mark.parametrize('mutation', ['macro', 'points', 'rational', 'step', 'duplicate_raw',
    'missing_raw', 'alias', 'task_order', 'complete_scope', 'operational_scope',
    'operational_auth', 'worker_exit', 'worker_pid', 'worker_start', 'worker_lease',
    'worker_command', 'worker_alias', 'worker_missing', 'cache_key'])
def test_bad_record_semantics_refused_after_mock_reembedding(prior, mutation):
    """The re-embedding deliberately bypasses the separate external bundle anchor."""
    row = copy.deepcopy(prior['checkpoints'][0])
    if mutation in ('macro', 'points', 'rational'):
        field = {'macro': 'macro_ndcg_at_10', 'points': 'macro_score_0_to_100',
                 'rational': 'exact_binary64_input_mean'}[mutation]
        row[field] = 0
    elif mutation == 'step':
        row['step'] = 3126
    elif mutation == 'duplicate_raw':
        row['raw_score_and_metadata_snapshots'].append(row['raw_score_and_metadata_snapshots'][0])
    elif mutation == 'missing_raw':
        row['raw_score_and_metadata_snapshots'].pop()
    elif mutation in ('alias', 'task_order', 'complete_scope', 'cache_key'):
        complete = row['native_complete_reread']
        if mutation == 'alias':
            complete['tasks'][9]['task'] = 'Quora'
        elif mutation == 'task_order':
            complete['tasks'][0], complete['tasks'][1] = complete['tasks'][1], complete['tasks'][0]
        elif mutation == 'complete_scope':
            complete['scientific_completion'] = True
        else:
            complete['plan']['cache_key'] = '0' * 64
        replace_embedded(row['original_complete_receipt'], complete)
    elif mutation.startswith('operational_'):
        original = T.strict_json(row['original_operational_receipt']['text'])
        original['committed_source_release' if mutation == 'operational_scope' else 'authorization_sha256'] = True
        replace_embedded(row['original_operational_receipt'], original)
    elif mutation == 'worker_missing':
        row['original_workers'].pop()
    else:
        which = 'exited' if mutation in ('worker_exit', 'worker_pid') else 'started'
        item = row['original_workers'][0][which]
        original = T.strict_json(item['text'])
        if mutation == 'worker_exit':
            original['exit_code'] = 1
        elif mutation == 'worker_pid':
            original['pid'] += 1
        elif mutation == 'worker_start':
            original['start_ticks'] = 0
        elif mutation == 'worker_lease':
            original['both_gpu_lease_namespaces_inherited'] = False
        elif mutation == 'worker_alias':
            original['job'][2] = 'Quora'
        else:
            original['command'][original['command'].index('--worker') + 1] = 'Quora'
        replace_embedded(item, original)
    with pytest.raises(ValueError):
        T.validate_record(row)


def test_actual_partial_grid_cannot_create_output(tmp_path):
    output = tmp_path / 'must-not-exist'
    args = SimpleNamespace(repository=HERE / 'source-original', bundle=PRIOR,
                           bundle_sha256=T.PRIOR_54, prior_bundle=PRIOR,
                           endpoint_readout=ENDPOINT, output=output)
    with pytest.raises(ValueError, match='Require all 60 checkpoints'):
        T.build(args)
    assert not output.exists()


@pytest.mark.parametrize('raw', ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_noncanonical_json_is_refused(raw):
    with pytest.raises(ValueError):
        T.strict_json(raw)


def test_external_anchor_refuses_changed_bundle():
    with pytest.raises(ValueError, match='external SHA-256'):
        T.read_bound(PRIOR, '0' * 64)


def test_symlinked_supplied_file_refused(tmp_path):
    link = tmp_path / 'link.json'
    link.symlink_to(PRIOR)
    with pytest.raises(ValueError, match='Ordinary absolute'):
        T.read_bound(link, T.PRIOR_54)


def test_original_kernel_loads_without_importing_project_or_torch():
    import sys
    original = set(sys.modules)
    function, identities, functions, parent = T.load_kernel(HERE / 'source-original')
    assert function.__name__ == 'summarize_score_rows' and len(functions) == 5 and len(identities) == 5
    assert parent['inference']['bootstrap_seed'] == 20260903
    assert not any(name == 'torch' or name.startswith(('torch.', 'embed_optim.'))
                   for name in set(sys.modules) - original)


def test_only_declared_auxiliary_nan_is_disclosed_without_changing_source_bytes():
    raw = b'{"scores":{"test":[{"ndcg_at_10":0.4,"nauc_ndcg_at_10_max":NaN}]}}'
    original = bytes(raw)
    value, undefined = T.mteb_json(raw)
    assert value['scores']['test'][0]['ndcg_at_10'] == .4
    assert value['scores']['test'][0]['nauc_ndcg_at_10_max'] is None
    assert undefined == [{'split': 'test', 'subset_row': 0, 'metric': 'nauc_ndcg_at_10_max'}]
    assert raw == original


@pytest.mark.parametrize('raw', [
    '{"scores":{"test":[{"ndcg_at_10":NaN}]}}',
    '{"scores":{"test":[{"main_score":NaN}]}}',
    '{"scores":{"test":[{"nauc_ndcg_at_10_max":Infinity}]}}',
    '{"scores":{"test":[{"nauc_ndcg_at_2_max":NaN}]}}',
    '{"evaluation_time":NaN,"scores":{"test":[]}}',
    '{"scores":{"test":[{"ndcg_at_10":0.4,"ndcg_at_10":0.5}]}}',
])
def test_nonfinite_scored_unknown_or_metadata_fields_and_duplicate_keys_refused(raw):
    with pytest.raises(ValueError):
        T.mteb_json(raw)


def test_auxiliary_policy_is_the_exact_original_ast_not_a_broader_nan_exception():
    original = ast.parse(T.read_bound(
        HERE / 'source-original/src/embed_optim/primary_completion.py',
        T.SOURCES['src/embed_optim/primary_completion.py'])[0])
    candidate = ast.parse((HERE / 'complete_trajectory.py').read_text())

    def policy(tree, name):
        function, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        start, = [i for i, n in enumerate(function.body)
                  if isinstance(n, ast.Assign) and len(n.targets) == 1
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'undefined']
        return ast.dump(ast.Module(body=function.body[start:start + 3], type_ignores=[]),
                        include_attributes=False)

    assert policy(original, 'read_mteb_result') == policy(candidate, 'mteb_json')


@pytest.mark.parametrize('flag', T.FALSE_FLAGS)
def test_partial_bundle_cannot_upgrade_native_scope(prior, flag):
    changed = copy.deepcopy(prior)
    changed[flag] = True
    with pytest.raises(ValueError, match='Wrong native reader scope'):
        T.complete_rows(changed, prior)


def test_producer_paths_are_provenance_only(prior, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('An embedded producer path was opened')

    monkeypatch.setattr(Path, 'open', forbidden)
    monkeypatch.setattr(Path, 'resolve', forbidden)
    for record in prior['checkpoints']:
        assert len(T.validate_record(record)) == 14


def test_observed_auc_audit_on_explicitly_synthetic_flat_curves():
    """A math fixture only: no native evidence or scientific result is fabricated."""
    rows, run_stage, optimizer_stage, areas = [], [], [], []
    for i, run in enumerate(T.RUNS):
        value = (i + 1) / 16
        for stage in range(1, 6):
            rows.extend({'run_id': run, 'stage': stage, 'task': task, 'ndcg_at_10': value}
                        for task in T.TASKS)
            run_stage.append({'run_id': run, 'stage': stage, 'mean_ndcg_at_10': value})
        areas.append({'run_id': run, 'observed_auc_20_to_100': value * .8,
                      'observed_mean_20_to_100': value})
    for i, optimizer in enumerate(T.RATE_NAMES):
        value = sum((j + 1) / 16 for j in range(i * 4, i * 4 + 4)) / 4
        optimizer_stage.extend({'optimizer': optimizer, 'stage': stage,
                                'mean_ndcg_at_10_across_rates': value}
                               for stage in range(1, 6))
    tables = {'run_stage_scores': run_stage, 'optimizer_stage_scores': optimizer_stage,
              'run_observed_auc': areas}
    assert T.verify_dynamics(rows, tables)['initialization_imputed'] is False
    tables['run_observed_auc'][0]['observed_auc_20_to_100'] /= 2
    with pytest.raises(ValueError, match='observed-range area differs'):
        T.verify_dynamics(rows, tables)
