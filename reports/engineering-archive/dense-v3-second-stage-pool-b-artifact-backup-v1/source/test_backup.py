"""Bounded new-adapter checks; not training, statistical or GPU tests."""

import copy
import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('second_stage_pool_b_backup_under_test', Path(__file__).with_name('backup.py'))
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


@pytest.fixture(scope='module')
def actual():
    t = B.load_transport()
    bundle, rows, selected = B.source_inputs(t)
    return t, bundle, rows, selected


def test_actual_complete_second_stage_pool_b_selection_and_safety(actual):
    t, bundle, rows, selected = actual
    assert len(rows) == 6 and [r['run_id'] for r in rows] == list(B.RUNS)
    assert len(selected) == 277
    assert sum(n.startswith('native/') for n in selected) == 108
    assert sum(n.startswith('provenance/') for n in selected) == 168
    for name, (path, identity) in selected.items():
        assert path.suffix in ('.json', '.jsonl', '.md')
        t.compare_file(path, identity)
        t.scan_text(path)
    assert B.TASKS[9] == 'QuoraRetrieval'


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'wrong_stage', 'wrong_protocol',
    'wrong_completion', 'wrong_recompute', 'missing_task', 'alias', 'worker_missing', 'task_duplicate'])
def test_incomplete_or_wrong_cohort_refused(actual, mutation):
    bundle = copy.deepcopy(actual[1])
    row = next(r for r in bundle['checkpoints'] if r['step'] == 1563)
    if mutation == 'missing':
        bundle['checkpoints'].pop()
    elif mutation == 'duplicate':
        bundle['checkpoints'][-1] = copy.deepcopy(bundle['checkpoints'][0])
    elif mutation == 'wrong_stage':
        row['step'] = 2345
    elif mutation == 'wrong_protocol':
        bundle['protocol_sha256'] = '0' * 64
    elif mutation == 'wrong_completion':
        bundle['scientific_completion'] = True
    elif mutation == 'wrong_recompute':
        bundle['model_or_retrieval_recomputed'] = True
    elif mutation == 'missing_task':
        row['native_complete_reread']['tasks'].pop()
    elif mutation == 'alias':
        row['native_complete_reread']['tasks'][9]['task'] = 'Quora'
    elif mutation == 'worker_missing':
        row['original_workers'].pop()
    elif mutation == 'task_duplicate':
        row['native_complete_reread']['tasks'][0] = row['native_complete_reread']['tasks'][1]
    with pytest.raises(ValueError):
        B.select_rows(bundle)


def test_actual_data_tables_are_complete(actual):
    generated = B.derived_bytes(actual[1], actual[2])
    assert set(generated) == B.GENERATED
    means = list(csv.DictReader(io.StringIO(generated['tables/checkpoint_means.csv'].decode())))
    tasks = list(csv.DictReader(io.StringIO(generated['tables/task_scores.csv'].decode())))
    assert len(means) == 6 and len(tasks) == 84
    assert {(r['run_id'], r['task']) for r in tasks} == {(r, t) for r in B.RUNS for t in B.TASKS}
    assert json.loads(generated['provenance/acceptance.json'])['scientific_completion'] is False


@pytest.mark.parametrize('field,value', [('ndcg_at_10', float('nan')), ('ndcg_at_10', -1.0),
    ('ndcg_at_10', 1.01), ('mean', 0.0), ('points', 0.0)])
def test_undefined_or_changed_score_refused(actual, field, value):
    rows = copy.deepcopy(actual[2])
    if field == 'ndcg_at_10':
        rows[0]['native_complete_reread']['tasks'][0][field] = value
    else:
        rows[0]['macro_ndcg_at_10' if field == 'mean' else 'macro_score_0_to_100'] = value
    with pytest.raises(ValueError):
        B.derived_bytes(actual[1], rows)


def modes():
    names = [f'data/{i}.json' for i in range(281)]
    prefix = 'corrected-dense-correctness-v3/new-scope'
    return prefix, names, {prefix + '/' + n: {'mode': 'regular', 'ignored': False, 'remote_oid': None}
                           for n in names}


def test_regular_new_only_upload_modes():
    prefix, names, values = modes()
    B.check_modes(values, prefix, names)


@pytest.mark.parametrize('mutation', ['lfs', 'ignore', 'overwrite', 'missing', 'extra', 'wrong_count'])
def test_remote_mode_changes_refused(mutation):
    prefix, names, values = modes()
    key = next(iter(values))
    if mutation == 'lfs':
        values[key]['mode'] = 'lfs'
    elif mutation == 'ignore':
        values[key]['ignored'] = True
    elif mutation == 'overwrite':
        values[key]['remote_oid'] = 'a' * 40
    elif mutation == 'missing':
        values.pop(key)
    elif mutation == 'extra':
        values[prefix + '/extra'] = values[key].copy()
    elif mutation == 'wrong_count':
        names.pop()
    with pytest.raises(ValueError):
        B.check_modes(values, prefix, names)


def preservation():
    before = {B.NAMESPACE: {'kind': 'RepoFolder', 'tree_id': 'old'},
              'README.md': {'kind': 'RepoFile', 'blob_id': 'card'},
              '.gitattributes': {'kind': 'RepoFile', 'blob_id': 'attributes'}}
    after = copy.deepcopy(before)
    after[B.NAMESPACE]['tree_id'] = 'new'
    old = {B.NAMESPACE + '/' + n: {'kind': 'RepoFolder', 'tree_id': n}
           for n in ('complete-endpoint-evaluations', 'endpoint-statistics', 'training-dynamics', 'weight-space', 'complete-first-stage-evaluations')}
    new = {**copy.deepcopy(old), B.ADDITION: {'kind': 'RepoFolder', 'tree_id': 'added'}}
    return before, after, old, new


def test_exact_prior_root_and_subtrees_preserved():
    B.check_preservation(*preservation())


@pytest.mark.parametrize('mutation', ['card', 'attributes', 'root_deleted', 'root_added', 'sub_changed',
    'sub_deleted', 'sub_extra', 'addition_missing', 'addition_preexisting'])
def test_remote_preservation_changes_refused(mutation):
    before, after, old, new = preservation()
    key = next(iter(old))
    if mutation == 'card':
        after['README.md']['blob_id'] = 'changed'
    elif mutation == 'attributes':
        after['.gitattributes']['blob_id'] = 'changed'
    elif mutation == 'root_deleted':
        after.pop('README.md')
    elif mutation == 'root_added':
        after['new-root-entry'] = {}
    elif mutation == 'sub_changed':
        new[key]['tree_id'] = 'changed'
    elif mutation == 'sub_deleted':
        new.pop(key)
    elif mutation == 'sub_extra':
        new[B.NAMESPACE + '/extra'] = {}
    elif mutation == 'addition_missing':
        new.pop(B.ADDITION)
    elif mutation == 'addition_preexisting':
        old[B.ADDITION] = new[B.ADDITION]
    with pytest.raises(ValueError):
        B.check_preservation(before, after, old, new)

