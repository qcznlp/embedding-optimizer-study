import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('statistics_backup_under_test', Path(__file__).with_name('backup.py'))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def modes():
    names = {f'tables/{n}' for n in b.TABLES} | {f'figures/{n}' for n in b.FIGURES} | {
        'provenance/source-relocated-readout.json', 'provenance/original-verification.json',
        'README.md', 'artifact_manifest.json'}
    return {f'prefix/{n}': {'mode': 'lfs' if n.endswith('.png') else 'regular',
                          'ignored': False, 'remote_oid': None} for n in names}


def test_declared_modes_pass():
    b.check_modes(modes(), 'prefix')


@pytest.mark.parametrize('key,value', [('mode', 'lfs'), ('ignored', True), ('remote_oid', 'existing')])
def test_changed_mode_refused(key, value):
    value_map = modes()
    value_map['prefix/tables/readout.json'][key] = value
    with pytest.raises(ValueError):
        b.check_modes(value_map, 'prefix')


@pytest.mark.parametrize('change', ['missing', 'foreign', 'extra', 'png_regular'])
def test_changed_population_refused(change):
    value_map = modes()
    if change == 'missing':
        value_map.pop('prefix/README.md')
    elif change == 'foreign':
        value_map['foreign/README.md'] = value_map.pop('prefix/README.md')
    elif change == 'extra':
        value_map['prefix/program.py'] = {'mode': 'regular', 'ignored': False, 'remote_oid': None}
    else:
        value_map['prefix/figures/endpoint_contrasts.png']['mode'] = 'regular'
    with pytest.raises(ValueError):
        b.check_modes(value_map, 'prefix')


def roots():
    before = {b.NAMESPACE: {'kind': 'RepoFolder', 'tree_id': 'before'},
              '.gitattributes': {'kind': 'RepoFile', 'blob_id': 'attrs'},
              'README.md': {'kind': 'RepoFile', 'blob_id': 'card'}}
    after = copy.deepcopy(before)
    after[b.NAMESPACE]['tree_id'] = 'after'
    old_sub = {f'{b.NAMESPACE}/{n}': {'kind': 'RepoFolder', 'tree_id': n}
               for n in ['weight-space', 'training-dynamics', 'complete-endpoint-evaluations']}
    new_sub = {**copy.deepcopy(old_sub), b.ADDITION: {'kind': 'RepoFolder', 'tree_id': 'new'}}
    return before, after, old_sub, new_sub


def test_only_declared_addition_passes():
    b.check_preservation(*roots())


@pytest.mark.parametrize('change', ['attrs', 'card', 'root_add', 'root_drop', 'old_subtree',
                                  'extra_subtree', 'no_addition', 'missing_namespace'])
def test_preserved_remote_counterexamples(change):
    before, after, old_sub, new_sub = roots()
    if change in ('attrs', 'card'):
        after['.gitattributes' if change == 'attrs' else 'README.md']['blob_id'] = 'changed'
    elif change == 'root_add':
        after['other'] = {'kind': 'RepoFile'}
    elif change == 'root_drop':
        del after['README.md']
    elif change == 'old_subtree':
        new_sub[f'{b.NAMESPACE}/weight-space']['tree_id'] = 'changed'
    elif change == 'extra_subtree':
        new_sub[f'{b.NAMESPACE}/unexpected'] = {'kind': 'RepoFolder'}
    elif change == 'no_addition':
        del new_sub[b.ADDITION]
    else:
        del before[b.NAMESPACE]
        del after[b.NAMESPACE]
    with pytest.raises(ValueError):
        b.check_preservation(before, after, old_sub, new_sub)


def test_original_payloads_all_authenticate_and_scan():
    t = b.load_transport()
    selected = b.select_sources(t)
    assert len(selected) == 15
    for name, (path, identity) in selected.items():
        b.inspect_payload(t, name, path, identity)
