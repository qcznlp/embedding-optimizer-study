"""Test the new independent data reader on copied accepted data, never originals."""

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

WORK = Path(__file__).parent.parent
STAGING = WORK / 'staging'
ANCHOR = json.loads((WORK / 'preflight.json').read_bytes())['manifest']['sha256']
SPEC = importlib.util.spec_from_file_location('independent_second_stage_pool_b_reader', Path(__file__).with_name('verify_recovered.py'))
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def test_actual_staged_native_scores_and_all_workers():
    proof = R.verify(STAGING, ANCHOR)
    assert proof['files_verified'] == 281
    assert proof['raw_task_scores_reconstructed'] == 84
    assert proof['native_exit_zero_workers_checked'] == 84
    assert proof['complete_checkpoint_means_reconstructed'] == 6
    assert proof['network_transport_verified_by_this_program'] is False


@pytest.mark.parametrize('name', ['', '/etc/passwd', '../escape', 'a/../b', './x', 'a//b', 'a\\b'])
def test_untrusted_paths_refused(name):
    with pytest.raises(ValueError):
        R.safe_relative(name)


@pytest.mark.parametrize('mutation', ['task_table', 'mean_table', 'worker_exit', 'worker_pid',
    'worker_alias', 'checkpoint_step', 'task_alias', 'manifest_stage'])
def test_semantic_corruption_refused_even_after_complete_rehash(tmp_path, mutation):
    root = tmp_path / 'snapshot'
    shutil.copytree(STAGING, root)
    manifest_path = root / 'artifact_manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    if mutation in ('task_table', 'mean_table'):
        path = root / ('tables/task_scores.csv' if mutation == 'task_table' else 'tables/checkpoint_means.csv')
        lines = path.read_text().splitlines()
        cells = lines[1].split(',')
        cells[-1] = '0.0'
        lines[1] = ','.join(cells)
        path.write_text('\n'.join(lines) + '\n')
    elif mutation.startswith('worker_'):
        suffix = '*QuoraRetrieval.started.json' if mutation == 'worker_alias' else '*.exited.json'
        path = next((root / 'provenance/beir-workers').rglob(suffix))
        value = json.loads(path.read_bytes())
        if mutation == 'worker_exit':
            value['exit_code'] = 1
        elif mutation == 'worker_pid':
            value['pid'] += 1
        else:
            value['job'][2] = 'Quora'
        path.write_text(json.dumps(value, indent=2) + '\n')
    elif mutation in ('checkpoint_step', 'task_alias'):
        path = next((root / 'native').rglob('all-fourteen-tasks-verified.json'))
        value = json.loads(path.read_bytes())
        if mutation == 'checkpoint_step':
            value['plan']['checkpoint']['step'] = 2345
        else:
            value['tasks'][9]['task'] = 'Quora'
        path.write_text(json.dumps(value, indent=2) + '\n')
    else:
        manifest['checkpoint_stages'] = [2345]
    for name in manifest['files']:
        manifest['files'][name] = R.identity(root / name)
    manifest['payload_bytes'] = sum(i['bytes'] for i in manifest['files'].values())
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    new_anchor = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        R.verify(root, new_anchor)


def test_external_manifest_anchor_required():
    with pytest.raises(ValueError, match='external anchor'):
        R.verify(STAGING, '0' * 64)


def test_extra_symlink_refused(tmp_path):
    root = tmp_path / 'snapshot'
    shutil.copytree(STAGING, root)
    (root / 'extra-link').symlink_to(root / 'README.md')
    with pytest.raises(ValueError, match='Symlink'):
        R.verify(root, ANCHOR)

