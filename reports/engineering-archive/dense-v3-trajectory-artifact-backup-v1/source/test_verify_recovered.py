"""Actual data replay and bounded corruption checks, never model execution."""
import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

WORK = Path(__file__).parents[1]
STAGING = WORK / 'staging'
ANCHOR = 'f2e3aec64101721e337dc13fc1bae5cdb3799ca76e0faece0fe7fbafabdf4e65'
SPEC = importlib.util.spec_from_file_location('trajectory_reader', Path(__file__).with_name('verify_recovered.py'))
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def test_actual_staged_reconstruction():
    report = R.verify(STAGING, ANCHOR)
    assert report['files_verified'] == 26
    assert report['task_scores_crosschecked_with_index_and_observer'] == 840
    assert report['endpoint_contrasts_matched'] == 6
    assert report['plotted_checkpoints_reconstructed'] == 60
    assert report['new_bootstrap_or_model_execution'] is False


@pytest.mark.parametrize('name', ['', '/etc/passwd', '../escape', 'a/../b', './x', 'a//b', 'a\\b'])
def test_unsafe_relative_paths_refused(name):
    with pytest.raises(ValueError):
        R.safe(name)


@pytest.fixture(scope='module')
def tables():
    return {name: R.table(STAGING / 'tables' / name, count) for name, count in R.COUNTS.items()}


@pytest.mark.parametrize('table_name,column', [
    ('run_stage_scores.csv', 'mean_ndcg_at_10'),
    ('run_stage_scores.csv', 'median_ndcg_at_10'),
    ('optimizer_stage_scores.csv', 'mean_ndcg_at_10_across_rates'),
    ('optimizer_stage_scores.csv', 'median_ndcg_at_10_across_rates'),
    ('run_observed_auc.csv', 'observed_auc_20_to_100'),
    ('run_observed_auc.csv', 'observed_mean_20_to_100'),
])
def test_independent_arithmetic_refuses_changed_values(tables, table_name, column):
    scores = R.load_scores(tables['all_task_scores.csv'])
    changed = copy.deepcopy(tables)
    changed[table_name][0][column] = '0.0'
    with pytest.raises(ValueError, match='scalar differs'):
        R.reconstruct_dynamics(scores, changed)


def test_duplicated_run_stage_refused(tables):
    changed = copy.deepcopy(tables)
    changed['run_stage_scores.csv'][1] = copy.deepcopy(changed['run_stage_scores.csv'][0])
    with pytest.raises(ValueError):
        R.reconstruct_dynamics(R.load_scores(tables['all_task_scores.csv']), changed)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'nonfinite'])
def test_score_grid_refusals(tables, mutation):
    rows = copy.deepcopy(tables['all_task_scores.csv'])
    if mutation == 'missing':
        rows.pop()
    elif mutation == 'duplicate':
        rows[-1] = rows[0].copy()
    else:
        rows[0]['ndcg_at_10'] = 'nan'
    with pytest.raises(ValueError):
        R.load_scores(rows)


@pytest.mark.parametrize('filename', ['tables/run_stage_scores.csv', 'tables/primary_summary.csv',
                                    'provenance/raw-score-index.json', 'tables/readout.json'])
def test_rehashed_payload_cannot_replace_accepted_inputs(tmp_path, filename):
    root = tmp_path / 'snapshot'
    shutil.copytree(STAGING, root)
    path = root / filename
    path.write_bytes(path.read_bytes() + b'\n')
    manifest_path = root / 'artifact_manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    manifest['files'][filename] = R.identity(path)
    manifest['payload_bytes'] = sum(f['bytes'] for f in manifest['files'].values())
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    with pytest.raises(ValueError):
        R.verify(root, R.identity(manifest_path)['sha256'])


def test_external_manifest_anchor_required():
    with pytest.raises(ValueError, match='anchor differs'):
        R.verify(STAGING, '0' * 64)


def test_extra_symlink_refused(tmp_path):
    root = tmp_path / 'snapshot'
    shutil.copytree(STAGING, root)
    (root / 'extra-link').symlink_to(root / 'README.md')
    with pytest.raises(ValueError, match='symlink'):
        R.verify(root, ANCHOR)
