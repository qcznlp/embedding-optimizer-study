"""Synthetic renderer-contract fixtures only; no accepted result or figure is made."""
import copy
import csv
import io
import json
from pathlib import Path

import pytest

import complete_trajectory as T
import render_trajectories as R


def as_csv(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


@pytest.fixture
def synthetic_tables():
    tables = {name: [{'synthetic_unused_placeholder': i} for i in range(count)]
              for name, count in R.COUNTS.items()}
    tasks, summaries = [], []
    for i, run in enumerate(T.RUNS):
        optimizer, rate = run.removeprefix('verified-v3-').split('-', 1)
        for stage in range(1, 6):
            score = (i + stage) / 32
            tasks.extend({'run_id': run, 'optimizer': optimizer, 'learning_rate': float(rate),
                          'model_family': 'dense', 'stage': stage, 'step': T.STEPS[stage - 1],
                          'fraction': stage / 5, 'task': task, 'ndcg_at_10': score}
                         for task in T.TASKS)
            summaries.append({'run_id': run, 'optimizer': optimizer, 'learning_rate': float(rate),
                              'stage': stage, 'progress_fraction': stage / 5,
                              'tasks': 14, 'mean_ndcg_at_10': score})
    tables['all_task_scores.csv'], tables['run_stage_scores.csv'] = tasks, summaries
    return tables


def materialize_synthetic_reader_fixture(root, tables, change=None):
    report = {'scope': 'complete_retained_retrieval_trajectory_readout',
              'checkpoint_count': 60, 'task_score_count': 840,
              'prior_54_records_exactly_unchanged': True, 'all_six_endpoint_contrasts_unchanged': True,
              'selected': copy.deepcopy(R.SELECTED),
              'accepted_endpoint': {'sha256': T.ENDPOINT}, 'prior_54_bundle': {'sha256': T.PRIOR_54},
              'model_or_retrieval_recomputed': False, 'new_inferential_rule': False,
              'original_whole_grid_consumer_called': False, 'original_whole_grid_admission_passed': False,
              'functional_or_causal_acceptance': False, 'source_release': False,
              'scientific_completion': False, 'outputs': {},
              'synthetic_unit_fixture_not_an_accepted_readout': True}
    for name, table in tables.items():
        raw = as_csv(table)
        (root / name).write_bytes(raw)
        report['outputs'][name] = {'bytes': len(raw), 'sha256': T.sha(raw), 'rows': len(table)}
    if change:
        change(report)
    raw = (json.dumps(report, allow_nan=False) + '\n').encode()
    path = root / 'synthetic-readout.json'
    path.write_bytes(raw)
    return path, T.sha(raw)


def test_synthetic_complete_reader_maps_every_cell_without_dropping_rates(tmp_path, synthetic_tables):
    path, digest = materialize_synthetic_reader_fixture(tmp_path, synthetic_tables)
    rows, report, _ = R.load_plot_data(path, digest)
    assert report['synthetic_unit_fixture_not_an_accepted_readout'] is True
    assert len(rows) == 60 and {row['run_id'] for row in rows} == set(T.RUNS)
    assert sum(row['validation_selected'] for row in rows) == 15
    assert {row['training_progress_percent'] for row in rows} == {20, 40, 60, 80, 100}
    assert all(row['plotted_score_0_to_100'] == row['mean_ndcg_at_10'] * 100 for row in rows)
    assert not any(path.suffix in ('.pdf', '.png', '.svg') for path in tmp_path.iterdir())


@pytest.mark.parametrize('mutation', ['missing_task', 'duplicate_task', 'unknown_task', 'wrong_stage',
    'wrong_step', 'wrong_fraction', 'wrong_optimizer', 'wrong_rate', 'score_nonfinite', 'score_range',
    'missing_summary', 'duplicate_summary', 'summary_mean', 'summary_task_count', 'summary_optimizer',
    'summary_rate', 'summary_progress'])
def test_reembedded_synthetic_bad_plot_data_refused(tmp_path, synthetic_tables, mutation):
    tasks, summaries = synthetic_tables['all_task_scores.csv'], synthetic_tables['run_stage_scores.csv']
    if mutation == 'missing_task':
        tasks.pop()
    elif mutation == 'duplicate_task':
        tasks[0] = copy.deepcopy(tasks[1])
    elif mutation == 'unknown_task':
        tasks[0]['task'] = 'Quora'
    elif mutation == 'wrong_stage':
        tasks[0]['stage'] = 6
    elif mutation == 'wrong_step':
        tasks[0]['step'] = 3907
    elif mutation == 'wrong_fraction':
        tasks[0]['fraction'] = 0
    elif mutation == 'wrong_optimizer':
        tasks[0]['optimizer'] = 'normuon'
    elif mutation == 'wrong_rate':
        tasks[0]['learning_rate'] = .5
    elif mutation == 'score_nonfinite':
        tasks[0]['ndcg_at_10'] = float('nan')
    elif mutation == 'score_range':
        tasks[0]['ndcg_at_10'] = 1.1
    elif mutation == 'missing_summary':
        summaries.pop()
    elif mutation == 'duplicate_summary':
        summaries[0] = copy.deepcopy(summaries[1])
    else:
        field, value = {'summary_mean': ('mean_ndcg_at_10', .8),
                        'summary_task_count': ('tasks', 13),
                        'summary_optimizer': ('optimizer', 'normuon'),
                        'summary_rate': ('learning_rate', .5),
                        'summary_progress': ('progress_fraction', .5)}[mutation]
        summaries[0][field] = value
    path, digest = materialize_synthetic_reader_fixture(tmp_path, synthetic_tables)
    with pytest.raises(ValueError):
        R.load_plot_data(path, digest)


@pytest.mark.parametrize('field,value', [('checkpoint_count', 54), ('task_score_count', 756),
    ('all_six_endpoint_contrasts_unchanged', False), ('source_release', True),
    ('scientific_completion', True), ('new_inferential_rule', True)])
def test_partial_or_upgraded_mock_readout_is_refused(tmp_path, synthetic_tables, field, value):
    path, digest = materialize_synthetic_reader_fixture(
        tmp_path, synthetic_tables, lambda report: report.update({field: value}))
    with pytest.raises(ValueError):
        R.load_plot_data(path, digest)


def test_mock_retrieval_selected_rate_is_refused(tmp_path, synthetic_tables):
    def changed(report):
        report['selected']['adamw'] = 'verified-v3-adamw-1e-5'
    path, digest = materialize_synthetic_reader_fixture(tmp_path, synthetic_tables, changed)
    with pytest.raises(ValueError, match='selection scope'):
        R.load_plot_data(path, digest)


def test_changed_csv_bytes_refused_by_anchor(tmp_path, synthetic_tables):
    path, digest = materialize_synthetic_reader_fixture(tmp_path, synthetic_tables)
    (tmp_path / 'run_stage_scores.csv').write_bytes(b'changed\n')
    with pytest.raises(ValueError, match='external SHA-256'):
        R.load_plot_data(path, digest)


def test_actual_partial_native_bundle_is_not_a_complete_plot_input():
    path = Path('/root/embedding-optimizer-story-refactor/reports/engineering-archive/'
                'dense-v3-fourth-stage-pool-b-evaluations-v1/actual/complete-checkpoint-readback.json')
    with pytest.raises(ValueError, match='trajectory or selection scope'):
        R.load_plot_data(path, T.PRIOR_54)
