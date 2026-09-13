"""Render every retained retrieval curve only after a complete accepted readout.

This is a descriptive figure, not another inferential comparison. No interpolation
is exported as measured data and no learning rate is selected from retrieval.
"""
import argparse
import csv
import io
import json
import math
import os
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import complete_trajectory as trajectory

COUNTS = {'primary_task_effects.csv': 14, 'primary_summary.csv': 3,
          'secondary_task_effects.csv': 14, 'secondary_summary.csv': 3,
          'run_stage_scores.csv': 60, 'optimizer_stage_scores.csv': 15,
          'run_observed_auc.csv': 12, 'all_task_scores.csv': 840}
SELECTED = {'adamw': 'verified-v3-adamw-3e-5', 'muon': 'verified-v3-muon-3e-4',
            'normuon': 'verified-v3-normuon-3e-4'}


def read_rows(path, identity):
    raw, _ = trajectory.read_bound(path, identity['sha256'])
    trajectory.require(len(raw) == identity['bytes'], 'CSV bytes differ')
    reader = csv.DictReader(io.StringIO(raw.decode()))
    trajectory.require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)),
                       'Duplicate or missing CSV header')
    rows = list(reader)
    trajectory.require(len(rows) == identity['rows'] and
                       all(None not in row and None not in row.values() for row in rows),
                       'Incomplete CSV rows')
    return rows


def plot_rows(readout):
    trajectory.require(readout['scope'] == 'complete_retained_retrieval_trajectory_readout'
                       and readout['checkpoint_count'] == 60 and readout['task_score_count'] == 840
                       and readout['prior_54_records_exactly_unchanged'] is True
                       and readout['all_six_endpoint_contrasts_unchanged'] is True
                       and readout['selected'] == SELECTED, 'Wrong trajectory or selection scope')
    for flag in ('model_or_retrieval_recomputed', 'new_inferential_rule',
                 'original_whole_grid_consumer_called', 'original_whole_grid_admission_passed',
                 'functional_or_causal_acceptance', 'source_release', 'scientific_completion'):
        trajectory.require(readout[flag] is False, 'Unexpected scope upgrade')
    trajectory.require(set(readout['outputs']) == set(COUNTS), 'Incomplete table inventory')
    trajectory.require(readout['accepted_endpoint']['sha256'] == trajectory.ENDPOINT
                       and readout['prior_54_bundle']['sha256'] == trajectory.PRIOR_54,
                       'Different accepted parents')


def load_plot_data(path, expected):
    raw, identity = trajectory.read_bound(path, expected)
    report = trajectory.strict_json(raw)
    plot_rows(report)
    tables = {}
    for name, count in COUNTS.items():
        item = report['outputs'][name]
        trajectory.require(item['rows'] == count, 'Wrong declared table population')
        tables[name] = read_rows(path.parent / name, item)
    scores = {}
    for row in tables['all_task_scores.csv']:
        run, stage, task = row['run_id'], int(row['stage']), row['task']
        key = run, stage, task
        trajectory.require(run in trajectory.RUNS and stage in range(1, 6)
                           and task in trajectory.TASKS and key not in scores,
                           'Unexpected or duplicate plotted score cell')
        optimizer, rate = run.removeprefix('verified-v3-').split('-', 1)
        score = float(row['ndcg_at_10'])
        trajectory.require(row['model_family'] == 'dense' and row['optimizer'] == optimizer
                           and float(row['learning_rate']) == float(rate)
                           and int(row['step']) == trajectory.STEPS[stage - 1]
                           and float(row['fraction']) == stage / 5
                           and math.isfinite(score) and 0 <= score <= 1,
                           'Wrong plotted score identity/value')
        scores[key] = Fraction.from_float(score)
    expected_cells = {(run, stage, task) for run in trajectory.RUNS
                      for stage in range(1, 6) for task in trajectory.TASKS}
    trajectory.require(set(scores) == expected_cells, 'Plot needs all 840 raw scores')
    summaries = {}
    for row in tables['run_stage_scores.csv']:
        key = row['run_id'], int(row['stage'])
        trajectory.require(key not in summaries, 'Duplicate run-stage summary')
        summaries[key] = row
    trajectory.require(set(summaries) == {(run, stage) for run in trajectory.RUNS
                                          for stage in range(1, 6)}, 'Missing plotted checkpoint')
    data = []
    for optimizer, rates in trajectory.RATE_NAMES.items():
        for rate in rates:
            run = 'verified-v3-' + optimizer + '-' + rate
            for stage in range(1, 6):
                row = summaries[(run, stage)]
                mean = sum((scores[(run, stage, task)] for task in trajectory.TASKS), Fraction()) / 14
                measured = float(row['mean_ndcg_at_10'])
                trajectory.require(row['optimizer'] == optimizer
                                   and float(row['learning_rate']) == float(rate)
                                   and float(row['progress_fraction']) == stage / 5
                                   and int(row['tasks']) == 14
                                   and abs(measured - float(mean)) <= 1e-15,
                                   'Plotted summary differs from complete raw tasks')
                data.append({'optimizer': optimizer, 'learning_rate': float(rate), 'run_id': run,
                             'stage': stage, 'step': trajectory.STEPS[stage - 1],
                             'training_progress_percent': stage * 20,
                             'mean_ndcg_at_10': measured, 'plotted_score_0_to_100': measured * 100,
                             'validation_selected': run == SELECTED[optimizer]})
    return data, report, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readout', type=Path, required=True)
    parser.add_argument('--readout-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    trajectory.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only rendering required')
    trajectory.require(args.output.is_absolute() and not args.output.exists()
                       and not any(p.is_symlink() for p in args.output.parents), 'Use a new ordinary directory')
    data, report, input_id = load_plot_data(args.readout, args.readout_sha256)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                                'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    colors = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')
    names = {'adamw': 'AdamW', 'muon': 'Muon', 'normuon': 'NorMuon'}
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.0), sharex=True, sharey=True)
    for ax, (optimizer, rates) in zip(axes, trajectory.RATE_NAMES.items(), strict=True):
        for rate, color in zip(rates, colors, strict=True):
            run = 'verified-v3-' + optimizer + '-' + rate
            curve = [row for row in data if row['run_id'] == run]
            selected = run == SELECTED[optimizer]
            ax.plot([row['training_progress_percent'] for row in curve],
                    [row['plotted_score_0_to_100'] for row in curve],
                    label=rate + (' *' if selected else ''), color=color,
                    linewidth=2.3 if selected else 1.15, marker='o', markersize=3.1)
        ax.set_title(names[optimizer])
        ax.set_xticks((20, 40, 60, 80, 100))
        ax.set_xlabel('Training progress (%)')
        ax.grid(axis='y', alpha=.22)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(title='Learning rate', frameon=False, fontsize=8, title_fontsize=8,
                  ncols=2, loc='upper center', bbox_to_anchor=(.5, -.20))
    axes[0].set_ylabel('14-task mean nDCG@10 × 100')
    fig.text(.5, .025, '* Selected by validation loss; all four rates are retained. '
             'Lines connect measured checkpoints.', ha='center', fontsize=8)
    fig.subplots_adjust(left=.07, right=.995, top=.9, bottom=.34, wspace=.13)
    args.output.mkdir()
    for suffix in ('pdf', 'png', 'svg'):
        fig.savefig(args.output / ('retrieval_trajectories.' + suffix), dpi=220)
    plt.close(fig)
    with (args.output / 'figure_data.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(data)
    outputs = {}
    for path in sorted(args.output.iterdir()):
        value = path.read_bytes()
        outputs[path.name] = {'bytes': len(value), 'sha256': trajectory.sha(value)}
    receipt = {'scope': 'complete_retained_retrieval_trajectory_figure',
               'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'input': input_id,
               'source_sha256': trajectory.sha(Path(__file__).read_bytes()),
               'matplotlib_version': matplotlib.__version__, 'outputs': outputs,
               'plotted_actual_checkpoint_count': 60, 'independently_read_task_score_count': 840,
               'selected_by_validation': report['selected'], 'rates_omitted': 0,
               'uncertainty_or_new_inference_added': False, 'manuscript_modified': False,
               'source_release': False, 'scientific_completion': False}
    with (args.output / 'figure.json').open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
