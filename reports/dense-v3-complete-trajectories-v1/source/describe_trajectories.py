"""Generate descriptive tables from all 840 accepted scores; no new inference."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import complete_trajectory as trajectory
import render_trajectories_v2 as renderer


def describe(readout, digest):
    data, report, identity = renderer.load_plot_data(readout, digest)
    areas = renderer.read_rows(readout.parent / 'run_observed_auc.csv',
                               report['outputs']['run_observed_auc.csv'])
    area_by_run = {row['run_id']: row for row in areas}
    trajectory.require(set(area_by_run) == set(trajectory.RUNS), 'Incomplete area table')
    indexed = {(row['run_id'], row['stage']): row for row in data}
    names = {'adamw': 'AdamW', 'muon': 'Muon', 'normuon': 'NorMuon'}
    selected = report['selected']
    lines = ['# Complete retained retrieval trajectories', '',
             'All scores are fourteen-task macro nDCG@10 multiplied by 100. Every declared rate',
             'and retained stage is included. The last column is the trapezoidal mean over',
             'the observed 20–100% range, not an initialization-to-endpoint or wall-time area.', '',
             '| Optimizer | Learning rate | 20% | 40% | 60% | 80% | 100% | Observed-range mean |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for optimizer, rates in trajectory.RATE_NAMES.items():
        for rate in rates:
            run = 'verified-v3-' + optimizer + '-' + rate
            values = [indexed[(run, s)]['plotted_score_0_to_100'] for s in range(1, 6)]
            mean = float(area_by_run[run]['observed_mean_20_to_100']) * 100
            marker = ' *' if selected[optimizer] == run else ''
            lines.append('| ' + names[optimizer] + ' | ' + rate + marker + ' | ' +
                         ' | '.join(f'{v:.3f}' for v in [*values, mean]) + ' |')
    lines += ['', '* indicates selection by final validation loss, never by BEIR.', '',
              '## Validation-selected trajectories', '',
              '| Progress | AdamW | Muon | NorMuon | Muon − AdamW | NorMuon − AdamW |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    comparisons = []
    adam_final = indexed[(selected['adamw'], 5)]['plotted_score_0_to_100']
    for stage in range(1, 6):
        a, m, n = [indexed[(selected[o], stage)]['plotted_score_0_to_100']
                   for o in ('adamw', 'muon', 'normuon')]
        row = {'stage': stage, 'training_progress_percent': 20 * stage,
               'adamw_score_0_to_100': a, 'muon_score_0_to_100': m,
               'normuon_score_0_to_100': n, 'muon_minus_adamw': m - a,
               'normuon_minus_adamw': n - a,
               'muon_above_selected_adamw_final': m > adam_final,
               'normuon_above_selected_adamw_final': n > adam_final}
        comparisons.append(row)
        lines.append(f'| {stage * 20}% | {a:.3f} | {m:.3f} | {n:.3f} | {m-a:+.3f} | {n-a:+.3f} |')
    facts = {}
    for optimizer in ('muon', 'normuon'):
        wins = sum(row[optimizer + '_minus_adamw'] > 0 for row in comparisons)
        above = [row['training_progress_percent'] for row in comparisons
                 if row[optimizer + '_above_selected_adamw_final']]
        facts[optimizer] = {'stages_above_selected_adamw_at_same_stage': wins,
                            'first_retained_percent_above_selected_adamw_final': min(above) if above else None}
    lines += ['', '## Descriptive interpretation and limits', '']
    for optimizer in ('muon', 'normuon'):
        fact = facts[optimizer]
        first = fact['first_retained_percent_above_selected_adamw_final']
        crossing = f'{first}%' if first is not None else 'not observed'
        lines += [f"- Selected {names[optimizer]} exceeds selected AdamW at "
                  f"{fact['stages_above_selected_adamw_at_same_stage']}/5 retained stages. "
                  f'Its first retained point above selected AdamW’s final score is {crossing}.']
    lines += ['', 'These are post-hoc descriptions of fixed validation-selected configurations, not',
              'new significance tests or deployable early-stopping rules. Selection uses full-horizon',
              'validation results, so a 40% crossing does not establish 60% compute or wall-time savings.',
              'Four rates are not four training seeds. No additional uncertainty band is inferred.', '',
              'The existing final-stage primary four-rate contrasts remain inconclusive. The secondary',
              'validation-selected NorMuon–AdamW contrast retains its positive task-level simultaneous',
              'interval; Muon–AdamW and NorMuon–Muon remain inconclusive. The complete trajectory',
              'does not alter those tests, establish useful-dimension allocation, or explain a mechanism.', '']
    return '\n'.join(lines), comparisons, facts, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readout', type=Path, required=True)
    parser.add_argument('--readout-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    trajectory.require(args.output.is_absolute() and not args.output.exists()
                       and not any(p.is_symlink() for p in args.output.parents), 'Use a new output directory')
    markdown, comparisons, facts, identity = describe(args.readout, args.readout_sha256)
    args.output.mkdir()
    with (args.output / 'summary.md').open('x') as stream:
        stream.write(markdown)
    with (args.output / 'selected_stage_contrasts.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(comparisons)
    outputs = {}
    for path in sorted(args.output.iterdir()):
        raw = path.read_bytes()
        outputs[path.name] = {'bytes': len(raw), 'sha256': trajectory.sha(raw)}
    receipt = {'scope': 'descriptive_complete_trajectory_tables', 'input': identity,
               'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'outputs': outputs,
               'source_sha256': trajectory.sha(Path(__file__).read_bytes()), 'derived_facts': facts,
               'new_inference': False, 'model_execution': False, 'manuscript_modified': False,
               'scientific_completion': False}
    with (args.output / 'description.json').open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
