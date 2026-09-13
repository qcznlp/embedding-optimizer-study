"""Render all six unchanged endpoint intervals, with no new statistical choices."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

READOUT_SHA = '592c902f94b8f190f3165f4075dfddc1e8c3f52add760805da639b541d27716f'


def identity(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert identity(args.input)['sha256'] == READOUT_SHA
    result = json.loads(args.input.read_text())
    assert result['task_cells'] == 168 and result['scientific_completion'] is False
    for name, expected in result['outputs'].items():
        got = identity(args.input.parent / name)
        assert all(got[k] == expected[k] for k in got)
    assert args.output.is_absolute() and not args.output.exists()
    assert not any(p.is_symlink() for p in args.output.parents)
    args.output.mkdir()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'dense-v3-final-inference'})
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), sharex=True, sharey=True)
    names = {'adamw': 'AdamW', 'muon': 'Muon', 'normuon': 'NorMuon'}
    figure_rows = []
    for axis, family, title in zip(axes, ('primary', 'secondary'),
                                   ('Four-rate average (primary)', 'Validation-selected (secondary)')):
        rows = result['tables'][family + '_summary.csv']
        assert [(r['treatment'], r['baseline']) for r in rows] == [('muon', 'adamw'), ('normuon', 'adamw'), ('normuon', 'muon')]
        for index, row in enumerate(rows):
            y = 2 - index
            mean, lower, upper = (100 * row[k] for k in
                                  ('mean_delta_ndcg_at_10', 'simultaneous_ci_95_lower', 'simultaneous_ci_95_upper'))
            axis.errorbar(mean, y, xerr=[[mean-lower], [upper-mean]], fmt='o', color='#264653',
                          ecolor='#264653', markersize=6, elinewidth=1.7, capsize=4, zorder=3)
            axis.text(-1.20, y-0.31, f'{mean:+.3f}  [{lower:+.3f}, {upper:+.3f}]', fontsize=9,
                      ha='left', va='center', color='#333333')
            figure_rows.append({'family': family, 'treatment': row['treatment'], 'baseline': row['baseline'],
                                'difference_points': mean, 'simultaneous_lower_points': lower,
                                'simultaneous_upper_points': upper, 'y': y, 'support': row['support']})
        axis.axvline(0, linestyle='--', color='#8a8a8a', linewidth=1, zorder=1)
        axis.set_title(title, fontsize=11, pad=15)
        axis.set_xlim(-1.25, 1.60)
        axis.set_ylim(-0.55, 2.55)
        axis.set_xticks(np.arange(-1.0, 1.51, 0.5))
        axis.set_yticks([2, 1, 0], [f"{names[r['treatment']]} − {names[r['baseline']]}" for r in rows])
        axis.tick_params(axis='y', length=0, pad=9)
        axis.set_xlabel('Difference in nDCG@10 × 100', labelpad=9)
        axis.grid(axis='x', color='#eeeeee', linewidth=0.7)
        axis.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            axis.spines[side].set_visible(False)
    figure.suptitle('DenseOn: complete final-checkpoint contrasts', fontsize=13, y=0.98)
    figure.subplots_adjust(left=0.19, right=0.98, top=0.80, bottom=0.28, wspace=0.13)
    figure.text(0.19, 0.11, '95% simultaneous max-T intervals across 3 contrasts per panel; 50,000 paired-task draws.', fontsize=9)
    figure.text(0.19, 0.055, 'All 168 final task cells. One training seed; intervals do not measure seed-to-seed robustness.', fontsize=9)
    for suffix, metadata in (('pdf', {'CreationDate': None, 'ModDate': None}), ('svg', {'Date': None}), ('png', {})):
        figure.savefig(args.output / ('endpoint_contrasts.' + suffix), dpi=180, metadata=metadata)
    plt.close(figure)
    with (args.output / 'figure_data.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(figure_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(figure_rows)
    receipt = {'scope': 'all_six_frozen_endpoint_intervals_visualization',
               'input': identity(args.input), 'source': identity(Path(__file__)),
               'matplotlib_version': matplotlib.__version__, 'plotted_rows': figure_rows,
               'outputs': {p.name: identity(p) for p in sorted(args.output.iterdir())},
               'new_statistical_inference': False, 'manuscript_installed': False, 'scientific_completion': False}
    with (args.output / 'rendering.json').open('x') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
