"""Display all 60 admitted states; no new statistic, fitted projection or selection.

This is a development-paper display consumer of the complete native assembly.
It does not waive the original publication or final-document admission gates.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPLETION = '1941d86b42972a53dbca8ff86c8cc346282cc10f17446d62f66d67fcedb826cc'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def identity(path):
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Ordinary file required')
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    need(args.output.is_absolute() and not args.output.exists(), 'New absolute display output required')
    root = HERE / 'scientific-publication-v3'
    complete_path = root / 'completed.json'
    need(identity(complete_path)['sha256'] == COMPLETION, 'Wrong complete native assembly')
    complete = json.loads(complete_path.read_bytes())
    need(complete['current_primary_scientific_result_assembly_complete'] is True, 'Incomplete primary assembly')
    table_name = 'exact-publication/tables.json'
    path = root / table_name
    need(identity(path) == complete['outputs'][table_name], 'Native display inputs changed')
    tables = json.loads(path.read_bytes())
    scores = tables['outcomes']['run_stage_scores']
    geometry = tables['geometry']['checkpoint_geometry']
    need(len(scores) == len(geometry) == 60, 'Require all 60 states in both families')
    indexed = {(r['run_id'], r['stage']): r for r in geometry}
    need(len(indexed) == 60, 'Duplicate geometric state')
    rows = []
    keys = set()
    for score in scores:
        key = score['run_id'], score['stage']
        need(key not in keys and key in indexed, 'Duplicate or absent retrieval state')
        keys.add(key)
        geom = indexed[key]
        need(all(score[k] == geom[k] for k in ('run_id', 'stage', 'optimizer', 'learning_rate', 'progress_fraction')), 'State join differs')
        need(score['tasks'] == 14 and geom['hidden_tensors'] == 88 and geom['hidden_parameters'] == 110297088, 'Scientific population differs')
        x, y = geom['cumulative_displacement_to_weight_ratio'], score['mean_ndcg_at_10']
        need(isinstance(x, float) and math.isfinite(x) and x > 0 and math.isfinite(y) and 0 <= y <= 1, 'Invalid display value')
        rows.append({k: score[k] for k in ('run_id', 'optimizer', 'learning_rate', 'stage', 'progress_fraction', 'mean_ndcg_at_10')} | {'cumulative_displacement_to_weight_ratio': x})
    need(keys == set(indexed), 'Geometric state omitted')
    families = ('adamw', 'muon', 'normuon')
    expected_rates = {'adamw': [1e-6, 3e-6, 1e-5, 3e-5], 'muon': [1e-4, 3e-4, 1e-3, 3e-3], 'normuon': [1e-4, 3e-4, 1e-3, 3e-3]}
    groups = {}
    for optimizer in families:
        selected = [r for r in rows if r['optimizer'] == optimizer]
        rates = sorted(set(r['learning_rate'] for r in selected))
        need(rates == expected_rates[optimizer] and len(selected) == 20, 'Rate surface changed')
        for rate in rates:
            group = sorted((r for r in selected if r['learning_rate'] == rate), key=lambda r: r['stage'])
            need([r['stage'] for r in group] == [1, 2, 3, 4, 5] and [r['progress_fraction'] for r in group] == [.2, .4, .6, .8, 1.0], 'Missing or reordered stage')
            need(len(set(r['run_id'] for r in group)) == 1, 'Multiple runs in trajectory')
            groups[optimizer, rate] = group

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'DejaVu Sans',
        'font.size': 7.5, 'axes.titlesize': 9, 'axes.labelsize': 8, 'legend.fontsize': 6.8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False})
    args.output.mkdir()
    rows.sort(key=lambda r: (families.index(r['optimizer']), r['learning_rate'], r['stage']))
    write(args.output / 'all-sixty-state-points.json', rows)
    colors = {'adamw': '#2364ad', 'muon': '#d36a12', 'normuon': '#21836a'}
    labels = {'adamw': 'AdamW', 'muon': 'Muon', 'normuon': 'NorMuon'}
    styles = ('-', '--', '-.', ':')
    ymin = min(r['mean_ndcg_at_10'] for r in rows) * 100
    ymax = max(r['mean_ndcg_at_10'] for r in rows) * 100
    xlow = min(r['cumulative_displacement_to_weight_ratio'] for r in rows) / 1.3
    xhigh = max(r['cumulative_displacement_to_weight_ratio'] for r in rows) * 1.3
    for mode, filename in [('weight', 'weight-to-retrieval-map.pdf'), ('progress', 'full-rate-retrieval-trajectories.pdf')]:
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True, layout='constrained')
        for ax, optimizer in zip(axes, families, strict=True):
            for rate, style in zip(expected_rates[optimizer], styles, strict=True):
                group = groups[optimizer, rate]
                xs = [r['cumulative_displacement_to_weight_ratio'] if mode == 'weight' else 100 * r['progress_fraction'] for r in group]
                ys = [100 * r['mean_ndcg_at_10'] for r in group]
                ax.plot(xs, ys, color=colors[optimizer], linestyle=style, marker='o', markersize=2.7, linewidth=1.15, label=f'{rate:.0e}')
                ax.scatter([xs[-1]], [ys[-1]], marker='s', s=18, facecolors='white', edgecolors=colors[optimizer], linewidths=.9, zorder=4)
            ax.set_title(labels[optimizer])
            ax.set_ylim(math.floor(ymin * 2) / 2 - .2, math.ceil(ymax * 2) / 2 + .2)
            ax.grid(axis='y', alpha=.20, linewidth=.5)
            ax.legend(title='Learning rate', title_fontsize=7, loc='lower right', frameon=True, framealpha=.85, edgecolor='none')
            if mode == 'weight':
                ax.set_xscale('log')
                ax.set_xlim(xlow, xhigh)
                ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.set_xlabel('Relative hidden-weight displacement')
            else:
                ax.set_xlim(15, 105)
                ax.set_xticks([20, 40, 60, 80, 100])
                ax.set_xlabel('Training progress (%)')
        axes[0].set_ylabel('BEIR nDCG@10 (points)')
        fig.savefig(args.output / filename, metadata={'CreationDate': None, 'ModDate': None})
        fig.savefig(args.output / filename.replace('.pdf', '.png'), dpi=160)
        plt.close(fig)
    need(identity(path) == complete['outputs'][table_name] and identity(complete_path)['sha256'] == COMPLETION, 'Assembly changed while displaying')
    outputs = {p.name: identity(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    write(args.output / 'completed.json', {'source': identity(Path(__file__)), 'native_assembly': identity(complete_path),
        'native_tables': identity(path), 'outputs': outputs, 'displayed_states_per_figure': 60, 'rates_per_optimizer': 4,
        'model_svds_repeated': False, 'new_statistical_test': False, 'points_not_fitted_projections': True,
        'exact_raw_values_retained': True, 'manuscript_installed': False, 'final_publication_gate_passed': False,
        'scientific_completion': False})
    print(json.dumps({'display_complete': True, 'states_per_figure': 60, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
