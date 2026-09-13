"""Export all validation recipes, then join the fixed selection to complete BEIR endpoints."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--validation-sha', required=True)
    parser.add_argument('--beir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(identity(args.validation)['sha256'] == args.validation_sha, 'Validation readback changed')
    validation = json.loads(args.validation.read_text())
    require(validation['scope'] == 'all_twelve_original_validation_native_readback' and
            validation['run_count'] == 12 and validation['scored_rows'] == 49152 and
            validation['scientific_completion'] is False and
            validation['beir_scores_read_by_this_reader'] is False, 'Wrong selection source')
    native = validation['native_selection_reread']
    rows = native['run_metrics']
    require(len(rows) == len({r['run_id'] for r in rows}) == 12, 'Incomplete validation population')
    selected = {}
    for optimizer in ('adamw', 'muon', 'normuon'):
        group = [row for row in rows if row['optimizer'] == optimizer]
        require(len(group) == 4, 'Incomplete optimizer validation grid')
        selected[optimizer] = min(group, key=lambda r: (r['contrastive_loss'], r['learning_rate']))['run_id']
    require(selected == native['selected'] == validation['independently_selected_from_validation_loss'],
            'Fixed validation-only selection differs')
    # BEIR is opened only after the immutable validation selection has been reconstructed.
    require(identity(args.beir)['sha256'] ==
            'b8c9aff0f580e3fee1d8f239b2943ef649ec154ae5767ee576ee47b38e027fd6', 'BEIR readback changed')
    beir = json.loads(args.beir.read_text())
    require(beir['scope'] == 'descriptive_complete_fourteen_task_checkpoint_readback' and
            len(beir['checkpoints']) == 12 and beir['scientific_completion'] is False,
            'Require the complete original endpoint grid')
    endpoints = {r['run_id']: r for r in beir['checkpoints']}
    require(set(endpoints) == {r['run_id'] for r in rows} and
            all(r['step'] == 3907 and r['task_count'] == 14 for r in endpoints.values()),
            'Endpoint/validation population differs')
    order = {'adamw': 0, 'muon': 1, 'normuon': 2}
    all_rows = [dict(row, selected_by_validation=(selected[row['optimizer']] == row['run_id']))
                for row in sorted(rows, key=lambda r: (order[r['optimizer']], r['learning_rate']))]
    chosen = []
    adam_score = endpoints[selected['adamw']]['macro_score_0_to_100']
    for optimizer, run in selected.items():
        row = next(row for row in all_rows if row['run_id'] == run)
        score = endpoints[run]['macro_score_0_to_100']
        chosen.append({'optimizer': optimizer, 'run_id': run, 'learning_rate': row['learning_rate'],
                       'validation_rows': 4096, 'validation_mean_loss': row['contrastive_loss'],
                       'beir_complete_tasks': 14, 'beir_macro_ndcg_x100': score,
                       'difference_from_selected_adamw_points': score - adam_score})
    args.output.mkdir(exist_ok=False)
    for name, values in (('all-validation-metrics.csv', all_rows), ('selected-endpoints.csv', chosen)):
        with (args.output / name).open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    lines = ['# Validation-selected final checkpoint results', '',
             'Selection uses all 4096 validation rows for every one of twelve runs: lowest mean contrastive loss within each optimizer, with exact ties resolved by lower learning rate.',
             'BEIR is joined after this fixed selection. Scores are equally weighted full-corpus nDCG@10 across all fourteen tasks, multiplied by 100.', '',
             '| Optimizer | Selected LR | Validation loss | BEIR score | Difference from AdamW |',
             '| --- | ---: | ---: | ---: | ---: |']
    lines.extend(f"| {r['optimizer']} | {r['learning_rate']:.0e} | {r['validation_mean_loss']:.6f} | {r['beir_macro_ndcg_x100']:.4f} | {r['difference_from_selected_adamw_points']:+.4f} |" for r in chosen)
    lines += ['', 'All twelve validation recipes are retained in [all-validation-metrics.csv](all-validation-metrics.csv).',
              'This is a descriptive endpoint comparison, not a significance test, a multi-seed training result, a complete five-stage trajectory comparison or a causal mechanism finding.',
              'The original scientific-completion and source-release flags remain false.', '']
    with (args.output / 'selected-endpoints.md').open('x') as handle:
        handle.write('\n'.join(lines))
    result = {'scope': 'fixed_validation_selection_then_complete_endpoint_join',
              'validation': identity(args.validation), 'beir': identity(args.beir), 'source': identity(Path(__file__)),
              'outputs': {p.name: identity(p) for p in sorted(args.output.iterdir())},
              'selected': selected, 'selected_endpoint_rows': chosen, 'all_validation_rows': 12,
              'beir_used_for_selection': False, 'significance_test': False,
              'source_release': False, 'scientific_completion': False}
    with (args.output / 'manifest.json').open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
