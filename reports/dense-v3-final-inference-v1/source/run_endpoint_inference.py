"""Frozen final-checkpoint statistics; explicitly not whole-grid outcome admission."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import numpy as np

MANIFEST = 'bf443d5623c1312f8661a918ab93f9aea5f119058307733d477c8eac0c6da74f'
PRIMARY = '4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b'
SOURCES = {
    'configs/dense_primary_v3_protocol.json': PRIMARY,
    'configs/dense_primary_v3_outcome_protocol.json': '914996b3423ba01329c03844419cca61e2a33452fe026665bb940d5b82e2cffe',
    'configs/dense_no_packing_outcome_protocol.json': '7f321f44b73bf18e88321204781814ff143e4072acf1f26780db17c341188b35',
    'src/embed_optim/corrected_outcome_summary.py': 'b22f5a9b14f3be0bb31c750aa2d32f220948e2f7a08d88c55617d1a6a47732d0',
    'src/embed_optim/decontamination.py': '58ecb5e44f0d9c9d2c3b8f63905299cd77ed94802e9da9024ed9b93b32ca71b6',
    'scripts/audit_dense_v3_outcomes.py': 'a66b45b4352b060912a7a33e7b5a5b07571586b1b7c6495e9bb16124f59a968d',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Require ordinary bound input')
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'git_blob_sha1': hashlib.sha1(b'blob ' + str(len(raw)).encode() + bytes([0]) + raw).hexdigest()}


def read(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def authenticate_snapshot(root):
    require(root.is_absolute() and root.is_dir(), 'Require actual recovered snapshot')
    require(identity(root / 'artifact_manifest.json')['sha256'] == MANIFEST, 'Immutable manifest changed')
    manifest = read(root / 'artifact_manifest.json')
    expected = set(manifest['files']) | {'artifact_manifest.json'}
    actual = set()
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'Symlinked snapshot entry')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    require(actual == expected and len(actual) == 660, 'Incomplete or changed recovered snapshot')
    for name, record in manifest['files'].items():
        relative = PurePosixPath(name)
        require(relative.as_posix() == name and not relative.is_absolute() and '..' not in relative.parts,
                'Noncanonical snapshot input')
        require(identity(root / name) == record, 'Recovered payload identity differs')
    require(manifest['scientific_completion'] is False and manifest['beir_task_cells'] == 168,
            'Wrong endpoint scope')
    return manifest


def load_pure_functions(repository, tasks):
    bindings = {}
    for name, digest in SOURCES.items():
        bindings[name] = identity(repository / name)
        require(bindings[name]['sha256'] == digest, 'Frozen statistical source changed: ' + name)
    primary = read(repository / 'configs/dense_primary_v3_protocol.json')
    outcome = read(repository / 'configs/dense_primary_v3_outcome_protocol.json')
    parent = read(repository / 'configs/dense_no_packing_outcome_protocol.json')
    require(outcome['scientific_rules']['inference'] == parent['inference'], 'Statistical rules changed')
    require(tasks == primary['evaluation']['tasks'], 'Wrong task order')
    source = (repository / 'src/embed_optim/corrected_outcome_summary.py').read_text()
    tree = ast.parse(source)
    namespace = {'np': np, 'math': math, 'statistics': statistics, 'Any': Any, 'RunConfig': object,
                 'DECONTAMINATED_TASK_NAMES': tuple(tasks)}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('OPTIMIZERS', 'CONTRASTS'):
                namespace[node.targets[0].id] = ast.literal_eval(node.value)
    decontamination = ast.parse((repository / 'src/embed_optim/decontamination.py').read_text())
    pinned = next(ast.literal_eval(n.value) for n in decontamination.body
                  if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == 'DECONTAMINATED_BEIR')
    require(list(pinned) == tasks and all(tuple(pinned[t]) ==
            (primary['beir_task_revisions'][t]['repo'], primary['beir_task_revisions'][t]['revision']) for t in tasks),
            'Original task list/revisions differ')
    function_text = {}
    for name in ('_task_effects', 'paired_max_t_intervals'):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        require(not node.decorator_list, 'Unexpected decorated statistical function')
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(repository / 'src/embed_optim/corrected_outcome_summary.py'), 'exec'), namespace)
        function_text[name] = ast.get_source_segment(source, node)
    oracle_source = (repository / 'scripts/audit_dense_v3_outcomes.py').read_text()
    node = next(n for n in ast.parse(oracle_source).body if isinstance(n, ast.FunctionDef) and n.name == 'independent_max_t')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(repository / 'scripts/audit_dense_v3_outcomes.py'), 'exec'), namespace)
    function_text['independent_max_t'] = ast.get_source_segment(oracle_source, node)
    return namespace, bindings, function_text, parent['inference']


def scored_population(root, tasks):
    # Reconstruct the original loss-only decision before opening any BEIR score.
    selection = read(root / 'native/validation-selection/all-twelve-validation-selection.json')
    completed = read(root / 'native/validation-selection/completed.json')
    metrics = selection['run_metrics']
    require(len(metrics) == len({r['run_id'] for r in metrics}) == 12, 'Incomplete validation recipes')
    require(selection['primary_protocol_sha256'] == PRIMARY and selection['scientific_completion'] is False,
            'Wrong validation parent')
    expected_rates = {'adamw': (1e-6, 3e-6, 1e-5, 3e-5),
                      'muon': (1e-4, 3e-4, 1e-3, 3e-3), 'normuon': (1e-4, 3e-4, 1e-3, 3e-3)}
    selected = {}
    for optimizer, rates in expected_rates.items():
        members = [r for r in metrics if r['optimizer'] == optimizer]
        require(tuple(sorted(r['learning_rate'] for r in members)) == rates and
                all(math.isfinite(r['contrastive_loss']) for r in members), 'Wrong rate/loss grid')
        selected[optimizer] = min(members, key=lambda r: (r['contrastive_loss'], r['learning_rate']))['run_id']
    require(selected == selection['selected'] == completed['selection'], 'Original selector differs')
    configs = [SimpleNamespace(run_id=r['run_id'], optimizer=SimpleNamespace(name=r['optimizer'], lr=r['learning_rate']))
               for r in sorted(metrics, key=lambda r: (tuple(expected_rates).index(r['optimizer']), r['learning_rate']))]
    indexed = {}
    original_root = Path('/root/embedding-optimizer-v3-experiment/evaluations/dense-primary-v3')
    for path in sorted((root / 'native/beir-final').glob('*/all-fourteen-tasks-verified.json')):
        native = read(path)
        checkpoint = native['plan']['checkpoint']
        run = checkpoint['run_id']
        require(checkpoint['step'] == 3907 and native['plan']['tasks'] == tasks and
                native['complete_task_files_verified'] is True and len(native['tasks']) == 14,
                'Incomplete native final-checkpoint result')
        for row in native['tasks']:
            task = row['task']
            files = [f for f in row['files'] if Path(f['path']).name.endswith('Decontaminated.json')]
            require(len(files) == 1, 'Nonunique task result')
            relative = Path(files[0]['path']).relative_to(original_root)
            payload = read(root / 'native/beir-final' / relative)
            split = 'dev' if task == 'MSMARCO' else 'test'
            require(payload['task_name'] == task + 'Decontaminated' and
                    set(payload['scores']) == {split} and len(payload['scores'][split]) == 1, 'Wrong original task split')
            score = payload['scores'][split][0]['ndcg_at_10']
            key = (run, 5, task)
            require(key not in indexed and score == row['ndcg_at_10'] and math.isfinite(score) and 0 <= score <= 1,
                    'Changed/duplicated native task score')
            indexed[key] = {'ndcg_at_10': score}
    require(set(indexed) == {(r.run_id, 5, t) for r in configs for t in tasks}, 'Require all 168 original final cells')
    csv_rows = read_csv(root / 'tables/final-checkpoints/task-scores.csv')
    require(len(csv_rows) == 168, 'Wrong complete task table')
    for row in csv_rows:
        require(indexed[(row['run_id'], 5, row['task'])]['ndcg_at_10'] == float(row['ndcg_at_10']) and
                int(row['step']) == 3907, 'Raw task score table differs')
    return indexed, configs, selected


def write_csv(path, rows):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--evaluation-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute() and not args.output.exists() and
            not any(p.is_symlink() for p in args.output.parents), 'Use a new ordinary output directory')
    authenticate_snapshot(args.evaluation_root)
    tasks = read(args.repository / 'configs/dense_primary_v3_protocol.json')['evaluation']['tasks']
    namespace, bindings, functions, inference = load_pure_functions(args.repository, tasks)
    indexed, configs, selected = scored_population(args.evaluation_root, tasks)
    tables, means, checks = {}, [], []
    for family, selection in (('primary', None), ('secondary', selected)):
        task_rows, effects = namespace['_task_effects'](indexed, configs, selection)
        for row in task_rows:
            for optimizer in namespace['OPTIMIZERS']:
                members = [r for r in configs if r.optimizer.name == optimizer and
                           (selection is None or r.run_id == selection[optimizer])]
                rational = sum((Fraction.from_float(indexed[(r.run_id, 5, row['task'])]['ndcg_at_10'])
                                for r in members), Fraction()) / len(members)
                require(row[optimizer + '_ndcg_at_10'] == float(rational), 'Independent exact task mean differs')
        actual = namespace['paired_max_t_intervals'](effects, samples=inference['bootstrap_samples'], seed=inference['bootstrap_seed'])
        oracle = namespace['independent_max_t'](effects)
        maximum_error = 0.0
        for contrast in namespace['CONTRASTS']:
            for key, value in oracle[contrast].items():
                error = abs(actual[contrast][key] - value)
                maximum_error = max(maximum_error, error)
                require(math.isclose(actual[contrast][key], value, rel_tol=1e-12, abs_tol=1e-12),
                        'Existing independent scalar bootstrap replay differs')
            lower, upper = oracle[contrast]['simultaneous_ci_95_lower'], oracle[contrast]['simultaneous_ci_95_upper']
            require(actual[contrast]['support'] == ('positive' if lower > 0 else 'negative' if upper < 0 else 'inconclusive'),
                    'Support decision differs')
        checks.append({'family': family, 'task_optimizer_means_checked': 42, 'interval_numeric_fields_checked': 24,
                       'maximum_absolute_scalar_replay_difference': maximum_error, 'support_decisions_checked': 3})
        tables[family + '_task_effects.csv'] = task_rows
        tables[family + '_summary.csv'] = [{'treatment': a, 'baseline': b, **actual[(a, b)]}
                                         for a, b in namespace['CONTRASTS']]
        for optimizer in namespace['OPTIMIZERS']:
            means.append({'family': family, 'optimizer': optimizer, 'rates_per_task': 4 if selection is None else 1,
                          'tasks': 14, 'selected_run_id': '' if selection is None else selection[optimizer],
                          'macro_ndcg_at_10': statistics.fmean(r[optimizer + '_ndcg_at_10'] for r in task_rows)})
    tables['optimizer_means.csv'] = means
    args.output.mkdir()
    for name, rows in tables.items():
        write_csv(args.output / name, rows)
    report = ['# Complete final-checkpoint statistical readout', '',
              'All scores/differences below are nDCG@10 multiplied by 100. Each interval is simultaneous',
              'over three contrasts within its declared family, not across both families together.', '',
              '| Estimand | Contrast | Difference | Simultaneous 95% interval | Decision |',
              '| --- | --- | ---: | --- | --- |']
    for family in ('primary', 'secondary'):
        for row in tables[family + '_summary.csv']:
            report.append(f"| {family} | {row['treatment']} − {row['baseline']} | {100*row['mean_delta_ndcg_at_10']:+.4f} | "
                          f"[{100*row['simultaneous_ci_95_lower']:+.4f}, {100*row['simultaneous_ci_95_upper']:+.4f}] | {row['support']} |")
    report += ['', 'Primary: all four predeclared rates, equally averaged within each task.',
               'Secondary: minimum-loss validation-only recipe per optimizer; BEIR never selects rates.',
               'Both use 50,000 paired-task draws, seed 20260903 and the original fixed-observed-SE max-T procedure.',
               'Intervals describe task variation on this fixed trained grid, not training-seed or per-query uncertainty.',
               'Inconclusive does not establish equality, equivalence or absence of an effect.', '',
               'All 168 final cells are present. This is not the complete 840-cell outcome campaign,',
               'full trajectory analysis, functional/causal evidence, source release or completed manuscript.',
               'The original whole-grid acceptance/publication gates remain unchanged and unpassed.', '']
    with (args.output / 'summary.md').open('x') as handle:
        handle.write('\n'.join(report))
    result = {'scope': 'complete_final_checkpoint_frozen_inference_readout',
              'observed_at_utc': datetime.now(timezone.utc).isoformat(),
              'input_root': str(args.evaluation_root), 'manifest_sha256': MANIFEST,
              'source': identity(Path(__file__)), 'frozen_sources': bindings,
              'original_function_sha256': {k: hashlib.sha256(v.encode()).hexdigest() for k,v in functions.items()},
              'numpy_version': np.__version__, 'inference': inference, 'selected': selected,
              'runs': 12, 'task_cells': 168, 'checkpoint_step': 3907, 'tables': tables,
              'independent_scalar_checks': checks,
              'outputs': {p.name: identity(p) for p in sorted(args.output.iterdir())},
              'original_whole_grid_consumer_called': False, 'original_whole_grid_admission_passed': False,
              'intermediate_scores_imputed': False, 'primary_campaign_complete': False,
              'model_execution': False, 'source_release': False, 'scientific_completion': False}
    with (args.output / 'readout.json').open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output':str(args.output),'observed_at_utc':result['observed_at_utc'],
                      'primary': tables['primary_summary.csv'], 'secondary': tables['secondary_summary.csv'],
                      'checks':checks,'scientific_completion':False},sort_keys=True))


if __name__ == '__main__':
    main()
