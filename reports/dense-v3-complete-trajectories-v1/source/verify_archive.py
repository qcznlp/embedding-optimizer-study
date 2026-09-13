"""Check complete actual retrieval evidence, replay, presentation and preservation."""
import ast
import csv
import io
import json
import math
import re
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import complete_trajectory as trajectory
import describe_trajectories as description

ROOT = Path(__file__).parents[1]
STORY = Path('/root/embedding-optimizer-story-refactor')
EXPERIMENT = Path('/root/embedding-optimizer-v3-experiment')
BUNDLE_SHA = '4795dca63d36f62b38d7c76e62b3ba9982eb3d6a7b3d20b7a610e057fbce0c50'
READOUT_SHA = 'dd48e622c9cd1f28f8e93cab670349fc937e8e64aa50b98947104151dbda42c2'
PARENT_SHA = '41d2c73fb4581ed24237a72e0dc9e8edc5b34c9cba0fd351c4c57d7bc0cc31d1'
OBSERVATIONS = {
    'observer-completed.json': 'eb022f77ad9eb1b58d929e14103b48aae388de1d67730b92d939e46fd9773928',
    'pool-a-completed.json': '3b88bc3df3acd6de0a86ff68f243e7e67dae9453223849b0755997835946276a',
    'pool-b-completed.json': '8bea8c969afb13166d930a26e9402d844407edc03b8df026d5825a1a55a8ffdd',
}


def identity(path):
    assert path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), str(path)
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': trajectory.sha(raw)}


def bound_json(path, digest):
    raw, _ = trajectory.read_bound(path, digest)
    return trajectory.strict_json(raw)


def main():
    output = ROOT / 'verification.json'
    assert not output.exists(), 'Preserve existing verification'
    bundle = bound_json(ROOT / 'actual/complete-checkpoint-readback.json', BUNDLE_SHA)
    prior = bound_json(ROOT / 'inputs/prior-54-readback.json', trajectory.PRIOR_54)
    readout = bound_json(ROOT / 'tables/readout.json', READOUT_SHA)
    assert identity(ROOT / 'source/complete_trajectory.py') == readout['source']
    assert identity(ROOT / 'source/readback.py') == {
        k: bundle['source'][k] for k in ('bytes', 'sha256')}
    bound_json(ROOT / 'inputs/accepted-endpoint-readout.json', trajectory.ENDPOINT)
    rows = trajectory.complete_rows(bundle, prior)
    scores = {(r['run_id'], r['step'], r['task']): r['ndcg_at_10'] for r in rows}
    assert len(scores) == 840
    observations = {name: bound_json(ROOT / 'observations' / name, digest)
                    for name, digest in OBSERVATIONS.items()}
    observer = observations['observer-completed.json']
    assert observer['primary_tasks_accepted'] == observer['primary_tasks_required'] == 840
    assert observer['baseline_tasks_accepted'] == observer['baseline_tasks_required'] == 14
    assert not observer['exact_worker_observations'] and not observer['deferred_observations']
    observed = [r for r in observer['task_rows'] if r['baseline'] is False]
    baseline = [r for r in observer['task_rows'] if r['baseline'] is True]
    assert len(observed) == 840 and len(baseline) == 14
    assert {(r['run_id'], r['step'], r['task']): r['ndcg_at_10'] for r in observed} == scores
    assert {r['task'] for r in baseline} == set(trajectory.TASKS)
    assert all(r['step'] == 0 and r['run_id'] == 'denseon-unsupervised-context8192'
               and math.isfinite(r['ndcg_at_10']) for r in baseline)
    for name in ('pool-a-completed.json', 'pool-b-completed.json'):
        pool = observations[name]
        assert pool['verified_checkpoints'] == 30 and pool['verified_checkpoint_task_cells'] == 420
        assert pool['scientific_completion'] is False

    native_count = 0
    allowed_roots = (EXPERIMENT / 'evaluations/dense-primary-v3',
                     EXPERIMENT / 'launch/evaluation-handoff')
    for record in bundle['checkpoints']:
        snapshots = [record['original_complete_receipt'], record['original_operational_receipt'],
                     *record['raw_score_and_metadata_snapshots']]
        snapshots += [item for worker in record['original_workers'] for item in worker.values()]
        assert len(snapshots) == 46
        for item in snapshots:
            path = Path(item['path'])
            assert any(path.is_relative_to(base) for base in allowed_roots), str(path)
            original, _ = trajectory.read_bound(path, item['sha256'])
            assert original == trajectory.embedded(item)
            native_count += 1
    assert native_count == 2760

    relocated = json.loads((ROOT / 'actual/source-relocated/readout.json').read_text())
    changed_provenance = {'observed_at_utc', 'native_bundle', 'prior_54_bundle',
                          'accepted_endpoint', 'frozen_sources'}
    assert set(readout) == set(relocated)
    for key in readout:
        if key not in changed_provenance:
            assert readout[key] == relocated[key], key
    for key in ('native_bundle', 'prior_54_bundle', 'accepted_endpoint'):
        assert {k: readout[key][k] for k in ('bytes', 'sha256')} == {
            k: relocated[key][k] for k in ('bytes', 'sha256')}
    for name, item in readout['frozen_sources'].items():
        assert identity(ROOT / 'source-original' / name) == {
            k: item[k] for k in ('bytes', 'sha256')}
    for name, item in readout['outputs'].items():
        expected = {k: item[k] for k in ('bytes', 'sha256')}
        assert identity(ROOT / 'tables' / name) == expected
        assert identity(ROOT / 'actual/source-relocated' / name) == expected

    stage_means = {(run, stage): sum((Fraction.from_float(scores[(run, step, task)])
                    for task in trajectory.TASKS), Fraction()) / 14
                   for run in trajectory.RUNS for stage, step in enumerate(trajectory.STEPS, 1)}
    observed_tables = {name.removesuffix('.csv'): list(csv.DictReader(io.StringIO(
        (ROOT / 'tables' / name).read_text()))) for name in readout['outputs']}
    scalar_checks = 0
    for row in observed_tables['run_stage_scores']:
        expected = stage_means[(row['run_id'], int(row['stage']))]
        assert abs(float(row['mean_ndcg_at_10']) - float(expected)) <= 1e-15
        scalar_checks += 1
    for row in observed_tables['optimizer_stage_scores']:
        members = [run for run in trajectory.RUNS if run.startswith('verified-v3-' + row['optimizer'] + '-')]
        expected = sum((stage_means[(run, int(row['stage']))] for run in members), Fraction()) / 4
        assert abs(float(row['mean_ndcg_at_10_across_rates']) - float(expected)) <= 1e-15
        scalar_checks += 1
    for row in observed_tables['run_observed_auc']:
        run = row['run_id']
        expected = sum(((stage_means[(run, i)] + stage_means[(run, i+1)]) / 10
                        for i in range(1, 5)), Fraction())
        assert abs(float(row['observed_auc_20_to_100']) - float(expected)) <= 1e-15
        assert abs(float(row['observed_mean_20_to_100']) - float(expected * Fraction(5, 4))) <= 1e-15
        scalar_checks += 2
    assert scalar_checks == 99

    for directory in ('figures', 'first-layout'):
        figure = json.loads((ROOT / directory / 'figure.json').read_text())
        for name, item in figure['outputs'].items():
            assert identity(ROOT / directory / name) == item
        assert figure['plotted_actual_checkpoint_count'] == 60
        assert figure['rates_omitted'] == 0 and figure['input']['sha256'] == READOUT_SHA
    assert (ROOT / 'figures/figure_data.csv').read_bytes() == (ROOT / 'first-layout/figure_data.csv').read_bytes()
    old_source = (ROOT / 'source/render_trajectories_v1.py').read_text()
    new_source = (ROOT / 'source/render_trajectories_v2.py').read_text()
    assert trajectory.sha(old_source.encode()) == '71d59234dcba662f75ebc6a9e2e6376aeb242e8e69b80a8ad6686dd7d8986b38'
    assert trajectory.sha(new_source.encode()) == '2854d8996a2531da3e7016db45dc48d0fc9dbab46401d0165844141787ac954b'
    for name in ('read_rows', 'plot_rows', 'load_plot_data'):
        def function_tree(source):
            return ast.dump(next(n for n in ast.parse(source).body
                                if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)
        assert function_tree(old_source) == function_tree(new_source)
    markdown, comparisons, facts, _ = description.describe(ROOT / 'tables/readout.json', READOUT_SHA)
    assert markdown == (ROOT / 'descriptive/summary.md').read_text()
    for name in ('summary.md', 'selected_stage_contrasts.csv'):
        assert (ROOT / 'descriptive' / name).read_bytes() == (ROOT / 'actual/source-relocated-description' / name).read_bytes()
    for row in comparisons:
        stage = row['stage']
        selected = readout['selected']
        for optimizer in ('adamw', 'muon', 'normuon'):
            exact = float(stage_means[(selected[optimizer], stage)] * 100)
            assert math.isclose(row[optimizer + '_score_0_to_100'], exact, rel_tol=0, abs_tol=1e-12)
        for optimizer in ('muon', 'normuon'):
            difference = float((stage_means[(selected[optimizer], stage)] -
                                stage_means[(selected['adamw'], stage)]) * 100)
            assert math.isclose(row[optimizer + '_minus_adamw'], difference, rel_tol=0, abs_tol=1e-12)

    parent_path = STORY / 'reports/engineering-archive/dense-v3-complete-trajectory-candidate-v1/verification.json'
    parent = bound_json(parent_path, PARENT_SHA)
    for name, item in parent['payloads'].items():
        assert identity(parent_path.parent / name) == item
    for name, item in parent['protected_inputs'].items():
        assert identity(Path(name)) == item
    assembly = trajectory.strict_json(trajectory.embedded(bundle['assembly']))
    assembly_count = 0
    for base in (Path('/root/embedding-optimizer-primary-v3'), EXPERIMENT / 'launch/source-snapshot'):
        for name, item in assembly['files'].items():
            assert identity(base / name) == item['identity']
            assembly_count += 1
    assert assembly_count == 112
    assert identity(ROOT / 'before/CURRENT_EXPERIMENT.md') == parent['current_handoff']
    payloads, links = {}, 0
    secret = re.compile(rb'(?:wandb_v1_[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})')
    for path in sorted(ROOT.rglob('*')):
        assert not path.is_symlink(), str(path)
        if not path.is_file():
            continue
        raw = path.read_bytes()
        assert not secret.search(raw), 'Credential-like string in archive'
        payloads[path.relative_to(ROOT).as_posix()] = identity(path)
        if path.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)', raw.decode()):
                if target.startswith(('http:', 'https:', '#')):
                    continue
                local = target.split('#', 1)[0]
                origin = STORY if path.parent == ROOT / 'before' else path.parent
                target_path = (origin / local).resolve()
                assert target_path.exists() or target_path == output, (str(path), local)
                links += 1
    record = {'scope': 'complete_actual_retrieval_readout_and_host_local_preservation',
              'observed_at_utc': datetime.now(timezone.utc).isoformat(),
              'native_bundle': identity(ROOT / 'actual/complete-checkpoint-readback.json'),
              'actual_readout': identity(ROOT / 'tables/readout.json'),
              'checkpoint_count': 60, 'task_score_count': 840,
              'prior_54_checkpoint_records_exactly_unchanged': True,
              'native_original_files_reopened_and_matched': native_count,
              'native_original_worker_exit_zero_count': 840,
              'observer_primary_scores_exactly_matched': 840, 'observer_baseline_rows': 14,
              'copied_source_csvs_byte_identical': 8, 'scalar_dynamics_checks': scalar_checks,
              'all_six_endpoint_contrasts_unchanged': True,
              'descriptive_tables_byte_identical_after_replay': 2,
              'selected_stage_scalar_checks': 25, 'descriptive_facts': facts,
              'figure_layouts_have_identical_data': True, 'figure_input_functions_ast_unchanged': 3,
              'original_source_assembly_files_unchanged': assembly_count,
              'protected_inputs': parent['protected_inputs'], 'candidate_archive_unchanged': identity(parent_path),
              'payloads': payloads, 'markdown_local_links_checked': links, 'credential_findings': 0,
              'current_handoff': identity(STORY / 'CURRENT_EXPERIMENT.md'),
              'current_readme': identity(STORY / 'README.md'),
              'current_project_status': identity(STORY / 'PROJECT_STATUS.md'),
              'model_or_scheduler_started': False, 'functional_recovery': False,
              'original_whole_grid_consumer_called': False, 'original_whole_grid_admission_passed': False,
              'manuscript_modified': False, 'remote_or_source_release': False, 'scientific_completion': False}
    with output.open('x') as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'verification': str(output), 'identity': identity(output),
                      'payloads': len(payloads), 'native_files_matched': native_count,
                      'checkpoint_count': 60, 'task_score_count': 840, 'csvs_reproduced': 8,
                      'scalar_dynamics_checks': scalar_checks, 'protected_inputs': len(parent['protected_inputs']),
                      'source_assembly_files_unchanged': assembly_count,
                      'scientific_completion': False}, sort_keys=True))


if __name__ == '__main__':
    main()
