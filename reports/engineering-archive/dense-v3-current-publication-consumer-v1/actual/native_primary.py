"""Fresh read-only primary admission with the accepted exact view-history amendment.

No old contract method is replaced. The unchanged deep checkpoint, validation,
and BEIR readers run in their original primary/validation source namespaces.
The obsolete one-selection guard remains failed, and source release is separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

PRIMARY = Path('/root/embedding-optimizer-primary-v3')
EXPERIMENT = Path('/root/embedding-optimizer-v3-experiment')
VALIDATION = EXPERIMENT / 'launch/validation-handoff/validation.py'
VALIDATION_SHA = 'd2fc55c67670ea5b42cacb055f48379ef278a465ce86d95808231ab86049d913'
VALIDATION_AUTH_SHA = '6a8dcdd579bd6c2f6ea82e3f4cd808f88e92e9a28b9e97f41ffa6447eb0e1f6f'
VIEW_SHA = '13cb27173e48b8262aae77cba6ed5ebb03d9839f10fec409af06beac570af473'
SCOPE = 'dense_v3_current_complete_primary_view_history_readout_v1'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
         f'Nonordinary input: {path}')
    before = path.stat()
    with path.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    need(all(getattr(before, key) == getattr(after, key) for key in keys), 'Input raced')
    return {'bytes': after.st_size, 'sha256': checksum}


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def event(name, **fields):
    print(json.dumps({'event': name, 'at_utc': datetime.now(timezone.utc).isoformat(),
                      **fields}, sort_keys=True, allow_nan=False), flush=True)


def load_original():
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only original admission')
    need(identity(VALIDATION)['sha256'] == VALIDATION_SHA, 'Original validation entry changed')
    spec = importlib.util.spec_from_file_location('original_primary_publication_validation', VALIDATION)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    context = module.authenticate(SimpleNamespace(source_sha=VALIDATION_SHA,
                                                   authorization_sha=VALIDATION_AUTH_SHA))
    return module, context


def compare_native_proof(fresh, accepted):
    """Every original native field must agree, including deep states and timing."""
    need(fresh.get('whole_run_artifacts_verified') is True, 'Missing native whole-run read')
    need(fresh.get('scientific_completion') is False, 'Native scope changed')
    need(fresh.get('steps') == [782, 1563, 2345, 3126, 3907], 'Incomplete native horizon')
    need(fresh.get('dataset_fingerprint') == '0c29f5d460d1b4d7', 'Wrong training history')
    need(accepted.get('explicit_two_selection_view_audit_passed') is True
         and accepted.get('original_single_selection_guard_passed') is False
         and accepted.get('input_view_audit_sha256') == VIEW_SHA,
         'Missing accepted content/history amendment')
    need(all(key in accepted and accepted[key] == value for key, value in fresh.items()),
         'Fresh native proof differs from original accepted evidence')


def require_coverage(primary, runs, evaluations, scores):
    run_ids = {row['run_id'] for row in primary.inputs['runs']}
    need(len(run_ids) == 12 and len(runs) == 12
         and {row['run_id'] for row in runs} == run_ids, 'Incomplete primary population')
    states = {(run, stage) for run in run_ids for stage in range(1, 6)}
    need(len(evaluations) == 60
         and {(r['run_id'], r['stage']) for r in evaluations} == states,
         'Incomplete checkpoint evaluation population')
    cells = {(run, stage, task) for run, stage in states
             for task in primary.payload['evaluation']['tasks']}
    need(len(cells) == len(scores) == 840
         and {(r['run_id'], r['stage'], r['task']) for r in scores} == cells,
         'Incomplete or duplicated task population')


def collect(output, source_sha):
    need(identity(__file__)['sha256'] == source_sha, 'Collector source changed')
    output = Path(output)
    need(output.is_absolute() and not output.exists() and not output.is_symlink(),
         'Use a fresh absolute output directory')
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'started.json', {'scope': SCOPE, 'source': identity(__file__),
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'original_single_selection_guard_passed': False, 'scientific_completion': False})
    try:
        original, context = load_original()
        from embed_optim.primary_completion import inspect_complete_run
        from embed_optim.primary_contract import read_json, require_same
        from embed_optim.primary_v3_completion import inspect_evaluation
        primary = context.original
        audit_path = EXPERIMENT / 'launch/view-history-continuation/input-view-audit.json'
        need(identity(audit_path)['sha256'] == VIEW_SHA, 'Original view audit changed')
        audit = context.continuation.view_audit(primary)
        require_same(audit, read_json(audit_path)['audit'])
        write_new(output / 'current-view-audit.json', audit)
        event('full_500k_content_history_audit_passed')
        runs, bindings = [], []
        roots = {}
        for recipe in primary.inputs['runs']:
            run_id = recipe['run_id']
            root = EXPERIMENT / primary.payload['output_root'] / 'dense' / run_id
            roots[run_id] = root
            accepted = original.completion(context, run_id)
            need(accepted is not None, 'Missing completed primary run')
            fresh = inspect_complete_run(root, primary.expected_identity(run_id),
                                         primary.payload['checkpoint_steps'])
            context.continuation.check_completed_view(fresh, audit)
            compare_native_proof(fresh, accepted['proof'])
            run = {'run_id': run_id, **fresh}
            runs.append(run)
            bindings.append(accepted['binding'])
            write_new(output / f'{run_id}.fresh-native.json',
                      {'native': run, 'original_completion': accepted['binding']})
            event('run_native_complete', run_id=run_id, count=len(runs))
        # No BEIR information is read until the original all-twelve validation
        # selection is freshly reconstructed from every held-out sample score.
        dataset, data = original.data_for(context)
        rows, receipts = [], []
        for recipe in primary.inputs['runs']:
            run_id = recipe['run_id']
            plan, binding, checkpoint = original.admitted_plan(context, run_id, data)
            require_same(binding, next(b for b in bindings if b['path'].endswith(
                f'{run_id}.view-verified.json') or b['path'].endswith(f'{run_id}.artifacts-verified.json')))
            result = context.io.inspect_saved(original.result_path(plan), plan, data['row_identities'])
            overall = [row for row in result['summary']['groups'] if row['group'] == '__all__']
            need(len(overall) == 1 and overall[0]['samples'] == 4096, 'Incomplete validation rows')
            optimizer = primary.expected_identity(run_id)['recipe']['optimizer']
            rows.append({'run_id': run_id, 'optimizer': optimizer['name'],
                         'learning_rate': optimizer['lr'],
                         **{key: overall[0][key] for key in context.numerical.METRICS}})
            receipts.append(result)
        selection = {'scope': 'dense_primary_v3_validation_selection',
            'primary_protocol_sha256': primary.sha256,
            'validation_protocol_sha256': context.validation.sha256,
            'selected': context.numerical.select_recipes(rows, primary.inputs['runs']),
            'run_metrics': rows, 'source_receipts': receipts, 'scientific_completion': False,
            'boundary': 'Held-out selection only. No BEIR input; not retrieval inference or paper release.'}
        write_new(output / 'validation-selection.json', selection)
        event('full_validation_selection_complete', runs=len(rows))
        scores, evaluations = [], []
        for recipe in primary.inputs['runs']:
            run_id = recipe['run_id']
            expected = primary.expected_identity(run_id)
            accepted = original.completion(context, run_id)
            for stage, step in enumerate(primary.payload['checkpoint_steps'], 1):
                evaluated = inspect_evaluation(primary, roots[run_id] / f'checkpoint-{step}',
                    run_id, step, EXPERIMENT / 'evaluations/dense-primary-v3')
                context.evaluation.bind_plan_to_completion(evaluated['plan'], accepted['proof'], step)
                evaluations.append({'run_id': run_id, 'stage': stage, 'step': step, **evaluated})
                for row in evaluated['tasks']:
                    scores.append({'run_id': run_id, 'model_family': 'dense',
                        'optimizer': expected['recipe']['optimizer']['name'],
                        'learning_rate': expected['recipe']['optimizer']['lr'],
                        'stage': stage, 'step': step,
                        'fraction': expected['recipe']['checkpoint_fractions'][stage - 1],
                        'task': row['task'], 'ndcg_at_10': row['ndcg_at_10']})
            event('run_beir_complete', run_id=run_id, checkpoints=len(evaluations), tasks=len(scores))
        require_coverage(primary, runs, evaluations, scores)
        require_same(context.continuation.view_audit(primary), audit)
        # Authentication again checks all immutable source/authority/runtime
        # parents without changing any original contract or process state.
        original.authenticate(SimpleNamespace(source_sha=VALIDATION_SHA,
                                               authorization_sha=VALIDATION_AUTH_SHA))
        need(identity(__file__)['sha256'] == source_sha, 'Collector changed during native reading')
        import torch
        need(not torch.cuda.is_initialized(), 'Native CPU consumer initialized CUDA')
        modules = {}
        for name, module in tuple(sys.modules.items()):
            if name == 'embed_optim' or name.startswith('embed_optim.') or name.startswith('_frozen_dense_v3_validation.'):
                path = Path(module.__file__).resolve()
                modules[name] = {'path': str(path), **identity(path)}
        grid = {'scope': 'dense_primary_v3_complete_grid', 'artifact_grid_verified': True,
            'scientific_completion': False, 'protocol_sha256': primary.sha256,
            'runs': runs, 'evaluations': evaluations, 'score_rows': scores,
            'boundary': 'Current whole-run and BEIR admission with the explicit content/history amendment; not original draft-guard or source-release acceptance.'}
        value = {'scope': SCOPE, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'source': identity(__file__), 'native_modules': modules,
            'original_completions': bindings, 'current_view_audit': audit,
            'grid': grid, 'validation_selection': selection,
            'current_native_primary_admission_complete': True,
            'original_single_selection_guard_passed': False,
            'committed_source_release': False, 'manuscript_installed': False,
            'scientific_completion': False}
        write_new(output / 'evidence.json', value)
        write_new(output / 'completed.json', {'scope': SCOPE, 'evidence': identity(output / 'evidence.json'),
            'complete_runs': 12, 'native_checkpoints': 60, 'beir_cells': 840, 'validation_runs': 12,
            'current_native_primary_admission_complete': True, 'original_single_selection_guard_passed': False,
            'scientific_completion': False})
        event('current_native_primary_admission_complete', **identity(output / 'evidence.json'))
    except Exception as error:
        write_new(output / 'failed.json', {'at_utc': datetime.now(timezone.utc).isoformat(),
            'exception_type': type(error).__name__, 'message': str(error),
            'original_guards_modified': False, 'scientific_completion': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args()
    collect(args.output, args.source_sha)
