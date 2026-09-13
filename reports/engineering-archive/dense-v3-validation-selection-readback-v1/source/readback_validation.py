"""Read all original validation outputs and replay the unchanged loss-only selector."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/root/embedding-optimizer-v3-experiment/launch/validation-handoff')
SOURCE_SHA = 'd2fc55c67670ea5b42cacb055f48379ef278a465ce86d95808231ab86049d913'
AUTH_SHA = '6a8dcdd579bd6c2f6ea82e3f4cd808f88e92e9a28b9e97f41ffa6447eb0e1f6f'
SELECTION_SHA = '580321b217bb443e739656196858b4a87a094b3c4c883aa9baa9686eed219be6'
COMPLETED_SHA = '81f39a1c0ae9123ddcbc226e93021b25cbc34537112b31e109f3610ee6c421a5'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f'Not a regular file: {path}')
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'Moving input')
    return {'path': str(path), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Require CPU-only readback')
    require(not args.output.exists(), 'Preserve prior output')
    original = {}
    bindings = {}
    for name, expected in (('validation.py', SOURCE_SHA), ('authorization.json', AUTH_SHA),
                           ('run/all-twelve-validation-selection.json', SELECTION_SHA),
                           ('run/completed.json', COMPLETED_SHA)):
        binding, raw = snapshot(ROOT / name)
        require(binding['sha256'] == expected, f'Changed original input: {name}')
        bindings[name] = binding
        if name.endswith('.json'):
            original[name] = json.loads(raw)
    require(not (ROOT / 'run/failed.json').exists(), 'Original validation has a failure receipt')
    started_binding, started_raw = snapshot(ROOT / 'run/started.json')
    started = json.loads(started_raw)
    require(started['pid'] == 13007 and started['start_ticks'] == 298006520 and
            started['source_sha256'] == SOURCE_SHA and
            started['authorization_sha256'] == AUTH_SHA, 'Different coordinator receipt')
    bindings['run/started.json'] = started_binding
    spec = importlib.util.spec_from_file_location('original_validation_readback_source', ROOT / 'validation.py')
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    context = native.authenticate(SimpleNamespace(source_sha=SOURCE_SHA, authorization_sha=AUTH_SHA))
    _, data = native.data_for(context)
    require(len(data['row_identities']) == 4096, 'Incomplete original validation view')
    plans, workers = {}, []
    for run in context.runs:
        plan, proof, _ = native.admitted_plan(context, run, data)
        prefix = ROOT / 'run/jobs' / run
        records, files = {}, {}
        for kind in ('admission', 'started', 'exited', 'scored'):
            binding, raw = snapshot(prefix.with_suffix(f'.{kind}.json'))
            records[kind], files[kind] = json.loads(raw), binding
        admission, start, end, scored = (records[k] for k in ('admission', 'started', 'exited', 'scored'))
        require(admission['plan'] == plan and admission['training_completion'] == proof and
                admission['source_sha256'] == SOURCE_SHA and admission['authorization_sha256'] == AUTH_SHA and
                admission['committed_source_release'] is False and admission['scientific_completion'] is False,
                'Wrong original operational admission')
        require(start['run_id'] == end['run_id'] == scored['run_id'] == run and
                start['pid'] == end['pid'] and end['exit_code'] == 0 and
                start['source_sha256'] == scored['source_sha256'] == SOURCE_SHA and
                start['authorization_sha256'] == scored['authorization_sha256'] == AUTH_SHA and
                scored['rows'] == 4096 and scored['model_updates'] == 0 and
                scored['scientific_completion'] is False and
                scored['operational_admission_file'] == files['admission'],
                'Missing or inconsistent original worker completion')
        plans[run] = plan
        workers.append({'run_id': run, 'records': records, 'files': files})
    reread = native.collect(context, plans, data['row_identities'])
    require(reread == original['run/all-twelve-validation-selection.json'],
            'Native all-row metric replay or loss-only selection differs')
    completed = original['run/completed.json']
    require(completed['all_twelve_validations_verified'] is True and
            completed['scientific_completion'] is False and
            completed['selection'] == reread['selected'], 'Wrong completion/selection link')
    raw_outputs, mean_rows = [], []
    for receipt, worker in zip(reread['source_receipts'], workers, strict=True):
        run = worker['run_id']
        require(receipt == worker['records']['scored']['verified'], 'Saved worker/native result differs')
        directory = native.result_path(plans[run])
        for name in ('admission.json', 'manifest.json', 'sample_scores.jsonl', 'summary.json'):
            raw_outputs.append(snapshot(directory / name)[0])
        records = context.io.read_record_file(directory / 'sample_scores.jsonl')
        require(len(records) == 4096, 'Wrong raw record count')
        means = {name: sum(row['metrics'][name] for row in records) / 4096
                 for name in context.numerical.METRICS}
        selected_row = next(row for row in reread['run_metrics'] if row['run_id'] == run)
        require(all(selected_row[name] == value for name, value in means.items()),
                'Direct ordered raw-metric mean differs')
        mean_rows.append(selected_row)
    independently_selected = {
        name: min((r for r in mean_rows if r['optimizer'] == name),
                  key=lambda row: (row['contrastive_loss'], row['learning_rate']))['run_id']
        for name in ('adamw', 'muon', 'normuon')
    }
    require(independently_selected == reread['selected'], 'Independent loss-only selector differs')
    for name, binding in bindings.items():
        require(snapshot(ROOT / name)[0] == binding, 'Original source/receipt changed')
    for worker in workers:
        for binding in worker['files'].values():
            require(snapshot(binding['path'])[0] == binding, 'Original worker receipt changed')
    for binding in raw_outputs:
        require(snapshot(binding['path'])[0] == binding, 'Original scored input changed')
    import torch
    require(not torch.cuda.is_initialized(), 'Readback initialized CUDA')
    result = {
        'scope': 'all_twelve_original_validation_native_readback',
        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': snapshot(__file__)[0], 'original_bindings': bindings,
        'original_completed': completed, 'original_started': started,
        'native_selection_reread': reread, 'workers': workers, 'raw_output_files': raw_outputs,
        'validation_rows_per_run': 4096, 'run_count': 12, 'scored_rows': 49152,
        'eight_candidate_score_values': 393216, 'independently_replayed_scalar_metrics': 294912,
        'direct_ordered_raw_metric_means_verified': 72,
        'independently_selected_from_validation_loss': independently_selected,
        'all_twelve_original_worker_exit_codes_zero': True,
        'coordinator_os_exit_code': None,
        'beir_scores_read_by_this_reader': False, 'model_or_forward_recomputed': False,
        'cuda_initialized': False, 'source_release': False, 'scientific_completion': False,
        'boundary': 'Original validation score replay and loss-only recipe selection. Not retrieval inference, full primary completion, or a source release.',
    }
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': str(args.output), 'observed_at_utc': result['observed_at_utc'],
                      'run_count': 12, 'scored_rows': 49152, 'selected': independently_selected,
                      'run_metrics': reread['run_metrics'], 'scientific_completion': False}))


if __name__ == '__main__':
    main()
