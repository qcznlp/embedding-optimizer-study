"""Independently check saved tracking observations; never query or modify W&B."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT_SHA = '9eccd575e9da494f94ccf9202b2c1667719f9dab74cfa566c7fec7fadb849761'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Nonordinary file')
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('Nonfinite JSON value')
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=invalid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute() and not args.output.exists(), 'Use a new absolute output')
    actual = ROOT / 'actual'
    readout = load(actual / 'readout.json')
    inputs = load(actual / 'input-bindings.json')
    require(binding(actual / 'input-bindings.json') == readout['input_bindings'], 'Input manifest changed')
    require(binding(ROOT / 'tracking.py') == readout['source'] == inputs['source'], 'Audit source changed')
    require(readout['source']['sha256'] == AUDIT_SHA, 'Wrong audit source')
    for path, expected in inputs['files'].items():
        require(binding(path) == expected, 'Original metadata changed: ' + path)
    rows = readout['runs']
    require(len(rows) == 24 and len({r['run_id'] for r in rows}) == 24, 'Wrong run population')
    require(readout['query_failures'] == [], 'An online read failed')
    require(sum(r['family'] == 'primary' for r in rows) == 12, 'Wrong primary count')
    require(sum(r['family'] == 'continuation' for r in rows) == 12, 'Wrong continuation count')
    all_rows, differences, primary_checks, continuation_fields = 0, [], 0, []
    for row in rows:
        run = row['run_id']
        record_path = actual / 'runs' / (run + '.json')
        require(binding(record_path) == row['record'], 'Saved online observation changed')
        record = load(record_path)
        require(record['run_id'] == run and record['remote_id'] == row['remote_id'], 'Wrong tracking identity')
        require(record['state'] == row['state'] == 'finished', 'Wrong remote state')
        require(record['native_whole_run_verified_upstream'] is True, 'Upstream native completion absent')
        primary = row['family'] == 'primary'
        total = 3907 if primary else 391
        suffix = '/' + run + ('/trainer_state_final.json' if primary else '/checkpoint-391/trainer_state.json')
        state_paths = [p for p in inputs['files'] if p.endswith(suffix)]
        require(len(state_paths) == 1, 'Native final-state binding ambiguous')
        state = load(state_paths[0])
        require(state['global_step'] == state['max_steps'] == total and state['epoch'] == 1, 'Native endpoint differs')
        require(state['log_history'] == record['native_history'], 'Copied native history differs')
        require(record['summary'] == {'train/global_step': total, 'train/epoch': 1}, 'Remote endpoint differs')
        left = [v for v in state['log_history'] if 'loss' in v]
        right = record['online_history']
        expected = [1] + list(range(10, total + 1, 10))
        require([v['step'] for v in left] == expected, 'Native ordered logging population differs')
        require([v['train/global_step'] for v in right] == expected, 'Online ordered logging population differs')
        require(all(type(v['train/global_step']) is int for v in right), 'Nonintegral online step')
        require(len(left) == len(right) == row['history_rows'], 'Metric row count differs')
        all_rows += len(left)
        base = record['native_recipe']['optimizer']['lr']
        warmup = (total + 9) // 10
        for native, online in zip(left, right):
            for metric in ('loss', 'grad_norm', 'learning_rate'):
                a, b = native[metric], online['train/' + metric]
                require(type(a) in (int, float) and type(b) in (int, float), 'Nonnumeric metric')
                require(math.isfinite(a) and math.isfinite(b) and a == b, 'Metric differs or is nonfinite')
            before = native['step'] - 1
            # Derive the same declared linear schedule without using tracking.py.
            fraction = before / warmup if before < warmup else (total - before) / (total - warmup)
            require(native['learning_rate'] == base * fraction, 'Native linear schedule differs')
        require(record['metrics']['numeric_differences'] == [], 'Original audit reported differences')
        require(row['exact_history_match'] is True and row['native_schedule_exact'] is True, 'Summary differs')
        if primary:
            check = record['primary_original_per_run_audit']
            require(check['status'] == 'valid' and check['problems'] == [], 'Original primary per-run check failed')
            primary_checks += 1
        else:
            require(record['argument_fields_absent_online'] == [], 'A requested argument was absent')
            require('optimizer' in record['research_recipe_fields_absent_online'], 'Unexpected research-config coverage')
            continuation_fields.append({'run_id': run, 'missing_research_fields': record['research_recipe_fields_absent_online'],
                                        'online_standard_optim': record['online_config'].get('optim'),
                                        'actual_native_optimizer': record['native_recipe']['optimizer']['name']})
            mismatches = record['argument_exact_mismatches']
            require(set(mismatches) <= {'learning_rate'}, 'Unexpected argument difference')
            for field, values in mismatches.items():
                a, b = values['native'], values['online']
                require(a != b, 'Inexact argument was incorrectly labelled')
                # Measure, but do not tolerance-collapse, the retained discrepancy.
                differences.append({'run_id': run, 'field': field, **values,
                                    'absolute_difference': abs(a - b),
                                    'adjacent_binary64_values': math.nextafter(a, b) == b})
        require(record['online_mutation'] is False and record['fresh_native_model_read'] is False, 'Scope changed')
    require(all_rows == readout['total_metric_rows'] == 5172, 'Incomplete metric population')
    require(primary_checks == readout['primary_config_and_terminal_passes'] == 12, 'Primary check count differs')
    require(len(differences) == 9, 'Argument discrepancy population differs')
    for path, expected in inputs['files'].items():
        require(binding(path) == expected, 'Original metadata raced')
    result = {'scope': 'independent-offline-readback-of-real-tracking-observations',
              'observed_at_utc': datetime.now(timezone.utc).isoformat(),
              'upstream_readout': binding(actual / 'readout.json'), 'source': binding(__file__),
              'native_metadata_bindings_rehashed': len(inputs['files']), 'runs': 24,
              'metric_rows': all_rows, 'exact_scalar_comparisons': all_rows * 3,
              'all_recorded_lr_schedules_exact': True, 'original_primary_per_run_checks_passed': primary_checks,
              'argument_discrepancies_preserved': differences,
              'continuation_online_metadata_limitations': continuation_fields,
              'no_network_or_model_payload_read': True, 'online_mutations': False,
              'source_release': False, 'scientific_completion': False}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in (
        'argument_discrepancies_preserved', 'continuation_online_metadata_limitations')}, sort_keys=True))


if __name__ == '__main__':
    main()
