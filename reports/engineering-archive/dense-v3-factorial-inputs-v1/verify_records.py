"""Verify durable copies and exact source/data/state evidence; no new model execution."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from audit_data import file_identity, read_json, require


def main():
    archive = Path(__file__).resolve().parent
    original = Path('/tmp/dense-v3-factorial-input-audit.LNUormRA')
    output = archive / 'verification.json'
    require(not output.exists(), 'Preserve previous input-verification attempts')
    for filename, digest in (
        ('data-first.json', '94ea1b27631cf9270f445e6406e79c0c470bd346564a4057cee1f742923d103f'),
        ('states-first.json', '9958ecc4f23d9165838632c6bd7e24f1e64e3753eb5632206f2228411179e932'),
    ):
        require(file_identity(archive / filename) == file_identity(original / filename) and
                file_identity(archive / filename)['sha256'] == digest, 'Original audit copy differs')
    data, states = (read_json(archive / name) for name in ('data-first.json', 'states-first.json'))
    require(data['scientific_admission'] is states['scientific_admission'] is False,
            'Input evidence became scientific admission')
    require(data['branch']['rows'] == 50000 and data['calibration']['rows'] == 32 and
            len(data['branch']['columns']) == len(data['calibration']['columns']) == 21 and
            data['all_selected_field_values_equal_to_revised_primary'] is True and
            data['branch']['differing_rows'] == data['calibration']['differing_rows'] == [] and
            data['branch']['ledger_differing_sample_ids'] == [] and
            data['branch']['revised_positions_in_branch'] == [] and
            data['calibration']['revised_positions_in_calibration'] == [],
            'Actual full-field branch/calibration relation differs')
    for path, expected in data['inputs'].items():
        require(file_identity(path) == expected, 'An original data/protocol/source input changed')
    require(len(states['states']) == 2 and states['calibration_performed'] is False and
            states['formal_branch_started'] is False, 'Source-state scope differs')
    source_files = dict(states['source_files'])
    for key in ('source_manifest', 'data_relation_receipt', 'complete_primary_input', 'audit_source'):
        record = states[key]
        require(file_identity(record['path']) == {k: record[k] for k in ('bytes', 'sha256')},
                'An original source-state input changed')
    for path, expected in source_files.items():
        require(file_identity(path) == expected, 'Loaded numerical source changed')
    native_files = 0
    for state, run in zip(states['states'], ('verified-v3-adamw-3e-5', 'verified-v3-muon-3e-4'), strict=True):
        require(state['genuine_v3_run_id'] == run and state['native_checkpoint']['step'] == 2345 and
                state['model_tensors'] == 134 and state['model_parameters'] == 149014272 and
                state['loaded_weights_bitwise_equal_before_and_after'] is True and
                state['actual_forward_or_backward_executed'] is False and
                state['formal_gpu_execution_verified'] is False, 'Genuine source/loading coverage differs')
        require(set(state['fresh_resets']) == {'hybrid_adamw', 'muon'}, 'Missing reset operator')
        for reset in state['fresh_resets'].values():
            require(reset['optimizer_steps'] == reset['parameter_states'] == 0,
                    'Source-state reset was not empty')
        for row in state['native_checkpoint']['files']:
            require(file_identity(Path(state['checkpoint']) / row['path']) ==
                    {k: row[k] for k in ('bytes', 'sha256')}, 'A native source-checkpoint file changed')
            native_files += 1
        for key in ('original_durability_receipt', 'original_remote_audit_receipt'):
            record = state[key]
            require(file_identity(record['path']) == {k: record[k] for k in ('bytes', 'sha256')},
                    'Original durability evidence changed')
    require(file_identity(archive / 'data-comparison-first.xml') == file_identity(original / 'data-comparison-first.xml'),
            'Original focused test result changed')
    cases = ET.parse(archive / 'data-comparison-first.xml').getroot().findall('.//testcase')
    require(len(cases) == 24 and all(not any(row.find(key) is not None for key in ('failure', 'error', 'skipped'))
                                   for row in cases), 'Focused comparison controls did not pass')
    files = {str(path.resolve()): file_identity(path) for path in sorted(archive.iterdir()) if path.is_file()}
    result = {
        'scope': 'durable-genuine-factorial-input-and-loading-evidence-readback',
        'scientific_admission': False, 'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'data_input_files_rehashed': len(data['inputs']), 'source_checkpoint_files_rehashed': native_files,
        'actual_branch_rows': 50000, 'actual_calibration_rows': 32,
        'actual_source_states': 2, 'fresh_empty_reset_pairs': 4, 'comparison_tests_passed': 24,
        'files': files, 'fresh_model_or_gradient_replay': False,
        'formal_branch_or_calibration_admission': False,
    }
    with output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'verification': str(output), **file_identity(output),
                      'bound_archive_files': len(files), 'data_inputs': len(data['inputs']),
                      'checkpoint_files': native_files, 'focused_tests': 24}), flush=True)


if __name__ == '__main__':
    main()
