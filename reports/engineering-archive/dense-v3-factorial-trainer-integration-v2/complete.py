"""Collect bounded actual CPU evidence, retaining failures; never admit science."""

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from integration import identity, now, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    names = (
        'loader-second', 'loader-seed271828', 'loader-seed161803',
        'train-adamw-second', 'train-muon-first',
        'train-adamw-stop79', 'train-muon-stop313',
        'resume-adamw-first', 'resume-muon-first',
        'gradient-hybrid_adamw-first', 'gradient-muon-first',
    )
    observations = {}
    for name in names:
        observation = json.loads((root / f'{name}-readback.json').read_text())
        if (observation['scientific_admission'] is not False or
                observation['row_order_exact'] is not True or
                observation['original_exited'] != identity(root / name / 'exited.json') or
                observation['original_started'] != identity(root / name / 'started.json')):
            raise ValueError('Actual readback scope or original execution changed')
        if name.startswith('resume-') and observation.get(
                'endpoint_model_optimizer_scheduler_bitwise_equal') is not True:
            raise ValueError('Actual resumed/uninterrupted equality remains unverified')
        observations[name] = observation
    if {observations[name]['seed'] for name in names[:3]} != {314159, 271828, 161803}:
        raise ValueError('Actual three-seed loader coverage differs')
    gradient_observations = {}
    for name in names[-2:]:
        ranks = []
        for rank in range(4):
            record = json.loads((root / name / 'output' / f'rank-{rank}.json').read_text())
            probes = record['independent_gradient_reference']
            if len(probes) != 2 or record['component_identity']['max_grad_norm'] != 0.0:
                raise ValueError('Missing actual unclipped gradient probes')
            for probe, update, count in zip(probes, (1, 391), (128, 80), strict=True):
                errors = probe['all_parameter_max_absolute_errors']
                if (probe['update'] != update or probe['global_groups'] != count or
                    probe['actual_micro_batches_per_rank'] != 4 or
                    probe['absolute_tolerance'] != 5e-6 or probe['relative_tolerance'] != 5e-4 or
                    probe['quarter_gradient_rejected'] is not True or
                    probe['fourfold_gradient_rejected'] is not True or
                    set(errors) != set(record['shapes']) or
                    any(not math.isfinite(v) or v < 0 for v in errors.values())):
                    raise ValueError('Actual FP64 oracle coverage, controls or tolerance differs')
            ranks.append(probes)
        gradient_observations[name] = {
            'rank_probes': ranks,
            'maximum_absolute_error': max(value for probes in ranks for probe in probes
                                          for value in probe['all_parameter_max_absolute_errors'].values()),
            'comparison_ran_inside_original_four_rank_worker': True,
            'raw_gradient_tensors_persisted': False,
            'fresh_numeric_gradient_replay_by_this_collector': False,
        }
    failures = {}
    for name, expected_text in (
            ('loader-first', 'Factorial requires all three nonempty parameter partitions'),
            ('train-adamw-first', 'Externally admitted resume step is required before state loading')):
        exited = json.loads((root / name / 'exited.json').read_text())
        if exited['exit_code'] != 1 or expected_text not in (root / name / 'worker.log').read_text():
            raise ValueError('Original diagnostic failure evidence is missing')
        failures[name] = exited
    suites = {}
    for filename, expected_total, expected_failures in (
            ('placement-first.xml', 18, 2), ('placement-retry.xml', 18, 0),
            ('optimizer-regression.xml', 82, 0)):
        xml = ET.parse(root / filename).getroot()
        cases = xml.findall('.//testcase')
        counts = {'cases': len(cases), 'failures': sum(c.find('failure') is not None for c in cases),
                  'errors': sum(c.find('error') is not None for c in cases),
                  'skipped': sum(c.find('skipped') is not None for c in cases)}
        if counts != {'cases': expected_total, 'failures': expected_failures, 'errors': 0, 'skipped': 0}:
            raise ValueError('Actual original test coverage differs')
        suites[filename] = counts
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError('Diagnostic archive contains a link or special file')
        if path.is_file():
            files[str(path)] = identity(path)
    for path in sorted(Path(__file__).parent.glob('*')):
        if path.is_file():
            files[str(path.resolve())] = identity(path)
    result = {
        'scope': 'completed-bounded-cpu-factorial-trainer-integration',
        'scientific_admission': False, 'observed_at_utc': now(),
        'actual_successful_four_rank_attempts': 11, 'preserved_failed_four_rank_attempts': 2,
        'observations': observations, 'live_worker_gradient_comparisons': gradient_observations,
        'preserved_failures': failures, 'test_suites': suites,
        'immutable_files': files,
        'still_required': [
            'Genuine corrected DenseOn source-state, calibration and branch-data binding',
            'Actual default-topology four-GPU factorial admission',
            'Whole-run scientific consumer and twelve formal branches',
            'Complete held-out retrieval and functional analysis; publication/release gates',
        ],
        'boundary': 'All toy work used hidden CUDA and separate sources. Primary training, live evaluation kernels, protocols and manuscript were unchanged.',
    }
    save(args.output, result)
    print(json.dumps({'handoff': str(args.output), **identity(args.output),
                      'immutable_files': len(files), 'actual_successful_four_rank_attempts': 11,
                      'gradient_maxima': {k: v['maximum_absolute_error']
                                          for k, v in gradient_observations.items()}}), flush=True)


if __name__ == '__main__':
    main()
