"""Check the actual currently complete cohort without producing trajectory results."""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import complete_trajectory as trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-bundle', type=Path, required=True)
    parser.add_argument('--endpoint-readout', type=Path, required=True)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    trajectory.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU only')
    trajectory.require(args.output.is_absolute() and not args.output.exists(), 'Use a new output')
    raw, identity = trajectory.read_bound(args.prior_bundle, trajectory.PRIOR_54)
    endpoint_raw, endpoint_id = trajectory.read_bound(args.endpoint_readout, trajectory.ENDPOINT)
    prior, endpoint = trajectory.strict_json(raw), trajectory.strict_json(endpoint_raw)
    _, sources, functions, rules = trajectory.load_kernel(args.repository)
    rows = [row for record in prior['checkpoints'] for row in trajectory.validate_record(record)]
    actual = {(row['run_id'], row['step']) for row in rows}
    required = {(run, step) for run in trajectory.RUNS for step in trajectory.STEPS}
    missing = sorted(required - actual)
    trajectory.require(len(actual) == 54 and len(rows) == 756 and len(missing) == 6,
                       'Different currently accepted cohort')
    try:
        trajectory.complete_rows(prior, prior)
    except ValueError as error:
        refusal = str(error)
        trajectory.require(refusal.startswith('Require all 60 checkpoints'), 'Unexpected refusal')
    else:
        raise ValueError('Partial cohort incorrectly admitted as a complete trajectory')
    trajectory.require(not any(n == 'torch' or n.startswith(('torch.', 'embed_optim.'))
                               for n in sys.modules), 'Model/project import occurred')
    report = {
        'scope': 'complete_trajectory_adapter_available_input_check',
        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'candidate_source': {'bytes': Path(trajectory.__file__).stat().st_size,
                             'sha256': trajectory.sha(Path(trajectory.__file__).read_bytes())},
        'input': identity, 'accepted_endpoint': endpoint_id,
        'frozen_sources': sources, 'unchanged_function_sha256': functions,
        'actual_complete_checkpoints_checked': 54, 'actual_task_scores_checked': 756,
        'missing_checkpoint_states': [{'run_id': run, 'step': step} for run, step in missing],
        'incomplete_grid_refused': True, 'refusal': refusal,
        'original_validation_selection': endpoint['selected'],
        'frozen_dynamics': rules['dynamics'],
        'full_grid_summary_executed': False, 'trajectory_tables_or_figures_generated': False,
        'new_model_or_retrieval_execution': False, 'source_release': False,
        'scientific_completion': False,
    }
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
