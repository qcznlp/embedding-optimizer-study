"""Independent full row-order and actual save/resume readback for CPU diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from integration import WORKER, identity, now, save


MARKS = (79, 157, 235, 313, 391)


def equal(left, right, path='root'):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or not torch.equal(left, right):
            error = None
            if isinstance(right, torch.Tensor) and left.shape == right.shape:
                error = float((left.double() - right.double()).abs().max())
            raise ValueError(f'Tensor mismatch at {path}; max absolute error={error}')
        if not torch.isfinite(left).all():
            raise ValueError(f'Nonfinite tensor at {path}')
    elif isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError(f'Mapping mismatch at {path}')
        for key in left:
            equal(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError(f'Sequence mismatch at {path}')
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            equal(a, b, f'{path}[{index}]')
    elif type(left) is not type(right) or left != right:
        raise ValueError(f'Scalar mismatch at {path}')


def order(seed):
    # Direct permutation and positional split; no production sampler or shard.
    permutation = torch.randperm(50000, generator=torch.Generator().manual_seed(seed)).tolist()
    result = [[] for _ in range(4)]
    for step in range(391):
        group = permutation[128 * step: min(128 * (step + 1), 50000)]
        width = len(group) // 16
        for micro in range(4):
            for rank in range(4):
                offset = (4 * micro + rank) * width
                result[rank].append(group[offset: offset + width])
    return permutation, result


def read(attempt):
    attempt = Path(attempt)
    started = json.loads((attempt / 'started.json').read_text())
    exited = json.loads((attempt / 'exited.json').read_text())
    if exited['exit_code'] != 0 or exited['started'] != identity(attempt / 'started.json'):
        raise ValueError('Require actual exit-zero original launch evidence')
    if exited['log'] != identity(attempt / 'worker.log'):
        raise ValueError('Original worker output changed after completion')
    manifest_binding = started['source_manifest']
    manifest_path = Path(manifest_binding['path'])
    if identity(manifest_path) != {k: manifest_binding[k] for k in ('bytes', 'sha256')}:
        raise ValueError('Original source manifest changed')
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest['root'])
    for relative, row in manifest['files'].items():
        if identity(source / relative) != {k: row[k] for k in ('bytes', 'sha256')}:
            raise ValueError(f'Original assembled source changed: {relative}')
    output = attempt / 'output'
    records = [json.loads((output / f'rank-{rank}.json').read_text()) for rank in range(4)]
    seed, mode, step = (records[0][k] for k in ('seed', 'mode', 'completed_steps'))
    binding = records[0]['resume_binding']
    begin = 0
    if binding:
        component = Path(binding['path']) / 'factorial_trainer_component.json'
        if identity(component)['sha256'] != binding['sha256']:
            raise ValueError('Original resumed checkpoint binding changed')
        begin = json.loads(component.read_text())['step']
    permutation, expected = order(seed)
    end = 391 if mode == 'loader' else step
    for rank, record in enumerate(records):
        if (record['scientific_admission'] is not False or record['rank'] != rank or
            record['world_size'] != 4 or record['seed'] != seed or record['mode'] != mode or
            record['completed_steps'] != step or record['resume_binding'] != binding or
            record['source']['worker_sha256'] != manifest['files'][WORKER]['sha256'] or
            record['component_identity'] != records[0]['component_identity'] or
            record['consumed_batches'] != expected[rank][4 * begin:4 * end] or
            record['consumed_groups'] != sum(map(len, record['consumed_batches']))):
            raise ValueError(f'Actual rank {rank} source, identity or full row order differs')
        if record['normalization'] != {
            'owner': 'Trainer.training_step/current_gradient_accumulation_steps',
            'trainer_scheduled_accumulation': 4, 'accelerator_divisor_before': 4,
            'accelerator_divisor_after': 1,
        }:
            raise ValueError('Actual normalization ownership differs')
    consumed = [value for record in records for batch in record['consumed_batches'] for value in batch]
    target = permutation[128 * begin:min(128 * end, 50000)]
    if len(consumed) != len(set(consumed)) or sorted(consumed) != sorted(target):
        raise ValueError('Actual four-rank coverage duplicates or omits query groups')
    saved = json.loads((output / 'saved-component-bindings.json').read_text())
    checkpoint_steps = []
    states = []
    if mode == 'train':
        from embed_optim import factorial_v3_checkpoint as checkpoint
        if Path(checkpoint.__file__).resolve() != source / 'src/embed_optim/factorial_v3_checkpoint.py':
            raise ValueError('Native checkpoint reader imported from a foreign source')
        for seal in saved:
            payload = checkpoint.read(seal, records[0]['component_identity'])
            checkpoint_steps.append(payload['step'])
        if checkpoint_steps != [mark for mark in MARKS if begin < mark <= end]:
            raise ValueError('Actual saved checkpoints differ from the five declared stages')
        states = [torch.load(output / f'rank-{rank}.pt', map_location='cpu', weights_only=True)
                  for rank in range(4)]
        for rank, state in enumerate(states):
            equal(states[0], state, f'rank-{rank}')
        if states[0]['optimizer']['factorial_v3']['steps'] != step:
            raise ValueError('Actual saved named optimizer counter differs')
    elif saved:
        raise ValueError('A loader diagnostic may not claim training checkpoints')
    result = {
        'attempt': str(attempt.resolve()), 'mode': mode, 'seed': seed,
        'optimizer': records[0]['optimizer'], 'begin_step': begin, 'end_step': end,
        'completed_training_steps': step, 'groups_consumed': len(consumed),
        'distinct_groups': len(set(consumed)), 'rank_groups': [r['consumed_groups'] for r in records],
        'native_checkpoint_steps': checkpoint_steps, 'row_order_exact': True,
        'rank_states_bitwise_equal': True if states else None,
        'original_started': identity(attempt / 'started.json'),
        'original_exited': identity(attempt / 'exited.json'),
    }
    return result, records, states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result, records, states = read(args.attempt)
    if args.reference:
        reference, reference_records, reference_states = read(args.reference)
        if result['end_step'] != reference['end_step'] or not states or not reference_states:
            raise ValueError('Require actual equal-endpoint training payloads for comparison')
        if records[0]['component_identity'] != reference_records[0]['component_identity']:
            raise ValueError('Reference source/data/execution identity differs')
        equal(states, reference_states, 'continued-versus-uninterrupted')
        result['reference'] = reference
        result['endpoint_model_optimizer_scheduler_bitwise_equal'] = True
    result.update(scope='cpu-toy-full-factorial-trainer-readback', scientific_admission=False,
                  observed_at_utc=now(), reader={'path': str(Path(__file__).resolve()), **identity(__file__)})
    save(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
