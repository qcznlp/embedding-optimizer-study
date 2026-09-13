"""Independent actual-array algebra readback; does not repeat literal ablations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main(root, destination, payload_root=None, frozen_source_root=None):
    root, destination = Path(root), Path(destination)
    if destination.exists():
        raise ValueError('Use a new verification receipt')
    complete = json.loads((root / 'completed.json').read_text())
    if complete['states'] != 61 or not complete['original_task_mean_attributions_exact']:
        raise ValueError('Require the actual complete literal-deletion analysis')
    source = root.parent / 'cosine_readout.py'
    if digest(source) != complete['source_sha256']:
        raise ValueError('Completed source differs')
    input_record = json.loads((root / 'input_bindings.json').read_text())
    resolved_inputs = {}
    for name in input_record['files']:
        if payload_root is None:
            resolved_inputs[name] = Path(name)
            continue
        prefixes = {
            '/root/embedding-optimizer-v3-experiment/analyses/dense-primary-v3-functional-dimensions-recovery-v1/vectors/': 'vectors/',
            '/root/embedding-optimizer-v3-experiment/analyses/dense-primary-v3-functional-features-recovery-v3/': 'features/',
            '/root/embedding-optimizer-story-refactor/reports/dense-v3-functional-inference-v1/actual/': 'results/functional-inference/',
        }
        matches = [(old, new) for old, new in prefixes.items() if name.startswith(old)]
        if len(matches) != 1:
            raise ValueError('Unknown restored input role')
        old, new = matches[0]
        resolved_inputs[name] = Path(payload_root) / (new + name.removeprefix(old))
    if payload_root is not None:
        restored_manifest = Path(payload_root) / 'artifact_manifest.json'
        if digest(restored_manifest) != input_record['immutable_hf_manifest_sha256']:
            raise ValueError('Restored immutable HF manifest differs')
    for name, binding in input_record['files'].items():
        actual_path = resolved_inputs[name]
        if digest(actual_path) != binding['sha256'] or actual_path.stat().st_size != binding['bytes']:
            raise ValueError('Previously bound actual input changed')
    vectors = [resolved_inputs[p] for p in input_record['files'] if p.endswith('/vectors.npz')]
    if len(vectors) != 61:
        raise ValueError('Require all 61 actual states')
    atol = 1e-12
    rows, checks = [], []
    for path in sorted(vectors):
        # Resolve the label from the authenticated state manifest, never a guessed depth.
        metadata = json.loads(path.with_name('manifest.json').read_text())['plan']['state']
        label = metadata['cell']
        with np.load(path, allow_pickle=False) as a:
            rawq = a['query_embeddings'].astype(np.float64)
            rawd = a['document_embeddings'].astype(np.float64)
            groups = a['sample_groups']
        q = rawq / np.sqrt(np.sum(rawq * rawq, axis=1))[:, None]
        d = rawd / np.sqrt(np.sum(rawd * rawd, axis=2))[:, :, None]
        energy_q, energy_d = q * q, d * d
        retained_q, retained_d = 1 - energy_q, 1 - energy_d
        if np.any(retained_q <= 0) or np.any(retained_d <= 0):
            raise ValueError('Subtractive algebra reference is undefined on this state')
        score = np.sum(q[:, None, :] * d, axis=2)
        ix = np.arange(q.shape[0])
        nidx = 1 + np.argmax(score[:, 1:], axis=1)
        pos, neg = score[:, 0], score[ix, nidx]
        margin = pos - neg
        pair_contribution = q[:, None, :] * d
        # Independent finite deletion identity, not the production literal kernel.
        deleted_score = (score[:, :, None] - pair_contribution) / np.sqrt(retained_q[:, None, :] * retained_d)
        fixed_deleted_margin = deleted_score[:, 0, :] - deleted_score[ix, nidx, :]
        dynamic_deleted_margin = deleted_score[:, 0, :] - np.max(deleted_score[:, 1:, :], axis=1)
        direct = pair_contribution[ix, nidx, :] - pair_contribution[:, 0, :]
        g_positive = -pair_contribution[:, 0, :] + pos[:, None] * (energy_q + energy_d[:, 0, :]) / 2
        g_negative = -pair_contribution[ix, nidx, :] + neg[:, None] * (energy_q + energy_d[ix, nidx, :]) / 2
        local = {
            'actual': dynamic_deleted_margin - margin[:, None],
            'fixed_negative': fixed_deleted_margin - margin[:, None],
            'direct': direct,
            'renormalization': fixed_deleted_margin - margin[:, None] - direct,
            'switching': dynamic_deleted_margin - fixed_deleted_margin,
            'first_order': g_positive - g_negative,
        }
        names = sorted(set(groups.tolist()))
        task = {key: np.stack([value[groups == g].mean(axis=0) for g in names]) for key, value in local.items()}
        with np.load(root / 'states' / label / 'decomposition.npz', allow_pickle=False) as previous:
            if previous['task_groups'].tolist() != names:
                raise ValueError('Task identities differ')
            errors = {key: float(np.max(np.abs(previous[key] - value))) for key, value in task.items()}
            if any(error > atol for error in errors.values()):
                raise ValueError(f'Independent finite/derivative algebra differs: {label}')
            # The following scalar verification uses saved original task arrays.
            for t, name in enumerate(names):
                a, g = previous['actual'][t], previous['first_order'][t]
                total = np.sum(np.abs(a))
                if total <= 0:
                    raise ValueError('Undefined task-level relative summary')
                rows.append({
                    'state': label, 'task': name,
                    'helpful_mass': float(np.sum(-a[a < 0])),
                    'degrading_mass': float(np.sum(a[a > 0])),
                    'helpful_share': float(np.sum(-a[a < 0]) / total),
                    'first_order_half_l1': float(np.sum(np.abs(g)) / 2),
                    'first_order_helpful_mass': float(np.sum(-g[g < 0])),
                    'first_order_degrading_mass': float(np.sum(g[g > 0])),
                    'direct_sum': float(np.sum(previous['direct'][t])),
                    'renormalization_sum': float(np.sum(previous['renormalization'][t])),
                    'switching_sum': float(np.sum(previous['switching'][t])),
                    'actual_sum': float(np.sum(a)),
                    'first_order_sum': float(np.sum(g)),
                    'finite_vs_first_order_l1': float(np.sum(np.abs(a - g))),
                    'finite_vs_first_order_relative_l1': float(np.sum(np.abs(a - g)) / total),
                })
        checks.append({'state': label, 'component_max_abs_errors': errors,
                       'minimum_retained_coordinate_energy': float(min(retained_q.min(), retained_d.min()))})
    with (root / 'task_summary.csv').open() as stream:
        original_rows = {(r['state'], r['task']): r for r in csv.DictReader(stream)}
    if len(original_rows) != 854 or len(rows) != 854:
        raise ValueError('Incomplete scalar task population')
    scalar_error = 0.0
    for row in rows:
        saved = original_rows[(row['state'], row['task'])]
        for key, value in row.items():
            if key not in ('state', 'task'):
                scalar_error = max(scalar_error, abs(value - float(saved[key])))
    if scalar_error > atol:
        raise ValueError('Task scalar reconstruction differs')
    for name, binding in input_record['files'].items():
        if digest(resolved_inputs[name]) != binding['sha256']:
            raise ValueError('Input changed during independent verification')
    for name, expected in input_record['frozen_sources'].items():
        original_root = '/root/embedding-optimizer-story-refactor/'
        if not name.startswith(original_root):
            raise ValueError('Unknown frozen source role')
        current = Path(name) if frozen_source_root is None else Path(frozen_source_root) / name.removeprefix(original_root)
        if digest(current) != expected:
            raise ValueError('Original frozen source changed')
    record = {
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'independent_actual_cosine_algebra_reference_not_primary_inference',
        'source_sha256': digest(__file__), 'parent_completion_sha256': digest(root / 'completed.json'),
        'states': 61, 'task_states': 854, 'checks': checks,
        'diagnostic_absolute_tolerance': atol, 'scalar_reconstruction_max_abs_error': scalar_error,
        'literal_deletion_repeated': False, 'model_encoding_repeated': False,
        'scientific_completion': False, 'manuscript_changed': False,
        'restored_payload_root': None if payload_root is None else str(Path(payload_root).resolve()),
        'frozen_source_root': None if frozen_source_root is None else str(Path(frozen_source_root).resolve()),
        'original_producer_inputs_used': payload_root is None,
        'resolved_inputs': {old: str(new) for old, new in resolved_inputs.items()},
    }
    with destination.open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in record.items() if k not in ('checks', 'resolved_inputs')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--payload-root', type=Path)
    parser.add_argument('--frozen-source-root', type=Path)
    args = parser.parse_args()
    main(args.input, args.output, args.payload_root, args.frozen_source_root)
