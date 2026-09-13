"""Post-result measurement interpretation; no model, training or primary gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

STORY = Path('/root/embedding-optimizer-story-refactor')
EXP = Path('/root/embedding-optimizer-v3-experiment/analyses')
VECTOR = EXP / 'dense-primary-v3-functional-dimensions-recovery-v1/vectors'
FEATURE = EXP / 'dense-primary-v3-functional-features-recovery-v3'
MANIFEST = STORY / 'reports/engineering-archive/dense-v3-close-loop-handoff-v1/functional-durability/artifact_manifest.json'
TABLES = STORY / 'reports/dense-v3-functional-inference-v1/actual/tables.json'
MANIFEST_SHA = '0b498557040c2329e6935fdc539bce6d029b8bbf02f8ffcbc194912203b8cc38'
SOURCE = STORY / 'src/embed_optim/dimension_utilization.py'
SOURCE_SHA = 'b07d77c32fb061a6c8e4d5384a5943846146de79a06093fe8973170897f6a9cb'
FROZEN = {
    SOURCE: SOURCE_SHA,
    STORY / 'src/embed_optim/dimension_interventions.py': '3728c1cb6155ea3200173c5816098581a787a861ec3598b8a6b49640d37deee4',
    STORY / 'configs/dense_dimension_utilization_protocol.json': 'de56772ef84745c074940527d621a932a06a20e865a10bc3faad823bcb0e651e',
    STORY / 'paper/main.tex': '45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e',
    STORY / 'paper/references.bib': 'fca6bf5762a2cecab795a1a60c1338ab4eb999acb5d78c437b9df0e5f3553218',
}
DIAGNOSTIC_ATOL = 1e-12


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def read_bound(path, expected):
    if Path(path).is_symlink() or sha(path) != expected['sha256'] or Path(path).stat().st_size != expected['bytes']:
        raise ValueError(f'Input binding differs: {path}')
    return Path(path)


def load_original():
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Original numerical source changed')
    spec = importlib.util.spec_from_file_location('original_dimension_utilization', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unit_vectors(q, d):
    q, d = np.asarray(q, dtype=np.float64), np.asarray(d, dtype=np.float64)
    qn, dn = np.linalg.norm(q, axis=-1, keepdims=True), np.linalg.norm(d, axis=-1, keepdims=True)
    if np.any(qn == 0) or np.any(dn == 0):
        raise ValueError('Zero vector')
    return q / qn, d / dn


def derivative(q, d):
    q, d = unit_vectors(q, d)
    scores = np.einsum('nd,ncd->nc', q, d, optimize=True)
    values = -q[:, None, :] * d + 0.5 * scores[:, :, None] * (q[:, None, :] ** 2 + d ** 2)
    return values


def synthetic_controls():
    original = load_original()
    rng = np.random.default_rng(20260913)
    q, d = rng.normal(size=(12, 8)), rng.normal(size=(12, 8, 8))
    g = derivative(q, d)
    epsilon = 1e-5
    finite_difference = np.empty_like(g)
    for j in range(8):
        pairs = []
        for delta in (epsilon, -epsilon):
            qj, dj = q.copy(), d.copy()
            qj[:, j] *= np.sqrt(1 - delta)
            dj[:, :, j] *= np.sqrt(1 - delta)
            pairs.append(original._cosine_scores(qj, dj))
        finite_difference[:, :, j] = (pairs[0] - pairs[1]) / (2 * epsilon)
    error = float(np.max(np.abs(g - finite_difference)))
    balance = float(np.max(np.abs(g.sum(axis=-1))))
    invariant = float(np.max(np.abs(original._cosine_scores(q, d) - original._cosine_scores(3 * q, 3 * d))))
    if error > 2e-8 or balance > DIAGNOSTIC_ATOL or invariant > DIAGNOSTIC_ATOL:
        raise ValueError('Synthetic derivative/scale control failed')
    return {'scope': 'synthetic_controls_not_scientific_outcomes', 'seed': 20260913,
            'derivative_central_difference_max_abs_error': error,
            'zero_sum_max_abs_error': balance, 'scale_invariance_max_abs_error': invariant,
            'derivative_bound': 2e-8, 'algebra_bound': DIAGNOSTIC_ATOL}


def analyze(q_raw, d_raw, groups, original):
    q_raw, d_raw = q_raw.astype(np.float64), d_raw.astype(np.float64)
    full_scores = original._cosine_scores(q_raw, d_raw)
    full_ndcg, margin, _ = original._query_metrics(full_scores)
    nidx = 1 + np.argmax(full_scores[:, 1:], axis=1)
    ix = np.arange(len(q_raw))
    negative_ties = np.sum(full_scores[:, 1:] == np.max(full_scores[:, 1:], axis=1)[:, None], axis=1) > 1
    q, d = unit_vectors(q_raw, d_raw)
    direct = -q * (d[:, 0, :] - d[ix, nidx, :])
    all_derivatives = derivative(q_raw, d_raw)
    gradient = all_derivatives[:, 0, :] - all_derivatives[ix, nidx, :]
    actual, fixed, switching, renormalization, ndcg_gain = [np.empty(q.shape, dtype=np.float64) for _ in range(5)]
    switches = np.zeros(q.shape, dtype=bool)
    for j in range(q.shape[1]):
        scores = original._cosine_scores(q_raw, d_raw, np.arange(q.shape[1]) != j)
        ablated_ndcg, ablated_margin, _ = original._query_metrics(scores)
        actual[:, j] = ablated_margin - margin
        fixed_margin = scores[:, 0] - scores[ix, nidx]
        fixed[:, j] = fixed_margin - margin
        switching[:, j] = ablated_margin - fixed_margin
        renormalization[:, j] = fixed[:, j] - direct[:, j]
        ndcg_gain[:, j] = ablated_ndcg - full_ndcg
        switches[:, j] = np.max(scores[:, 1:], axis=1) > scores[ix, nidx]
    exact_error = float(np.max(np.abs(actual - direct - renormalization - switching)))
    zero_sum_error = float(np.max(np.abs(gradient.sum(axis=1))))
    if exact_error > DIAGNOSTIC_ATOL or zero_sum_error > DIAGNOSTIC_ATOL or np.any(switching > 0):
        raise ValueError('Decomposition/zero-sum/sign check failed')
    names = sorted(set(groups.tolist()))
    arrays = {'task_groups': np.asarray(names)}
    raw = {'actual': actual, 'fixed_negative': fixed, 'direct': direct,
           'renormalization': renormalization, 'switching': switching,
           'first_order': gradient, 'ndcg_removal_gain': ndcg_gain}
    for key, a in raw.items():
        arrays[key] = np.stack([a[groups == name].mean(axis=0) for name in names])
    rows = []
    for t, name in enumerate(names):
        selected = groups == name
        a, g = arrays['actual'][t], arrays['first_order'][t]
        h, b = np.maximum(-a, 0).sum(), np.maximum(a, 0).sum()
        hlin, blin = np.maximum(-g, 0).sum(), np.maximum(g, 0).sum()
        total = np.abs(a).sum()
        error_l1 = np.abs(a - g).sum()
        row = {
            'task': name, 'baseline_margin': float(margin[selected].mean()),
            'baseline_shortlist_ndcg': float(full_ndcg[selected].mean()),
            'helpful_mass': float(h), 'degrading_mass': float(b),
            'helpful_share': float(h / total) if total > 0 else 0.0,
            'first_order_half_l1': float(np.abs(g).sum() / 2),
            'first_order_helpful_mass': float(hlin), 'first_order_degrading_mass': float(blin),
            'direct_sum': float(arrays['direct'][t].sum()),
            'renormalization_sum': float(arrays['renormalization'][t].sum()),
            'switching_sum': float(arrays['switching'][t].sum()),
            'fixed_negative_sum': float(arrays['fixed_negative'][t].sum()),
            'actual_sum': float(a.sum()), 'first_order_sum': float(g.sum()),
            'finite_vs_first_order_l1': float(error_l1),
            'finite_vs_first_order_relative_l1': float(error_l1 / total) if total > 0 else None,
            'switching_fraction': float(switches[selected].mean()),
            'queries_with_any_switch_fraction': float(switches[selected].any(axis=1).mean()),
            'baseline_negative_tie_queries': int(negative_ties[selected].sum()),
            'max_coordinate_energy': float(max(np.max(q[selected] ** 2), np.max(d[selected] ** 2))),
            'switching_to_total_mass': float(-arrays['switching'][t].sum() / total) if total > 0 else None,
        }
        rows.append(row)
    return arrays, rows, {'exact_decomposition_max_abs_error': exact_error,
                         'per_query_first_order_zero_sum_max_abs_error': zero_sum_error}


def write_csv(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This reader must have CUDA hidden')
    output = Path(output)
    output.mkdir(parents=False, exist_ok=False)
    dump(output / 'started.json', {'time_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': sha(__file__), 'plan_sha256': sha(Path(__file__).with_name('analysis-plan.md')),
        'scope': 'post-result-cosine-deletion-interpretation', 'scientific_completion': False})
    for p, h in FROZEN.items():
        if sha(p) != h:
            raise ValueError(f'Frozen source changed: {p}')
    if sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError('Trusted immutable HF manifest differs')
    manifest = json.loads(MANIFEST.read_text())['files']
    bindings = {}

    def bound(logical):
        if logical.startswith('vectors/'):
            path = VECTOR / logical.removeprefix('vectors/')
        elif logical.startswith('features/'):
            path = FEATURE / logical.removeprefix('features/')
        elif logical == 'results/functional-inference/tables.json':
            path = TABLES
        else:
            raise ValueError('Unexpected input role')
        read_bound(path, manifest[logical])
        bindings[str(path)] = {k: manifest[logical][k] for k in ('bytes', 'sha256')}
        return path

    vectors = json.loads(bound('vectors/manifest.json').read_text())
    features = json.loads(bound('features/manifest.json').read_text())
    tables = json.loads(bound('results/functional-inference/tables.json').read_text())
    labels = sorted(vectors['states'])
    if len(labels) != 61 or set(labels) != set(features['states']):
        raise ValueError('Complete paired population required')
    bridge = {(r['run_id'], r['stage']): r for r in tables['bridge_rows']}
    if len(bridge) != 60:
        raise ValueError('Complete original retrieval bridge required')
    original = load_original()
    dump(output / 'synthetic-controls.json', synthetic_controls())
    (output / 'states').mkdir()
    task_rows, state_rows, checks = [], [], []
    sample_identity = None
    for label in labels:
        vm = json.loads(bound(f'vectors/states/{label}/manifest.json').read_text())
        fm = json.loads(bound(f'features/states/{label}/manifest.json').read_text())
        state = vm['plan']['state']
        if fm['plan']['state'] != state or state['cell'] != label:
            raise ValueError('State identities disagree')
        vp = bound(f'vectors/states/{label}/vectors.npz')
        ap = bound(f'features/states/{label}/coordinate_attribution.npz')
        with np.load(vp, allow_pickle=False) as pack:
            q, d, groups, ids = (pack[k] for k in ('query_embeddings', 'document_embeddings', 'sample_groups', 'sample_ids'))
        if q.shape != (224, 768) or d.shape != (224, 8, 768) or q.dtype != np.float32 or d.dtype != np.float32:
            raise ValueError('Vector shape or storage differs')
        if len(set(groups.tolist())) != 14 or any(np.sum(groups == g) != 16 for g in set(groups.tolist())):
            raise ValueError('Task balance differs')
        if sample_identity is None:
            sample_identity = (groups.copy(), ids.copy())
        if not np.array_equal(groups, sample_identity[0]) or not np.array_equal(ids, sample_identity[1]):
            raise ValueError('Paired query identities differ')
        arrays, rows, errors = analyze(q, d, groups, original)
        with np.load(ap, allow_pickle=False) as previous:
            checks_here = {
                'task_groups_exact': np.array_equal(arrays['task_groups'], previous['task_groups']),
                'margin_attributions_exact': np.array_equal(arrays['actual'], previous['margin_removal_gain']),
                'ndcg_attributions_exact': np.array_equal(arrays['ndcg_removal_gain'], previous['ndcg_removal_gain']),
            }
        if not all(checks_here.values()):
            dump(output / 'failed-state-comparison.json', {'state': label, **checks_here, **errors})
            raise ValueError(f'Original literal attribution differs: {label}')
        destination = output / 'states' / label
        destination.mkdir(parents=True, exist_ok=False)
        with (destination / 'decomposition.npz').open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        augmented = [{'state': label, **state['meta'], 'stage': state['stage'], **r} for r in rows]
        task_rows.extend(augmented)
        state_row = {'state': label, **state['meta'], 'stage': state['stage']}
        for key in rows[0]:
            if key != 'task':
                values = [r[key] for r in rows]
                state_row[key] = float(np.mean(values)) if all(v is not None for v in values) else None
        state_row['beir_ndcg_at_10'] = None if label == 'pretrained' else bridge[(state['meta']['run_id'], state['stage'])]['mean_ndcg_at_10']
        if label != 'pretrained' and state_row['degrading_mass'] != bridge[(state['meta']['run_id'], state['stage'])]['margin_degrading_attribution_mass']:
            raise ValueError('Original checkpoint degrading mass differs')
        state_rows.append(state_row)
        checks.append({'state': label, **checks_here, **errors})
        dump(destination / 'readback.json', checks[-1])
        print(json.dumps({'states_complete': len(state_rows), 'state': label}), flush=True)
    if len(task_rows) != 854 or len(state_rows) != 61:
        raise ValueError('Incomplete population')
    write_csv(output / 'task_summary.csv', task_rows)
    write_csv(output / 'state_summary.csv', state_rows)
    keys = ['degrading_mass', 'first_order_half_l1', 'baseline_margin', 'renormalization_sum', 'switching_sum', 'beir_ndcg_at_10']
    trained = [r for r in state_rows if r['optimizer'] != 'pretrained']
    associations = []
    for i, x in enumerate(keys):
        for y in keys[i + 1:]:
            associations.append({'x': x, 'y': y, 'n': 60,
                'pearson': float(np.corrcoef([r[x] for r in trained], [r[y] for r in trained])[0, 1]),
                'scope': 'unadjusted_post_result_descriptive_no_hypothesis_test'})
    write_csv(output / 'descriptive_associations.csv', associations)
    for path, binding in bindings.items():
        read_bound(path, binding)
    for p, h in FROZEN.items():
        if sha(p) != h:
            raise ValueError(f'Frozen source changed during analysis: {p}')
    dump(output / 'input_bindings.json', {'immutable_hf_manifest_sha256': MANIFEST_SHA,
        'files': bindings, 'frozen_sources': {str(p): h for p, h in FROZEN.items()}})
    dump(output / 'completed.json', {'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'actual-post-result-cosine-deletion-interpretation', 'states': 61, 'task_states': 854,
        'original_task_mean_attributions_exact': True, 'checks': checks,
        'source_sha256': sha(__file__), 'scientific_completion': False,
        'primary_inference_modified': False, 'manuscript_modified': False,
        'gpu_or_model_used': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
