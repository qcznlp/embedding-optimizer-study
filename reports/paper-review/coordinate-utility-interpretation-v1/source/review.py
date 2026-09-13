"""Synthetic definition audit, not a DenseOn measurement or inference run.

Find an integer-vector counterexample, then check it independently with Decimal
cosines and the unchanged literal-deletion production helper at 768 dimensions.
Only stdout is written. No model, dataset, optimizer, or GPU is used.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

ROOT = Path('/root/embedding-optimizer-story-refactor')
SOURCES = {
    'src/embed_optim/dimension_interventions.py':
        '3728c1cb6155ea3200173c5816098581a787a861ec3598b8a6b49640d37deee4',
    'src/embed_optim/dimension_utilization.py':
        'b07d77c32fb061a6c8e4d5384a5943846146de79a06093fe8973170897f6a9cb',
}
for name, expected in SOURCES.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected

from embed_optim.dimension_interventions import leave_one_out_metrics
from embed_optim.dimension_utilization import _attribution_summary, _cosine_scores, _query_metrics


def cosine(q, docs):
    return (docs @ q) / (np.linalg.norm(docs, axis=1) * np.linalg.norm(q))


def summaries(a):
    h, b = np.maximum(-a, 0), np.maximum(a, 0)
    H, B = float(h.sum()), float(b.sum())
    return {
        'helpful_mass': H,
        'degrading_mass': B,
        'helpful_share': H / (H + B) if H + B else 0.0,
        'normalized_helpful_participation':
            H * H / (a.size * float(h @ h)) if h @ h else 0.0,
    }


def small_case(q, docs):
    full = cosine(q, docs)
    star = int(np.argmax(full[1:])) + 1
    margin = full[0] - full[star]
    actual, fixed, penalties = [], [], []
    for j in range(len(q)):
        keep = np.arange(len(q)) != j
        scores = cosine(q[keep], docs[:, keep])
        actual.append(scores[0] - scores[1:].max() - margin)
        fixed.append(scores[0] - scores[star] - margin)
        penalties.append(scores[1:].max() - scores[star])
    a, f, r = map(np.asarray, (actual, fixed, penalties))
    assert np.all(r >= 0) and np.allclose(a, f - r, atol=1e-14, rtol=0)
    return {'full': full, 'a': a, 'fixed': f, 'penalties': r, 'summary': summaries(a)}


def find_witness():
    rng = np.random.default_rng(20260910)
    q = np.ones(4, dtype=np.float64)
    for trial in range(1, 5001):
        p, n, v = rng.integers(-3, 4, size=(3, 4))
        if any(np.count_nonzero(x) < 2 for x in (p, n, v)):
            continue
        base = cosine(q, np.stack([p, n, v]))
        if not (base[0] > base[1] + 0.05 and base[1] > base[2] + 0.05):
            continue
        cases = []
        for perm in sorted(set(itertools.permutations(v.tolist()))):
            docs = np.stack([p, n, perm, -q, -q, -q, -q, -q]).astype(np.float64)
            case = small_case(q, docs)
            cases.append((docs, case))
        for docs_a, a in cases:
            if a['penalties'].max() > 1e-12:
                continue
            for docs_b, b in cases:
                sa, sb = a['summary'], b['summary']
                if (b['penalties'].max() > 0.001
                    and sb['helpful_share'] > sa['helpful_share'] + 0.001
                    and sb['normalized_helpful_participation'] > sa['normalized_helpful_participation'] + 0.001
                    and sb['degrading_mass'] < sa['degrading_mass'] - 0.001):
                    return trial, q, docs_a, docs_b, a, b
    raise RuntimeError('No witness in the declared finite search; do not claim one.')


def decimal_cosine(q, d, remove=None):
    # Exact integer products and high-precision roots, independent of NumPy.
    indices = [i for i in range(len(q)) if i != remove]
    dot = sum(int(q[i]) * int(d[i]) for i in indices)
    q2 = sum(int(q[i]) ** 2 for i in indices)
    d2 = sum(int(d[i]) ** 2 for i in indices)
    assert q2 > 0 and d2 > 0
    return Decimal(dot) / Decimal(q2 * d2).sqrt()


def independent_reference(q, docs):
    with localcontext() as ctx:
        ctx.prec = 80
        full = [decimal_cosine(q, d) for d in docs]
        star = max(range(1, 8), key=lambda k: full[k])
        baseline = full[0] - full[star]
        changes, fixed, penalties = [], [], []
        for j in range(4):
            scores = [decimal_cosine(q, d, j) for d in docs]
            m = scores[0] - max(scores[1:])
            f = scores[0] - scores[star]
            r = max(scores[1:]) - scores[star]
            assert r >= 0 and abs((m - baseline) - ((f - baseline) - r)) < Decimal('1e-75')
            changes.append(m - baseline)
            fixed.append(f - baseline)
            penalties.append(r)
        return {
            'full_scores': [str(x) for x in full],
            'baseline_margin': str(baseline),
            'coordinate_margin_changes': [str(x) for x in changes],
            'fixed_original_negative_changes': [str(x) for x in fixed],
            'switch_penalties': [str(x) for x in penalties],
        }


def actual_768(q4, docs4, reference):
    q = np.zeros((1, 768), dtype=np.float64)
    docs = np.zeros((1, 8, 768), dtype=np.float64)
    q[0, :4], docs[0, :, :4] = q4, docs4
    full = _cosine_scores(q, docs)
    baseline = _query_metrics(full)
    ndcg, margin = leave_one_out_metrics(q, docs)
    a = (margin - baseline[1][:, None])[0]
    expected = np.zeros(768, dtype=np.float64)
    expected[:4] = [float(x) for x in reference['coordinate_margin_changes']]
    np.testing.assert_allclose(a, expected, atol=1e-12, rtol=0)
    actual = _attribution_summary(a, 1e-12)
    independent = summaries(expected)
    for name, key in [('helpful_share','helpful_mass_share'),
                      ('normalized_helpful_participation','helpful_attribution_participation_ratio'),
                      ('degrading_mass','degrading_attribution_mass')]:
        assert abs(independent[name] - actual[key]) < 1e-12
    H = actual['helpful_attribution_mass']
    B = actual['degrading_attribution_mass']
    assert H + B > 0
    assert abs(actual['helpful_mass_share'] - 0.5 * (1 - float(a.sum()) / (H + B))) < 1e-12
    return {
        'full_scores': full[0].tolist(),
        'full_positive_rank': int(baseline[2][0]),
        'full_shortlist_ndcg': float(baseline[0][0]),
        'full_margin': float(baseline[1][0]),
        'attribution': actual,
        'margin_changes_first_four': a[:4].tolist(),
        'maximum_decimal_reference_error': float(np.max(np.abs(a - expected))),
        'maximum_absolute_change_remaining_764_coordinates': float(np.max(np.abs(a[4:]))),
    }


trial, q, docs_a, docs_b, small_a, small_b = find_witness()
ref_a, ref_b = independent_reference(q, docs_a), independent_reference(q, docs_b)
assert ref_a['full_scores'] == ref_b['full_scores']
assert ref_a['fixed_original_negative_changes'] == ref_b['fixed_original_negative_changes']
assert all(Decimal(x) == 0 for x in ref_a['switch_penalties'])
assert any(Decimal(x) > 0 for x in ref_b['switch_penalties'])
a, b = actual_768(q, docs_a, ref_a), actual_768(q, docs_b, ref_b)
np.testing.assert_allclose(a['full_scores'], b['full_scores'], atol=1e-14, rtol=0)
assert a['full_positive_rank'] == b['full_positive_rank'] == 1
assert a['full_shortlist_ndcg'] == b['full_shortlist_ndcg'] == 1
for key in ('helpful_mass_share','helpful_attribution_participation_ratio'):
    assert b['attribution'][key] > a['attribution'][key] + 1e-10
assert b['attribution']['degrading_attribution_mass'] < a['attribution']['degrading_attribution_mass'] - 1e-10
for name, expected in SOURCES.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
result = {
    'scope':'synthetic_coordinate_sensitivity_interpretation_not_model_evidence',
    'observed_at_utc':datetime.now(timezone.utc).isoformat(),
    'search':{'seed':20260910,'maximum_trials':5000,'first_witness_trial':trial,
              'purpose':'Existence counterexample selected by the stated property, not a random-sample performance estimate.'},
    'integer_query_first_four':q.astype(int).tolist(),
    'integer_documents_a_first_four':docs_a.astype(int).tolist(),
    'integer_documents_b_first_four':docs_b.astype(int).tolist(),
    'remaining_coordinates':'All 764 additional coordinates are zero in both mathematical constructions.',
    'decimal_reference':{'precision':80,'a':ref_a,'b':ref_b},
    'actual_unchanged_literal_helper_768':{'a':a,'b':b},
    'equal_full_scores_exact_integer_radical_inputs':True,
    'fixed_original_negative_effects_identical':True,
    'three_raw_metric_directions_improve':True,
    'actual_source_hashes':SOURCES,
    'test_tolerance_absolute':1e-12,
    'no_zero_remaining_vector':True,
    'model_or_dataset_loaded':False,
    'gpu_used':False,
    'formal_task_bootstrap_performed':False,
    'scientific_claim_rule_passed':False,
    'scientific_completion':False,
    'interpretation':'Changing a lower-ranked competitor orientation can alter all three native-coordinate sensitivity summaries while preserving every full-vector shortlist score. This is not evidence that any real optimizer causes this pattern, that the measured sensitivities are invalid, or that full-corpus performance is unchanged after actual training.',
}
print(json.dumps(result,indent=2,allow_nan=False))
