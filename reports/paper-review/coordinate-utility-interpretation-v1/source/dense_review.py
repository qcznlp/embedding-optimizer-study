"""Fixed dense-vector witness; no model inference or primary statistical claim."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path('/root/embedding-optimizer-story-refactor')
SOURCES = {
    'src/embed_optim/dimension_interventions.py':'3728c1cb6155ea3200173c5816098581a787a861ec3598b8a6b49640d37deee4',
    'src/embed_optim/dimension_utilization.py':'b07d77c32fb061a6c8e4d5384a5943846146de79a06093fe8973170897f6a9cb',
    'configs/dense_dimension_utilization_protocol.json':'de56772ef84745c074940527d621a932a06a20e865a10bc3faad823bcb0e651e',
}
for name, wanted in SOURCES.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == wanted

from embed_optim.dimension_interventions import leave_one_out_metrics
from embed_optim.dimension_utilization import _attribution_summary, _cosine_scores, _query_metrics, _shared_masks

FIRST_FOUR = np.asarray([[2,-1,0,2],[1,-3,-1,3],[0,-3,-2,3],
                        [-1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,-1,-1],
                        [-1,-1,-1,-1],[-1,-1,-1,-1]],dtype=np.int64)
query = np.ones(768,dtype=np.int64)
query[:4] = 1000
docs_a = np.ones((8,768),dtype=np.int64)
docs_a[:,:4] = 1000 * FIRST_FOUR
docs_b = docs_a.copy()
docs_b[2,:4] = 1000 * np.asarray([-3,0,3,-2])
assert np.count_nonzero(query) == 768
assert min(np.count_nonzero(d) for d in docs_a) >= 767
assert min(np.count_nonzero(d) for d in docs_b) >= 767


def rational_rank(rows):
    matrix = [[Fraction(int(x)) for x in row] for row in rows]
    rank = 0
    for column in range(len(matrix[0])):
        pivot = next((r for r in range(rank,len(matrix)) if matrix[r][column]),None)
        if pivot is None:
            continue
        matrix[rank],matrix[pivot] = matrix[pivot],matrix[rank]
        scale=matrix[rank][column]
        matrix[rank] = [x/scale for x in matrix[rank]]
        for r in range(len(matrix)):
            if r != rank:
                scale=matrix[r][column]
                matrix[r]=[a-scale*b for a,b in zip(matrix[r],matrix[rank],strict=True)]
        rank += 1
    return rank


def exact_cosine(d, remove=None):
    # Integer dot products and squared norms are exact; no binary64 cancellation.
    dot=sum(int(a)*int(b) for a,b in zip(query,d,strict=True))
    q2=sum(int(x)**2 for x in query)
    d2=sum(int(x)**2 for x in d)
    if remove is not None:
        dot -= int(query[remove])*int(d[remove])
        q2 -= int(query[remove])**2
        d2 -= int(d[remove])**2
    assert q2 > 0 and d2 > 0
    return Decimal(dot)/Decimal(q2*d2).sqrt()


def verify(docs):
    with localcontext() as context:
        context.prec=80
        full=[exact_cosine(d) for d in docs]
        star=max(range(1,8),key=lambda k:full[k])
        assert star==1
        baseline=full[0]-full[star]
        # Coordinates 4..767 are exactly equal in every vector; check one
        # representative analytically and every coordinate with literal deletion.
        effects,fixed,penalties=[],[],[]
        for j in range(5):
            scores=[exact_cosine(d,j) for d in docs]
            penalty=max(scores[1:])-scores[star]
            value=scores[0]-max(scores[1:])-baseline
            f=scores[0]-scores[star]-baseline
            assert penalty>=0 and abs(value-(f-penalty))<Decimal('1e-75')
            effects.append(value);fixed.append(f);penalties.append(penalty)
        values=effects[:4]+[effects[4]]*764
        h=[max(-x,Decimal(0)) for x in values]
        b=[max(x,Decimal(0)) for x in values]
        H,B=sum(h),sum(b)
        expected_summary={
            'helpful_mass_share':H/(H+B),
            'helpful_attribution_participation_ratio':H*H/(Decimal(768)*sum(x*x for x in h)),
            'degrading_attribution_mass':B,
        }
        # Propagate the same 1e-12 coordinate bound through the nonlinear
        # summaries. An unpropagated 1e-12 bound on a 768-term sum is not implied.
        delta=Decimal('1e-12')
        hlo=[max(-x-delta,Decimal(0)) for x in values]
        hhi=[max(-x+delta,Decimal(0)) for x in values]
        blo=[max(x-delta,Decimal(0)) for x in values]
        bhi=[max(x+delta,Decimal(0)) for x in values]
        HL,HU,BL,BU=sum(hlo),sum(hhi),sum(blo),sum(bhi)
        bounds={
            'helpful_mass_share':(HL/(HL+BU),HU/(HU+BL)),
            'helpful_attribution_participation_ratio':
                (HL*HL/(Decimal(768)*sum(x*x for x in hhi)),
                 HU*HU/(Decimal(768)*sum(x*x for x in hlo))),
            'degrading_attribution_mass':(BL,BU),
        }
        assert abs(H/(H+B)-(1-sum(values)/(H+B))/2)<Decimal('1e-75')
        decimal_record={
            'precision':80,'full_scores':[str(x) for x in full],
            'full_margin':str(baseline),'first_four_changes':[str(x) for x in effects[:4]],
            'each_remaining_coordinate_change':str(effects[4]),
            'first_four_fixed_negative_changes':[str(x) for x in fixed[:4]],
            'each_remaining_fixed_negative_change':str(fixed[4]),
            'first_four_switch_penalties':[str(x) for x in penalties[:4]],
            'each_remaining_switch_penalty':str(penalties[4]),
            'three_metric_values':{k:str(v) for k,v in expected_summary.items()},
            'metric_bounds_from_coordinate_error':{k:[str(x) for x in v] for k,v in bounds.items()},
        }
    qf,df=query[None,:].astype(np.float64),docs[None,:,:].astype(np.float64)
    scores=_cosine_scores(qf,df)
    metrics=_query_metrics(scores)
    _,margin=leave_one_out_metrics(qf,df)
    a=(margin-metrics[1][:,None])[0]
    ref=np.asarray([float(x) for x in values])
    np.testing.assert_allclose(a,ref,atol=1e-12,rtol=0)
    summary=_attribution_summary(a,1e-12)
    aggregate_checks={}
    with localcontext() as context:
        context.prec=80
        assert all(abs(Decimal.from_float(float(got))-wanted)<Decimal('1e-12')
                   for got,wanted in zip(a,values,strict=True))
    for name,expected in expected_summary.items():
        error=abs(summary[name]-float(expected))
        aggregate_checks[name]={'absolute_error':error,
                                'original_unpropagated_1e_12_check_passed':error<1e-12}
        assert bounds[name][0] <= Decimal.from_float(summary[name]) <= bounds[name][1]
    masks=_shared_masks(768,json.loads((ROOT/'configs/dense_dimension_utilization_protocol.json').read_text()))
    assert len(masks)==80
    for mask in masks:
        removed=_cosine_scores(qf,df,mask['keep'])
        assert np.isfinite(removed).all()
    # All trailing columns are identical. One represents their exact column span.
    basis_rows=np.concatenate([query[None,:5],docs[:,:5]],axis=0)
    rank=rational_rank(basis_rows)
    return {
        'decimal':decimal_record,'actual_full_scores':scores[0].tolist(),
        'actual_full_margin':float(metrics[1][0]),'actual_full_rank':int(metrics[2][0]),
        'actual_full_ndcg':float(metrics[0][0]),'actual_attribution':summary,
        'maximum_decimal_error_all_768':float(np.max(np.abs(a-ref))),
        'original_unpropagated_aggregate_checks':aggregate_checks,
        'coordinate_absolute_bound_unchanged':'1e-12',
        'inside_propagated_deterministic_bounds':True,
        'original_shared_random_masks_checked':len(masks),
        'minimum_nonzero_document_coordinates':min(int(np.count_nonzero(d)) for d in docs),
        'query_and_document_span_rank_exact_rational':rank,
    }


a,b=verify(docs_a),verify(docs_b)
assert a['decimal']['full_scores']==b['decimal']['full_scores']
assert a['actual_full_scores']==b['actual_full_scores']
assert a['decimal']['first_four_fixed_negative_changes']==b['decimal']['first_four_fixed_negative_changes']
assert a['decimal']['each_remaining_fixed_negative_change']==b['decimal']['each_remaining_fixed_negative_change']
assert all(Decimal(x)==0 for x in a['decimal']['first_four_switch_penalties'])
assert Decimal(a['decimal']['each_remaining_switch_penalty'])==0
assert any(Decimal(x)>0 for x in b['decimal']['first_four_switch_penalties'])
assert Decimal(b['decimal']['each_remaining_switch_penalty'])==0
assert a['query_and_document_span_rank_exact_rational']==b['query_and_document_span_rank_exact_rational']==5
for name in ('helpful_mass_share','helpful_attribution_participation_ratio'):
    assert b['actual_attribution'][name]>a['actual_attribution'][name]+1e-8
assert b['actual_attribution']['degrading_attribution_mass']<a['actual_attribution']['degrading_attribution_mass']-1e-8
for name in ('helpful_mass_share','helpful_attribution_participation_ratio'):
    assert Decimal(b['decimal']['metric_bounds_from_coordinate_error'][name][0]) > Decimal(a['decimal']['metric_bounds_from_coordinate_error'][name][1])
assert Decimal(b['decimal']['metric_bounds_from_coordinate_error']['degrading_attribution_mass'][1]) < Decimal(a['decimal']['metric_bounds_from_coordinate_error']['degrading_attribution_mass'][0])
for name,wanted in SOURCES.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==wanted
print(json.dumps({
    'scope':'dense_synthetic_coordinate_sensitivity_counterexample_not_model_evidence',
    'observed_at_utc':datetime.now(timezone.utc).isoformat(),
    'construction':{'query_first_four':[1000]*4,'document_a_first_four':docs_a[:,:4].tolist(),
                    'document_b_first_four':docs_b[:,:4].tolist(),'all_764_remaining_entries_of_every_vector':1},
    'a':a,'b':b,'full_scores_exactly_equal':True,'three_raw_metric_directions_improve':True,
    'span_rank_equal':5,'source_hashes':SOURCES,'model_or_dataset_loaded':False,'gpu_used':False,
    'metric_direction_bounds_disjoint':True,'bounds_are_statistical_intervals':False,
    'initial_unpropagated_aggregate_check_failure_retained':True,
    'full_primary_analysis_or_bootstrap_performed':False,'scientific_claim_rule_passed':False,
    'scientific_completion':False,
    'boundary':'A fixed constructed shortlist, not 224 real probe queries, trained encoders, full-corpus BEIR, a population frequency estimate, or evidence against the predeclared conditional-sensitivity interpretation.'
},indent=2,allow_nan=False))
