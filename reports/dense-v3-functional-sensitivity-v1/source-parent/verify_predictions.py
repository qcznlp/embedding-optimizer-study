"""Independent full augmented-matrix rational OLS on the actual saved panel.

No producer module is imported. This verifies predictions/statistics of the same
measurement, not a second training run or independent scientific replication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    need(not any(p.is_symlink() for p in (path, *path.parents)), "Symlinked evidence")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def q(value):
    f = Fraction.from_float(float(value))
    return sp.Rational(f.numerator, f.denominator)


def solve(design, outcomes, indices):
    matrix = sp.Matrix([design[i] for i in indices])
    y = sp.Matrix([outcomes[i] for i in indices])
    gram = matrix.T * matrix
    need(gram.det(method="domain-ge") != 0, "Independent full system is singular")
    beta = gram.inv(method="DM") * matrix.T * y
    need(matrix.T * (y - matrix * beta) == sp.zeros(matrix.cols, 1), "Independent normal equations differ")
    return list(sp.Matrix(design) * beta)


def error(y, prediction):
    return sum(((a-b)**2 for a,b in zip(y,prediction,strict=True)), sp.Rational(0)) / len(y)


def correlation(a,b):
    am,bm=sum(a)/len(a),sum(b)/len(b)
    ac,bc=[v-am for v in a],[v-bm for v in b]
    numerator=sum(x*y for x,y in zip(ac,bc,strict=True))
    denominator=sp.sqrt(sum(x*x for x in ac)*sum(y*y for y in bc))
    return float(sp.N(numerator/denominator,80))


def ranks(values):
    ordered=sorted(enumerate(values),key=lambda p:p[1])
    result=[None]*len(values)
    position=0
    while position<len(values):
        stop=position+1
        while stop<len(values) and ordered[stop][1]==ordered[position][1]:
            stop+=1
        average=sp.Rational(position+1+stop,2)
        for i,_ in ordered[position:stop]:
            result[i]=average
        position=stop
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--readout',type=Path,required=True)
    p.add_argument('--receipt-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU only')
    need(not a.output.exists(),'Refuse existing independent output')
    receipt_path=a.readout/'readout.json'
    need(sha(receipt_path)==a.receipt_sha256,'Readout receipt differs')
    receipt=json.loads(receipt_path.read_text())
    need(receipt['formal_consumer_called'] is False and receipt['scientific_completion'] is False,
         'Unexpected producer scope')
    for name,b in receipt['outputs'].items():
        path=a.readout/name
        need(path.stat().st_size==b['bytes'] and sha(path)==b['sha256'],'Stored output changed')
    tables=json.loads((a.readout/'tables.json').read_text())
    original=tables['original']['bridge_rows']
    exact=tables['exact']['bridge_rows']
    need(len(original)==len(exact)==60,'Wrong complete population')
    need(all(all(r[k]==e[k] for k in r) for r,e in zip(original,exact,strict=True)),
         'Exact family overwrites original panel')
    baseline=[[q(1),q(r['optimizer']=='muon'),q(r['optimizer']=='normuon'),
               *[q(r['stage']==s) for s in range(2,6)],q(r['centered_log10_learning_rate'])] for r in original]
    y=[q(r['mean_ndcg_at_10']) for r in original]
    splits={f:([i for i,r in enumerate(original) if r['dose_index']!=f],
               [i for i,r in enumerate(original) if r['dose_index']==f]) for f in range(1,5)}
    need(all(len(tr)==45 and len(te)==15 for tr,te in splits.values()),'Wrong held-dose folds')
    base_predictions={f:solve(baseline,y,tr) for f,(tr,te) in splits.items()}
    full_base=solve(baseline,y,range(60))
    residual_y=[v-p for v,p in zip(y,full_base,strict=True)]
    prediction_checks, fold_checks, summary_checks, association_checks=0,0,0,0
    summaries=[]
    for family in ['original','exact']:
        t=tables[family]
        rows=t['bridge_rows']
        features=[r['feature'] for r in t['feature_prediction_summary']]
        need(len(features)==(9 if family=='original' else 5),'Wrong named feature population')
        fold_index={(r['feature'],r['held_out_dose_index']):r for r in t['leave_dose_fold_metrics']}
        pred_index={(r['feature'],r['run_id'],r['stage']):r for r in t['held_out_predictions']}
        for summary in t['feature_prediction_summary']:
            feature=summary['feature']
            values=[q(r[feature]) for r in rows]
            design=[b+[x] for b,x in zip(baseline,values,strict=True)]
            combined_base,combined_feature=[None]*60,[None]*60
            improved=0
            for fold,(train,test) in splits.items():
                actual=fold_index[feature,fold]
                need(actual['status']=='resolved','Independent actual-population oracle requires resolved fits')
                predictions=solve(design,y,train)
                base=base_predictions[fold]
                be=error([y[i] for i in test],[base[i] for i in test])
                fe=error([y[i] for i in test],[predictions[i] for i in test])
                need(be==sp.Rational(actual['baseline_mse_exact']) and fe==sp.Rational(actual['feature_mse_exact']),
                     'Fold rational MSE differs')
                improves=bool(be>fe)
                need(actual['feature_improves'] is improves,'Fold comparison differs')
                improved+=improves
                fold_checks+=1
                for i in test:
                    saved=pred_index[feature,rows[i]['run_id'],rows[i]['stage']]
                    need(saved['held_out_dose_index']==fold and saved['observed']==rows[i]['mean_ndcg_at_10'],'Prediction identity differs')
                    need(sp.Rational(saved['baseline_prediction_exact'])==base[i]
                         and sp.Rational(saved['feature_prediction_exact'])==predictions[i],'Exact held-out prediction differs')
                    need(saved['baseline_prediction']==float(base[i]) and saved['feature_prediction']==float(predictions[i]),
                         'Prediction display differs')
                    combined_base[i],combined_feature[i]=base[i],predictions[i]
                    prediction_checks+=1
            be,fe=error(y,combined_base),error(y,combined_feature)
            need(be==sp.Rational(summary['pooled_baseline_mse_exact']) and fe==sp.Rational(summary['pooled_feature_mse_exact']),
                 'Pooled rational MSE differs')
            support=bool(be>fe and improved>=3)
            need(summary['folds_improved']==improved and summary['folds_defined']==4
                 and summary['predictively_useful'] is support,'Predictive decision differs')
            for name,expected in [('pooled_baseline_rmse',float(sp.N(sp.sqrt(be),80))),
                                  ('pooled_feature_rmse',float(sp.N(sp.sqrt(fe),80))),
                                  ('pooled_rmse_reduction',float(sp.N(sp.sqrt(be)-sp.sqrt(fe),80)))]:
                need(summary[name]==expected, 'RMSE display differs')
            summary_checks+=1
            fitted_x=solve(baseline,values,range(60))
            residual_x=[v-p for v,p in zip(values,fitted_x,strict=True)]
            assoc=next(r for r in t['residual_associations'] if r['feature']==feature)
            need(assoc['status']=='resolved','Unresolved association')
            need(sp.Rational(assoc['feature_residual_energy_exact'])==sum(x*x for x in residual_x)
                 and sp.Rational(assoc['outcome_residual_energy_exact'])==sum(y*y for y in residual_y),
                 'Full residual energy differs')
            need(assoc['pearson_residual_association']==correlation(residual_x,residual_y)
                 and assoc['spearman_residual_association']==correlation(ranks(residual_x),ranks(residual_y)),
                 'Independent residual association differs')
            association_checks+=1
            summaries.append(dict(family=family,feature=feature,improved_folds=improved,predictive_support=support))
            print(json.dumps(summaries[-1]),flush=True)
    need((prediction_checks,fold_checks,summary_checks,association_checks)==(840,56,14,14),'Independent coverage incomplete')
    result=dict(scope='independent-full-augmented-rational-OLS-on-actual-weight-retrieval-panel',
         completed_at_utc=datetime.now(timezone.utc).isoformat(),producer_readout_sha256=a.receipt_sha256,
         exact_held_out_prediction_rows=prediction_checks,exact_fold_mse_comparisons=fold_checks,
         pooled_summaries_and_decisions=summary_checks,residual_associations=association_checks,
         solver='SymPy rational full augmented normal equations; not producer FWL implementation',
         sympy_version=sp.__version__,same_data_not_independent_experiment=True,
         formal_primary_admission=False,scientific_completion=False,feature_summaries=summaries)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='feature_summaries'}))


if __name__=='__main__':
    main()
