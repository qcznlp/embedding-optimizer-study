"""Independent design reconstruction and full rational OLS for all diagnostics."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from datetime import datetime, timezone
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
import sympy as sp

PARENT_SHA='243b001980feca61e6cca4852f544e9922fa7ca90b2a157981836d47b16d8621'
STEPS=[782,1563,2345,3126,3907]
OPTIMIZERS=['adamw','muon','normuon']
NORM='cumulative_displacement_to_weight_ratio'


def need(v,msg):
    if not v:raise ValueError(msg)


def bound(path,expected):
    need(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)),'Missing/symlinked evidence')
    actual=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    need(actual['sha256']==expected if isinstance(expected,str) else actual==expected,'Bound evidence differs')
    return json.loads(path.read_text()) if path.suffix=='.json' else path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','analysis','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--receipt-sha256',required=True)
    a=p.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU only')
    need(not a.output.exists(),'Preserve prior independent result')
    parent=bound(a.parent/'verification.json',PARENT_SHA)
    prior=bound(a.parent/'actual/tables.json',parent['payloads']['actual/tables.json'])
    receipt=bound(a.analysis/'readout.json',a.receipt_sha256)
    need(receipt['parent_verification_sha256']==PARENT_SHA and receipt['exploratory_post_result'] is True,'Wrong declared analysis')
    for name,b in receipt['outputs'].items():bound(a.analysis/name,b)
    result=json.loads((a.analysis/'result.json').read_text())
    source='source/verify_predictions.py'
    path=bound(a.parent/source,parent['payloads'][source])
    spec=importlib.util.spec_from_file_location('_independent_original_full_matrix_solver',path)
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    rows=prior['exact']['bridge_rows'];y=[oracle.q(r['mean_ndcg_at_10']) for r in rows]
    features=[r['feature'] for family in ['original','exact'] for r in prior[family]['feature_prediction_summary']]
    need(len(rows)==60 and len(features)==14,'Wrong complete panel')
    matrices={k:[] for k in result['design_values']}
    covariates=[]
    for row in rows:
        o,s=row['optimizer'],row['stage'];x=row['centered_log10_learning_rate']
        unit={'adamw':1e-5,'muon':1e-3,'normuon':1e-3}[o]
        raw=row['learning_rate']/unit
        # Independently sum every nominal schedule coefficient, not the closed form.
        j=STEPS[s-1]
        mass=sum([Q(i,391) if i<391 else Q(3907-i,3516) for i in range(j)],Q())/Q(3907,2)
        budget=raw*float(mass)
        b0=[1.,float(o=='muon'),float(o=='normuon')]+[float(s==t) for t in range(2,6)]+[x]
        rate_block=[raw*int(o==t) for t in OPTIMIZERS]
        optimizer_stage=[float(o==t and s==v) for t in OPTIMIZERS for v in range(1,6)]
        b2=optimizer_stage+[x*int(o==t) for t in OPTIMIZERS]+rate_block
        data={'B0_original':b0,'B1_raw_rate':b0+rate_block,'B2_optimizer_stage_rate':b2,
              'B3_nominal_schedule':b2+[budget*int(o==t) for t in OPTIMIZERS]}
        for k,v in data.items():matrices[k].append(v)
        covariates.append(dict(run_id=row['run_id'],stage=s,step=j,raw_rate_unit=unit,scaled_raw_rate=raw,
             nominal_schedule_mass=float(mass),scaled_nominal_budget=budget))
    need(matrices==result['design_values'] and covariates==result['covariates'],'Independently reconstructed recipe design differs')
    splits={f:([i for i,r in enumerate(rows) if r['dose_index']!=f],[i for i,r in enumerate(rows) if r['dose_index']==f]) for f in range(1,5)}
    folds={(r['baseline'],r['conditioned_on_displacement'],r['feature'],r['fold']):r for r in result['folds']}
    predictions={(r['baseline'],r['conditioned_on_displacement'],r['feature'],r['run_id'],r['stage']):r for r in result['predictions']}
    need(len(folds)==336 and len(predictions)==5040,'Wrong or duplicate stored population')
    base_context={};count=0;fold_count=0;summary_count=0;equivalent=0
    for r in result['summaries']:
        key=r['baseline'],r['conditioned_on_displacement']
        if key not in base_context:
            design=[[oracle.q(v) for v in b]+([oracle.q(row[NORM])] if key[1] else []) for b,row in zip(matrices[key[0]],rows,strict=True)]
            base_context[key]=(design,{f:oracle.solve(design,y,tr) for f,(tr,te) in splits.items()})
        design,base=base_context[key];feature=r['feature'];values=[oracle.q(row[feature]) for row in rows]
        aug=[b+[v] for b,v in zip(design,values,strict=True)]
        pooled_b=[None]*60;pooled_a=[None]*60;improved=0
        for fold,(tr,te) in splits.items():
            recorded=folds[*key,feature,fold]
            if key[1] and feature==NORM:
                need(recorded['status']=='baseline_equivalent','Self-addition is not an equivalent control')
                need(all(row[-1]==row[-2] for row in aug),'Redundant columns differ')
                ap=base[fold];equivalent+=1
            else:
                need(recorded['status']=='resolved','Actual solver has an unresolved fit requiring explicit audit')
                ap=oracle.solve(aug,y,tr)
                # Independent resolution check with the same stated affine convention.
                train_values=[values[i] for i in tr]
                mean=sum(train_values)/len(tr);centered=[v-mean for v in train_values]
                amplitude=max(abs(v) for v in centered)
                z=np.asarray([float(v/amplitude) for v in centered]);z=z/np.sqrt(np.mean(z*z))
                health=np.column_stack([np.asarray([design[i] for i in tr],dtype=float),z])
                singular=np.linalg.svd(health,compute_uv=False)
                need(np.count_nonzero(singular>np.finfo(float).eps*max(health.shape)*singular[0])==health.shape[1],
                     'Actual augmented fit is numerically unresolved')
            bp=base[fold]
            bm=oracle.error([y[i] for i in te],[bp[i] for i in te]);am=oracle.error([y[i] for i in te],[ap[i] for i in te])
            need(bm==sp.Rational(recorded['baseline_mse_exact']) and am==sp.Rational(recorded['feature_mse_exact']),'Fold exact MSE differs')
            improves=bool(bm>am);need(recorded['improves'] is improves,'Fold decision differs');improved+=improves
            fold_count+=1
            for i in te:
                saved=predictions[*key,feature,rows[i]['run_id'],rows[i]['stage']]
                need(saved['fold']==fold and saved['observed']==rows[i]['mean_ndcg_at_10'],'Prediction row identity differs')
                need(sp.Rational(saved['baseline_prediction_exact'])==bp[i] and sp.Rational(saved['feature_prediction_exact'])==ap[i],
                     'Independent full-matrix prediction differs')
                pooled_b[i]=bp[i];pooled_a[i]=ap[i];count+=1
        bm,am=oracle.error(y,pooled_b),oracle.error(y,pooled_a)
        need(bm==sp.Rational(r['pooled_baseline_mse_exact']) and am==sp.Rational(r['pooled_feature_mse_exact']),'Pooled exact MSE differs')
        need(r['defined_folds']==4 and r['improved_folds']==improved and r['diagnostic_flag'] is bool(bm>am and improved>=3),'Pooled diagnostic differs')
        for field,value in [('pooled_baseline_rmse',sp.sqrt(bm)),('pooled_feature_rmse',sp.sqrt(am)),('pooled_rmse_reduction',sp.sqrt(bm)-sp.sqrt(am))]:
            need(r[field]==float(sp.N(value,80)),'Display arithmetic differs')
        summary_count+=1
        if summary_count%14==0:print(json.dumps({'verified_comparisons':summary_count,'verified_predictions':count,'baseline':key}),flush=True)
    need((count,fold_count,summary_count,equivalent)==(5040,336,84,8),'Incomplete independent scope')
    result=dict(scope='independent-all-designs-and-full-rational-solve-of-post-result-sensitivity',
         completed_at_utc=datetime.now(timezone.utc).isoformat(),readout_sha256=a.receipt_sha256,
         independently_reconstructed_design_rows=240,independently_reconstructed_nominal_covariates=60,
         exact_prediction_rows=5040,exact_fold_mse_comparisons=336,pooled_comparisons=84,redundant_self_addition_folds=8,
         exploratory_post_result=True,independent_experiment=False,formal_primary_admission=False,scientific_completion=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:f.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
