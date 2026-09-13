"""Explicitly post-result recipe-baseline and conditional predictor diagnostics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from datetime import datetime, timezone
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

PARENT_SHA='243b001980feca61e6cca4852f544e9922fa7ca90b2a157981836d47b16d8621'
READOUT_SHA='6891e82a46df11a6d142e79fdf495afa329f40912569337f4b1d6da33d5e8046'
ARITHMETIC_SHA='0d66ce6e99aa42b8d93c8295cb0f5f3015ea7b7907a9f10fb095d95be2d646b4'
OPTIMIZERS=('adamw','muon','normuon')
STEPS=(782,1563,2345,3126,3907)
DISPLACEMENT='cumulative_displacement_to_weight_ratio'
BASELINES=('B0_original','B1_raw_rate','B2_optimizer_stage_rate','B3_nominal_schedule')


def need(condition,message):
    if not condition:raise ValueError(message)


def identity(path):
    need(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)),'Missing or symlinked input')
    before=path.stat()
    with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    after=path.stat()
    need(all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'Input changed during read')
    return dict(bytes=after.st_size,sha256=sha)


def bound(path,expected):
    actual=identity(path)
    need(actual['sha256']==expected if isinstance(expected,str) else actual==expected,'Input identity differs: '+path.name)
    return actual


def dump(path,value):
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def table(path,rows):
    need(bool(rows),'Empty table')
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def load_parent(root):
    bindings={}
    def read(name,expected):
        path=root/name;bindings[str(path)]=bound(path,expected)
        return json.loads(path.read_text())
    verification=read('verification.json',PARENT_SHA)
    receipt=read('actual/readout.json',READOUT_SHA)
    need(receipt['formal_consumer_called'] is False and receipt['scientific_completion'] is False,'Parent scope differs')
    tables=read('actual/tables.json',receipt['outputs']['tables.json'])
    recipes=read('actual/recipes.json',receipt['outputs']['recipes.json'])
    joined=read('actual/joined-checkpoints.json',receipt['outputs']['joined-checkpoints.json'])
    original=tables['original']['bridge_rows'];rows=tables['exact']['bridge_rows']
    need(len(original)==len(rows)==len(joined)==60 and len(recipes)==12,'Incomplete parent')
    need(all(all(r[k]==e[k] for k in r) for r,e in zip(original,rows,strict=True)),'Original feature/outcome overwritten')
    need({r['run_id'] for r in rows}==set(recipes),'Recipe grid differs')
    need({(r['run_id'],r['stage']) for r in rows}=={(run,s) for run in recipes for s in range(1,6)},'Incomplete stage grid')
    need({(r['run_id'],r['stage'],r['step']) for r in joined}=={(run,s,STEPS[s-1]) for run in recipes for s in range(1,6)},'Wrong retained steps')
    need(all(r['warmup_ratio']==.1 and math.ceil(STEPS[-1]*r['warmup_ratio'])==391 for r in recipes.values()),'Different recipe warmup')
    features=[r['feature'] for r in tables['original']['feature_prediction_summary']]+[r['feature'] for r in tables['exact']['feature_prediction_summary']]
    need(len(features)==len(set(features))==14,'Wrong feature inventory')
    for row in rows:
        recipe=recipes[row['run_id']]
        need(row['learning_rate']==recipe['optimizer']['lr'] and row['optimizer']==recipe['optimizer']['name'],'Recipe row differs')
        need(all(type(row[f]) in (int,float) and math.isfinite(row[f]) for f in features),'Undefined/nonfinite parent measurement')
    source='source-original/src/embed_optim/bridge_exact_arithmetic.py'
    path=root/source
    need(verification['payloads'][source]['sha256']==ARITHMETIC_SHA,'Arithmetic source binding differs')
    bindings[str(path)]=bound(path,verification['payloads'][source])
    spec=importlib.util.spec_from_file_location('_unchanged_bridge_arithmetic',path)
    arithmetic=importlib.util.module_from_spec(spec);spec.loader.exec_module(arithmetic)
    return dict(rows=rows,features=features,recipes=recipes,tables=tables,bindings=bindings,arithmetic=arithmetic)


def nominal_mass(step,total=3907,warmup=391):
    """Exact nominal sum over schedule indices 0..step-1, normalized by total/2."""
    need(type(step) is int and 0<=step<=total and 0<warmup<total,'Invalid nominal schedule')
    if step<=warmup:
        value=Q(step*(step-1),2*warmup)
    else:
        count=step-warmup
        value=Q(warmup-1,2)+Q(count*(2*total-warmup-step+1),2*(total-warmup))
    return value/Q(total,2)


def designs(rows):
    names={
        'B0_original':['intercept','muon','normuon',*[f'stage_{s}' for s in range(2,6)],'centered_log_rate'],
        'B2_optimizer_stage_rate':[f'{o}_stage_{s}' for o in OPTIMIZERS for s in range(1,6)]+[f'{o}_log_rate' for o in OPTIMIZERS]+[f'{o}_raw_rate' for o in OPTIMIZERS],
    }
    names['B1_raw_rate']=names['B0_original']+[f'{o}_raw_rate' for o in OPTIMIZERS]
    names['B3_nominal_schedule']=names['B2_optimizer_stage_rate']+[f'{o}_nominal_budget' for o in OPTIMIZERS]
    output={name:[] for name in BASELINES};covariates=[]
    for r in rows:
        o,s=r['optimizer'],r['stage'];x=r['centered_log10_learning_rate']
        unit=1e-5 if o=='adamw' else 1e-3
        rate=r['learning_rate']/unit
        mass=float(nominal_mass(STEPS[s-1]));budget=rate*mass
        base=[1.,float(o=='muon'),float(o=='normuon'),*[float(s==z) for z in range(2,6)],x]
        raw=[rate if o==z else 0. for z in OPTIMIZERS]
        structured=[float(o==z and s==j) for z in OPTIMIZERS for j in range(1,6)]
        structured += [x if o==z else 0. for z in OPTIMIZERS]+raw
        new={'B0_original':base,'B1_raw_rate':base+raw,'B2_optimizer_stage_rate':structured,
             'B3_nominal_schedule':structured+[budget if o==z else 0. for z in OPTIMIZERS]}
        for key,value in new.items():output[key].append(value)
        covariates.append(dict(run_id=r['run_id'],stage=s,step=STEPS[s-1],raw_rate_unit=unit,
             scaled_raw_rate=rate,nominal_schedule_mass=mass,scaled_nominal_budget=budget))
    need([len(names[k]) for k in BASELINES]==[8,11,21,24],'Baseline width differs')
    return output,names,covariates


class ExactBaseline:
    """New generic-width solver; the original eight-column class is not edited."""
    def __init__(self,design,train,k):
        self.k=k;self.design=design;self.train=tuple(train)
        self.numeric=np.asarray([design[i] for i in train],dtype=np.float64)
        singular=np.linalg.svd(self.numeric,compute_uv=False)
        self.width=self.numeric.shape[1]
        cutoff=float(np.finfo(np.float64).eps*max(self.numeric.shape)*singular[0])
        rank=int(np.count_nonzero(singular>cutoff))
        self.health=dict(columns=self.width,train_rows=len(train),rank=rank,cutoff=cutoff,singular_values=singular.tolist())
        need(rank==self.width,'Numerically rank-deficient baseline; no silent column removal')
        self.columns=list(map(list,zip(*(design[i] for i in train),strict=True)))
        self.inverse=k.inverse([[k.dot(a,b) for b in self.columns] for a in self.columns])

    def project(self,values):
        k=self.k
        rhs=[k.dot(c,[values[i] for i in self.train]) for c in self.columns]
        beta=[k.dot(r,rhs) for r in self.inverse]
        prediction=[k.dot(r,beta) for r in self.design]
        residual=[values[i]-prediction[i] for i in self.train]
        need(all(k.dot(c,residual)==0 for c in self.columns),'Exact normal equations failed')
        return prediction,residual

    def add(self,values,y):
        k=self.k;base,yr=self.project(y);fit,xr=self.project(values)
        energy=k.dot(xr,xr)
        extension=[v-p for v,p in zip(values,fit,strict=True)]
        health=k.design_health(self.numeric,[values[i] for i in self.train])
        if energy==0:
            status='baseline_equivalent' if all(v==0 for v in extension) else 'unidentified_extension'
            added=base if status=='baseline_equivalent' else None
        elif health['augmented_numeric_rank']!=self.width+1:
            status,added='unresolved_nonzero_direction',None
        else:
            coefficient=k.dot(xr,yr)/energy
            added=[p+coefficient*v for p,v in zip(base,extension,strict=True)]
            status='resolved'
        return dict(status=status,baseline=base,added=added,health=health,energy=str(energy))


def calculate(parent):
    rows,features,k=parent['rows'],parent['features'],parent['arithmetic']
    numeric,names,covariates=designs(rows)
    rational={name:[[k.rational(v) for v in row] for row in design] for name,design in numeric.items()}
    y=[k.rational(r['mean_ndcg_at_10']) for r in rows]
    values={f:[k.rational(r[f]) for r in rows] for f in features}
    splits={fold:([i for i,r in enumerate(rows) if r['dose_index']!=fold],
                  [i for i,r in enumerate(rows) if r['dose_index']==fold]) for fold in range(1,5)}
    need(all(len(tr)==45 and len(te)==15 for tr,te in splits.values()),'Different holdout population')
    summaries=[];folds=[];predictions=[];baselines=[];diagnostics=[]
    for name in BASELINES:
        for conditioned in ([False,True] if name in ('B0_original','B3_nominal_schedule') else [False]):
            design=[b+([v] if conditioned else []) for b,v in zip(rational[name],values[DISPLACEMENT],strict=True)]
            contexts={fold:ExactBaseline(design,tr,k) for fold,(tr,te) in splits.items()}
            base_all=[None]*60
            for fold,(tr,te) in splits.items():
                bp,_=contexts[fold].project(y)
                for i in te:base_all[i]=bp[i]
            baseline_mse=k.mse(y,base_all)
            baselines.append(dict(baseline=name,conditioned_on_displacement=conditioned,columns=len(design[0]),
                   pooled_rmse=k.root(baseline_mse),pooled_mse_exact=str(baseline_mse),rows=60))
            for feature in features:
                added_all=[None]*60;defined=0;improved=0
                for fold,(tr,te) in splits.items():
                    result=contexts[fold].add(values[feature],y)
                    base=[result['baseline'][i] for i in te]
                    added=None if result['added'] is None else [result['added'][i] for i in te]
                    bm=k.mse([y[i] for i in te],base)
                    am=None if added is None else k.mse([y[i] for i in te],added)
                    improves=None if am is None else bm>am
                    defined+=int(am is not None);improved+=int(improves is True)
                    common=dict(baseline=name,conditioned_on_displacement=conditioned,feature=feature,fold=fold)
                    folds.append(dict(**common,status=result['status'],baseline_mse_exact=str(bm),feature_mse_exact=None if am is None else str(am),
                         baseline_rmse=k.root(bm),feature_rmse=None if am is None else k.root(am),improves=improves,
                         training_residual_energy_exact=result['energy'],train_rows=45,test_rows=15))
                    diagnostics.append(dict(**common,baseline_health=contexts[fold].health,feature_health=result['health']))
                    for i in te:
                        ap=None if result['added'] is None else result['added'][i]
                        added_all[i]=ap
                        predictions.append(dict(**common,run_id=rows[i]['run_id'],stage=rows[i]['stage'],status=result['status'],
                              observed=rows[i]['mean_ndcg_at_10'],baseline_prediction_exact=str(result['baseline'][i]),
                              feature_prediction_exact=None if ap is None else str(ap)))
                am=k.mse(y,added_all) if defined==4 else None
                summary=dict(baseline=name,conditioned_on_displacement=conditioned,feature=feature,columns=len(design[0]),
                     pooled_baseline_mse_exact=str(baseline_mse),pooled_feature_mse_exact=None if am is None else str(am),
                     pooled_baseline_rmse=k.root(baseline_mse),pooled_feature_rmse=None if am is None else k.root(am),
                     pooled_rmse_reduction=None if am is None else k.rmse_reduction(baseline_mse,am),
                     improved_folds=improved,defined_folds=defined,total_folds=4,
                     diagnostic_flag=None if am is None else baseline_mse>am and improved>=3)
                summaries.append(summary)
            print(json.dumps({'baseline':name,'conditioned':conditioned,'completed_comparisons':len(summaries)}),flush=True)
    need((len(summaries),len(folds),len(predictions))==(84,336,5040),'Incomplete declared sensitivity')
    old={r['feature']:r for family in ('original','exact') for r in parent['tables'][family]['feature_prediction_summary']}
    old_predictions={(r['feature'],r['run_id'],r['stage']):r for family in ('original','exact') for r in parent['tables'][family]['held_out_predictions']}
    for r in summaries:
        if r['baseline']=='B0_original' and not r['conditioned_on_displacement']:
            prior=old[r['feature']]
            for key in ('pooled_baseline_mse_exact','pooled_feature_mse_exact','pooled_baseline_rmse','pooled_feature_rmse','pooled_rmse_reduction'):
                need(r[key]==prior[key],'B0 does not exactly reproduce original: '+key)
            need(r['diagnostic_flag'] is prior['predictively_useful'] and r['improved_folds']==prior['folds_improved'],'Original B0 decision differs')
    for r in predictions:
        if r['baseline']=='B0_original' and not r['conditioned_on_displacement']:
            prior=old_predictions[r['feature'],r['run_id'],r['stage']]
            need(r['baseline_prediction_exact']==prior['baseline_prediction_exact'] and r['feature_prediction_exact']==prior['feature_prediction_exact'],
                 'B0 exact held-out prediction differs')
    return dict(summaries=summaries,folds=folds,predictions=predictions,baseline_scores=baselines,
                covariates=covariates,design_columns=names,design_values=numeric,diagnostics=diagnostics)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU only')
    need(args.parent.is_absolute() and args.output.is_absolute() and not args.output.exists(),'Require absolute fresh output')
    need(not any(p.is_symlink() for p in (args.output,*args.output.parents)),'Symlinked output')
    parent=load_parent(args.parent)
    result=calculate(parent)
    for path,b in parent['bindings'].items():bound(Path(path),b)
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ('summaries','folds','predictions','baseline_scores','covariates'):table(args.output/(name+'.csv'),result[name])
    dump(args.output/'result.json',result)
    lines=['# Post-result baseline and conditional sensitivity','',
        'Exploratory only. All errors are nDCG@10 points; lower is better. No original result is overwritten.','']
    def fmt(value):return 'undefined' if value is None else f'{100*value:.6f}'
    for base in BASELINES:
        for conditional in ([False,True] if base in ('B0_original','B3_nominal_schedule') else [False]):
            lines += ['## '+base+(' + cumulative displacement comparator' if conditional else ''),'',
                 '| Added feature | Comparator RMSE | Augmented RMSE | Improved folds | Defined folds | Diagnostic flag |',
                 '| --- | ---: | ---: | ---: | ---: | --- |']
            for r in result['summaries']:
                if r['baseline']==base and r['conditioned_on_displacement']==conditional:
                    lines.append(f"| {r['feature']} | {fmt(r['pooled_baseline_rmse'])} | {fmt(r['pooled_feature_rmse'])} | {r['improved_folds']}/4 | {r['defined_folds']}/4 | {r['diagnostic_flag']} |")
            lines.append('')
    with (args.output/'summary.md').open('x') as stream:stream.write('\n'.join(lines))
    outputs={p.name:identity(p) for p in args.output.iterdir()}
    receipt=dict(scope='post-result-recipe-baseline-and-conditional-predictor-sensitivity',
         completed_at_utc=datetime.now(timezone.utc).isoformat(),parent_verification_sha256=PARENT_SHA,
         source=identity(Path(__file__)),input_bindings=parent['bindings'],outputs=outputs,
         comparisons=84,folds=336,heldout_prediction_rows=5040,original_prediction_rows_exactly_reproduced=840,
         exploratory_post_result=True,original_protocol_or_result_modified=False,model_or_retrieval_execution=False,
         functional_or_causal_evidence=False,formal_primary_admission=False,scientific_completion=False)
    dump(args.output/'readout.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('input_bindings','outputs')}))


if __name__=='__main__':main()
