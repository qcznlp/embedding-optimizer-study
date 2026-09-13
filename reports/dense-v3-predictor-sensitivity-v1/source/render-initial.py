"""Show all 84 post-result diagnostic comparisons without selecting a baseline."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS=[
    'Log segment displacement / weight','Segment stable rank','Segment entropy rank (top-64)',
    'Segment row-norm CV','Segment top-1% row energy','Cumulative displacement / weight',
    'Cumulative stable rank','Segment overlap with AdamW','Cumulative overlap with AdamW',
    'Exact segment stable rank','Segment entropy rank (full spectrum)',
    'Exact cumulative stable rank','Exact segment overlap with AdamW','Exact cumulative overlap with AdamW']
GROUPS=[('B0_original',False),('B1_raw_rate',False),('B2_optimizer_stage_rate',False),
        ('B3_nominal_schedule',False),('B0_original',True),('B3_nominal_schedule',True)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--receipt-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    receipt_path=a.analysis/'readout.json'
    assert not receipt_path.is_symlink() and hashlib.sha256(receipt_path.read_bytes()).hexdigest()==a.receipt_sha256
    receipt=json.loads(receipt_path.read_text())
    result_path=a.analysis/'result.json'
    assert hashlib.sha256(result_path.read_bytes()).hexdigest()==receipt['outputs']['result.json']['sha256']
    result=json.loads(result_path.read_text())
    features=[r['feature'] for r in result['summaries'][:14]]
    assert len(features)==len(set(features))==14
    indexed={(r['baseline'],r['conditioned_on_displacement'],r['feature']):r for r in result['summaries']}
    assert len(indexed)==84
    baseline_scores={(r['baseline'],r['conditioned_on_displacement']):r for r in result['baseline_scores']}
    data=[];matrix=np.empty((14,6));annotations={}
    for j,group in enumerate(GROUPS):
        for i,feature in enumerate(features):
            r=indexed[*group,feature]
            assert r['defined_folds']==4 and r['pooled_rmse_reduction'] is not None
            delta=-100*r['pooled_rmse_reduction']
            matrix[i,j]=delta
            annotations[i,j]=f'{delta:+.2f}'+('*' if r['diagnostic_flag'] else '')
            data.append(dict(comparator_column=j,feature_row=i,baseline=group[0],conditioned_on_displacement=group[1],
                 feature=feature,baseline_rmse_points=100*r['pooled_baseline_rmse'],feature_rmse_points=100*r['pooled_feature_rmse'],
                 feature_minus_baseline_rmse_points=delta,improved_folds=r['improved_folds'],diagnostic_flag=r['diagnostic_flag']))
    a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/'figure_data.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
    all_four=[f for f in features if all(indexed[b,False,f]['diagnostic_flag'] for b in ['B0_original','B1_raw_rate','B2_optimizer_stage_rate','B3_nominal_schedule'])]
    descriptive=dict(scope='post-result-all-comparator-descriptive-accounting',
         features_passing_all_four_unconditioned_baselines=all_four,
         groups=[dict(baseline=b,conditioned_on_displacement=c,
                flags_true=sum(indexed[b,c,f]['diagnostic_flag'] for f in features),
                baseline_rmse_points=100*baseline_scores[b,c]['pooled_rmse']) for b,c in GROUPS],
         significance_or_causal_claim=False)
    with (a.output/'descriptive.json').open('x') as f:f.write(json.dumps(descriptive,indent=2,sort_keys=True)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
                         'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'dense-v3-predictor-sensitivity-v1'})
    fig,ax=plt.subplots(figsize=(12.5,8.7))
    limit=max(abs(matrix.min()),abs(matrix.max()))
    im=ax.imshow(matrix,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_yticks(range(14),LABELS)
    labels=[]
    for b,c in GROUPS:
        row=baseline_scores[b,c]
        labels.append(b[:2]+(' + disp.' if c else '')+f"\nRMSE {100*row['pooled_rmse']:.3f}\np = {row['columns']}")
    ax.set_xticks(range(6),labels)
    ax.xaxis.tick_top();ax.tick_params(axis='x',length=0,pad=12)
    ax.tick_params(axis='y',length=0)
    for (i,j),text in annotations.items():ax.text(j,i,text,ha='center',va='center',fontsize=9,color='white' if abs(matrix[i,j])>.58*limit else '#222222')
    ax.axhline(8.5,color='black',lw=1.2);ax.axvline(3.5,color='black',lw=1.2)
    ax.set_xticks(np.arange(-.5,6,1),minor=True);ax.set_yticks(np.arange(-.5,14,1),minor=True)
    ax.grid(which='minor',color='white',lw=.8);ax.tick_params(which='minor',bottom=False,left=False)
    fig.suptitle('Predictive geometry claims are sensitive to the comparator',x=.025,ha='left',fontsize=14,y=.985)
    fig.text(.025,.945,'Post-result sensitivity: all 14 existing features, four recipe baselines and two displacement-conditioned comparators.',fontsize=10)
    fig.subplots_adjust(left=.40,right=.93,top=.80,bottom=.135)
    cax=fig.add_axes([.948,.24,.014,.43]);fig.colorbar(im,cax=cax,label='Added feature − comparator RMSE (points)')
    fig.text(.025,.065,'Negative values favor the added feature. *: lower pooled RMSE and improvements in ≥3/4 held-dose folds.\nB0: original; B1: + raw LR; B2: optimizer × stage + log/raw LR; B3: B2 + nominal cumulative LR budget.\nSame 60 states and training seed. All flags are exploratory, not significance or causal evidence.',fontsize=9)
    stamp=datetime(2026,9,12,12,33,52,tzinfo=timezone.utc)
    fig.savefig(a.output/'comparator_sensitivity.pdf',metadata={'CreationDate':stamp,'ModDate':stamp})
    fig.savefig(a.output/'comparator_sensitivity.png',dpi=180)
    fig.savefig(a.output/'comparator_sensitivity.svg',metadata={'Date':stamp.isoformat()})
    plt.close(fig)
    files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(a.output.iterdir())}
    manifest=dict(input_readout_sha256=a.receipt_sha256,comparisons=84,all_features_preserved=True,
         statistical_recalculation=False,exploratory_post_result=True,scientific_completion=False,files=files)
    with (a.output/'manifest.json').open('x') as f:f.write(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'comparisons':84,'features_passing_all_four_baselines':all_four,'files':files}))


if __name__=='__main__':main()
