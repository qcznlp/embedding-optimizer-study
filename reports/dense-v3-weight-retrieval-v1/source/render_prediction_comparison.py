"""Plot all fourteen declared features from the actual accepted numerical readout."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {
    'log_saved_segment_to_weight_ratio': 'Log segment displacement / weight',
    'saved_segment_stable_rank_fraction': 'Segment stable rank',
    'saved_segment_sketch_effective_rank_fraction': 'Segment entropy rank (top-64)',
    'saved_segment_row_norm_cv': 'Segment row-norm CV',
    'saved_segment_top_1pct_row_energy': 'Segment top-1% row energy',
    'cumulative_displacement_to_weight_ratio': 'Cumulative displacement / weight',
    'cumulative_stable_rank_fraction': 'Cumulative stable rank',
    'mean_saved_segment_subspace_overlap_to_adamw': 'Segment overlap with AdamW',
    'mean_cumulative_subspace_overlap_to_adamw': 'Cumulative overlap with AdamW',
    'exact_nonzero_saved_segment_stable_rank_fraction': 'Exact segment stable rank',
    'full_spectrum_nonzero_saved_segment_entropy_rank_fraction': 'Segment entropy rank (full spectrum)',
    'exact_nonzero_cumulative_stable_rank_fraction': 'Exact cumulative stable rank',
    'exact_mean_saved_segment_overlap_to_adamw': 'Exact segment overlap with AdamW',
    'exact_mean_cumulative_overlap_to_adamw': 'Exact cumulative overlap with AdamW',
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--readout',type=Path,required=True)
    p.add_argument('--receipt-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    def bound(path, expected):
        if any(x.is_symlink() for x in (path,*path.parents)) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Figure input identity differs')
    bound(a.readout/'readout.json',a.receipt_sha256)
    receipt=json.loads((a.readout/'readout.json').read_text())
    bound(a.readout/'tables.json',receipt['outputs']['tables.json']['sha256'])
    tables=json.loads((a.readout/'tables.json').read_text())
    rows=[]
    for family in ['original','exact']:
        for r in tables[family]['feature_prediction_summary']:
            rows.append(dict(family=family,feature=r['feature'],label=LABELS[r['feature']],
                 baseline_rmse_points=100*r['pooled_baseline_rmse'],feature_rmse_points=100*r['pooled_feature_rmse'],
                 improved_folds=r['folds_improved'],defined_folds=r['folds_defined'],predictive_support=r['predictively_useful']))
    if len(rows)!=14 or {r['feature'] for r in rows}!=set(LABELS):
        raise ValueError('Figure must include all fourteen distinct features')
    if any(r['defined_folds']!=4 for r in rows):
        raise ValueError('This actual-population plot must not omit an undefined fit')
    a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/'figure_data.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
                         'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'dense-v3-weight-retrieval-v1'})
    fig,axes=plt.subplots(2,1,figsize=(10.8,8.1),gridspec_kw={'height_ratios':[9,5]})
    for ax,family,title in zip(axes,['original','exact'],['Original nine features','Separate exact-measurement sensitivity']):
        part=[r for r in rows if r['family']==family]
        baseline=part[0]['baseline_rmse_points']
        for i,r in enumerate(part):
            c='#1565a7' if r['predictive_support'] else '#6d7177'
            x=r['feature_rmse_points']
            ax.plot([baseline,x],[i,i],color=c,alpha=.55,lw=2)
            ax.scatter(x,i,color=c,s=42,zorder=3)
            ax.text(4.12,i,f"{r['improved_folds']}/4",va='center',ha='center',fontsize=9)
        ax.axvline(baseline,color='#a34532',ls='--',lw=1.5)
        ax.set_yticks(range(len(part)),[r['label'] for r in part])
        ax.set_ylim(len(part)-.4,-.9)
        ax.set_xlim(.7,4.35)
        ax.set_xticks([1,1.5,2,2.5,3,3.5,4])
        ax.set_title(title,loc='left',fontsize=11,fontweight='bold',pad=12)
        ax.text(4.12,-.63,'Folds\nimproved',ha='center',va='center',fontsize=8)
        ax.grid(axis='x',alpha=.16)
        ax.tick_params(axis='y',length=0)
        ax.spines[['top','right','left']].set_visible(False)
        ax.set_xlabel('Held-dose RMSE (nDCG@10 points; lower is better)')
    fig.suptitle('Weight-space fingerprints do not all predict retrieval',x=.03,ha='left',fontsize=14,y=.985)
    fig.text(.03,.944,'All 60 states; four learning-rate-index holdouts; one feature added to the locked baseline.',fontsize=10)
    fig.text(.03,.025,'Dashed line: baseline RMSE 2.905. Blue: lower pooled RMSE and ≥3/4 improved folds.\nPredictive flags are descriptive, not significance tests or causal/functional-utility evidence.',fontsize=9)
    fig.subplots_adjust(left=.385,right=.975,top=.88,bottom=.13,hspace=.48)
    stamp=datetime(2026,9,12,12,15,2,tzinfo=timezone.utc)
    fig.savefig(a.output/'held_dose_prediction.pdf',metadata={'CreationDate':stamp,'ModDate':stamp})
    fig.savefig(a.output/'held_dose_prediction.png',dpi=180)
    fig.savefig(a.output/'held_dose_prediction.svg',metadata={'Date':stamp.isoformat()})
    plt.close(fig)
    files={f.name:{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(a.output.iterdir())}
    result=dict(input_readout_sha256=a.receipt_sha256,features=14,points=14,all_features_preserved=True,
                statistical_recalculation=False,scientific_completion=False,files=files)
    with (a.output/'figure_manifest.json').open('x') as f:f.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
