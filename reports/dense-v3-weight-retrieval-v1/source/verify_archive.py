"""Seal the actual analysis after independent feature, prediction and replay checks."""
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT=Path(__file__).parents[1]
STORY=ROOT.parents[1]


def need(value,message):
    if not value:raise ValueError(message)


def identity(path):
    need(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)),'Missing/symlinked file')
    with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return {'bytes':path.stat().st_size,'sha256':sha}


def read(path,expected=None):
    if expected is not None:
        actual=identity(path)
        need(actual['sha256']==expected if isinstance(expected,str) else actual==expected,'Bound file differs')
    return json.loads(path.read_text())


def csv_rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def main():
    output=ROOT/'verification.json'
    need(not output.exists(),'Preserve prior archive seal')
    first=read(ROOT/'actual/readout.json','6891e82a46df11a6d142e79fdf495afa329f40912569337f4b1d6da33d5e8046')
    second=read(ROOT/'source-relocated/readout.json')
    need(first['outputs']==second['outputs'] and first['source_ast_sha256']==second['source_ast_sha256'], 'Copied source/data replay differs')
    need(len(first['outputs'])==17 and len(first['input_bindings'])==167,'Wrong output/input coverage')
    for prefix,receipt in [('actual',first),('source-relocated',second)]:
        for name,b in receipt['outputs'].items():need(identity(ROOT/prefix/name)==b,'Actual/replayed payload differs')
        for name,b in receipt['input_bindings'].items():need(identity(Path(name))==b,'Original/copied input changed')
        need(receipt['formal_consumer_called'] is False and receipt['formal_primary_admission'] is False
             and receipt['scientific_completion'] is False,'Unjustified admission flag')
    oracle=read(ROOT/'independent-predictions.json')
    need(oracle['producer_readout_sha256']==identity(ROOT/'actual/readout.json')['sha256'],'Oracle source differs')
    need([oracle[k] for k in ['exact_held_out_prediction_rows','exact_fold_mse_comparisons','pooled_summaries_and_decisions','residual_associations']]==[840,56,14,14], 'Incomplete independent predictions')
    tables=read(ROOT/'actual/tables.json')
    original=tables['original']['bridge_rows'];exact=tables['exact']['bridge_rows']
    approx={(r['run_id'],int(r['stage'])):r for r in csv_rows(ROOT/'inputs/weights/approximate/checkpoint_geometry.csv')}
    precise={(r['run_id'],int(r['stage'])):r for r in csv_rows(ROOT/'inputs/weights/exact/checkpoint_exact_geometry.csv')}
    original_columns={
        'saved_segment_stable_rank_fraction':'saved_segment_stable_rank_fraction_parameter_weighted',
        'saved_segment_sketch_effective_rank_fraction':'saved_segment_sketch_effective_rank_fraction_parameter_weighted',
        'saved_segment_row_norm_cv':'saved_segment_row_cv_parameter_weighted',
        'saved_segment_top_1pct_row_energy':'saved_segment_top_1pct_row_energy_parameter_weighted',
        'cumulative_displacement_to_weight_ratio':'cumulative_displacement_to_weight_ratio',
        'cumulative_stable_rank_fraction':'cumulative_stable_rank_fraction_parameter_weighted'}
    exact_columns={
        'exact_nonzero_saved_segment_stable_rank_fraction':'exact_saved_segment_stable_rank_fraction_parameter_weighted_nonzero',
        'full_spectrum_nonzero_saved_segment_entropy_rank_fraction':'exact_saved_segment_full_entropy_effective_rank_fraction_parameter_weighted_nonzero',
        'exact_nonzero_cumulative_stable_rank_fraction':'exact_cumulative_stable_rank_fraction_parameter_weighted_nonzero'}
    feature_checks=0
    for rows,source,mapping in [(original,approx,original_columns),(exact,precise,exact_columns)]:
        for row in rows:
            raw=source[row['run_id'],row['stage']]
            for name,column in mapping.items():
                need(row[name]==float(raw[column]),'Checkpoint feature mapping differs')
                feature_checks+=1
    for row in original:
        raw=approx[row['run_id'],row['stage']]
        need(row['log_saved_segment_to_weight_ratio']==math.log(float(raw['saved_segment_to_weight_ratio'])),'Log feature differs')
        feature_checks+=1
    for family,rows,path in [('original',original,'approximate/run_pair_subspace_overlap.csv'),
                             ('exact',exact,'exact/run_pair_exact_subspace_overlap.csv')]:
        pairs=csv_rows(ROOT/'inputs/weights'/path)
        need(len(pairs)==660,'Missing complete run-pair population')
        for row in rows:
            for kind in ['saved_segment','cumulative']:
                other_ids=[];values=[]
                for p in pairs:
                    if int(p['stage'])!=row['stage'] or p['displacement_kind']!=kind:continue
                    if p['first_run_id']==row['run_id'] and p['second_optimizer']=='adamw':
                        other_ids.append(p['second_run_id']);values.append(float(p['mean_subspace_overlap']))
                    elif p['second_run_id']==row['run_id'] and p['first_optimizer']=='adamw':
                        other_ids.append(p['first_run_id']);values.append(float(p['mean_subspace_overlap']))
                wanted={r['run_id'] for r in rows if r['optimizer']=='adamw' and r['run_id']!=row['run_id']}
                need(set(other_ids)==wanted and len(other_ids)==len(wanted),'Comparator set differs')
                name=(f'mean_{kind}_subspace_overlap_to_adamw' if family=='original' else f'exact_mean_{kind}_overlap_to_adamw')
                expected=float(np.mean(values)) if family=='original' else math.fsum(values)/len(values)
                need(row[name]==expected,'Complete comparator mean differs')
                feature_checks+=1
    need(feature_checks==840,'Not every 60-by-14 feature value was checked')
    old={(r['run_id'],r['stage']):r for r in original};new={(r['run_id'],r['stage']):r for r in exact}
    comparisons=tables['exact']['measurement_comparison']
    need(len(comparisons)==300,'Missing exact/original measurement correspondence')
    for r in comparisons:
        key=r['run_id'],r['stage'];x=old[key][r['original_feature']];y=new[key][r['exact_feature']]
        need(r['original_value']==x and r['exact_value']==y and r['exact_minus_original']==y-x and r['defined'] is True,
             'Measurement correspondence differs')
    old_s={r['feature']:r for r in tables['original']['feature_prediction_summary']}
    new_s={r['feature']:r for r in tables['exact']['feature_prediction_summary']}
    for r in tables['exact']['predictive_sensitivity_summary']:
        a,b=old_s[r['original_feature']],new_s[r['exact_feature']]
        difference=Fraction(a['pooled_feature_mse_exact'])-Fraction(b['pooled_feature_mse_exact'])
        need(Fraction(r['original_minus_exact_feature_mse_rational'])==difference
             and r['original_minus_exact_feature_mse']==float(difference),'Sensitivity MSE difference differs')
        need(r['original_predictively_useful'] is a['predictively_useful'] and r['exact_predictively_useful'] is b['predictively_useful'],
             'Sensitivity decision mapping differs')
    figure=read(ROOT/'figures/figure_manifest.json')
    need(figure['input_readout_sha256']==identity(ROOT/'actual/readout.json')['sha256'],'Figure input differs')
    for name,b in figure['files'].items():need(identity(ROOT/'figures'/name)==b,'Figure bytes differ')
    plotted=csv_rows(ROOT/'figures/figure_data.csv')
    need(len(plotted)==14,'Missing figure points')
    for r in plotted:
        value=(old_s if r['family']=='original' else new_s)[r['feature']]
        need(float(r['feature_rmse_points'])==100*value['pooled_feature_rmse']
             and float(r['baseline_rmse_points'])==100*value['pooled_baseline_rmse']
             and int(r['improved_folds'])==value['folds_improved'],'Figure value differs')
    prior=read(ROOT/'inputs/outcomes/verification.json')
    for name,b in prior['protected_inputs'].items():need(identity(Path(name))==b,'Protected original input changed')
    primary=Path('/root/embedding-optimizer-primary-v3')
    assembly=read(primary/'source-assembly.json','e603ef175c0f691a1dd83c4d8b474d90da2fc67c8c65f5a77f3c3180e78332b8')
    count=0
    for base in [primary,Path('/root/embedding-optimizer-v3-experiment/launch/source-snapshot')]:
        for name,b in assembly['files'].items():need(identity(base/name)==b['identity'],'Original primary source changed');count+=1
    need(count==112,'Source assembly coverage differs')
    need(identity(STORY/'paper/main.tex')['sha256']=='45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e', 'Manuscript changed')
    payloads={};links=0
    secret=re.compile(rb'(?:wandb_v1_[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})')
    for path in sorted(ROOT.rglob('*')):
        need(not path.is_symlink(),'Symlinked report payload')
        if not path.is_file():continue
        if path.suffix in ['.py','.json','.md','.csv']:
            need(not secret.search(path.read_bytes()),'Credential-like report payload; value suppressed')
        if path.suffix=='.md' and 'before' not in path.parts and 'inputs' not in path.parts:
            for ref in re.findall(r'\]\(([^)]+)\)',path.read_text()):
                if '://' in ref or ref.startswith('#'):continue
                target=ref.split('#',1)[0]
                need((path.parent/target).exists(),'Broken report link: '+target);links+=1
        payloads[path.relative_to(ROOT).as_posix()]=identity(path)
    result=dict(scope='actual-weight-retrieval-readout-archive-verification',verified_at_utc=datetime.now(timezone.utc).isoformat(),
         producer_readout=identity(ROOT/'actual/readout.json'),copied_source_readout=identity(ROOT/'source-relocated/readout.json'),
         independent_prediction_receipt=identity(ROOT/'independent-predictions.json'),
         independently_checked_feature_values=840,original_exact_correspondences=300,
         exact_held_out_predictions=840,fold_comparisons=56,pooled_decisions=14,residual_associations=14,
         byte_identical_replayed_outputs=17,native_common_checkpoint_identity_seal_joins=60,
         original_source_assembly_files_unchanged=count,protected_inputs_unchanged=len(prior['protected_inputs']),
         figure_points=14,credential_findings=0,local_links_checked=links,payloads=payloads,
         numerical_logic_changed=False,formal_primary_admission=False,model_or_retrieval_execution=False,
         functional_recovery=False,manuscript_modified=False,source_or_data_published=False,scientific_completion=False)
    with output.open('x') as f:f.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='payloads'}))


if __name__=='__main__':main()
