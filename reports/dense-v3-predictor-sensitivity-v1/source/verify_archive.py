"""Seal complete exploratory evidence without rewriting any parent result."""
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).parents[1]
STORY=ROOT.parents[1]
PARENT=STORY/'reports/dense-v3-weight-retrieval-v1'


def need(v,m):
    if not v:raise ValueError(m)


def identity(p):
    need(p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)),'Missing/symlinked evidence')
    with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(bytes=p.stat().st_size,sha256=sha)


def read(p,sha=None):
    if sha:need(identity(p)['sha256']==sha,'Evidence anchor differs')
    return json.loads(p.read_text())


def main():
    output=ROOT/'verification.json'
    need(not output.exists(),'Preserve previous verification')
    need(identity(ROOT/'plan.md')['sha256']=='0d2f1548dafebcd62c10e535607743e9f5ec3eba42d970e4ee507e9f3ccc835a','Post-result plan changed')
    parent=read(PARENT/'verification.json','243b001980feca61e6cca4852f544e9922fa7ca90b2a157981836d47b16d8621')
    for name,b in parent['payloads'].items():need(identity(PARENT/name)==b,'A preceding locked result/source changed')
    first=read(ROOT/'actual/readout.json','00009a1f0253d9dfa6de9938f016f20f339f1e0729e6ce31886aca9a96a2db64')
    second=read(ROOT/'source-relocated/readout.json','c7c3cd00021ab243e3ddcaa2b4a0b5debae04e7914a118d00d30b02772824dae')
    need(first['outputs']==second['outputs'] and len(first['outputs'])==7,'Replay output mismatch')
    need(first['source']==second['source']==identity(ROOT/'source/sensitivity.py'),'Source replay differs')
    for directory,receipt in [('actual',first),('source-relocated',second)]:
        for name,b in receipt['outputs'].items():need(identity(ROOT/directory/name)==b,'Stored output differs')
        for name,b in receipt['input_bindings'].items():need(identity(Path(name))==b,'Original/copied input differs')
        need(receipt['exploratory_post_result'] is True and receipt['original_protocol_or_result_modified'] is False
             and receipt['formal_primary_admission'] is False and receipt['scientific_completion'] is False,'Misstated scientific scope')
    oracle=read(ROOT/'independent.json','09384e9a579fc4fa1e24f8dcdda12ba46f0c12343e50821209670356afaa7050')
    need(oracle['readout_sha256']==identity(ROOT/'actual/readout.json')['sha256'],'Independent input differs')
    need([oracle[k] for k in ['independently_reconstructed_design_rows','independently_reconstructed_nominal_covariates',
          'exact_prediction_rows','exact_fold_mse_comparisons','pooled_comparisons','redundant_self_addition_folds']]==[240,60,5040,336,84,8],
         'Independent comparison coverage differs')
    result=read(ROOT/'actual/result.json')
    need([len(result[k]) for k in ['summaries','folds','predictions','baseline_scores','covariates']]==[84,336,5040,6,60],'Missing result population')
    index={(r['baseline'],r['conditioned_on_displacement'],r['feature']):r for r in result['summaries']}
    features={r['feature'] for r in result['summaries']}
    need(len(index)==84 and len(features)==14,'Duplicate/missing features')
    all_four=[f for f in features if all(index[b,False,f]['diagnostic_flag'] for b in ['B0_original','B1_raw_rate','B2_optimizer_stage_rate','B3_nominal_schedule'])]
    need(all_four==[],'Reported comparator sensitivity differs')
    for prefix in ['figures','figures-v2']:
        manifest=read(ROOT/prefix/'manifest.json')
        need(manifest['comparisons']==84 and manifest['input_readout_sha256']==oracle['readout_sha256'],'Figure input differs')
        for name,b in manifest['files'].items():need(identity(ROOT/prefix/name)==b,'Figure payload differs')
        with (ROOT/prefix/'figure_data.csv').open(newline='') as f:plot=list(csv.DictReader(f))
        need(len(plot)==84,'Missing figure comparisons')
        for r in plot:
            key=r['baseline'],r['conditioned_on_displacement']=='True',r['feature'];source=index[key]
            need(float(r['feature_minus_baseline_rmse_points'])==-100*source['pooled_rmse_reduction']
                 and float(r['baseline_rmse_points'])==100*source['pooled_baseline_rmse']
                 and float(r['feature_rmse_points'])==100*source['pooled_feature_rmse']
                 and int(r['improved_folds'])==source['improved_folds']
                 and (r['diagnostic_flag']=='True') is source['diagnostic_flag'],'Figure arithmetic/decision differs')
    need(identity(ROOT/'figures/figure_data.csv')==identity(ROOT/'figures-v2/figure_data.csv'),'Figure layout changed values')
    need(identity(ROOT/'figures/descriptive.json')==identity(ROOT/'figures-v2/descriptive.json'),'Figure layout changed description')
    original=read(PARENT/'inputs/outcomes/verification.json')
    for name,b in original['protected_inputs'].items():need(identity(Path(name))==b,'Protected input changed')
    primary=Path('/root/embedding-optimizer-primary-v3')
    assembly=read(primary/'source-assembly.json','e603ef175c0f691a1dd83c4d8b474d90da2fc67c8c65f5a77f3c3180e78332b8')
    count=0
    for base in [primary,Path('/root/embedding-optimizer-v3-experiment/launch/source-snapshot')]:
        for name,b in assembly['files'].items():need(identity(base/name)==b['identity'],'Original numerical source changed');count+=1
    need(count==112,'Primary source coverage differs')
    need(identity(STORY/'paper/main.tex')['sha256']=='45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e','Manuscript changed')
    payloads={};links=0
    secret=re.compile(rb'(?:wandb_v1_[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})')
    for p in sorted(ROOT.rglob('*')):
        need(not p.is_symlink(),'Symlinked report')
        if not p.is_file():continue
        if p.suffix in ['.py','.json','.md','.csv']:need(not secret.search(p.read_bytes()),'Credential-like content; suppressed')
        if p.suffix=='.md' and 'before' not in p.parts:
            for ref in re.findall(r'\]\(([^)]+)\)',p.read_text()):
                if '://' in ref or ref.startswith('#'):continue
                target=ref.split('#',1)[0];need((p.parent/target).exists(),'Broken local link');links+=1
        payloads[p.relative_to(ROOT).as_posix()]=identity(p)
    result=dict(scope='verified-actual-post-result-predictor-sensitivity-archive',verified_at_utc=datetime.now(timezone.utc).isoformat(),
         plan=identity(ROOT/'plan.md'),actual=identity(ROOT/'actual/readout.json'),copied_source=identity(ROOT/'source-relocated/readout.json'),
         independent=identity(ROOT/'independent.json'),all_comparisons=84,fold_comparisons=336,exact_prediction_rows=5040,
         original_B0_prediction_rows_exactly_reproduced=840,byte_identical_replayed_outputs=7,
         independently_reconstructed_recipe_design_rows=240,nominal_schedule_covariates=60,
         all_four_baseline_supported_features=[],original_parent_payloads_unchanged=len(parent['payloads']),
         original_source_assembly_files_unchanged=count,protected_inputs_unchanged=len(original['protected_inputs']),
         plotted_comparisons=84,layout_versions_have_identical_data=True,credential_findings=0,local_links_checked=links,
         payloads=payloads,exploratory_post_result=True,original_protocol_or_result_modified=False,
         model_or_retrieval_execution=False,functional_recovery=False,manuscript_modified=False,
         source_or_data_published=False,formal_primary_admission=False,scientific_completion=False)
    with output.open('x') as f:f.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='payloads'}))


if __name__=='__main__':main()
