"""Independent filesystem/provenance verification of the offline full-flow fixture."""
import ast
import hashlib
import json
import re
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).parents[1]
WORK=Path('/tmp/dense-v3-functional-flow-candidate.TD3eLq4a')
STORY=Path('/root/embedding-optimizer-story-refactor')


def need(v,m):
    if not v:raise ValueError(m)


def identity(p):
    need(p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)),'Nonordinary archive input')
    raw=p.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def read(p):return json.loads(p.read_bytes())


def main():
    output=ROOT/'verification.json';need(not output.exists(),'Preserve existing verification')
    third=read(ROOT/'actual/tests-third.json')
    need(identity(ROOT/'actual/tests-third.json')['sha256']=='6b704be5ad2377437f42778deee8ff97b79053b27f03ae1c8b3963d4d09dfdc3','Final receipt changed')
    need([third[k] for k in ('tests','failures','errors','skipped')]==[18,0,0,0],'Final tests failed')
    for attempt,source,test in [('first','flow_candidate-initial.py','test_flow-first.py'),
                              ('second','flow_candidate-initial.py','test_flow-second.py'),
                              ('third','flow_candidate.py','test_flow.py')]:
        record=read(ROOT/f'actual/tests-{attempt}.json')
        need(record['source']==identity(ROOT/'source'/source) and record['test_source']==identity(ROOT/'source'/test),'Attempt source binding differs')
        for key in ('actual_model_or_gpu_execution','native_numeric_acceptance','real_process_or_lease_inspection',
                    'execution_authority_created','original_sources_modified','production_launcher','scientific_completion'):
            need(record[key] is False,'Misstated simulation boundary')
    for name,item in third['protected_files'].items():
        need(Path(name).name!='gpu.py','Protected helper is outside this task')
        need(identity(Path(name))==item,'Original protected input changed')
    need(len(third['protected_files'])==66,'Original source coverage differs')
    for path in (WORK/'source').iterdir():need(identity(path)==identity(ROOT/'source'/path.name),'Archived source differs')
    need(identity(ROOT/'source-original/dispatch.py')['sha256']=='3b02f1d8486f3c2b41d0403cbf291580d0ca32868b8b29cb05514a22770257f5','Original dispatcher changed')
    need(identity(ROOT/'source-original/primary_v3_dimension_vector_io.py')['sha256']=='ea33869f8cf677ad0c15e8747b6d91045ba55a39301a8bf2c52f09bf642ed858','Original vector reader changed')
    fixture=ROOT/'synthetic-complete-flow';actual=Path(third['completed_full_fixture_flows'][0]['fixture_root'])
    files={p.relative_to(actual).as_posix():identity(p) for p in actual.rglob('*') if p.is_file()}
    need({p.relative_to(fixture).as_posix() for p in fixture.rglob('*') if p.is_file()}==set(files),'Incomplete representative fixture')
    for name,item in files.items():need(identity(fixture/name)==item,'Copied representative fixture differs')
    matrix=read(fixture/'output/vectors/manifest.json');cells=set(matrix['states'])
    need(len(cells)==61 and 'pretrained' in cells,'Incomplete synthetic state population')
    counters={k:0 for k in ('started','exited','encoded','verified','features-verified')}
    for cell in sorted(cells):
        for kind in counters:
            path=fixture/'run/jobs'/f'{cell}.{kind}.json'
            if cell=='pretrained' and kind!='features-verified':
                need(not path.exists(),'Invented new pretrained worker');continue
            record=read(path);need(record['cell']==cell,'Wrong nested synthetic record');counters[kind]+=1
            if kind=='started':need(record['pid']<0 and record['ppid']==-100,'A fixture claimed a real process')
            if kind=='exited':need(record['exit_code']==0,'Complete fixture has failed worker')
            if kind=='features-verified':
                item=record['manifest'];need(item['path']==f'states/{cell}/manifest.json','Wrong feature path')
                need(identity(fixture/'output/features'/item['path'])=={k:item[k] for k in ('bytes','sha256')},'Feature anchor differs')
        vector=fixture/'output/vectors/states'/cell/'vectors.npz'
        need(vector.read_bytes().startswith(b'NOT_AN_NPZ: synthetic model stub'),'Fixture is not explicitly synthetic')
    need(counters=={'started':60,'exited':60,'encoded':60,'verified':60,'features-verified':61},'Whole-flow record coverage differs')
    reuse=read(fixture/'run/pretrained.reused.json')
    need(reuse['new_worker_created'] is False and reuse['model_reencoded'] is False,'Reused fixture mislabeled')
    for name in ('manifest.json','vectors.npz'):
        need(identity(fixture/'prior/vectors/pretrained'/name)==identity(fixture/'output/vectors/states/pretrained'/name),'Baseline copy differs')
    final=read(fixture/'run/completed.json')
    need(final['features']['states']==61 and final['vectors']['all_states_encoded_and_native_readback_verified']==61
         and final['scope']=='synthetic_full_flow_fixture_only','Wrong synthetic finalization')
    need(not (fixture/'unread-models').exists(),'Unexpected actual model fixture')
    need(identity(STORY/'paper/main.tex')['sha256']=='45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e','Manuscript changed')
    payloads={};links=0
    secret=re.compile(rb'(?:wandb_v1_[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})')
    for p in sorted(ROOT.rglob('*')):
        need(not p.is_symlink(),'Symlinked archive')
        if not p.is_file():continue
        need(secret.search(p.read_bytes()) is None,'Credential-shaped content; suppressed')
        payloads[p.relative_to(ROOT).as_posix()]=identity(p)
    for target in re.findall(r'\]\(([^)]+)\)',(ROOT/'README.md').read_text()):
        resolved=(ROOT/target.split('#',1)[0]).resolve()
        need(resolved.exists() or resolved==output,'Broken local archive link');links+=1
    value={'scope':'independently-verified-offline-functional-full-flow-preparation',
        'verified_at_utc':datetime.now(timezone.utc).isoformat(),'tests':18,'failures':0,'errors':0,'skipped':0,
        'representative_synthetic_files':len(files),'record_counts':counters,'reused_synthetic_pretrained_states':1,
        'feature_stub_calculations':122,'original_bound_files_unchanged':66,
        'source':identity(ROOT/'source/flow_candidate.py'),'test_source':identity(ROOT/'source/test_flow.py'),
        'payloads':payloads,'local_links_checked':links,'credential_findings':0,
        'actual_model_or_gpu_execution':False,'native_numeric_acceptance':False,'production_launcher':False,
        'recovery_execution_authorized':False,'original_dispatcher_or_observer_modified':False,
        'scientific_completion':False,'manuscript_modified':False,'remote_publication':False}
    with output.open('x') as stream:stream.write(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k!='payloads'},sort_keys=True))


if __name__=='__main__':main()
