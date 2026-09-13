"""Archive the actual numerical result, retained computation chain and live handoff."""
import importlib.util
import json
from pathlib import Path
import shutil

WORK=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('functional_archive_parent',WORK/'inference.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
p.need(p.identity(WORK/'inference.py')['sha256']=='84e6aaf4dcb573ac04b74c6e97bf88b44fe4e7c410842eb80228239555168254','Changed parent')
ROOT=p.STORY/'reports/dense-v3-functional-inference-v1'
ENGINE=p.STORY/'reports/engineering-archive/dense-v3-functional-result-chain-v1'
for path in (ROOT,ENGINE):p.need(not path.exists(),'Preserve any archive');path.mkdir(parents=True)
copied={}


def copy(source,destination):
    p.need(source.name!='gpu.py','Protected helper excluded')
    bound=p.identity(source)
    destination.parent.mkdir(parents=True,exist_ok=True)
    p.need(not destination.exists(),'Existing archive target')
    shutil.copyfile(source,destination,follow_symlinks=False)
    p.need(p.identity(destination)==bound==p.identity(source),'Archive copy changed')
    copied[str(destination)]={'source':str(source),**bound}


def tree(source,destination,exclude=()):
    for path in sorted(source.rglob('*')):
        p.need(not path.is_symlink(),'Symlinked archive source')
        if path.is_file() and path.suffix not in exclude:copy(path,destination/path.relative_to(source))


actual=WORK/'reconciled'
r=p.read(actual/'readout.json','608965f5297fc196adcce6ca7417bd40e7f35d05e26133f20c0811dd1fd7027e')
v=p.read(actual/'independent_verification.json','03f9670ab2a8a2334db439c95f1ef6264485cb816addd48457df305f64b45e23')
p.need(v['all_checks_passed'] is True and v['independent_exact_predictions']['exact_predictions']==240,'Missing actual independent check')
for path,bound in r['source_bindings'].items():p.need(p.identity(path)==bound,'Changed actual source/data')
for name,bound in r['payloads'].items():p.need(p.identity(actual/name)==bound,'Changed actual output')
tree(actual,ROOT/'actual')
for name in ('readout.pdf','readout.tex','page-1.png','page-2.png'):copy(WORK/'visualization'/name,ROOT/'figures'/name)
for name in ('readout.log','readout.aux'):copy(WORK/'visualization'/name,ENGINE/'layout'/name)
for name in ('AGENTS.md','CURRENT_EXPERIMENT.md','PROJECT_STATUS.md','README.md'):
    copy(p.STORY/name,ENGINE/'before'/name)
for name in ('inference.py','reconcile.py','test_inference.py','tests.json','owner-approval.json',
    'preparation.json','terminal.json','prepare_reconciliation.py','reconciliation-inputs.json',
    'reconciliation-approval.json','archive.py'):
    copy(WORK/name,ENGINE/'entry'/name)
tree(WORK/'actual',ENGINE/'original-readback-prefix')
for path in r['source_bindings']:
    source=Path(path)
    if source.is_relative_to(p.STORY/'src') or source.is_relative_to(p.STORY/'configs'):
        copy(source,ROOT/'source-original'/source.relative_to(p.STORY))
for name in ('scripts/dimension_inference_reference.py','configs/dense_primary_v3_dimension_inference_protocol.json'):
    source=p.STORY/name;destination=ROOT/'source-original'/name
    if not destination.exists():copy(source,destination)
feature_entry=p.FEATURE_ENTRY
for name in ('features.py','test_features.py','tests.json','owner-approval.json'):
    copy(feature_entry/name,ENGINE/'feature-producer'/name)
tree(feature_entry/'run',ENGINE/'feature-producer/run')
references=[]
for label,source in [('vectors',p.EXP/'analyses/dense-primary-v3-functional-dimensions-recovery-v1/vectors'),
                     ('features',p.EXP/'analyses/dense-primary-v3-functional-features-recovery-v3')]:
    for path in sorted(source.rglob('*')):
        p.need(not path.is_symlink(),'Symlinked functional input')
        if not path.is_file():continue
        if path.suffix=='.npz':references.append({'role':label+'/'+str(path.relative_to(source)),
            'path':str(path),**p.identity(path)})
        else:copy(path,ENGINE/'input-metadata'/label/path.relative_to(source))
backup=Path('/tmp/dense-v3-factorial-backup-launch.75Chtt86')
for name in ('backup.py','test_backup.py','tests.json','owner-approval.json','preparation.json'):
    copy(backup/name,ENGINE/'backup-entry'/name)
copy(backup/'run/started.json',ENGINE/'backup-entry/run/started.json')
for run in ('factorial-v3-adamw_state-adamw-seed314159','factorial-v3-adamw_state-muon-seed314159'):
    tree(backup/'run'/run,ENGINE/'backup-entry/run'/run)
    pool='a' if '-adamw-seed' in run else 'b'
    job=p.EXP/'launch/factorial-training-v1/run'/('pool-'+pool)/run
    for name in ('completed.json','fresh-native-readback.json','ranks.exited.json','reader.exited.json'):
        copy(job/name,ENGINE/'complete-first-branches'/run/name)
observer=Path('/tmp/dense-v3-factorial-training-launch.KxwKP2lw/observation-ninth.json')
copy(observer,ENGINE/'training-observation-1553.json')
p.write(ENGINE/'retained-binary-references.json',{'files':references,'payloads_copied_into_git_tree':False,
    'new_off_host_functional_backup_claimed':False,'scientific_completion':False})
receipt={'scope':'actual_functional_inference_and_first_crossed_branch_chain_archive',
    'actual_readout_sha256':r and p.identity(actual/'readout.json')['sha256'],
    'independent_verification_sha256':p.identity(actual/'independent_verification.json')['sha256'],
    'genuine_raw_state_recomputations':61,'source_context_reconciliation_passed':True,
    'input_table_counts':r['input_table_counts'],'inference_table_counts':r['output_table_counts'],
    'independent_exact_predictions':240,'independent_task_contrasts':9,'independent_rotation_contrasts':27,
    'original_outer_readback_exit':1,'reconciled_calculation_exit':0,'independent_verifier_exit':0,
    'complete_crossed_branches':2,'new_hf_backed_checkpoints':10,
    'copied_files':copied,'retained_npz_references':len(references),'manuscript_changed':False,
    'scientific_completion':False}
p.write(ENGINE/'verification.json',receipt)
print(json.dumps({'archive':str(ROOT),'engineering':str(ENGINE),'copied_files':len(copied),
    'binary_references':len(references),'verification':p.identity(ENGINE/'verification.json')}))
