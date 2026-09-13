"""Download and check genuine terminal GitHub CPU verification receipts."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

work = Path(__file__).resolve().parent
root = Path('/root/embedding-optimizer-story-refactor')
run_id = '34766616352'
revision = '562eef8f34db15a58afe3a12580bbb58b6116a9e'
repo = 'qcznlp/embedding-optimizer-study'
out = work / ('terminal-ci-' + run_id)
out.mkdir(exist_ok=False)
def api(path):
    return json.loads(subprocess.check_output(['gh','api',f'repos/{repo}/'+path]))
run = api('actions/runs/' + run_id)
assert run['head_sha'] == revision and run['status'] == 'completed'
(out/'run.json').write_text(json.dumps(run,indent=2))
jobs = api('actions/runs/' + run_id + '/jobs')
(out/'jobs.json').write_text(json.dumps(jobs,indent=2))
assert len(jobs['jobs']) == 1
job = jobs['jobs'][0]
with (out/'job.log').open('xb') as stream:
    subprocess.run(['gh','api','--allow-escape-sequences',f'repos/{repo}/actions/jobs/{job["id"]}/logs'],stdout=stream,check=True)
artifacts = api('actions/runs/' + run_id + '/artifacts')
(out/'artifacts.json').write_text(json.dumps(artifacts,indent=2))
items = [a for a in artifacts['artifacts'] if a['name']=='verification-receipts' and not a['expired']]
assert len(items) == 1
artifact = items[0]
archive_path = out/'receipts.zip'
with archive_path.open('xb') as stream:
    subprocess.run(['gh','api',f'repos/{repo}/actions/artifacts/{artifact["id"]}/zip'],stdout=stream,check=True)
assert archive_path.stat().st_size == artifact['size_in_bytes']
assert 'sha256:'+hashlib.sha256(archive_path.read_bytes()).hexdigest() == artifact['digest']
assert run['conclusion'] == job['conclusion'] == 'success', 'terminal failure retained; inspect raw receipts'
assert all(s['conclusion']=='success' for s in job['steps']), 'a hosted gate did not pass'
with zipfile.ZipFile(archive_path) as archive:
    names=archive.namelist()
    assert len(names)==len(set(names))
    assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
    summary=json.loads(archive.read('source-role-tests/summary.json'))
    assert summary['complete'] is True and summary['cases']==3800 and summary['test_modules']==211
    assert summary['role_file_sha256']=='16fef314c00a848ce5e6e1592f8955ecb7a83f5c4d70c32a67c7d82d4fc7caf3'
    expected={'current':2873,'original-analysis':733,'original-factorial':194}
    assert {r['role'] for r in summary['roles']} == set(expected)
    for role in summary['roles']:
        label=role['role']
        result=json.loads(archive.read('source-role-tests/'+label+'-result.json'))
        cases=ET.fromstring(archive.read('source-role-tests/'+label+'.xml')).findall('.//testcase')
        assert len(cases)==role['cases']==result['cases']==expected[label]
        assert result['complete'] is role['complete'] is True
        for key in ['failure','error','skipped','exit_code']:
            assert result[key]==role[key]==0
        assert not any(case.find(k) is not None for case in cases for k in ['failure','error','skipped'])
    complete=json.loads(archive.read('complete-paper/complete.json'))
    assert complete['complete'] is complete['numerical_and_reviewed_document_complete'] is True
    assert complete['numerical_exit']['exit_code']==0
    assert complete['original_contracts_modified'] is False
    assert complete['input_manifest_sha256']=='746a545d3e59d86aab8d4ff6579abca3be9884f1ea824a702e4fa28bc85956b7'
    assert complete['numerical_entry_sha256']=='1a9c50dba2f58cc68b110749ccbc40d427445d53f0dc6c994b8f78c02096d566'
    assert complete['document_snapshot_sha256']=='6ef28bcf1f2cd873bcb0a663c2baea57ec6dc98a4e441d0850608a61f46ae6b9'
    pdf=archive.read('complete-paper/reviewed/paper/build/main.pdf')
    assert complete['pdf']=={'bytes':len(pdf),'sha256':hashlib.sha256(pdf).hexdigest()}
    document=complete['reviewed_document']
    assert document['document_reproduction_complete'] is True
    assert document['abstract_words']==158 and document['main_end_page']==8 and document['pdf_pages']==13
    assert len(complete['shared_inputs'])==8
    inputs=json.loads(archive.read('source-role-tests/inputs.json'))
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==revision
    for name,binding in inputs['files'].items():
        raw=(root/name).read_bytes()
        assert len(raw)==binding['bytes'] and hashlib.sha256(raw).hexdigest()==binding['sha256'], name
    archive.extractall(out/'receipts')
receipt={'complete':True,'scope':'independent-hosted-CPU-regressions-and-numerical-paper-reproduction',
         'run_id':run_id,'job_id':job['id'],'revision':revision,'artifact_digest':artifact['digest'],
         'cases':3800,'test_modules':211,'failures':0,'errors':0,'skipped':0,
         'source_inputs_byte_verified':len(inputs['files']),'numerical_and_paper_complete':True,
         'pdf':complete['pdf'],'new_gpu_execution':False,'new_scientific_training':False}
(out/'verified.json').write_text(json.dumps(receipt,indent=2,sort_keys=True))
print(json.dumps(receipt),flush=True)
