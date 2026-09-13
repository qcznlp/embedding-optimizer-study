"""Enumerate current-versus-historical source mismatches; change no binding."""
import hashlib
import json
from pathlib import Path

root=Path('/root/embedding-optimizer-story-refactor')
def identity(path):
    with path.open('rb') as stream:
        sha=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'bytes':path.stat().st_size,'sha256':sha}

result={}
for name in ('dense_primary_v3_publication_protocol.json','dense_primary_v3_exact_publication_protocol.json'):
    path=root/'configs'/name
    value=json.loads(path.read_bytes())
    differences=[]
    for relative,expected in value['sources'].items():
        actual=identity(root/relative)
        wanted={k:expected[k] for k in ('bytes','sha256')}
        if actual!=wanted:
            differences.append({'path':relative,'historical':wanted,'current':actual})
    result[name]={'original_protocol':identity(path),'source_count':len(value['sources']),
                  'differences':differences,'historical_protocol_changed':False}
value={'publication_source_context_actual_exit':1,'actual_terminal':'ff7df0','actual_session':20618,
       'source_context_script':identity(Path(__file__).parent/'inspect_source_context.py'),
       'first_failure':'File content identity differs: config.py',
       'source_binding_census':result,'historical_binding_replaced':False,'scientific_result_changed':False}
with (Path(__file__).parent/'source-binding-differences.json').open('x') as stream:
    json.dump(value,stream,indent=2,sort_keys=True)
print(json.dumps(value),flush=True)
