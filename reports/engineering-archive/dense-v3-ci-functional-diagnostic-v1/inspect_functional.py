"""Read-only diagnostic of the original functional computation; never an acceptance override."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

bundle=Path(sys.argv[1]).resolve()
output=Path(sys.argv[2]).resolve()
assert not output.exists() and not output.is_relative_to(bundle)
spec=importlib.util.spec_from_file_location('_original_primary_replay',bundle/'replay.py')
original=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=original
spec.loader.exec_module(original)
original.authenticate(bundle,'88861e52218f56b860851489106cd30e35dd79bb038eb9082967e3376db6457d')
assert not any(n=='embed_optim' or n.startswith('embed_optim.') for n in sys.modules)
sys.path.insert(0,str(bundle/'src'))
from embed_optim import dimension_inference as dimension, primary_v3_exact_bridge as bridge
from embed_optim.primary_contract import canonical
import numpy as np
import torch
torch.set_num_threads(1)
np.show_runtime()
catalog=original.SavedRecipeCatalog(original.read(bundle/'inputs/recipe-catalog.json'))
features={n:bridge.typed_csv(bundle/'inputs/functional/features'/(n+'.csv')) for n in ['checkpoint_summary','task_summary','random_removal','rotation_summary']}
tables,actual=dimension.summarize(catalog,features,original.read(bundle/'inputs/functional/original_panel.json'),
    original.read(bundle/'inputs/functional/scientific_protocol.json'),sorted(catalog.payload['evaluation']['tasks']))
expected=original.read(bundle/'inputs/functional/decisions.json')
expected_tables=original.read(bundle/'inputs/scientific-publication-v3/exact-publication/tables.json')['functional']
diffs=[]
def compare(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict) and set(a)==set(b):
        for key in sorted(a):compare(a[key],b[key],path+'/'+key)
    elif isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
    elif canonical(a)!=canonical(b):
        row={'path':path,'actual':a,'expected':b}
        if isinstance(a,float) and isinstance(b,float):row['absolute_difference']=abs(a-b)
        diffs.append(row)
compare(actual,expected)
assert all(Path(m.__file__).resolve().is_relative_to(bundle/'src') for n,m in sys.modules.items() if n=='embed_optim' or n.startswith('embed_optim.'))
receipt={'scope':'diagnostic-only-original-functional-reconstruction','acceptance_override':False,
         'openblas_coretype':os.environ.get('OPENBLAS_CORETYPE'),
         'functional_tables_exact':canonical(tables)==canonical(expected_tables),
         'decisions_exact':canonical(actual)==canonical(expected),'differences':diffs,'actual_decisions':actual,
         'original_decisions_sha256':hashlib.sha256((bundle/'inputs/functional/decisions.json').read_bytes()).hexdigest()}
with output.open('x') as stream:json.dump(receipt,stream,indent=2,sort_keys=True,allow_nan=False)
print(json.dumps({k:v for k,v in receipt.items() if k not in ['actual_decisions','differences']}))
print('differences',len(diffs))
print(json.dumps(diffs[:5]))
