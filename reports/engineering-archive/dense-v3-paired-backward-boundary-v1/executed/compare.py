"""Authenticate paired GPU evidence and quantify exact post-reduction differences."""
import hashlib
import json
import math
import os
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import torch
from safetensors import safe_open
HERE = Path(__file__).resolve().parent

def identity(path):
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream,'sha256').hexdigest()
    return {'bytes':path.stat().st_size,'sha256':value}

def read(path):
    return json.loads(path.read_bytes())

result = {}
for pool in ('a','b'):
    job = HERE/'run'/('pool-'+pool)
    done = read(job/'completed.json')
    assert done['actual_rank_exits'] == [0]*4 and done['optimizer_updates'] == 0
    assert done['scope'] == 'dense-v3-four-rank-paired-backward-boundary-v2'
    pairs, colds = [], []
    for rank in range(4):
        d = job/f'rank-{rank}'
        completed, cold, pair = read(d/'completed.json'), read(d/'backward.json'), read(d/'paired.json')
        for filename,key in (('loaded.json','loaded'),('backward.json','backward'),('paired.json','paired')):
            assert identity(d/filename) == completed[key]
        for filename,key,record in (
            ('ddp-preclip.safetensors','preclip',cold),('local-leaf-sum.safetensors','local_sum',cold),
            ('warm-ddp-preclip.safetensors','warm_preclip',pair),('warm-local-leaf-sum.safetensors','warm_local_sum',pair)):
            assert identity(d/filename) == record[key]
        assert pair['same_weights_optimizer_scheduler'] and pair['same_first_input_rng'] and pair['exact_token_features']
        assert pair['native_second_pass_item_count_is_none'] and pair['optimizer_updates'] == 0
        assert cold['local_gradient_event_fingerprints'] == pair['warm_gradient_event_fingerprints']
        assert len(cold['local_gradient_event_fingerprints']) == 134
        assert all(len(v)==4 for v in cold['local_gradient_event_fingerprints'].values())
        assert pair['local_gradient_changed_parameters'] == []
        with safe_open(d/'local-leaf-sum.safetensors',framework='pt') as x, safe_open(d/'warm-local-leaf-sum.safetensors',framework='pt') as y:
            assert x.keys() == y.keys()
            assert all(torch.equal(x.get_tensor(k).reshape(-1).view(torch.uint8),y.get_tensor(k).reshape(-1).view(torch.uint8)) for k in x.keys())
        pairs.append(pair)
        colds.append(cold)
    assert all(v['ddp_gradient_fingerprints']==colds[0]['ddp_gradient_fingerprints'] for v in colds)
    assert all(v['warm_ddp_gradient_fingerprints']==pairs[0]['warm_ddp_gradient_fingerprints'] for v in pairs)
    rows = []
    with safe_open(job/'rank-0/ddp-preclip.safetensors',framework='pt') as a, safe_open(job/'rank-0/warm-ddp-preclip.safetensors',framework='pt') as b:
        assert a.keys() == b.keys() and len(a.keys()) == 134
        for name in a.keys():
            x,y = a.get_tensor(name),b.get_tensor(name)
            assert x.dtype==y.dtype==torch.float32 and x.shape==y.shape
            delta = x.double()-y.double()
            rows.append({'name':name,'elements':x.numel(), 'different_elements':int(torch.count_nonzero(x!=y)),
                'max_abs_difference':float(delta.abs().max()),'squared_difference':float(delta.square().sum()),
                'cold_squared':float(x.double().square().sum()),'warm_squared':float(y.double().square().sum())})
    changed = [v['name'] for v in rows if v['different_elements']]
    assert sorted(changed) == sorted(pairs[0]['post_ddp_changed_parameters'])
    value = {'case':pool,'parameter_tensors':len(rows),'changed_tensors':len(changed),
        'elements':sum(v['elements'] for v in rows),'different_elements':sum(v['different_elements'] for v in rows),
        'max_abs_difference':max(v['max_abs_difference'] for v in rows),
        'relative_l2_difference':math.sqrt(sum(v['squared_difference'] for v in rows)/sum(v['cold_squared'] for v in rows)),
        'all_local_leaf_contributions_bit_equal':True,'all_local_cpu_sums_bit_equal':True,
        'both_passes_post_ddp_equal_across_ranks':True,
        'same_model_optimizer_scheduler_input_rng':True,
        'cold_rebuilt':colds[0]['ddp']['has_rebuilt_buckets'],
        'warm_rebuilt':pairs[0]['warm_ddp']['has_rebuilt_buckets'],
        'warm_bucket_sizes':pairs[0]['warm_ddp'].get('rebuilt_bucket_sizes'),
        'parameters':rows, 'difference_localized_after_leaf_contributions_in_this_pair':True,
        'original_endpoint_failure_fully_explained':False,'exact_endpoint_resume_accepted':False,
        'new_tolerance_defined':False,'scientific_completion':False}
    with (job/'comparison.json').open('x') as out:
        json.dump(value,out,indent=2,sort_keys=True,allow_nan=False)
    result[pool] = {k:v for k,v in value.items() if k!='parameters'}
    print(json.dumps(result[pool]),flush=True)
with (HERE/'comparison.json').open('x') as out:
    json.dump(result,out,indent=2,sort_keys=True,allow_nan=False)
