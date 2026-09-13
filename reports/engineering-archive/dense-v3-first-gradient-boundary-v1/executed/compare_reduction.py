"""Describe actual DDP gradients relative to an independent FP64 rank mean.

No new equality tolerance or GPU-resume acceptance is defined. CPU sums of four
observed leaf contributions are not claimed to expose the internal DDP buckets.
"""
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
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': sha}


def read(path):
    return json.loads(path.read_bytes())


results = {}
for pool in ('a', 'b'):
    job = HERE/'run'/('pool-'+pool)
    done = read(job/'completed.json')
    assert done['scope'] == 'dense-v3-four-rank-first-backward-boundary-v1'
    assert done['actual_rank_exits'] == [0]*4 and done['optimizer_updates'] == 0
    manifests = [read(job/f'rank-{r}/backward.json') for r in range(4)]
    for rank, manifest in enumerate(manifests):
        path = job/f'rank-{rank}'
        completion = read(path/'completed.json')
        assert identity(path/'backward.json') == completion['backward']
        assert identity(path/'loaded.json') == completion['loaded']
        assert identity(path/'ddp-preclip.safetensors') == manifest['preclip']
        assert identity(path/'local-leaf-sum.safetensors') == manifest['local_sum']
        assert manifest['ddp_gradient_fingerprints'] == manifests[0]['ddp_gradient_fingerprints']
    local = [safe_open(job/f'rank-{r}/local-leaf-sum.safetensors', framework='pt', device='cpu') for r in range(4)]
    actual = safe_open(job/'rank-0/ddp-preclip.safetensors', framework='pt', device='cpu')
    names = list(actual.keys())
    assert len(names) == 134 and all(list(v.keys()) == names for v in local)
    rows = []
    sums = dict(reference_squared=0., actual_squared=0., error_squared=0., dot=0.)
    for name in names:
        values = [file.get_tensor(name).to(torch.float64) for file in local]
        reference = (values[0]+values[1]+values[2]+values[3])/4
        observed = actual.get_tensor(name).to(torch.float64)
        error = observed-reference
        row = dict(name=name, elements=observed.numel(),
                   max_abs_error=float(error.abs().max()),
                   reference_squared=float((reference*reference).sum()),
                   actual_squared=float((observed*observed).sum()),
                   error_squared=float((error*error).sum()),
                   dot=float((reference*observed).sum()))
        assert all(math.isfinite(row[k]) for k in sums)
        for key in sums:
            sums[key] += row[key]
        rows.append(row)
    result = dict(pool=pool, tensors=len(rows), elements=sum(r['elements'] for r in rows),
        reference_norm=math.sqrt(sums['reference_squared']), actual_norm=math.sqrt(sums['actual_squared']),
        max_abs_error=max(r['max_abs_error'] for r in rows),
        relative_l2_error=math.sqrt(sums['error_squared']/sums['reference_squared']),
        least_squares_scale=sums['dot']/sums['reference_squared'], parameters=rows,
        comparison='observed DDP preclip gradient versus FP64 average of CPU leaf-contribution sums',
        establishes_original_uninterrupted_gradient=False, defines_new_tolerance=False,
        exact_endpoint_resume=False, scientific_completion=False)
    with (job/'reduction-comparison.json').open('x') as out:
        json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
    results[pool] = {k:v for k,v in result.items() if k != 'parameters'}
    print(json.dumps(results[pool]), flush=True)
    del local, actual
with (HERE/'reduction-comparison.json').open('x') as out:
    json.dump(results, out, indent=2, sort_keys=True, allow_nan=False)
