"""Narrow saved-state loading and resumed sampler boundaries without GPU training."""
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import torch
from accelerate.data_loader import BatchSamplerShard, DataLoaderShard, SeedableRandomSampler, skip_first_batches
from embed_optim import factorial_v3_batches as batching
from embed_optim import factorial_v3_optimizer as optim
from embed_optim import saved_factorial_checkpoint as saved
from embed_optim.config import OptimizerConfig
from embed_optim.runtime import verify_runtime_spec
from safetensors.torch import load_file
from sentence_transformers import SentenceTransformer
from sentence_transformers.base.trainer import BaseTrainer

def write(name, value):
    with (HERE / name).open('x') as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def bit_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
            a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(bit_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(bit_equal(x, y) for x, y in zip(a, b))
    return a == b

def loader(seed, rank, workers=0):
    dataset = range(50000)  # Index-only diagnostic, not synthetic training data admission.
    sampler = SeedableRandomSampler(dataset, generator=torch.Generator().manual_seed(seed), data_seed=seed)
    batches = batching.FactorialBatchSampler(sampler)
    shard = BatchSamplerShard(batches, num_processes=4, process_index=rank,
                              split_batches=False, even_batches=False)
    kwargs = dict(num_workers=workers, persistent_workers=workers > 0, pin_memory=False)
    if workers:
        kwargs['prefetch_factor'] = 4
    return DataLoaderShard(dataset, batch_sampler=shard, device=torch.device('cpu'), **kwargs)

def order(seed, rank, skip=0, workers=0):
    data = loader(seed, rank, workers)
    if skip:
        data = skip_first_batches(data, skip)
    data.set_epoch(0)
    return [batch.tolist() for batch in data]

def sequence_sha(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()

runtime = verify_runtime_spec(ROOT / 'configs/formal_runtime.json')
rows = []
for seed in batching.ORDER_SEEDS:
    for rank in range(4):
        full = order(seed, rank)
        resumed = order(seed, rank, 313 * 4)
        assert len(full) == 1564 and len(resumed) == 312 and full[1252:] == resumed
        assert sum(map(len, resumed)) == 2484 and all(len(v) == 5 for v in resumed[-4:])
        rows.append(dict(seed=seed, rank=rank, workers=0, remaining_microbatches=len(resumed),
                         remaining_groups=sum(map(len, resumed)), remaining_order_sha256=sequence_sha(resumed)))
        print(json.dumps({'sampler_seed': seed, 'rank': rank, 'exact_remaining_order': True}), flush=True)
for rank in range(4):
    expected = order(314159, rank, 1252)
    actual = order(314159, rank, 1252, workers=8)
    assert actual == expected
    rows.append(dict(seed=314159, rank=rank, workers=8, remaining_order_sha256=sequence_sha(actual)))
assert order(314159, 0, 1251) != order(314159, 0, 1252)
assert order(271828, 0, 1252) != order(314159, 0, 1252)
write('sampler.json', {'cases': rows, 'off_by_one_and_wrong_seed_controls_distinguished': True,
    'actual_text_tokenization_or_GPU_gradients_tested': False})

reference = ROOT / 'reports/engineering-archive/dense-v3-portable-factorial-checkpoint-v1/original-download.json'
assert sha(reference) == '3c48d5c7d10434fe48925ec223e55c85df59e23ee6767a9d4bffbf1f5901f332'
cases = json.loads(reference.read_bytes())['cases']
models = []
for label in ('a', 'b'):
    case = cases[label]
    binding, identity, component, _ = saved.authenticated_component(
        case['binding']['path'], case['binding']['sha256'], ROOT / 'configs/formal_runtime.json')
    path = Path(binding['path'])
    # This is the same constructor used by the installed upstream checkpoint loader.
    # CUDA is hidden and networking is offline; no forward/backward is performed.
    model = SentenceTransformer(str(path), trust_remote_code=True)
    expected_model = load_file(path / 'model.safetensors', device='cpu')
    assert len(expected_model) == 134 and bit_equal(dict(model[0].model.state_dict()), dict(expected_model))
    # Start from different values, then invoke the exact upstream loader, not a hand-copied implementation.
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    BaseTrainer._load_from_checkpoint(SimpleNamespace(model=model), str(path))
    assert bit_equal(dict(model[0].model.state_dict()), dict(expected_model))
    config = OptimizerConfig(**identity['optimizer'])
    optimizer = optim.FactorialOptimizer(model, config)
    scheduler = optim.create_scheduler(optimizer)
    saved_optimizer = torch.load(path / 'optimizer.pt', map_location='cpu', weights_only=True)
    saved_scheduler = torch.load(path / 'scheduler.pt', map_location='cpu', weights_only=True)
    optim.restore_training_state(optimizer, scheduler, saved_optimizer, saved_scheduler, step=313)
    assert bit_equal(optimizer.state_dict(), saved_optimizer)
    assert bit_equal(scheduler.state_dict(), saved_scheduler)
    assert optimizer.completed_steps == 313 and not torch.cuda.is_initialized()
    saved.checkpoints.read(binding, component)
    models.append({'case': label, 'component_sha256': binding['sha256'], 'step': 313,
                   'all_134_loaded_model_tensors_bit_exact': True,
                   'upstream_loader_after_zeroing_bit_exact': True,
                   'all_optimizer_and_scheduler_state_bit_exact_on_cpu': True,
                   'gpu_loaded_state_or_gradients_tested': False})
    write('loaded-' + label + '.json', models[-1])
    print(json.dumps(models[-1]), flush=True)
    del model, expected_model, optimizer, scheduler, saved_optimizer, saved_scheduler
    gc.collect()

assert not torch.cuda.is_initialized()
sources = {name: sha(module.__file__) for name, module in sys.modules.items()
           if name in ('embed_optim.factorial_v3_batches', 'embed_optim.factorial_v3_optimizer',
                       'embed_optim.saved_factorial_checkpoint', 'accelerate.data_loader',
                       'transformers.trainer', 'sentence_transformers.base.trainer')}
write('completed.json', {'scope': 'dense-v3-resume-cpu-boundary-diagnosis-v1', 'runtime': runtime,
    'source_sha256': sources, 'sampler_cases': len(rows), 'loaded_checkpoints': models,
    'scientific_results_changed': False, 'gpu_initialized': False,
    'exact_gpu_resume_accepted': False, 'full_cause_localized': False})
