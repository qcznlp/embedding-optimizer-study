"""Read genuine continuation completion in its unchanged training source namespace.

CPU-only artifact admission for a separate evaluator. No GPU lease, worker
launch, numerical modification, historical migration or source publication.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

TRAIN = Path('/root/embedding-optimizer-v3-experiment/launch/factorial-training-v1')
SOURCE_SHA = 'd11d3281523b1573ce7fcd650e10c538cfa84022b5db92117559d47213e57e53'
AUTH_SHA = '00c52f0bec979fb163a734ad0f9eadcd6ea09f7393133fd35c65bc62878ce11f'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--pool', choices=('a', 'b'), required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != a.source_sha256:
        raise ValueError('Evaluation reader source changed')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not a.output.is_absolute() or a.output.exists():
        raise ValueError('Require CPU-only new absolute reader output')
    path = TRAIN / 'factorial_dispatch.py'
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Original training entry changed')
    spec = importlib.util.spec_from_file_location('_unchanged_factorial_training_admission', path)
    entry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = entry
    spec.loader.exec_module(entry)
    auth, _ = entry.authenticate(SimpleNamespace(source_sha=SOURCE_SHA, authorization_sha=AUTH_SHA))
    entry.need(a.run_id in auth['queues'][a.pool], 'Wrong run/pool')
    directory = TRAIN / 'run' / ('pool-' + a.pool) / a.run_id
    completion_id = entry.identity(directory / 'completed.json')
    completion = entry.read(directory / 'completed.json')
    entry.need(completion['run_id'] == a.run_id and completion['source_sha256'] == SOURCE_SHA
        and completion['authorization_sha256'] == AUTH_SHA and completion['actual_rank_exits'] == [0, 0, 0, 0]
        and completion['actual_fresh_reader_exit'] == 0 and completion['full_horizon_391_steps_verified'] is True
        and completion['all_five_checkpoints_verified'] is True, 'Not a complete actual continuation')
    old = entry.read(directory / 'fresh-native-readback.json', completion['native_readback_binding']['sha256'])
    entry.need(entry.identity(directory / 'fresh-native-readback.json') == completion['native_readback_binding']
        and old['worker_completion'] == completion['worker_completion'], 'Original fresh reader differs')
    entry.need(entry.read(directory / 'ranks.exited.json')['actual_rank_exits'] == {str(i): 0 for i in range(4)}
        and entry.read(directory / 'reader.exited.json')['exit_code'] == 0, 'Original actual process exits differ')
    _, native, locations = entry.context(auth['source_files'])
    request = auth['requests'][a.run_id]
    checked = native.read_execution(locations, request, completion['worker_completion'])
    entry.need(checked == old['native'], 'Fresh original numerical artifact read differs')
    artifacts = checked['native_readback']['artifacts']
    entry.need([r['step'] for r in artifacts['checkpoints']] == [79, 157, 235, 313, 391]
        and artifacts['optimizer_steps'] == 391 and artifacts['whole_run_artifacts_verified'] is True,
        'Incomplete actual checkpoint horizon')
    cp = Path(request['run_root']) / 'checkpoint-391'
    files = artifacts['final_inference_files']
    entry.need(len(files) == 8 and len({r['path'] for r in files}) == 8, 'Unexpected inference input inventory')
    for row in files:
        entry.need(entry.identity(cp / row['path']) == {k: row[k] for k in ('bytes', 'sha256')},
                   'Final checkpoint differs from native inference files')
    entry.c.check_imports(auth['source_files'])
    entry.need(entry.source_files() == auth['source_files'] and entry.identity(directory / 'completed.json') == completion_id,
               'Source or completion changed while reading')
    import torch
    entry.need(not torch.cuda.is_initialized(), 'CPU admission initialized CUDA')
    record = dict(scope='actual-factorial-v3-evaluation-input-admission', verified_at_utc=entry.now(),
        run_id=a.run_id, pool=a.pool, checkpoint=str(cp), checkpoint_step=391,
        state=request['state'], operator=request['operator'], seed=request['seed'],
        training_source_sha256=SOURCE_SHA, training_authorization_sha256=AUTH_SHA,
        actual_completion=completion_id, original_fresh_readback=completion['native_readback_binding'],
        actual_rank_exits=[0, 0, 0, 0], actual_original_reader_exit=0,
        native_readback=checked, inference_files=files,
        component_file=entry.identity(cp / 'factorial_trainer_component.json'),
        reader_source=entry.identity(Path(__file__)), numerical_sources_unchanged=True,
        formal_historical_factorial_admission=False, source_release=False, scientific_completion=False)
    entry.write_new(a.output, record)
    print(json.dumps(dict(run_id=a.run_id, output=entry.identity(a.output), actual_native_run_verified=True,
                         model_files_verified=8, gpu_access=False, scientific_completion=False)), flush=True)


if __name__ == '__main__':
    main()
