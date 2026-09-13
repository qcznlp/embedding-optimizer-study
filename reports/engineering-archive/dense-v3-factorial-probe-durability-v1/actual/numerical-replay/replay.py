"""Replay all actual continuation-probe metrics using copied sources and HF data.

Only two unchanged pure numerical functions are extracted. No producer-directory
fallback, model import/encoding, native checkpoint admission, GPU, or BEIR data.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BINDINGS_SHA = '226feb59c1f50002d37a4a711e499e5457957983ae0cc89f11980c411ec6369c'


def need(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Ordinary input required')
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def sources():
    need(identity(HERE / 'source-bindings.json')['sha256'] == BINDINGS_SHA, 'Copied-source bindings changed')
    records = json.loads((HERE / 'source-bindings.json').read_text())
    need(set(records) == {'restore_factorial_probes.py', 'representation_geometry.py', 'state_operator_factorial_probe.py'},
         'Incomplete copied numerical sources')
    for name, value in records.items():
        need(identity(HERE / 'source' / name) == {k: value[k] for k in ('bytes', 'sha256')}, 'Copied source changed')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('PYTHONPATH', '') == '',
         'Hide CUDA and use no producer Python path')
    need(args.snapshot.is_absolute() and args.output.is_absolute() and not args.output.exists(),
         'Use an absolute downloaded snapshot and a new receipt')
    bound = sources()
    spec = importlib.util.spec_from_file_location('_copied_probe_snapshot_verifier', HERE / 'source/restore_factorial_probes.py')
    restore = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(restore)
    integrity = restore.verify(args.snapshot)
    checkpoints = restore.checkpoint_index(args.snapshot)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    namespace = {'torch': torch, 'Tensor': torch.Tensor, 'F': torch.nn.functional, 'math': math}
    for name, function in (('representation_geometry.py', 'dense_probe_scores'),
                           ('state_operator_factorial_probe.py', '_summary')):
        path = HERE / 'source' / name
        nodes = [n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == function]
        need(len(nodes) == 1 and not nodes[0].decorator_list, 'Original pure function differs')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    scorer, summarize = namespace['dense_probe_scores'], namespace['_summary']
    keys = {'query_embeddings', 'document_embeddings', 'sample_ids', 'sample_groups'}

    def arrays(path, dtype):
        with np.load(path, allow_pickle=False) as bundle:
            need(set(bundle.files) == keys, 'Wrong vector array population')
            result = {key: bundle[key] for key in bundle.files}
        need(result['query_embeddings'].shape == (224, 768)
             and result['document_embeddings'].shape == (224, 8, 768)
             and result['query_embeddings'].dtype == result['document_embeddings'].dtype == dtype
             and result['sample_ids'].dtype == np.int64 and result['sample_ids'].shape == (224,)
             and result['sample_groups'].shape == (224,) and len(set(result['sample_ids'].tolist())) == 224,
             'Incorrect actual vector shapes or dtypes')
        need(np.isfinite(result['query_embeddings']).all() and np.isfinite(result['document_embeddings']).all(),
             'Nonfinite probe vectors')
        return result

    reference = arrays(args.snapshot / 'pretrained/vectors-fp16.npz', np.dtype('float16'))
    groups = reference['sample_groups']
    tasks = sorted(set(groups.tolist()))
    need(len(tasks) == 14 and all(int((groups == task).sum()) == 16 for task in tasks), 'Wrong fixed task panel')
    reference_scores = scorer(torch.from_numpy(reference['query_embeddings']), torch.from_numpy(reference['document_embeddings']))
    labels = ['pretrained'] + [row['run_id'] + '/checkpoint-' + str(row['step']) for row in checkpoints]
    need(len(labels) == len(set(labels)) == 61, 'Incomplete or duplicate replay states')
    observations = []
    for label in labels:
        root = args.snapshot / label
        raw, stored = arrays(root / 'raw/vectors.npz', np.dtype('float32')), arrays(root / 'vectors-fp16.npz', np.dtype('float16'))
        for key in keys:
            expected = raw[key].astype(np.float16) if key in ('query_embeddings', 'document_embeddings') else raw[key]
            need(np.array_equal(stored[key], expected), 'Original FP32-to-FP16 data projection differs')
        need(np.array_equal(stored['sample_ids'], reference['sample_ids'])
             and np.array_equal(stored['sample_groups'], groups), 'Ordered probe identities differ')
        scores = scorer(torch.from_numpy(stored['query_embeddings']), torch.from_numpy(stored['document_embeddings']))
        with np.load(root / 'scores.npz', allow_pickle=False) as bundle:
            need(bundle.files == ['scores'] and bundle['scores'].dtype == np.float32
                 and np.array_equal(bundle['scores'], scores.numpy()), 'Original saved scores do not reproduce exactly')
        result = {'overall': summarize(scores, reference_scores),
            'by_task': {task: summarize(scores[groups == task], reference_scores[groups == task]) for task in tasks},
            'scoring': 'unchanged-frozen-factorial-six-metric-summary', 'score_dtype': 'float32',
            'stored_embedding_dtype': 'float16', 'ties': 'pessimistic-positive-rank; first-index-argmax-for-agreement',
            'temperature': .02, 'raw_fp32_vectors_retained': True, 'shortlist_candidates': 8, 'scientific_completion': False}
        need(result == json.loads((root / 'metrics.json').read_text()), 'All six original metric summaries do not reproduce exactly')
        observations.append({'state': label, 'all_scores_exact': True, 'all_six_metrics_exact': True,
            'raw_to_scoring_arrays_exact': True, 'groups': 15,
            'source_payloads': {name: identity(root / name) for name in ('raw/vectors.npz', 'vectors-fp16.npz', 'scores.npz', 'metrics.json')}})
    need(not torch.cuda.is_initialized() and not any(name == 'embed_optim' or name.startswith('embed_optim.') for name in sys.modules),
         'Replay imported a producer package or initialized CUDA')
    need(sources() == bound, 'Copied source changed during numerical replay')
    restore.verify(args.snapshot)
    result = {'scope': 'actual-anonymously-downloaded-continuation-probe-numerical-replay',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'source': identity(Path(__file__)),
        'source_bindings': identity(HERE / 'source-bindings.json'), 'snapshot_integrity': integrity,
        'actual_states': 61, 'continuation_checkpoints': 60, 'metric_summaries': 915,
        'exact_original_metric_values': 5490, 'observations': observations,
        'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'original_pure_functions_unchanged': True, 'producer_directory_fallback': False,
        'source_was_copied': True, 'snapshot_hf_revision': restore.REVISION,
        'anonymous_download_provenance': 'separate operational download receipt',
        'cross_host_equivalence_verified_here': False, 'native_model_provenance_revalidated_here': False,
        'model_encoding_repeated': False, 'gpu_resume_equivalence': False, 'beir_loaded': False,
        'scientific_completion': False}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'replayed_states': 61, 'exact_metric_values': 5490,
                      'receipt': identity(args.output), 'scientific_completion': False}), flush=True)


if __name__ == '__main__':
    main()
