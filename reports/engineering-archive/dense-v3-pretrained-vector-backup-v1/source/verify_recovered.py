"""Read an authenticated pretrained-vector snapshot, without a model or original host."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

PROTOCOL = '4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b'
PLAN = 'e41938b9b4b0607872465fbdea532e9b3a8ad5ed63fe52f80fc7e986ec503c01'
SOURCE = '3b02f1d8486f3c2b41d0403cbf291580d0ca32868b8b29cb05514a22770257f5'
AUTH = 'd72527595e2c963eae7fd46b0a1de2bd83d1ea15a863ae467a1065cfc5b2f336'
ORIGINALS = {
    'vectors/vectors.npz': (6208554, 'e57057107312363619ebb1ba19b55892fe539568d9545726f3c9b9277325112f'),
    'vectors/manifest.json': (217494, '90d9f044b53c05065b00d16a53a31a5c82a37d86e33552d797f6c38364e64fac'),
    'actual-payload-readback.json': (2770, '4c2456fe282616c2b90870c0b3a5c0033ed31cd4d9c3de1a1a7f5af68f57bf53'),
    'native/pretrained.admission.json': (1328, '61d8ee12eb6728ddaf89b50d165a7197ea38aa3de587a04d30a5bd502b2e6b23'),
    'native/pretrained.started.json': (926, '2da0ecee9ab23c2dcd87f77e41587c6c24d3285aafdc52b95f8c289cda4c3722'),
    'native/pretrained.exited.json': (793, '8a82b1f85a96e67f07e7e03ef9ae457bebed7b3c9d5e463d798bd504ae9b4e76'),
    'native/pretrained.encoded.json': (1594, 'd668916f56dfd7c4f8ee35f6f5f189a4f397e7c08f09456e0bca4e4d477d32c5'),
    'native/pretrained.verified.json': (1441, '8867643a9309af31b73e9b4f6bd9b3b8329389bb687b1e967d5043482cdd1dcb'),
}
TASKS = ('ArguAna', 'ClimateFEVER', 'DBPedia', 'FEVER', 'FiQA2018', 'HotpotQA',
         'MSMARCO', 'NFCorpus', 'NQ', 'QuoraRetrieval', 'SCIDOCS', 'SciFact', 'TRECCOVID', 'Touche2020')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def identity(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            'Require ordinary files, without symlinked parents')
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'git_blob_sha1': hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()}


def bound(path, expected):
    got = identity(path)
    require(all(k in got and got[k] == v for k, v in expected.items()), 'Payload identity differs')
    return got


def read(path):
    identity(path)
    return json.loads(Path(path).read_bytes())


def check_originals(root):
    for name, (size, sha) in ORIGINALS.items():
        bound(root / name, {'bytes': size, 'sha256': sha})
    m = read(root / 'vectors/manifest.json')
    plan = m['plan']
    require(m['status'] == 'complete' and canonical_sha(plan) == PLAN
            and plan['primary_protocol_sha256'] == PROTOCOL
            and plan['state']['cell'] == 'pretrained' and plan['state']['stage'] == 0
            and plan['scientific_completion'] is False, 'Wrong original encoding plan')
    rows = plan['probe']['row_identities']
    require(canonical_sha(rows) == plan['probe']['row_identities_sha256']
            and len(rows) == 224 and [r['position'] for r in rows] == list(range(224))
            and Counter(r['source'] for r in rows) == {t: 16 for t in TASKS}
            and all(len(r['candidate_ids_positive_first']) == 8 for r in rows), 'Probe identity differs')
    loading = m['observed_loading']
    require(loading['parameters_unchanged'] is True and loading['before'] == loading['after'],
            'Recorded model state changed during encoding')
    before = loading['before']
    require(before['saved_weights_matched'] is True and before['model_mode'] == 'eval'
            and before['saved_dtype_conversion'] == 'bfloat16' and before['max_length'] == 8192
            and len(before['state_tensors']) == 134
            and canonical_sha(before['state_tensors']) == before['state_tensors_sha256'],
            'Recorded loading provenance differs')
    native = {kind: read(root / f'native/pretrained.{kind}.json')
              for kind in ('admission', 'started', 'exited', 'encoded', 'verified')}
    a, s, e, enc, v = (native[k] for k in ('admission', 'started', 'exited', 'encoded', 'verified'))
    require(all(x['cell'] == 'pretrained' for x in native.values()), 'Wrong native cell')
    require(all(x['authorization_sha256'] == AUTH for x in (a, s, e, enc))
            and all(x['source_sha256'] == SOURCE for x in (a, s, enc))
            and a['plan_sha256'] == enc['plan_sha256'] == PLAN, 'Native authority binding differs')
    require(all(s[k] == e[k] for k in ('pid', 'ppid', 'start_ticks', 'command'))
            and (s['pid'], s['ppid'], s['start_ticks']) == (325637, 42916, 306837451)
            and e['exit_code'] == v['actual_exit_code'] == 0, 'Original exit provenance differs')
    command = ['/usr/bin/python', '-B', '/root/embedding-optimizer-v3-experiment/launch/functional-dimensions/dispatch.py',
               '--source-sha', SOURCE, '--authorization-sha', AUTH, '--worker', 'pretrained',
               '--plan-sha', PLAN, '--gpu-token', '0', '--lease-fd', '8', '--lease-fd', '9']
    require(s['command'] == command and a['both_lease_namespaces_inherited'] is True
            and s['both_lease_namespaces_inherited'] is True
            and enc['both_lease_namespaces_inherited'] is True, 'Original command or leases differ')
    require(enc['saved'] == v['saved'] and v['native_readback_passed'] is True
            and enc['saved']['array_metadata'] == m['array_metadata']
            and enc['saved']['output'] == m['output'], 'Native saved/readback binding differs')
    require(m['output']['path'] == 'vectors.npz', 'Unexpected output path')
    bound(root / 'vectors/vectors.npz', {k: m['output'][k] for k in ('bytes', 'sha256')})
    bound(root / 'vectors/manifest.json', {k: v['saved']['manifest'][k] for k in ('bytes', 'sha256')})
    bound(root / 'native/pretrained.encoded.json', {k: v['worker_receipt'][k] for k in ('bytes', 'sha256')})
    proof = read(root / 'actual-payload-readback.json')
    require(proof['encoded_and_native_verified_states'] == 1 and proof['expected_states'] == 61
            and proof['feature_states'] == 0 and proof['original_native_save_readback_passed'] is True
            and proof['coordinator_failed_before_second_worker'] is True
            and proof['scientific_completion'] is False, 'Original partial-campaign status differs')
    for field, name in [('raw_vectors', 'vectors/vectors.npz'), ('vector_manifest', 'vectors/manifest.json'),
                        *[(k, f'native/pretrained.{k}.json') for k in ('started', 'exited', 'encoded', 'verified')]]:
        bound(root / name, {k: proof[field][k] for k in ('bytes', 'sha256')})
    with np.load(root / 'vectors/vectors.npz', allow_pickle=False) as arrays:
        require(set(arrays.files) == set(m['array_metadata']) and len(arrays.files) == 4,
                'Wrong NPZ array inventory')
        for name in arrays.files:
            value = arrays[name]
            require({'shape': list(value.shape), 'dtype': str(value.dtype)} == m['array_metadata'][name],
                    'Array shape or dtype differs')
            if name.endswith('_embeddings'):
                require(value.dtype == np.float32 and np.isfinite(value).all()
                        and np.all(np.linalg.norm(value.astype(np.float64), axis=-1) > 0),
                        'Nonfinite or zero-norm embedding')
        require(arrays['sample_ids'].tolist() == [r['sample_id'] for r in rows]
                and len(np.unique(arrays['sample_ids'])) == 224
                and arrays['sample_groups'].tolist() == [r['source'] for r in rows], 'Array row ordering differs')
    return {'accepted_vector_states': 1, 'expected_campaign_states': 61, 'feature_states': 0,
            'probe_queries': 224, 'candidates_per_query': 8, 'embedding_dimensions': 768,
            'original_worker_exit_code': 0, 'array_metadata': m['array_metadata'],
            'original_worker_reexecuted': False, 'features_recomputed': False,
            'original_failure_reconciled_or_restarted': False, 'scientific_completion': False}


def verify(root, manifest_sha):
    root = Path(root)
    require(isinstance(manifest_sha, str) and len(manifest_sha) == 64, 'External manifest anchor required')
    bound(root / 'artifact_manifest.json', {'sha256': manifest_sha})
    m = read(root / 'artifact_manifest.json')
    require(m['scope'] == 'accepted_pretrained_functional_vectors_data_only'
            and m['primary_protocol_sha256'] == PROTOCOL
            and m['source_code_included'] is False and m['scientific_completion'] is False
            and m['accepted_vector_states'] == 1 and m['feature_states'] == 0,
            'Wrong backup scope')
    require(set(m['files']) == set(ORIGINALS) | {'README.md'}, 'Wrong snapshot payload population')
    for name, expected in m['files'].items():
        p = PurePosixPath(name)
        require(p.as_posix() == name and not p.is_absolute() and '..' not in p.parts, 'Unsafe path')
        bound(root / name, expected)
    require(not any(p.is_symlink() for p in root.rglob('*')), 'Symlink in recovered tree')
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
            == set(m['files']) | {'artifact_manifest.json'}, 'Extra or missing recovered file')
    return {'snapshot_files': 10, 'manifest_sha256': manifest_sha, **check_originals(root)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.manifest_sha256), sort_keys=True))
