"""Standalone verification of 39 original training-completion data records."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

PROTOCOL = '4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b'
ORIGINALS = json.loads(r'''{
  "anchors/evaluation-authorization.json": [
    6588,
    "2351e225387f42fe607d7e002a0717de79a78cba9f3c93f66652510083f0be7c"
  ],
  "anchors/input-view-audit.json": [
    4463,
    "13cb27173e48b8262aae77cba6ed5ebb03d9839f10fec409af06beac570af473"
  ],
  "anchors/source-assembly.json": [
    15970,
    "e603ef175c0f691a1dd83c4d8b474d90da2fc67c8c65f5a77f3c3180e78332b8"
  ],
  "receipts/verified-v3-adamw-1e-5.exited.json": [
    112,
    "b4553d65aa50f6ccb1a7169f28a7b64306471d53d92d41c1849b0d46b2703459"
  ],
  "receipts/verified-v3-adamw-1e-5.proof.json": [
    23764,
    "ce708b79780804b074e0c0fceea853a7e5d02fb31160e671693e01bb26b377af"
  ],
  "receipts/verified-v3-adamw-1e-5.started.json": [
    1086,
    "4c23c456dd9e926039c5a3516558daa6584a7384d19cc08c50764b6f93d34fad"
  ],
  "receipts/verified-v3-adamw-1e-6.exited.json": [
    112,
    "dec4a3e25f9d1f8e3db245510016f0520271f9f3ce1cd7f334261796311b9eee"
  ],
  "receipts/verified-v3-adamw-1e-6.proof.json": [
    23764,
    "86a3d6c1b247919f027aabcf5cb70ea0eaff39082c9198d3a05ec1f74b5ba565"
  ],
  "receipts/verified-v3-adamw-1e-6.started.json": [
    1086,
    "bcec21a5f878a64607df78b1446557d4a6f3df0044bb3a65c17849abe529a3b3"
  ],
  "receipts/verified-v3-adamw-3e-5.exited.json": [
    112,
    "0bc251d736cfadfd5f3a46e397ff940d8537345d786d03f00a6d72925a184746"
  ],
  "receipts/verified-v3-adamw-3e-5.proof.json": [
    23764,
    "4035f4c714a959a47d13ca5f027d39938e357606a983069023bf91cceae1885c"
  ],
  "receipts/verified-v3-adamw-3e-5.started.json": [
    984,
    "bfcf1a4c27ac014feddbb5003214ae817e19156072a0bb0fc7c9e0c1cb038308"
  ],
  "receipts/verified-v3-adamw-3e-6.exited.json": [
    112,
    "a168fe009591064db5743a799073c868ad9d32a03bffca0205402a89055b9568"
  ],
  "receipts/verified-v3-adamw-3e-6.proof.json": [
    23763,
    "9f06c88cbd2894c3d3c422ec9a1c2ac87c194179fe5f011ac7280b614da58ab1"
  ],
  "receipts/verified-v3-adamw-3e-6.started.json": [
    1086,
    "b622349c82f06574950369045572afd8ed69cc1d3724a18c6bf26513e22c163d"
  ],
  "receipts/verified-v3-muon-1e-3.exited.json": [
    111,
    "6d9c37d7ed0fce775bac2cd630ba6af756df71691be2b551ef9624294388987e"
  ],
  "receipts/verified-v3-muon-1e-3.proof.json": [
    23807,
    "e45ec40da17dd471ffccad8e1da5898a2f9baeeff948caf12e6ae9a502026f78"
  ],
  "receipts/verified-v3-muon-1e-3.started.json": [
    1084,
    "5f04d1e094794c513c33860d69edae0d03bea7f875f9a1e1cc5eca3ff9b59425"
  ],
  "receipts/verified-v3-muon-1e-4.exited.json": [
    111,
    "a1ec5d10ff14fe824381ef280482e1436f0c907d1636f19f909231e753539744"
  ],
  "receipts/verified-v3-muon-1e-4.proof.json": [
    23808,
    "e1859a94408f08c443c8babead5a8fb9fec6899aa34a837c3b817a501fdbe252"
  ],
  "receipts/verified-v3-muon-1e-4.started.json": [
    1084,
    "e472fb5ab7ec54677540b0c3706e49cb0ae8c4bd544aab1f1a272a2cf9039402"
  ],
  "receipts/verified-v3-muon-3e-3.exited.json": [
    111,
    "819095d6aa8c761841d49143a95daf86677ce490d58ac6a201ebece0080ac4a8"
  ],
  "receipts/verified-v3-muon-3e-3.proof.json": [
    23808,
    "25cd6b8b45a457d7761d446f11163862c82cd3c4b341d3eec6e80d3c6bdfaba0"
  ],
  "receipts/verified-v3-muon-3e-3.started.json": [
    1084,
    "ff06cf017f219575ce799fae80f16da1731e8b1622a5825f18b0b9531d41f315"
  ],
  "receipts/verified-v3-muon-3e-4.exited.json": [
    111,
    "dd731e0a02e35679de3f93a83c6211803ee2ae539b5c85c8d1ec4fed82a0b5f0"
  ],
  "receipts/verified-v3-muon-3e-4.proof.json": [
    23808,
    "5845f0e3129de0cee0f0706d2f53ab18f7758f23cc2f2cd4e0bf03c673e4a57d"
  ],
  "receipts/verified-v3-muon-3e-4.started.json": [
    982,
    "e7cdb0d008e7deb5122013f65b9b9879d83fbe01c61fb26b2e4336748e138699"
  ],
  "receipts/verified-v3-normuon-1e-3.exited.json": [
    114,
    "3cdbb71c67d647d81525e7143a290b10989bf919208b662aede72514efa3d308"
  ],
  "receipts/verified-v3-normuon-1e-3.proof.json": [
    23811,
    "1ec11b0f47f919031a32ca731dd0bbac5dde9bdbb20e6bcf89ef4dbcd35507e3"
  ],
  "receipts/verified-v3-normuon-1e-3.started.json": [
    1090,
    "a52ac1abec7509f988d45e1ba8ddfd1ea5880b027ae330a36b115dbd68250966"
  ],
  "receipts/verified-v3-normuon-1e-4.proof.json": [
    25230,
    "97217c4a2394d44b82c49c8c0b0d2dddfe306f264c09a423d91004c930594241"
  ],
  "receipts/verified-v3-normuon-1e-4.started.json": [
    1090,
    "98ff6ebf287b6bb177b4949a0a2cf58f4c80dc990c0528719b2cef1b4ecf6ca4"
  ],
  "receipts/verified-v3-normuon-1e-4.terminated.json": [
    745,
    "1d9dda131c2e1e150bfc98f45786f034c2acd57a5c7755f1cb63d9673dc113c2"
  ],
  "receipts/verified-v3-normuon-3e-3.proof.json": [
    25232,
    "6b52984def4b11732a934df00c9d400b1a0923fb44948f567a2642e4dce52b44"
  ],
  "receipts/verified-v3-normuon-3e-3.started.json": [
    1090,
    "8eec4b13a049ce62b8cc66ea520ab93e6fc906e570dfbd550277dd7b1e77ea4a"
  ],
  "receipts/verified-v3-normuon-3e-3.terminated.json": [
    745,
    "758e999d2fc495247a4a02bd17b4d81d208c9e85ba60f22443ed90015d58d246"
  ],
  "receipts/verified-v3-normuon-3e-4.exited.json": [
    114,
    "a4bd16cdef60231f291b051d2f8133cb98ed2ec4a8a14c215c5f27196a3328a0"
  ],
  "receipts/verified-v3-normuon-3e-4.proof.json": [
    23812,
    "ee4031e2a49b66e1fd5938acd2d3acc5df79eb7cc9573caed6fd19d737bc2535"
  ],
  "receipts/verified-v3-normuon-3e-4.started.json": [
    1090,
    "7ac1159825f2e5db0b4c3bc5a28d40f079667fdede5b6d9e32fa7c4b9c721696"
  ]
}''')

def require(ok, message):
    if not ok:
        raise ValueError(message)

def bound(path, expected):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts
            and not any(p.is_symlink() for p in (path, *path.parents))
            and path.is_file(), 'Require an ordinary absolute input')
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            'Input changed while reading')
    identity = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    require(all(identity[k] == v for k, v in expected.items()), 'Original byte binding differs')
    return identity

def unique(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, 'Duplicate JSON key')
        out[key] = value
    return out

def read(path, expected):
    bound(path, expected)
    raw = Path(path).read_bytes()
    actual = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    require(all(actual[k] == v for k, v in expected.items()), 'Input changed before JSON decoding')
    value = json.loads(raw, object_pairs_hook=unique)
    json.dumps(value, allow_nan=False)
    return value

def names(root):
    root = Path(root)
    require(root.is_absolute() and root.is_dir() and '..' not in root.parts
            and not any(p.is_symlink() for p in (root, *root.parents)), 'Invalid recovery root')
    out = set()
    for p in root.rglob('*'):
        require(not p.is_symlink() and (p.is_file() or p.is_dir()), 'Symlink or special entry')
        if p.is_file():
            out.add(p.relative_to(root).as_posix())
    return out

def check_originals(root):
    root = Path(root)
    for name, (size, sha) in ORIGINALS.items():
        bound(root / name, {'bytes': size, 'sha256': sha})
    def original(name):
        size, sha = ORIGINALS[name]
        return read(root / name, {'bytes': size, 'sha256': sha})
    proofs = {name.removeprefix('receipts/').removesuffix('.proof.json'): original(name)
              for name in ORIGINALS if name.endswith('.proof.json')}
    require(len(proofs) == 12, 'Wrong original run population')
    for run, proof in proofs.items():
        require(proof['run_id'] == run and proof['protocol_sha256'] == PROTOCOL
                and proof['whole_run_artifacts_verified'] is True
                and proof['explicit_two_selection_view_audit_passed'] is True
                and proof['original_single_selection_guard_passed'] is False
                and proof['committed_source_release'] is False
                and proof['scientific_completion'] is False
                and proof['steps'] == [782, 1563, 2345, 3126, 3907],
                'Original completion proof differs')
        require(original(f'receipts/{run}.started.json')['run_id'] == run, 'Wrong original start')
        if run in ('verified-v3-normuon-1e-4', 'verified-v3-normuon-3e-3'):
            term = original(f'receipts/{run}.terminated.json')
            require(proof['scope'] == 'dense_primary_v3_orphan_worker_artifact_completion'
                    and term == proof['process_observation']
                    and term['os_exit_code_observed'] is False
                    and term['training_exit_code'] is None
                    and term['process_termination_observed_via_pidfd'] is True,
                    'Do not fabricate unobserved OS exits')
        else:
            exited = original(f'receipts/{run}.exited.json')
            require(proof['scope'] == 'dense_primary_v3_explicit_view_history_completion'
                    and exited['run_id'] == run and type(exited['exit_code']) is int
                    and exited['exit_code'] == 0, 'Original observed exit differs')
    return proofs

def verify(root, manifest_sha256):
    require(isinstance(manifest_sha256, str) and re.fullmatch('[0-9a-f]{64}', manifest_sha256),
            'Require external manifest SHA-256')
    root = Path(root)
    first = names(root)
    require(first == set(ORIGINALS) | {'README.md', 'artifact_manifest.json'}, 'Wrong exact file population')
    directories = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_dir()}
    wanted_dirs = {p.as_posix() for n in first for p in PurePosixPath(n).parents if p.as_posix() != '.'}
    require(directories == wanted_dirs, 'Unexpected empty directory')
    bound(root / 'artifact_manifest.json', {'sha256': manifest_sha256})
    manifest = read(root / 'artifact_manifest.json', {'sha256': manifest_sha256})
    require(manifest['scope'] == 'original_primary_training_completion_evidence_data_only'
            and manifest['primary_protocol_sha256'] == PROTOCOL
            and manifest['training_runs'] == 12 and manifest['checkpoint_states'] == 60
            and manifest['original_exit_zero_runs'] == 10
            and manifest['original_exit_unobserved_runs'] == 2
            and manifest['source_code_included'] is False and manifest['scientific_completion'] is False,
            'Backup scope changed')
    require(set(manifest['files']) == set(ORIGINALS) | {'README.md'}, 'Wrong manifest population')
    for name, ident in manifest['files'].items():
        require(set(ident) == {'bytes','sha256'}, 'Wrong identity schema')
        bound(root / name, ident)
    for name, (size, sha) in ORIGINALS.items():
        require(manifest['files'][name] == {'bytes':size,'sha256':sha}, 'Original anchor changed')
    check_originals(root)
    require(names(root) == first, 'Recovery tree changed')
    return {'scope':'original-training-completion-data-readback','files':41,'original_records':39,
            'training_runs':12,'checkpoint_states':60,'original_exit_zero_runs':10,
            'original_exit_unobserved_runs':2,'source_code_included':False,'scientific_completion':False,
            'metadata_only_not_fresh_training_or_tensor_verification':True}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('manifest_sha256')
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.manifest_sha256), sort_keys=True))
