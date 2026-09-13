"""Download/verify the immutable actual DenseOn functional/calibration snapshot.

Verification uses only the standard library. No model, GPU, pickle or training
code is imported. This verifies artifact integrity, not scientific admission.
"""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REPO = 'qcz/embedding-optimizer-study-analysis-artifacts'
REVISION = '9b182d77b31c93277457a578c16389935bc3fdb9'
MANIFEST_SHA = '0b498557040c2329e6935fdc539bce6d029b8bbf02f8ffcbc194912203b8cc38'
PREFIX = 'corrected-dense-correctness-v3/functional-and-calibration-v1/' + MANIFEST_SHA
MANIFEST_BYTES = 161464
TOTAL_FILES = 624
TOTAL_BYTES = 7496655065


def need(ok, message):
    if not ok:
        raise ValueError(message)


def verify_file(path, expected):
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Missing or symlinked payload')
    first = path.stat()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    last = path.stat()
    need(all(getattr(first, k) == getattr(last, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')),
         'Payload changed during read')
    need(last.st_size == expected['bytes'] and digest == expected['sha256'], 'Payload checksum differs: ' + str(path))


def files(root):
    manifest = root / 'artifact_manifest.json'
    verify_file(manifest, dict(bytes=MANIFEST_BYTES, sha256=MANIFEST_SHA))
    value = json.loads(manifest.read_text())
    need(value['scope'] == 'actual-dense-v3-functional-and-calibration-durability-v1'
         and value['source_code_included'] is False and value['scientific_completion'] is False,
         'Unexpected snapshot scope')
    result = value['files']
    for name in result:
        path = PurePosixPath(name)
        need(path.parts and not path.is_absolute() and path.as_posix() == name and
             all(p not in ('.', '..') for p in path.parts) and '\\' not in name, 'Unsafe payload path')
    result = {**result, 'artifact_manifest.json': dict(bytes=MANIFEST_BYTES, sha256=MANIFEST_SHA)}
    need(len(result) == TOTAL_FILES and sum(r['bytes'] for r in result.values()) == TOTAL_BYTES, 'Incomplete snapshot inventory')
    return result


def verify(root):
    expected = files(root)
    need(not any(p.is_symlink() for p in root.rglob('*')), 'Symlinked snapshot')
    need({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} == set(expected),
         'Missing or extra snapshot payload')
    for name, binding in sorted(expected.items()):
        verify_file(root / name, binding)
    return dict(scope='actual-functional-calibration-artifact-integrity-verification',
        verified_at_utc=datetime.now(timezone.utc).isoformat(), repo_id=REPO, revision=REVISION,
        prefix=PREFIX, files=TOTAL_FILES, bytes=TOTAL_BYTES, all_checksums_match=True,
        source_release=False, gpu_execution=False, scientific_completion=False)


def download(destination):
    need(not destination.exists() and not any(p.is_symlink() for p in (destination, *destination.parents)),
         'Use a new destination; preserve partial attempts')
    destination.mkdir(parents=True, exist_ok=False)
    from huggingface_hub import hf_hub_download

    def fetch(name):
        path = Path(hf_hub_download(REPO, PREFIX + '/' + name, repo_type='dataset',
            revision=REVISION, token=False, local_dir=destination))
        need(path == destination / PREFIX / name, 'Unexpected downloaded location')
        return path

    root = fetch('artifact_manifest.json').parent
    expected = files(root)

    def transfer(name):
        verify_file(fetch(name), expected[name])

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(transfer, sorted(set(expected) - {'artifact_manifest.json'})))
    return root, verify(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'download'))
    parser.add_argument('--path', type=Path, required=True,
        help='verify: snapshot directory containing artifact_manifest.json; download: new destination')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.receipt:
        need(not args.receipt.exists(), 'Preserve previous verification receipt')
    if args.action == 'download':
        root, result = download(args.path.absolute())
    else:
        root = args.path.absolute()
        result = verify(root)
    result['snapshot_root'] = str(root)
    if args.receipt:
        with args.receipt.open('x') as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
