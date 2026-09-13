# Restore the six original pool-B third-stage BEIR evaluations

This public immutable snapshot preserves **281 files / 982,548 logical bytes**:
six corrected DenseOn configurations at step 2345, each with fourteen full-corpus
BEIR task results. The configurations are AdamW 3e-6 / 1e-5, Muon 3e-4 / 3e-3,
and NorMuon 1e-3 / 3e-3. Membership follows the original pool-B queue, not scores.
**This is not the complete twelve-configuration third stage.**

The [first-stage results](first-stage-evaluation-restoration.md),
[complete second-stage results](second-stage-evaluation-restoration.md) and
[final/validation results](evaluation-analysis-restoration.md) remain separate
unchanged snapshots. Model tensors use the [checkpoint recovery entry](checkpoint-restoration.md).

- Dataset: `qcz/embedding-optimizer-study-analysis-artifacts`.
- Revision: `dbd16adcc835c2c6b7537a61b8d511f027a2c351`.
- Trusted manifest SHA-256: `1d238cb93e9e2b6791ebf5b6b6e2cd236a12a253e13d7d8f4fa0ecef6523828a`.
- Exact prefix:

```text
corrected-dense-correctness-v3/partial-third-stage-pool-b-evaluations/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/1d238cb93e9e2b6791ebf5b6b6e2cd236a12a253e13d7d8f4fa0ecef6523828a
```

[Browse the exact snapshot](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts/tree/dbd16adcc835c2c6b7537a61b8d511f027a2c351/corrected-dense-correctness-v3/partial-third-stage-pool-b-evaluations/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/1d238cb93e9e2b6791ebf5b6b6e2cd236a12a253e13d7d8f4fa0ecef6523828a).

## Download and independently verify

The snapshot contains 108 original native files, 168 original started/exited
worker records, two CSV tables, a data-only acceptance summary, README and manifest.
All 84 task scores and six equal-task means are retained. No model tensors, raw
query/document examples, executable source bodies or credentials are included.
Historical commands, paths and process IDs are provenance, never instructions.

The following convenience block downloads anonymously into a **new absolute
directory**. Preserve any interrupted attempt and the fixed manifest digest;
do not overwrite another snapshot. Client cache metadata is outside the prefix.

```python
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from huggingface_hub import hf_hub_download

destination = Path('/absolute/path/to/new-third-stage-pool-b-download')
if not destination.is_absolute() or any(p.is_symlink() for p in (destination, *destination.parents)):
    raise ValueError('Use a new absolute non-symlink directory')
destination.mkdir(parents=True, exist_ok=False)
revision = 'dbd16adcc835c2c6b7537a61b8d511f027a2c351'
manifest_sha = '1d238cb93e9e2b6791ebf5b6b6e2cd236a12a253e13d7d8f4fa0ecef6523828a'
prefix = (
    'corrected-dense-correctness-v3/partial-third-stage-pool-b-evaluations/'
    '4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/'
    + manifest_sha
)

def download(name):
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or '..' in path.parts or path.as_posix() != name or '\\' in name:
        raise ValueError('Unsafe manifest path')
    return Path(hf_hub_download(
        repo_id='qcz/embedding-optimizer-study-analysis-artifacts', repo_type='dataset',
        revision=revision, filename=prefix + '/' + name, local_dir=destination,
        token=False, endpoint='https://huggingface.co',
    ))

raw = download('artifact_manifest.json').read_bytes()
if hashlib.sha256(raw).hexdigest() != manifest_sha:
    raise ValueError('Manifest differs; keep the trusted digest')
manifest = json.loads(raw)
if len(manifest['files']) != 280:
    raise ValueError('Incomplete manifest')
with ThreadPoolExecutor(max_workers=2) as executor:
    list(executor.map(download, sorted(manifest['files'])))
print(destination / prefix)
```

Before closing this host, copy this guide and the
[standalone stdlib reader](../reports/engineering-archive/dense-v3-third-stage-pool-b-artifact-backup-v1/source/verify_recovered.py).
Its SHA-256 is `b1dff4501b3aac7cde811766f2d5b9255ae9bf4837512804e83f1d2df9ee84bb`.
The reader is local source, not included in the public data snapshot or claimed
available from an updated GitHub checkout. Run it with the full path printed above:

```bash
python -I -B /absolute/path/to/verify_recovered.py \
  --root /absolute/path/to/downloaded-snapshot-prefix \
  --manifest-sha256 1d238cb93e9e2b6791ebf5b6b6e2cd236a12a253e13d7d8f4fa0ecef6523828a \
  --output /absolute/path/to/new-third-stage-pool-b-replay.json
```

Expected: **281 authenticated files, 84 raw task scores, 84 original exit-zero
workers, six exact macro means and 90 CSV rows**. The reader checks bytes,
SHA-256 and Git-blob identities, and reconstructs means using exact rational
arithmetic on the original binary64 scores. It uses only the supplied snapshot,
not the project environment, original absolute paths or uploaded commands.
The canonical task name is `QuoraRetrieval`; no alias substitution is needed.

## Actual evidence and remaining scope

Full anonymous recovery passed at **2026-09-11 22:54:21 UTC**. The byte-identical
copied reader, run with `python -I -B` from outside the project, passed recovered
score reconstruction at **22:54:42 UTC**. These were same-host operations, not
second-host model runs. The convenience download block above uses the same API
and immutable population; it is not an additional executed recovery experiment.
All **34 adapter tests and 18 reader tests** passed; the latter include semantic
corruption controls whose manifests and payload hashes were deliberately recomputed.
The first adapter-test invocation omitted the local module path and had seventeen
fixture setup errors; it made no remote write. With the explicit module path, the
unchanged adapter passed all 34 cases. The failed XML is preserved. A later console
warning caused a wrapper JSON parse error after a successful upload; the original
uploader exited zero and was not retried.

The remote audit confirms all twenty other root entries and all seven earlier
corrected subtrees unchanged, including the repository card and `.gitattributes`.
No existing file was overwritten or deleted. Previous backup exceptions retain
their original records; they are not retried or relabeled by this snapshot.
See the [evidence archive](../reports/engineering-archive/dense-v3-third-stage-pool-b-artifact-backup-v1/README.md).

These are aggregate task measurements, not per-query ranking traces. The other
six step-2345 configurations and all step-3126 states remain outside this
snapshot. Complete trajectories, functional dimension analysis, crossed
continuations, portable source release and the final paper remain required.
See [CURRENT_EXPERIMENT.md](../CURRENT_EXPERIMENT.md).
