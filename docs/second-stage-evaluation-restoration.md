# Restore all twelve second-stage BEIR evaluations

This public immutable snapshot preserves **557 files / 1,961,293 logical bytes**:
all twelve corrected DenseOn configurations at step 1563, each with fourteen
full-corpus BEIR task results. AdamW rates are 1e-6 / 3e-6 / 1e-5 / 3e-5;
Muon and NorMuon each use 1e-4 / 3e-4 / 1e-3 / 3e-3. Membership follows the
complete declared configuration set, not scores.

Six states / 84 task values overlap the [earlier pool-B snapshot](second-stage-pool-b-evaluation-restoration.md),
which remains unchanged. Only six pool-A states / 84 values are newly backed up;
they are not twelve additional training runs or independent replications.

The [first-stage results](first-stage-evaluation-restoration.md) and
[final/validation results](evaluation-analysis-restoration.md) remain separate
unchanged snapshots. Model tensors use the [checkpoint recovery entry](checkpoint-restoration.md).

- Dataset: `qcz/embedding-optimizer-study-analysis-artifacts`.
- Revision: `aa1515cbf98cb1b2af0be58cf05a4985bf1254cb`.
- Trusted manifest SHA-256: `8ead2e55fa3c800199fc1da8051f1b1d8c0fe15246f9d0f0e5f688e34942c277`.
- Exact prefix:

```text
corrected-dense-correctness-v3/complete-second-stage-evaluations/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/8ead2e55fa3c800199fc1da8051f1b1d8c0fe15246f9d0f0e5f688e34942c277
```

[Browse the exact snapshot](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts/tree/aa1515cbf98cb1b2af0be58cf05a4985bf1254cb/corrected-dense-correctness-v3/complete-second-stage-evaluations/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/8ead2e55fa3c800199fc1da8051f1b1d8c0fe15246f9d0f0e5f688e34942c277).

## Download and independently verify

The snapshot contains 216 original native files, 336 original started/exited
worker records, two CSV tables, a data-only acceptance summary, README and manifest.
All 168 task scores and twelve equal-task means are retained. No model tensors, raw
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

destination = Path('/absolute/path/to/new-complete-second-stage-download')
if not destination.is_absolute() or any(p.is_symlink() for p in (destination, *destination.parents)):
    raise ValueError('Use a new absolute non-symlink directory')
destination.mkdir(parents=True, exist_ok=False)
revision = 'aa1515cbf98cb1b2af0be58cf05a4985bf1254cb'
manifest_sha = '8ead2e55fa3c800199fc1da8051f1b1d8c0fe15246f9d0f0e5f688e34942c277'
prefix = (
    'corrected-dense-correctness-v3/complete-second-stage-evaluations/'
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
if len(manifest['files']) != 556:
    raise ValueError('Incomplete manifest')
with ThreadPoolExecutor(max_workers=2) as executor:
    list(executor.map(download, sorted(manifest['files'])))
print(destination / prefix)
```

Before closing this host, copy this guide and the
[standalone stdlib reader](../reports/engineering-archive/dense-v3-complete-second-stage-artifact-backup-v1/source/verify_recovered.py).
Its SHA-256 is `c69e3caee15a34745c39ed246c1d8f434fc6023a535d42ef331ae0b9b73372ce`.
The reader is local source, not included in the public data snapshot or claimed
available from an updated GitHub checkout. Run it with the full path printed above:

```bash
python -I -B /absolute/path/to/verify_recovered.py \
  --root /absolute/path/to/downloaded-snapshot-prefix \
  --manifest-sha256 8ead2e55fa3c800199fc1da8051f1b1d8c0fe15246f9d0f0e5f688e34942c277 \
  --output /absolute/path/to/new-complete-second-stage-replay.json
```

Expected: **557 authenticated files, 168 raw task scores, 168 original exit-zero
workers, twelve exact macro means and 180 CSV rows**. The reader checks bytes,
SHA-256 and Git-blob identities, and reconstructs means using exact rational
arithmetic on the original binary64 scores. It uses only the supplied snapshot,
not the project environment, original absolute paths or uploaded commands.
The canonical task name is `QuoraRetrieval`; no alias substitution is needed.

## Actual evidence and remaining scope

Full anonymous recovery passed at **2026-09-11 14:27:01 UTC**. The byte-identical
copied reader, run with `python -I -B` from outside the project, passed recovered
score reconstruction at **14:29:06 UTC**. These were same-host operations, not
second-host model runs. The convenience download block above uses the same API
and immutable population; it is not an additional executed recovery experiment.
All **34 adapter tests and 18 reader tests** passed; the latter include semantic
corruption controls whose manifests and payload hashes were deliberately recomputed.

The remote audit confirms all twenty other root entries and all six earlier
corrected subtrees unchanged, including the repository card and `.gitattributes`.
No existing file was overwritten or deleted. Previous backup exceptions retain
their original records; they are not retried or relabeled by this snapshot.
See the [evidence archive](../reports/engineering-archive/dense-v3-complete-second-stage-artifact-backup-v1/README.md).

These are aggregate task measurements, not per-query ranking traces. All steps 2345 / 3126 remain outside this
snapshot. Complete trajectories, functional dimension analysis, crossed
continuations, portable source release and the final paper remain required.
See [CURRENT_EXPERIMENT.md](../CURRENT_EXPERIMENT.md).
