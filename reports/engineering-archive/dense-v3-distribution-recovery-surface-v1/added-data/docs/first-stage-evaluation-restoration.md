# Restore all DenseOn first-stage BEIR results

This public immutable snapshot contains **557 files / 1,957,694 logical bytes**:
all twelve configurations at step 782, each with fourteen full-corpus task results.
It complements the separately backed-up [final and validation results](evaluation-analysis-restoration.md).
Stages 1563, 2345 and 3126 are not included.

- Dataset: `qcz/embedding-optimizer-study-analysis-artifacts`.
- Revision: `8555e5849b56862948b0fc0688085708ce3e67cf`.
- Trusted manifest SHA-256: `17b90cd5d8b2d9b47ba2849b5a9e3d0ffd17998e5ffcb9fc8b1f176e55ea0e7b`.
- Exact prefix:

```text
corrected-dense-correctness-v3/complete-first-stage-evaluations/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/17b90cd5d8b2d9b47ba2849b5a9e3d0ffd17998e5ffcb9fc8b1f176e55ea0e7b
```

[Browse the exact snapshot](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts/tree/8555e5849b56862948b0fc0688085708ce3e67cf/corrected-dense-correctness-v3/complete-first-stage-evaluations/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/17b90cd5d8b2d9b47ba2849b5a9e3d0ffd17998e5ffcb9fc8b1f176e55ea0e7b).

## Download and verify

The snapshot preserves 216 native files, all 336 original worker records, two
CSV tables, an acceptance summary, README and manifest. It includes all 168
scores and twelve equal-task means, without score-based selection. The native
task name is `QuoraRetrieval`, including worker records. No model tensors, raw
query/document texts, executable source or credentials are included. Historical
paths/commands/PIDs are provenance, never execution instructions.

Use `huggingface_hub` to download anonymously into a **new absolute directory**.
Keep partial files if interrupted; do not overwrite another snapshot or change
the expected manifest hash. Client cache metadata stays outside the prefix.

```python
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from huggingface_hub import hf_hub_download

destination = Path('/absolute/path/to/new-first-stage-download')
if not destination.is_absolute() or any(p.is_symlink() for p in (destination, *destination.parents)):
    raise ValueError('Use a new absolute non-symlink directory')
destination.mkdir(parents=True, exist_ok=False)
revision = '8555e5849b56862948b0fc0688085708ce3e67cf'
manifest_sha = '17b90cd5d8b2d9b47ba2849b5a9e3d0ffd17998e5ffcb9fc8b1f176e55ea0e7b'
prefix = (
    'corrected-dense-correctness-v3/complete-first-stage-evaluations/'
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

Copy the [standalone stdlib reader](../reports/engineering-archive/dense-v3-first-stage-artifact-backup-v1/source/verify_recovered.py)
along with this guide before closing the host. Its SHA-256 is
`98d21d6033c3d209aaf1c6fd8112f4fd4363eb1cfd744819dc225917d94114bf`.
It is local WIP, not included in the public data snapshot or claimed available
in a clean remote GitHub checkout. Set `--root` to the full path printed above:

```bash
python -I -B /absolute/path/to/verify_recovered.py \
  --root /absolute/path/to/downloaded-snapshot-prefix \
  --manifest-sha256 17b90cd5d8b2d9b47ba2849b5a9e3d0ffd17998e5ffcb9fc8b1f176e55ea0e7b \
  --output /absolute/path/to/new-first-stage-replay.json
```

Expected: **557 verified files, 168 raw task scores, 168 original exit-zero
workers, twelve means and 180 CSV rows**. Every payload's bytes, SHA-256 and
Git-blob identity must match. The reader imports no project package and reads
only the supplied snapshot, never original absolute paths or uploaded commands.
Its arithmetic uses exact rational means of the original binary64 task values.
Undefined auxiliary metrics are retained; no primary score is imputed.

## Actual evidence and limits

Full anonymous recovery passed at **2026-09-11 04:18:08 UTC**. The copied reader's
isolated recovered-score reconstruction passed at **04:18:43 UTC**. Both were
same-host operations, not second-host model runs. The convenience download block
uses the same API and immutable population; it is not an additional executed
cross-host experiment. All 34 new adapter tests and 18 reader tests passed,
including fully rehashed semantic corruption controls.

All twenty other root entries and all four previous corrected subtrees remained
unchanged, including the card and `.gitattributes`. No old file was overwritten
or deleted. Earlier training-backup storage exceptions remain separately recorded.
The [evidence archive](../reports/engineering-archive/dense-v3-first-stage-artifact-backup-v1/README.md)
preserves receipts, commands and the original local pre-upload alias failures.

These are aggregate task scores, not per-query ranking traces. First/final stages
alone do not establish a full trajectory, seed robustness or useful-dimension
mechanism. Complete middle-stage evaluation, functional measurements, crossed
continuations, portable source release and the final paper remain required.
See [CURRENT_EXPERIMENT.md](../CURRENT_EXPERIMENT.md) and the separate
[checkpoint guide](checkpoint-restoration.md).
