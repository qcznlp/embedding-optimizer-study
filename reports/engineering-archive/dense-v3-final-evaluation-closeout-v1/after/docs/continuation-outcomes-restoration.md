# Restore the complete continuation evaluation and statistics

The original backup process completed on **2026-09-13 at 06:48:52 UTC**:
all **593 files / 5,199,649 bytes** were anonymously downloaded and checksum-
verified after upload. This includes all 168 full-corpus task results, final
per-model metadata, native worker/collector provenance and all six statistical
tables. It does not include executable source, model weights or raw examples.

- [Immutable public snapshot](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts/tree/7766dd0ae83e36a4c16121703813f6a9eec63336/corrected-dense-correctness-v3/complete-continuation-outcomes-v1/296c688d47617986d297044f901f0c0e1e41e552a1e06110cff8f6bf01770230).
- Dataset: `qcz/embedding-optimizer-study-analysis-artifacts`.
- Revision: `7766dd0ae83e36a4c16121703813f6a9eec63336`.
- Manifest SHA-256: `296c688d47617986d297044f901f0c0e1e41e552a1e06110cff8f6bf01770230`.
- [Actual completion and bounded scientific results](../reports/engineering-archive/dense-v3-final-evaluation-closeout-v1/README.md).

## Download and verify

Use a separate environment with `huggingface_hub`. No GPU/model library or
pickle deserialization is needed. Choose a new destination; partial attempts
are preserved rather than overwritten. This uses the same anonymous,
revision-pinned per-file download API as the completed backup verification.

```python
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from huggingface_hub import hf_hub_download

destination = Path('/data/dense-continuation-outcomes-NEW')
assert not destination.exists()
assert not any(p.is_symlink() for p in (destination, *destination.parents))
destination.mkdir(parents=True)
repo = 'qcz/embedding-optimizer-study-analysis-artifacts'
revision = '7766dd0ae83e36a4c16121703813f6a9eec63336'
digest = '296c688d47617986d297044f901f0c0e1e41e552a1e06110cff8f6bf01770230'
prefix = 'corrected-dense-correctness-v3/complete-continuation-outcomes-v1/' + digest

def fetch(name):
    relative = PurePosixPath(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    assert relative.as_posix() == name and '\\' not in name
    path = Path(hf_hub_download(repo, prefix + '/' + name, repo_type='dataset',
                               revision=revision, token=False, local_dir=destination))
    assert path == destination / prefix / name
    assert not any(p.is_symlink() for p in (path, *path.parents))
    return path

manifest = fetch('artifact_manifest.json')
data = manifest.read_bytes()
assert len(data) == 185031 and hashlib.sha256(data).hexdigest() == digest
records = json.loads(data)['files']
records['artifact_manifest.json'] = {'bytes': 185031, 'sha256': digest}
assert len(records) == 593 and sum(v['bytes'] for v in records.values()) == 5199649

def transfer_and_verify(name):
    path = manifest if name == 'artifact_manifest.json' else fetch(name)
    payload = path.read_bytes()
    assert len(payload) == records[name]['bytes'], name
    assert hashlib.sha256(payload).hexdigest() == records[name]['sha256'], name

with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(transfer_and_verify, sorted(records)))
root = manifest.parent
assert {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} == set(records)
print('Verified snapshot:', root)
```

The manifest also carries Git blob hashes; the instructions above verify the
pinned manifest and every payload's exact byte length and SHA-256. Keep the
manifest beside the payloads. Absolute producer paths inside original receipts
are historical provenance, not download destinations or required local paths.

## Contents and interpretation

`native/beir/<run>/` contains fourteen raw MTEB results per final checkpoint
and the native complete-result receipt. `provenance/` preserves the original
worker exits, complete pools, collectors and inference evidence.
`related-artifacts.json` links the existing immutable checkpoint/probe artifacts.

| CSV under `tables/` | Rows |
| --- | ---: |
| `beir_seed_task_scores.csv` | 168 |
| `factorial_cell_summary.csv` | 4 |
| `estimand_seed_task_contrasts.csv` | 126 |
| `estimand_summary.csv` | 3 |
| `probe_checkpoint_metrics.csv` | 60 |
| `probe_task_metrics.csv` | 840 |

The tables use fractional nDCG, not percentage points: multiply effect estimates
and interval endpoints by 100 for points. The three intervals are marginal,
not simultaneous. They cover three data-order seeds and fourteen tasks for
two fixed source states; they do not replicate primary source training.

Use the [probe/checkpoint guide](continuation-probe-restoration.md) for all
sixty continuation models and five-stage vectors. The [paper reproduction
guide](paper-results-reproduction.md) describes the numerical source closure;
the current source release and final paper review are still pending.
Artifact integrity does not establish GPU-resume equivalence: the original
two GPU recovery attempts exposed a state-device bug and remain failed until
a separately verified repair succeeds.
