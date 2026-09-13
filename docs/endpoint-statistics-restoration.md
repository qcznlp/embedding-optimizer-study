# Restore the complete endpoint statistical readout

All six original final-checkpoint contrasts and their figures are available in an
[immutable public snapshot](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts/tree/13b110a0d067948ff518a0f720ee84c2e2f8c6b1/corrected-dense-correctness-v3/endpoint-statistics/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/cddee2dcfb68c25e42931c82f129c0f57590434cecf2b48bbc106da0dfc38c2e).
There are **16 files / 218,466 bytes**, including the manifest. Every file was
actually recovered anonymously and independently verified on September 10.
This is not a physical second-host experiment or public executable-source release.

## What is preserved

The snapshot contains all five CSV tables, the generated summary, the original
native readout, PDF/PNG/SVG figures, plotted data and rendering metadata, and the
original verification and source-relocated readout receipts. Fifteen payloads
are bound by `artifact_manifest.json`, SHA-256
`cddee2dcfb68c25e42931c82f129c0f57590434cecf2b48bbc106da0dfc38c2e`.

The [original endpoint analysis](../reports/dense-v3-final-inference-v1/README.md)
explains the primary four-rate averages and secondary validation-selected recipes.
All three primary intervals are inconclusive; only the secondary NorMuon-minus-
AdamW interval is wholly positive. Task intervals are not seed intervals, and
one positive and one inconclusive comparison do not establish a treatment difference.
The [separate raw-score snapshot](evaluation-analysis-restoration.md) preserves
all 168 final task scores and all twelve validations needed by the original analysis.

## Anonymous download with a fixed revision

Use a **new** absolute destination, not an existing checkpoint or repository root.
The following convenience block uses the same installed HF download API as the
actual transfer. Its syntax is checked; the recorded real transfer used the
archived transport program, not an additional duplicate download of this block.

```bash
python3 - /absolute/new/endpoint-statistics-download <<'PY'
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from huggingface_hub import hf_hub_download

destination = Path(sys.argv[1])
if not destination.is_absolute() or destination.exists():
    raise ValueError('Choose a new absolute destination')
repo = 'qcz/embedding-optimizer-study-analysis-artifacts'
revision = '13b110a0d067948ff518a0f720ee84c2e2f8c6b1'
manifest_sha = 'cddee2dcfb68c25e42931c82f129c0f57590434cecf2b48bbc106da0dfc38c2e'
prefix = ('corrected-dense-correctness-v3/endpoint-statistics/'
          '4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/'
          + manifest_sha)

def fetch(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or p.as_posix() != name:
        raise ValueError('Unsafe manifest path')
    path = Path(hf_hub_download(repo, repo_type='dataset', revision=revision,
                filename=prefix + '/' + name, local_dir=destination, token=False))
    if path != destination / prefix / name or any(x.is_symlink() for x in (path, *path.parents)):
        raise ValueError('Unexpected downloaded path')
    return path.read_bytes()

raw = fetch('artifact_manifest.json')
if hashlib.sha256(raw).hexdigest() != manifest_sha:
    raise ValueError('Manifest identity differs')
manifest = json.loads(raw)
if len(manifest['files']) != 15:
    raise ValueError('Unexpected payload population')
for name, identity in manifest['files'].items():
    raw = fetch(name)
    if len(raw) != identity['bytes'] or hashlib.sha256(raw).hexdigest() != identity['sha256']:
        raise ValueError('Payload identity differs: ' + name)
print(destination / prefix)
PY
```

With the local project source separately available, the standalone
[independent verifier](../reports/engineering-archive/dense-v3-endpoint-statistics-backup-v1/source/verify_recovered.py)
checks all bytes plus 246 numeric table fields, 24 figure fields, the six
interval-to-figure mappings and the original relocation receipt. It accepts
`--root /absolute/downloaded/prefix --output /absolute/new/result.json` and uses
only Python's standard library. It does not execute a model or regenerate
bootstrap draws. No publicly available source checkout is implied by this link.

## Actual evidence and limits

The one successful remote commit completed at 18:52:34 UTC; strict remote
verification completed at 18:52:35, anonymous recovery at 18:54:17, and the
independent recovered-input check at 18:55:02. All twenty other root entries
(including `.gitattributes` and the dataset card) and all three preceding
corrected subtrees stayed unchanged. Read the
[transport archive](../reports/engineering-archive/dense-v3-endpoint-statistics-backup-v1/README.md)
for the original pre-upload cache-symlink failure and successful new attempt.
That failure is retained; no numerical source, scientific rule or previous remote
payload was changed. The older training-backup root-byte exception also remains
unchanged and is not reinterpreted by this later success.

The 840-cell trajectory evaluation, 61-state functional measurements, crossed
continuation, complete portable source reconstruction and NAACL paper remain
required. This backup does not make any of them complete.
