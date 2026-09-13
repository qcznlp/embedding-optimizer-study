# Restore all corrected-v3 training observations

The complete training-observation snapshot is public and immutable: **149 files /
26,985,403 logical bytes**. It contains all twelve runs, 4,692 logged observations,
sixty stage/timing records, the original native metadata and independent proofs.
It is separate from the model checkpoints and weight-geometry snapshot.

- Dataset: `qcz/embedding-optimizer-study-analysis-artifacts`.
- Immutable revision: `3c95da08a4c817d5bcd58b5c84df777716a02ca9`.
- Transport manifest SHA-256:
  `9a69191e47b25eca459a08eb1a3ddb2958d0f652e94b3c089e1c5685a3cf7d6e`.
- Exact prefix:

```text
corrected-dense-correctness-v3/training-dynamics/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/9a69191e47b25eca459a08eb1a3ddb2958d0f652e94b3c089e1c5685a3cf7d6e
```

[Browse the exact snapshot](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts/tree/3c95da08a4c817d5bcd58b5c84df777716a02ca9/corrected-dense-correctness-v3/training-dynamics/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/9a69191e47b25eca459a08eb1a3ddb2958d0f652e94b3c089e1c5685a3cf7d6e).
Use this commit and prefix, not the moving default branch or historical analyses.

All 149 files were actually recovered anonymously into a fresh directory at
**2026-09-10 14:10:47 UTC**, and checked against original sizes, SHA-256 and remote
LFS/Git identities. At **14:12:24 UTC**, the recovered native inputs also passed
the existing independent full numerical auditor: all five tables, 132 original
JSON inputs and sixty saved log prefixes. The auditor used the retained local
reference source; this is not source publication or a physical second-host run.

## Contents and interpretation

The exact original 143-file bundle is under `native/`: five CSV tables,
`tables.json`, SVG/PDF/PNG curves, the original `manifest.json`, and
`inputs/admission.json` plus all 132 run/checkpoint JSON files under `inputs/native/`.
Nothing was filtered out of that bundle. `provenance/` adds both original independent
numerical receipts, its preservation receipt, and the sixty-model-checkpoint index.

These inputs contain metadata, not query/document examples or executable program
source. No GPU, model loading, pickle, W&B access or token is needed to read them.
Original machine paths and code hashes are provenance, not instructions to execute
those paths. Program source availability and full experiment reconstruction remain
separate requirements.

Logged loss ends at step 3900; actual training ends at 3907. Native full-run loss
and the trailing-ten-log mean are distinct. Stage deviations are not confidence
intervals, and sampled gradient norms are not an every-update clipping count.
Timing includes checkpoint saves and work inside the training segments; it excludes
model setup before training and external queue waits. It is not an optimizer-kernel
benchmark. Learning-rate cells are not independent training seeds. Training loss
does not determine retrieval quality, useful embedding dimensions or causality.

## Download a fresh copy

This convenience block needs `huggingface_hub` (actual recovery used 1.28.0).
Replace the destination with an explicit new absolute path. The actual recovery
used anonymous per-file downloads; this convenience block is syntax-checked, not
reported as an additional completed transfer.

```python
from pathlib import Path
from huggingface_hub import snapshot_download

destination = Path("/absolute/path/to/new-training-analysis-download")
if not destination.is_absolute() or any(p.is_symlink() for p in (destination, *destination.parents)):
    raise ValueError("Use an absolute non-symlink destination")
destination.mkdir(parents=True, exist_ok=False)
prefix = (
    "corrected-dense-correctness-v3/training-dynamics/"
    "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/"
    "9a69191e47b25eca459a08eb1a3ddb2958d0f652e94b3c089e1c5685a3cf7d6e"
)
snapshot_download(
    repo_id="qcz/embedding-optimizer-study-analysis-artifacts",
    repo_type="dataset",
    revision="3c95da08a4c817d5bcd58b5c84df777716a02ca9",
    allow_patterns=[prefix + "/*"],
    endpoint="https://huggingface.co",
    local_dir=destination,
    token=False,
    max_workers=2,
)
```

Preserve partial files if interrupted; inspect them before an explicit resume of
the same immutable selection. The client keeps cache metadata outside the selected
snapshot directory. Do not overwrite another snapshot or delete historical data.

## Verify every file offline

Use the same destination. This independent standard-library block does not import
the uploader, model or project, and does not execute any metadata contents.

```python
import hashlib
import json
from pathlib import Path, PurePosixPath

destination = Path("/absolute/path/to/new-training-analysis-download")
manifest_sha = "9a69191e47b25eca459a08eb1a3ddb2958d0f652e94b3c089e1c5685a3cf7d6e"
root = destination / (
    "corrected-dense-correctness-v3/training-dynamics/"
    "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/"
    + manifest_sha
)
if not root.is_absolute() or any(p.is_symlink() for p in (root, *root.parents)):
    raise ValueError("Require an absolute ordinary snapshot root")
manifest_path = root / "artifact_manifest.json"
if not manifest_path.is_file() or manifest_path.is_symlink():
    raise ValueError("Missing ordinary manifest")
raw = manifest_path.read_bytes()
if hashlib.sha256(raw).hexdigest() != manifest_sha:
    raise ValueError("Manifest identity differs; do not replace the trusted hash")
manifest = json.loads(raw)
expected = set(manifest["files"]) | {"artifact_manifest.json"}
actual = set()
for path in root.rglob("*"):
    if path.is_symlink():
        raise ValueError("Symlinked payload refused")
    if path.is_file():
        actual.add(path.relative_to(root).as_posix())
if actual != expected or len(actual) != 149:
    raise ValueError("Incomplete or changed file population")
for name, record in manifest["files"].items():
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
        raise ValueError("Invalid payload path")
    path = root / name
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if path.stat().st_size != record["bytes"] or digest.hexdigest() != record["sha256"]:
        raise ValueError("Payload differs: " + name)
print(json.dumps({"verified_files": len(actual), "manifest_sha256": manifest_sha}))
```

## Storage-metadata caveat and provenance

The upload returned the commit above, then its original strict unchanged-root
check failed: `.gitattributes` grew by two literal-path LFS rules for this new
snapshot's `native/inputs/admission.json` and `native/training-trajectories.pdf`.
All old attribute bytes remain an identical prefix. All nineteen other root
payload/card entries and the old corrected weight subtree retain their original
Git identities. No old model/analysis payload or repository card was overwritten.

The initial upload process's exit **1** and failed original root-byte guard are
preserved. No success receipt was forged for that guard, no second remote write
was made, and no rule was removed. A separate read-only reconciliation verified
the exact two additions and all 149 payload digests before the successful recovery.
Thus complete payload recovery is verified; universal root-byte preservation is
explicitly **not** claimed.

The [evidence archive](../reports/engineering-archive/dense-v3-training-artifact-backup-v1/README.md)
retains the exact source versions, original failure, separate reconciliation,
recovery and native numerical replay. See also the
[model checkpoint guide](checkpoint-restoration.md),
[weight-analysis guide](weight-analysis-restoration.md), and
[current handoff](../CURRENT_EXPERIMENT.md). Full retrieval, functional analysis,
crossed continuations, portable source and the final NAACL paper remain unfinished.
