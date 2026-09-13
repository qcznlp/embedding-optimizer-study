# Restore completed DenseOn endpoint and validation scores

The immutable public snapshot contains **660 files / 29,526,987 logical bytes**:
every final checkpoint's fourteen-task BEIR outputs, all twelve full validation
outputs, original worker records and derived score tables. It does not contain
the 48 intermediate-checkpoint evaluations or a completed scientific release.

All twelve step-782 evaluations now have a separate
[first-stage recovery entry](first-stage-evaluation-restoration.md). The original
endpoint/validation snapshot and its immutable revision remain unchanged.

- Dataset: `qcz/embedding-optimizer-study-analysis-artifacts`.
- Immutable revision: `3883b677f87b1982f06016e9fadb8bb95e0cfc96`.
- Manifest SHA-256: `bf443d5623c1312f8661a918ab93f9aea5f119058307733d477c8eac0c6da74f`.
- Exact prefix:

```text
corrected-dense-correctness-v3/complete-endpoint-evaluations/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/bf443d5623c1312f8661a918ab93f9aea5f119058307733d477c8eac0c6da74f
```

[Browse the exact snapshot](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts/tree/3883b677f87b1982f06016e9fadb8bb95e0cfc96/corrected-dense-correctness-v3/complete-endpoint-evaluations/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/bf443d5623c1312f8661a918ab93f9aea5f119058307733d477c8eac0c6da74f).
Use this revision, not a moving default branch or historical experiment folder.

## What can be reconstructed?

`native/beir-final/` preserves all 168 task-score JSON files, model/run metadata,
and original complete endpoint receipts. `native/validation/` preserves 49,152
query records, each with eight candidate cosine scores and six original metrics.
No raw query/document texts or model/optimizer tensors are included. Their IDs,
row-content digests, source/data identities and all original worker records remain.

`tables/final-checkpoints/` retains every declared optimizer/rate endpoint.
`tables/validation/` retains all validation metrics and the comparison selected
by lowest mean validation loss (exact ties choose lower learning rate).
Selection uses all 4,096 rows per model and never reads BEIR scores.

You can reconstruct validation metrics, validation-only choices and the complete
endpoint score tables without a GPU or W&B. The native BEIR outputs contain
aggregate task metrics, not per-query rankings: they cannot alone reconstruct
query-level BEIR scores from embeddings. Full forward reproduction additionally
requires the exact model, program and dataset versions. The sixty-model download
index is included under `provenance/primary-v3-checkpoints.json`.

Original source/scientific-completion flags remain false. Learning-rate cells are
not independent training seeds; this is neither a significance test nor a causal
weight-space explanation. The snapshot's paths/PIDs are historical metadata, not
commands to execute or processes to signal on a new host.

## Download into a new directory

This convenience block uses `huggingface_hub` and anonymous access. Replace only
the destination with a new absolute directory. It selects the exact immutable
prefix and preserves client cache metadata outside the selected snapshot.

```python
from pathlib import Path
from huggingface_hub import snapshot_download

destination = Path("/absolute/path/to/new-evaluation-download")
if not destination.is_absolute() or any(p.is_symlink() for p in (destination, *destination.parents)):
    raise ValueError("Use an absolute non-symlink destination")
destination.mkdir(parents=True, exist_ok=False)
prefix = (
    "corrected-dense-correctness-v3/complete-endpoint-evaluations/"
    "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/"
    "bf443d5623c1312f8661a918ab93f9aea5f119058307733d477c8eac0c6da74f"
)
snapshot_download(
    repo_id="qcz/embedding-optimizer-study-analysis-artifacts",
    repo_type="dataset",
    revision="3883b677f87b1982f06016e9fadb8bb95e0cfc96",
    allow_patterns=[prefix + "/*"],
    endpoint="https://huggingface.co",
    local_dir=destination,
    token=False,
    max_workers=2,
)
```

If interrupted, preserve partial files and inspect them before explicitly resuming
the same immutable selection. Do not delete another snapshot, overwrite existing
outputs or substitute a new expected digest.

## Verify every byte offline

Use the same destination. This block uses only the standard library, does not
import the uploader or project, and does not execute any downloaded metadata.

```python
import hashlib
import json
from pathlib import Path, PurePosixPath

destination = Path("/absolute/path/to/new-evaluation-download")
manifest_sha = "bf443d5623c1312f8661a918ab93f9aea5f119058307733d477c8eac0c6da74f"
root = destination / (
    "corrected-dense-correctness-v3/complete-endpoint-evaluations/"
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
    raise ValueError("Manifest identity differs; keep the trusted digest")
manifest = json.loads(raw)
expected = set(manifest["files"]) | {"artifact_manifest.json"}
actual = set()
for path in root.rglob("*"):
    if path.is_symlink():
        raise ValueError("Symlinked payload refused")
    if path.is_file():
        actual.add(path.relative_to(root).as_posix())
if actual != expected or len(actual) != 660:
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

Actual anonymous recovery completed at **2026-09-10 17:28:18 UTC**. All 660 files
matched their immutable hashes. Independent numerical replay from that recovered
copy passed at **17:37:09 UTC**: 168 BEIR task scores, twelve endpoint means,
49,152 validation rows / 294,912 scalar checks, 576 group-metric means and all
four CSV tables. The three validation-only choices match the original selector.
These timestamps are observer-reported; this was a same-host anonymous recovery.

The local [evidence archive](../reports/engineering-archive/dense-v3-evaluation-artifact-backup-v1/README.md)
contains actual transfer/check receipts and a separate standard-library numerical
replay script. That script checks all scalar validation metrics using the original
reference tolerances without replacing recorded FP32 values; it reconstructs
BEIR means and both all-rate and selected tables. The program source was not
uploaded in this dataset snapshot. A same-host recovery/replay is not a physical
second-host training or evaluation experiment.

## Preservation and remaining work

All 660 paths were new under one independent prefix. The commit preserved all
twenty other root entries (including the existing card and `.gitattributes`) and
both earlier corrected training/weight subtrees. No old file was overwritten or
deleted. This new addition does not reinterpret the separately retained storage-
metadata exception from the earlier training-artifact upload.

See [checkpoint-restoration.md](checkpoint-restoration.md),
[weight-analysis-restoration.md](weight-analysis-restoration.md),
[training-analysis-restoration.md](training-analysis-restoration.md) and
[CURRENT_EXPERIMENT.md](../CURRENT_EXPERIMENT.md) for their distinct scopes.
Intermediate retrieval, functional measurements, crossed continuations, full
source availability and the final NAACL paper remain unfinished.
