# Restore the complete current weight-analysis artifacts

The corrected v3 weight measurements are backed up publicly, separately from the
model checkpoints and historical analyses. This snapshot contains **298 files /
4,981,840,933 bytes**, including all 291 native numerical files, original provenance,
the checkpoint index, a readable description and the complete artifact manifest.

- Repository: `qcz/embedding-optimizer-study-analysis-artifacts` (dataset).
- Immutable revision: `209b4517e64ac5373db47b04ada39104e9080016`.
- Manifest SHA-256: `9f53cdb8260f7b9d8f4c29db04159d02dfb7ff4e272cf9c9ba49cba842a1a841`.
- Exact snapshot prefix:

```text
corrected-dense-correctness-v3/weight-space/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/9f53cdb8260f7b9d8f4c29db04159d02dfb7ff4e272cf9c9ba49cba842a1a841
```

[Browse the exact snapshot](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts/tree/209b4517e64ac5373db47b04ada39104e9080016/corrected-dense-correctness-v3/weight-space/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/9f53cdb8260f7b9d8f4c29db04159d02dfb7ff4e272cf9c9ba49cba842a1a841).
Use this revision/prefix, not the moving default branch or the mixed-history repository.

The actual full anonymous recovery into a fresh directory completed at
**2026-09-10 12:19:06 UTC**: all 298 files matched their size, SHA-256 and original
remote digest. The [evidence archive](../reports/engineering-archive/dense-v3-weight-artifact-backup-v1/README.md)
keeps the original upload, remote-audit and recovery records separately. This is
a same-host fresh-directory check, not a second-physical-host experiment.

## What is included

`approximate/` and `exact/` each retain all twelve runs, five stages and 88 hidden
matrices, including every original raw record, retained basis, table and manifest.
The exact branch retains complete spectra. `update-map/` retains the FP64 Gram
arrays, all projected paths, full pair/path tables and original figures;
`update-map-readout/` retains full-dimensional endpoint comparisons and all
5,310 pair/projection-retention rows. Neither geometry branch replaces the other.

`provenance/` contains the original numerical completion records, map declaration
and all sixty source-checkpoint locations/hashes. Original metadata, flags, source
hashes and machine paths are unchanged; a recorded producer path is not a command
or a dependency location that must exist on the new machine.

## Download into a new directory

The following needs `huggingface_hub` (the actual backup/recovery uses 1.28.0),
but no project installation, GPU, model code or HF token. Replace the absolute
destination with a new directory; existing destinations are intentionally refused.

```python
from pathlib import Path
from huggingface_hub import snapshot_download

destination = Path("/absolute/path/to/new-weight-analysis-download")
if not destination.is_absolute() or any(p.is_symlink() for p in (destination, *destination.parents)):
    raise ValueError("Use an absolute, non-symlinked new destination")
destination.mkdir(parents=True, exist_ok=False)
prefix = (
    "corrected-dense-correctness-v3/weight-space/"
    "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/"
    "9f53cdb8260f7b9d8f4c29db04159d02dfb7ff4e272cf9c9ba49cba842a1a841"
)
snapshot_download(
    repo_id="qcz/embedding-optimizer-study-analysis-artifacts",
    repo_type="dataset",
    revision="209b4517e64ac5373db47b04ada39104e9080016",
    allow_patterns=[prefix + "/*"],
    local_dir=destination,
    token=False,
    max_workers=2,
)
```

Do not remove valid partial files after an interrupted transfer. Preserve and
inspect the destination before explicitly resuming that same immutable selection.
The client creates its own cache metadata outside the snapshot directory.

## Verify all files offline

This standard-library check authenticates the manifest against the trusted digest
above, then hashes **every** payload and refuses missing, extra or symlinked files.
It never executes a model, pickle, source command or the contents of metadata.
Use the same destination as the download:

```python
import hashlib
import json
from pathlib import Path

destination = Path("/absolute/path/to/new-weight-analysis-download")
manifest_sha = "9f53cdb8260f7b9d8f4c29db04159d02dfb7ff4e272cf9c9ba49cba842a1a841"
root = destination / (
    "corrected-dense-correctness-v3/weight-space/"
    "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/"
    + manifest_sha
)
if any(p.is_symlink() for p in (root, *root.parents)):
    raise ValueError("Refuse a symlinked snapshot root")
manifest_path = root / "artifact_manifest.json"
if manifest_path.is_symlink() or not manifest_path.is_file():
    raise ValueError("Require an ordinary manifest file")
raw = manifest_path.read_bytes()
if hashlib.sha256(raw).hexdigest() != manifest_sha:
    raise ValueError("Manifest authentication failed; do not replace the trusted digest")
manifest = json.loads(raw)
expected = set(manifest["files"]) | {"artifact_manifest.json"}
actual = set()
for path in root.rglob("*"):
    if path.is_symlink():
        raise ValueError("Refuse symlinked payloads")
    if path.is_file():
        actual.add(path.relative_to(root).as_posix())
if actual != expected or len(actual) != 298:
    raise ValueError("Snapshot inventory differs")
for name, record in manifest["files"].items():
    path = root / name
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if path.stat().st_size != record["bytes"] or digest.hexdigest() != record["sha256"]:
        raise ValueError(f"Payload differs; preserve and investigate: {name}")
print(json.dumps({"verified_files": len(actual), "manifest_sha256": manifest_sha}))
```

## Boundary and related artifacts

This is complete **weight-artifact recovery**, not complete experiment recovery.
Full retrieval, functional interventions, crossed continuations and the paper are
not in this snapshot. Spectral energy or a two-dimensional map alone does not
establish useful embedding dimensions, retrieval superiority or causality.

The [model checkpoint guide](checkpoint-restoration.md) addresses all sixty
model/optimizer checkpoints. Current execution status is in
[CURRENT_EXPERIMENT.md](../CURRENT_EXPERIMENT.md). Source publication and full
portable analysis reconstruction remain separate work; no WIP program source
was uploaded with this numerical snapshot. Recovery on this physical host does
not establish an actual second-host experiment or subsequent bitwise GPU replay.
