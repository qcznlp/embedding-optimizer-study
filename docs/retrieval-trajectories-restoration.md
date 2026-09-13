# Restore the complete corrected retrieval trajectories

The complete data-only snapshot preserves all twelve optimizer/rate trajectories
at five checkpoints: **840 task scores, eight numerical CSVs, all twelve plotted
curves, selected-stage descriptive tables and the original six endpoint contrasts**.
It also includes a recovery index for the five separate raw-score snapshots covering
all sixty checkpoints. No source code, model tensors or training text is included.

## Exact immutable target

- Dataset: `qcz/embedding-optimizer-study-analysis-artifacts`.
- Revision: `dbcd12376f483347cc570584abc588503c0a5fa0`.
- Prefix:
  `corrected-dense-correctness-v3/complete-retrieval-trajectories/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/f2e3aec64101721e337dc13fc1bae5cdb3799ca76e0faece0fe7fbafabdf4e65`.
- Manifest: `artifact_manifest.json`, 6,223 bytes, SHA-256
  `f2e3aec64101721e337dc13fc1bae5cdb3799ca76e0faece0fe7fbafabdf4e65`.
- Complete snapshot: **26 files / 1,093,911 bytes**.

Get the external digest from the trusted project handoff. A manifest is not
authenticated merely by listing its own hashes. Use this exact revision/prefix;
do not download the moving default branch or the mixed historical repository root.

All 26 files were anonymously recovered at **2026-09-12 11:40:10 UTC**. A copied
standalone reader returned zero at **11:40:28 UTC**, independently reconstructing
the stage summaries, areas, point effects, selection contrasts and figure values.
Existing interval values/decisions match the accepted endpoint analysis exactly;
bootstrap sampling was not rerun by this recovery reader.

## Anonymous download

The actual transport used `huggingface_hub` 1.28.0. This convenience snippet downloads
the same explicit prefix to a new directory; it was not a separate transport trial:

```python
from huggingface_hub import snapshot_download

prefix = (
    "corrected-dense-correctness-v3/complete-retrieval-trajectories/"
    "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/"
    "f2e3aec64101721e337dc13fc1bae5cdb3799ca76e0faece0fe7fbafabdf4e65"
)
snapshot_download(
    repo_id="qcz/embedding-optimizer-study-analysis-artifacts",
    repo_type="dataset",
    revision="dbcd12376f483347cc570584abc588503c0a5fa0",
    allow_patterns=[prefix + "/**"],
    local_dir="/absolute/path/to/new-trajectory-download",
    token=False,
    max_workers=2,
)
```

## Reconstruct the recovered data independently

Copy the local
[verify_recovered.py](../reports/engineering-archive/dense-v3-trajectory-artifact-backup-v1/source/verify_recovered.py)
off-host with this guide. Its SHA-256 is
`30c71d09767b29cd9e6164bc9e0e8ccd609b9773d116f79185337e64a61078ea`.
The reader uses only the Python standard library and the supplied snapshot:

```bash
python -I -B /absolute/path/to/verify_recovered.py \
  --root /absolute/path/to/new-trajectory-download/corrected-dense-correctness-v3/complete-retrieval-trajectories/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/f2e3aec64101721e337dc13fc1bae5cdb3799ca76e0faece0fe7fbafabdf4e65 \
  --manifest-sha256 f2e3aec64101721e337dc13fc1bae5cdb3799ca76e0faece0fe7fbafabdf4e65 \
  --output /absolute/path/to/new-trajectory-reconstruction.json
```

The actual recovered read checks:

- all 26 payload/manifest files and all 840 scores against the separate native-score
  index and original observer record;
- sixty run-stage means and medians, fifteen optimizer-stage means and medians,
  twelve areas and twelve normalized observed-range means;
- both endpoint task-effect tables, all six original endpoint comparisons,
  sixty plotted checkpoints and five selected-stage contrasts.

It refuses missing, extra, altered or symlinked files. It does not execute producer
paths, import model code, rerun retrieval/bootstrap sampling, or open the original
training machine. It checks the recovered index but does not download the five raw
snapshots itself.

## Recover all five original raw-score snapshots

After verifying the snapshot, read `provenance/raw-score-index.json`. Its external
SHA-256 is `0418c68b9a78cc80173c9cf6af5d5bb57d223d7e763bc6da03d2fe4f682bbf96`.
The five entries contain exact revisions, prefixes and manifest identities; each
also lists the native receipt and fourteen raw score-file identities for every
checkpoint. Index construction reopened all **2,888 previously recovered files /
37,364,806 logical bytes**, rebuilt all 840 raw scores, and matched the full table.

| Step | Progress | Raw snapshot | Files |
| ---: | ---: | --- | ---: |
| 782 | 20% | [Complete first stage](first-stage-evaluation-restoration.md) | 557 |
| 1563 | 40% | [Complete second stage](second-stage-evaluation-restoration.md) | 557 |
| 2345 | 60% | [Complete third stage](third-stage-evaluation-restoration.md) | 557 |
| 3126 | 80% | [Complete fourth stage](fourth-stage-evaluation-restoration.md) | 557 |
| 3907 | 100% | [Endpoint plus validation](evaluation-analysis-restoration.md) | 660 |

Download the exact `repo_id`, `repo_type`, `revision` and `prefix` in each index
entry, using a new destination and its own manifest digest. Verify each complete
snapshot using the corresponding guide. The older partial-stage snapshots overlap
these populations and need not be added again. This index is data, not a program;
do not execute strings found in provenance records.

The [model checkpoint guide](checkpoint-restoration.md) and
[weight-analysis guide](weight-analysis-restoration.md) cover separate artifacts.
Source and restoration scripts are still local WIP, not a released GitHub clone.
Actual recovery was on the same physical host; a second-host model experiment is
not claimed. Functional analysis, crossed continuation and the complete paper
remain unfinished. See the [evidence archive](../reports/engineering-archive/dense-v3-trajectory-artifact-backup-v1/README.md).
