---
tags:
  - sentence-transformers
  - information-retrieval
  - dense-retrieval
  - muon
  - adamw
  - normuon
---

# Embedding optimizer study checkpoints

The active study compares AdamW, Muon and NorMuon when adapting
[`lightonai/DenseOn-unsupervised`](https://huggingface.co/lightonai/DenseOn-unsupervised).
Code, the current execution status and provenance receipts are in
[`qcznlp/embedding-optimizer-study`](https://github.com/qcznlp/embedding-optimizer-study).

## Use the current replication only

Current checkpoints are under `corrected-dense-no-packing-v1/dense/`.
This is an incremental backup of a 12-run study, not a completed scientific result.
Each complete run has five scheduled stages at steps 782, 1563, 2345, 3126 and 3907.
Intermediate stages can be resumable before the corresponding run finishes.
Use the source repository's execution protocol and backup receipts for exact coverage.

```bash
hf download qcz/embedding-optimizer-study-checkpoints \
  --local-dir /path/to/restore \
  --include 'corrected-dense-no-packing-v1/**'
```

Preserve each checkpoint's model, optimizer, scheduler, trainer and rank-specific RNG
files together. Do not combine state files across runs or stages.

## Withdrawal of obsolete artifacts

At the project owner's request, artifacts from the invalidated historical Dense
implementation have been removed from the current repository tree, including the
old discovery sweep, routing controls, confirmation runs, shared-start branches,
affected Dense smoke tests and quarantined runs. Their former scores must not be
used as the current study's results. The earlier card's claims that these were
formal or confirmatory evidence are withdrawn.

This deletion commit does **not** purge Git history. Old revisions can still
contain withdrawn artifacts until a separately verified history cleanup occurs.
Do not use an old revision to restore the withdrawn experiment as valid evidence.

Separately retained `late/`, `hybrid-adamw/late/` and LateOn diagnostic paths are
legacy backups, outside the active Dense-only paper. Retention is not a new
correctness certification. They were not classified as affected by the Dense
implementation defect merely because they are old.

The repository is no longer a complete mirror of the producer's old `outputs/`
directory. Future uploads must not restore withdrawn paths from a broad snapshot.
This cleanup does not delete the producer's local engineering records.

Derived weights remain subject to the upstream model's terms. Code is licensed
separately in the source repository.
