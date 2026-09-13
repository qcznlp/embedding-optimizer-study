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

The current DenseOn study compares AdamW, Muon and NorMuon from
[`lightonai/DenseOn-unsupervised`](https://huggingface.co/lightonai/DenseOn-unsupervised).
The [source repository and paper](https://github.com/qcznlp/embedding-optimizer-study/tree/b09b334973e69d6990e2811d5de5c1d0f471f170)
contain the complete results, protocols, analysis code and restoration tools.

## Restore the current scientific checkpoints

All **120 current scientific checkpoints** are backed up: sixty from twelve
500K-query primary runs and sixty from twelve crossed 50K-query continuations.
They are not 120 independently trained models. Each run retains five stages.

- [Primary checkpoint index and restoration instructions](https://github.com/qcznlp/embedding-optimizer-study/blob/b09b334973e69d6990e2811d5de5c1d0f471f170/docs/checkpoint-restoration.md):
  sixty checkpoints under `corrected-dense-correctness-v3/dense/`, with exact
  immutable revisions, paths and hashes in `docs/primary-v3-checkpoints.json`.
- [Continuation checkpoint locations and probe restoration](https://github.com/qcznlp/embedding-optimizer-study/blob/b09b334973e69d6990e2811d5de5c1d0f471f170/docs/continuation-probe-restoration.md):
  sixty additional checkpoints with their original eighteen-file manifests.

Use those immutable indexes, not a download of the entire mixed-history
repository or a substituted moving `main` revision. Preserve model, optimizer,
scheduler and rank-specific RNG files together. Do not combine runs or stages.
File-integrity verification is not a blanket guarantee of bitwise training
resumption on arbitrary hardware; the source repository documents tested cases.

The older `corrected-dense-no-packing-v1/` namespace is **not the current v3
primary experiment**. Retained historical files are not admitted as paper evidence
merely because they remain available here.

## Owner-approved withdrawal of obsolete results

This commit removes **3,790 explicitly reviewed old files** from the current
tree, following the owner's confirmation. These contain the invalidated Dense
discovery, confirmation, routing-control, shared-start and affected diagnostic
artifacts. Their former scores and the previous model card's formal-evidence
claims are withdrawn; do not use them as results of the current experiment.

The [exact reviewed deletion manifest](https://github.com/qcznlp/embedding-optimizer-study/blob/b09b334973e69d6990e2811d5de5c1d0f471f170/reports/hf-obsolete-cleanup/plan-v2.json)
has SHA-256 `1193dbc2a1a483cba79b3532b4ef47f66eb9198fefc7492ea4521a9d00f1d026`.
The commit changes only those paths and this README. It does **not** rewrite Git
history, purge LFS objects, delete local records or remove valid current files.
Old revisions may still expose withdrawn files; that availability is not validity.
Shared objects and original immutable restoration references remain preserved.

Separately retained LateOn files are out-of-scope legacy backups, not a new
correctness certification. Shared inputs and unknown/new namespaces are retained.
The repository is no longer a complete mirror of an old local `outputs/` folder;
future uploads must not restore withdrawn paths from broad historical snapshots.

Derived weights remain subject to the upstream model's terms. Code is licensed
separately in the source repository.
