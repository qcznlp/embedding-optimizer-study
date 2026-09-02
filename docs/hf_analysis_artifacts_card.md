---
pretty_name: Embedding Optimizer Study Analysis Artifacts
tags:
  - information-retrieval
  - dense-retrieval
  - optimization
  - reproducibility
---

# Embedding optimizer study analysis artifacts

This dataset repository backs up the large derived artifacts used by the
AdamW, Muon, and NorMuon embedding-optimizer study. Its root mirrors the local
project's `results/` directory at source commit
`5d1a0fe9daca8b077137410125294cc32f2f3e48` (with any finalizer outputs recorded
by the accompanying project reports).

The separate model repository
[`qcz/embedding-optimizer-study-checkpoints`](https://huggingface.co/qcz/embedding-optimizer-study-checkpoints)
contains the complete `outputs/` checkpoint tree. The source repository
`qcznlp/embedding-optimizer-study` contains code, frozen configurations,
human-readable reports, and the paper.

## Contents

The artifact tree includes full-corpus and checkpoint-level retrieval outputs,
shared-start/query-disjoint validation data, exact-SVD summaries, common-state
representations, functional interventions, and spectral-transplant results.
It retains legacy LateOn paths for migration completeness, but the current
paper's formal scope is Dense retrieval only.

## Restore

```bash
hf download qcz/embedding-optimizer-study-analysis-artifacts \
  --repo-type dataset \
  --local-dir /path/to/embedding-optimizer-study/results \
  --exclude README.md
```

Do not treat directory names alone as evidence status. The project reports and
scope amendment identify formal, descriptive, diagnostic, legacy, and
quarantined artifacts and verify their source hashes.
