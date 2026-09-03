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

This repository is the complete checkpoint backup for the AdamW, Muon, and
NorMuon embedding-optimizer study based on `lightonai/DenseOn-unsupervised`.
It preserves the original contents of the project's local `outputs/` directory
so that training-state, optimizer-state, and five-stage weight-space analyses
can be resumed on another machine.

## Frozen source version

- Project repository: `qcznlp/embedding-optimizer-study`
- Source commit at backup start: `5d1a0fe9daca8b077137410125294cc32f2f3e48`
- Checkpoint directories: 251
- Local checkpoint/output footprint at backup start: approximately 389 GiB

The project repository contains the exact configurations, evaluation code,
scope amendment, reports, and paper. The checkpoint repository is intentionally
larger and contains both confirmatory artifacts and transparent diagnostic or
legacy runs.

## Directory guide

| Directory | Checkpoint directories | Role |
|---|---:|---|
| `confirmatory/` | 45 | Three optimizers, three fresh seeds, five stages; formal Dense evidence |
| `dense/` | 60 | Dense discovery sweep over four settings per optimizer, five stages |
| `hybrid-adamw/dense/` | 20 | Routing-matched AdamW controls |
| `short-branch/` | 45 | Shared-start Dense branch used for local/global and weight-space analysis |
| `late/` | 60 | Legacy LateOn runs, excluded by the Dense-only scope amendment |
| `hybrid-adamw/late/` | 2 | Incomplete legacy LateOn control, excluded from formal evidence |
| `quarantine/` | 13 | Invalidated diagnostic runs; never use as formal evidence |
| `smoke/`, `ddp-smoke/`, `stress-long/` | 6 | Engineering diagnostics only |

Each run retains its original Hugging Face/Trainer checkpoint layout. The
checkpoint directories include model weights and, where produced by the run,
optimizer, scheduler, RNG, scaler, and trainer state required for resumption.

## Restore

Install a recent `huggingface_hub` and download into the project's `outputs/`
directory:

```bash
hf download qcz/embedding-optimizer-study-checkpoints \
  --local-dir /path/to/embedding-optimizer-study/outputs \
  --exclude README.md
```

For a smaller transfer, use `--include`, for example:

```bash
hf download qcz/embedding-optimizer-study-checkpoints \
  --local-dir /path/to/embedding-optimizer-study/outputs \
  --include 'confirmatory/**' \
  --include 'short-branch/**'
```

Do not mix checkpoint directories across runs. Configuration and provenance
checks in the project repository bind each checkpoint to its run id, seed,
optimizer, learning rate, training corpus, and stage.

## Scope and interpretation

The current paper scope is Dense retrieval only. Legacy LateOn files are kept
solely because this is a complete machine-migration backup. The `quarantine/`
directory records invalidated experiments for auditability and must not be
treated as scientific evidence.

The base model and any derived weights remain subject to the terms stated by
the upstream model repository. Project code is licensed separately under the
Apache License 2.0 in the source repository.
