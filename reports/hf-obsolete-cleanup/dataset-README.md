---
pretty_name: Embedding Optimizer Study Analysis Artifacts
tags:
  - information-retrieval
  - dense-retrieval
  - optimization
  - reproducibility
---

# Embedding optimizer study analysis artifacts

The active project studies AdamW, Muon and NorMuon for DenseOn adaptation.
See the [source repository](https://github.com/qcznlp/embedding-optimizer-study)
for code, frozen protocols and the current progress. Model and optimizer states
are stored in the separate
[checkpoint repository](https://huggingface.co/qcz/embedding-optimizer-study-checkpoints).

## Active artifacts

`corrected-dense-no-packing-v1/` contains the current replication's incremental
analysis backup. Partial geometry coverage is not a complete optimizer comparison
and must not be presented as a final retrieval or mechanism result.

```bash
hf download qcz/embedding-optimizer-study-analysis-artifacts \
  --repo-type dataset \
  --local-dir /path/to/restore \
  --include 'corrected-dense-no-packing-v1/**'
```

## Withdrawn historical Dense results

At the owner's request, outputs of the invalidated historical Dense implementation
have been removed from the current tree. This includes its retrieval/validation
scores, affected representation and weight-space analyses, follow-up branches and
interventions, mixed summaries containing those results, and corresponding result
reports and identifiable Dense execution logs from the old project snapshot.
Do not cite their former values as results of the current study.

This deletion commit does **not** purge Git history. Withdrawn artifacts can still
exist in old revisions until a separately verified history cleanup occurs.

Shared `project/data/` files and historical `project/configs/` are retained;
configuration retention does not make an old experiment valid or current.
Separately identified LateOn artifacts are retained as out-of-scope legacy records,
without certifying their correctness. Independently encoded, untrained Dense BEIR
baseline exports are also retained; they are not trained optimizer results.

This repository is no longer a complete mirror of the producer's old `results/`
directory. Do not re-upload withdrawn files from broad local snapshots. Local
engineering records are not deleted by this HF-only operation.
